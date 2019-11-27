import tensorflow as tf
from library.util import FederatedLearningUtil, NetUtil, FileSystemUtil
import math
from models.cnn import SimpleCNN


class _FederatedLearningHook(tf.train.SessionRunHook):
    """
    Custom hook of TF MonitoredTrainingSession. This hook contain the core functions which operate federated learning
    processes, e.g. initialized weight propagation to every nodes after session created, sending local updated weights
    from client nodes, receiving local weights and averaging at the edge, mid (fog) and cloud levels, etc.
    """

    def __init__(self, device_type, device_index, parent_id, event_handler, private_ip, private_port, interval_steps,
                 federated_learning_util, logger, n_batches, n_child_nodes=None, public_ip=None, public_port=None):
        """
        Init function for federated learning hook class
        :param device_type: the node type [client_node / edge_node / mid_node / cloud]
        :param device_index: the device index within its layer, e.g.first edge node will have device index 0 (with
        regard to edge layer. Similarly the first client node also has the device index 0 regarding to client layer
        :param parent_id: the unique ID of device's parent node, in the form of {device-type_device-index}. For example,
         if the parent is the first edge node, then parent_id=edge_node_0
        :param event_handler: The event handler object, which is responsible for communication and signaling between
        nodes, e.g. sending signal whenever an aggregation step comes
        :param private_ip: the socket IP address of device
        :param private_port: the socket port of device
        :param interval_steps: the interval steps, after which an aggregation step happens
        :param federated_learning_util: the federated learning utility object reference
        :param n_child_nodes: number of child nodes
        :param public_ip: the published socket IP address of parent node
        :param public_port: the published socket port of parent node
        """
        self._federated_learning_util = federated_learning_util
        self._net_util = NetUtil()
        self._device_type = device_type
        self._device_index = device_index
        self._parent_id = parent_id
        self._private_ip = private_ip
        self._private_port = private_port
        self._public_ip = public_ip
        self._public_port = public_port
        self._n_batches = n_batches
        self._placeholders = []
        self._n_child_nodes = n_child_nodes
        self._interval_steps = interval_steps
        self._event_handler = event_handler
        self._logger = logger
        if device_type != "client_node":
            self._server_socket = self._net_util.start_socket_server(private_ip, private_port)

    def _client_after_create_session(self, session):
        """
        After the TF training session is created, this function will be called on client nodes. The goal is to receive
        the initialized weights from its parent node (here is from its connected edge node)
        :param session: current session reference
        :return:
        """
        self._event_handler.socket_server_start.wait()
        client_socket = self._net_util.start_socket_client(self._public_ip, self._public_port)
        received_weight = self._federated_learning_util.receive_initialized_weight(client_socket, self._logger)
        self._update_local_vars(received_weight, session)
        self._event_handler.socket_server_start.clear()

    def _higher_level_node_after_create_session(self, session):
        """
        After the TF training session is created, this function will be called on the higher level nodes (i.e. any nodes
        which is at the higher level than the client nodes, e.g. edge node, mid node or cloud). The following process
        will be performed, based on the type of node:
        - if the node is edge node or mid node:
            - it will try to receive the initialized weights from its parent node. The parent node could be a higher mid
             node or cloud.
            - Once it has received the initialized weight successfully, it sends the received weight to child nodes, so
            that they all start with the same initial ones.
        - if the node is cloud:
            - it sends directly the initialized weight to its child nodes. Important notes here: Because we are using
            the TensorFlow MonitoredTrainingSession, all of the variables, including model weights, were initialized
            randomly along with the session creation process
        :param session: current session reference
        :return:
        """
        # Check if this node is not cloud. If yes, try to receive the inititalized weights from higher level one, to
        # which it connect
        if self._device_type != "cloud":
            self._event_handler.socket_server_start.wait()
            client_socket = self._net_util.start_socket_client(self._public_ip, self._public_port)
            local_initialized_weight = self._federated_learning_util.receive_initialized_weight(client_socket,
                                                                                                self._logger)
            self._event_handler.socket_server_start.clear()
            # Update local vars to received initialized weight
            self._update_local_vars(local_initialized_weight, session)
        else:
            local_initialized_weight = session.run(tf.trainable_variables())
        # Send the (received) initialized weights to it's children. Note: before starting sending weights via socket,
        # the node will publish a server socket starting message to notify its children that they are ready to send data
        self._event_handler.publish("server_socket/start/{}_{}".format(self._device_type, self._device_index), b'',
                                    False)
        self._federated_learning_util.send_initialized_weight(self._server_socket, self._n_child_nodes,
                                                              local_initialized_weight, self._logger)

    def begin(self):
        self._placeholders = []
        for var in tf.trainable_variables():
            self._placeholders.append(tf.placeholder_with_default(var, var.shape,
                                                                  name="%s/%s" % ("FedAvg",
                                                                                  var.op.name)))
        self._update_local_vars_op = self._federated_learning_util.assign_vars(self._placeholders,
                                                                               tf.trainable_variables())
        if self._device_type == "client_node":
            self._global_step = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0]

    def after_create_session(self, session, coord):
        if self._device_type == "client_node":
            self._client_after_create_session(session)
        else:
            self._higher_level_node_after_create_session(session)

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        """
        The processing logic in this function will ONLY TAKE EFFECT AT THE CLIENT NODE. This is because only at client
        node, we have a real training session, in which the model is trained directly on data.
        :param run_context:
        :param run_values:
        :return:
        """
        if self._device_type == "client_node":
            session = run_context.session
            global_step = session.run(self._global_step)
            if global_step % self._interval_steps == 0 and global_step != 0:
                # Signaling upper layer node for the upcoming aggregation step
                self._event_handler.publish("training/aggregate/{}".format(self._parent_id), 0, False)
                local_updated_weight = session.run(tf.trainable_variables())
                self.send_and_update_local_weight(local_updated_weight, session)
                self._logger.log_info('Weights succesfully updated on client, iter {}'.format(global_step))

    def send_and_update_local_weight(self, local_weight, session):
        client_socket = self._net_util.start_socket_client(self._public_ip, self._public_port)
        self._federated_learning_util.send_np_array(local_weight, client_socket)
        received_weight = self._federated_learning_util.get_np_array(client_socket)
        self._update_local_vars(received_weight, session)
        client_socket.close()

    def combine_nets(self, session, parent_id, agg_step, epoch_pass: int = 0):
        """
        Function to combine multiple models receiving from child nodes. The averaged model will then be sent back to
        children for updating
        :param epoch_pass:
        :param session: current session reference
        :param parent_id: the unique ID of device's parent node
        :param agg_step: the current number aggregation steps, which has been completed. This variable is only applied
        at the edge nodes, mid nodes or cloud, since the aggregation process will be performed at these node types.
        :return:
        """
        # Clear the aggregation signaling event, otherwise next time when the child node send another message to start
        # another aggregation step, the event won't be fired since the old event is still not cleared
        self._event_handler.aggregation_start.clear()

        # Connect and receive weights sending from its children
        clients, received_weight_arr = self._federated_learning_util.receive_weight_list_from_children(
            self._server_socket, self._n_child_nodes, self._logger)

        # Rearrange weight arrays in the received weight list, then average them to get the newly aggregated local
        # weight
        aggregated_local_weight = self._federated_learning_util.rearrange_and_average(received_weight_arr)

        # Check if the server node is either edge_node or mid_node. If yes, forward the averaged weight to parent
        # node for averaging
        if self._device_type != "cloud":
            if agg_step % self._interval_steps == 0 or epoch_pass == 1:
                # Signaling upper layer node for the upcoming aggregation step
                self._event_handler.publish("training/aggregate/{}".format(parent_id), epoch_pass, False)
                if epoch_pass == 1:
                    self._event_handler.socket_server_start.wait()
                # Start sending received weights from connect nodes to the upper level node
                client_socket = self._net_util.start_socket_client(self._public_ip, self._public_port)
                self._logger.log_info('Sending weights from intermediate server node, iter {}'.format(agg_step))
                self._federated_learning_util.send_np_array(aggregated_local_weight, client_socket)
                # Try to receive the averaged weights from upper level node
                received_weight = self._federated_learning_util.get_np_array(client_socket)
                self._logger.log_info('Received weights from upper layer node successfully, iter {}'.format(agg_step))
                self._federated_learning_util.send_averaged_weight(clients, received_weight, self._logger)
            else:
                self._federated_learning_util.send_averaged_weight(clients, aggregated_local_weight, self._logger)
            self._update_local_vars(aggregated_local_weight, session)
            if epoch_pass == 1:
                self._event_handler.socket_server_start.clear()
        else:
            self._federated_learning_util.send_averaged_weight(clients, aggregated_local_weight, self._logger)
            self._update_local_vars(aggregated_local_weight, session)

    def _update_local_vars(self, received_weight, session):
        """
        Assign newly received weight to local variables
        :param received_weight:
        :param session:
        :return:
        """
        feed_dict = {}
        for placeholder, received_weight in zip(self._placeholders, received_weight):
            feed_dict[placeholder] = received_weight
        session.run(self._update_local_vars_op, feed_dict=feed_dict)


class TrainingSession(tf.train.SessionRunHook):
    """
    Main class for a training session. In the implementation we are using the TF MonitoredTrainingSession
    """
    def __init__(self, net_config, learning_rate, device_type, device_index, event_handler, parent_id, n_epochs,
                 batch_size, X_train, y_train, X_valid, y_valid, test_ds, interval_steps, private_ip, private_port,
                 n_child_nodes, experiment, logger):
        """
        :param net_config: the type of Deep learning model to be used. E.g. a SimpleCNN network [simple],
        or Inception Network [inception], etc.
        :param learning_rate: the learning rate applied when training
        :param device_type: device type [client_node/mid_node/edge_node/cloud]
        :param device_index: the device unique index, with regard to it's belonging layer
        :param event_handler: the event handler object, used to send / receive information or signaling messages from / to
        other nodes
        :param parent_id: the unique ID of device's parent node, in the form of {device-type_device-index}. For example,
         if the parent is the first edge node, then parent_id=edge_node_0
        :param n_epochs: number of epochs to train in
        :param batch_size: mini-batch size
        :param X_train: train images numpy array
        :param y_train: train labels numpy array
        :param X_valid: validation images numpy array, used to measure accuracy during training. In fact, with regard to
        some specific experiments, X_valid is the same dataset as X_train
        :param y_valid: validation labels numpy array, used to measure accuracy durin training. In fact, with regard to
        some specific experiments, y_valid is the same dataset as y_train
        :param test_ds: test dataset, used for final accuracy testing after the training is finished
        :param interval_steps: the interval steps, after which an aggregation step happens
        :param private_ip: the socket IP address of device
        :param private_port: the socket port of device
        :param n_child_nodes: number of child nodes
        :param experiment: the type of experiment to run. [case_a/case_b/random]
        :param logger: the logger object
        """
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._interval_steps = interval_steps
        self._n_child_nodes = n_child_nodes
        self._device_type = device_type
        self._device_index = device_index
        self._event_handler = event_handler
        self._lr = learning_rate
        self._net_config = net_config
        self._experiment = experiment
        self._X_train = X_train
        self._y_train = y_train
        self._X_valid = X_valid
        self._y_valid = y_valid
        self._private_ip = private_ip
        self._private_port = private_port
        if self._device_type == "client_node":
            self._n_batches = math.ceil(self._X_train.shape[0] / self._batch_size)
        else:
            self._n_batches = None
        self._test_ds = test_ds
        self._parent_id = parent_id
        self._logger = logger
        self._checkpoint_dir = "device/model_checkpoint/{}_{}_{}".format(self._device_type, self._device_index, experiment)
        FileSystemUtil().make_dirs(self._checkpoint_dir)
        self._checkpoint_path = "{}/trained_model".format(self._checkpoint_dir)

    @staticmethod
    def _init_net(image, label, net_config, learning_rate, global_step, n_classes):
        if net_config == "simple":
            return SimpleCNN(image=image, label=label, learning_rate=learning_rate, global_step=global_step,
                             n_output_classes=n_classes)

    def _init_tf_data_iterator(self):
        with tf.variable_scope("dataset"), tf.device('/cpu:0'):
            self._handle = tf.placeholder(tf.string, shape=[], name='handle')
            if self._device_type == "client_node":
                ds_train = tf.data.Dataset.from_tensor_slices(
                    (self._X_train.astype("float32") / 255, self._y_train.astype("int8")))
                ds_train = ds_train.shuffle(self._X_train.shape[0], reshuffle_each_iteration=True).batch(
                    self._batch_size)
            ds_valid = tf.data.Dataset.from_tensor_slices(
                (self._X_valid.astype("float32") / 255, self._y_valid.astype("int8")))
            ds_valid = ds_valid.shuffle(1, reshuffle_each_iteration=True).batch(self._X_valid.shape[0])
            if self._device_type == "client_node":
                iterator = tf.data.Iterator.from_string_handle(self._handle, ds_train.output_types,
                                                               ds_train.output_shapes)
                self._train_iterator = ds_train.make_initializable_iterator()
            else:
                iterator = tf.data.Iterator.from_string_handle(self._handle, ds_valid.output_types,
                                                               ds_valid.output_shapes)
            self._valid_iterator = ds_valid.make_initializable_iterator()
            if "paper" in self._experiment:
                ds_test = tf.data.Dataset.from_tensor_slices(
                    (self._test_ds[0].astype("float32") / 255, self._test_ds[1].astype("int8")))
                ds_test = ds_test.shuffle(1, reshuffle_each_iteration=True).batch(self._test_ds[0].shape[0])
                self._test_iterator = ds_test.make_initializable_iterator()
            else:
                self._test_iterator = None
            return iterator.get_next()

    def _get_sock_server_info(self):
        return self._event_handler.parent_sock_info['sock_addr'], self._event_handler.parent_sock_info['sock_port']

    def _measure_accuracy(self, epoch, session, test_handle):
        if self._device_type != "client_node":
            self._event_handler.epoch_completion_signal.clear()
        if self._test_iterator is not None:
            session.run(self._test_iterator.initializer)
        else:
            session.run(self._valid_iterator.initializer)
        acc = session.run(self._net_model.accuracy, feed_dict={self._handle: test_handle})
        self._logger.log_training_acc(epoch, acc)
        self._trained_model_saver.save(session._tf_sess(), self._checkpoint_path)

    def train_net(self, mode):
        """
        Train the model
        :param mode: if mode == "train": train model from scratch. If mode == "train_from_checkpoint": train model from
        previous checkpoint
        :return:
        """
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        image, label = self._init_tf_data_iterator()
        global_step = tf.train.get_or_create_global_step()
        if self._device_type != "client_node":
            agg_step = 0
        self._net_model = self._init_net(image, label, self._net_config, self._lr, global_step, 10)
        if self._device_type == "client_node":
            self._event_handler.publish("client/join/{}".format(self._parent_id), b'', False)
        # Wait for starting training session
        self._event_handler.training_sess_start.wait()
        self._logger.log_info("Start training session \n")
        if self._device_type != "cloud":
            public_ip, public_port = self._get_sock_server_info()
        else:
            public_ip, public_port = None, None
        federated_learning_util = FederatedLearningUtil()
        self._trained_model_saver = tf.train.Saver()
        federated_hook = _FederatedLearningHook(self._device_type, self._device_index, self._parent_id,
                                                self._event_handler, self._private_ip, self._private_port,
                                                self._interval_steps, federated_learning_util,
                                                self._logger, self._n_batches, self._n_child_nodes, public_ip,
                                                public_port)
        with tf.name_scope('monitored_session'):
            with tf.train.MonitoredTrainingSession(
                    hooks=[federated_hook],
                    config=sess_config) as mon_sess:
                ckpt = tf.train.get_checkpoint_state(self._checkpoint_dir)
                if mode == "train_from_checkpoint" and ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    self._trained_model_saver.restore(mon_sess, ckpt.model_checkpoint_path)
                if self._device_type == "client_node":
                    train_handle = mon_sess.run(self._train_iterator.string_handle())
                if self._test_iterator is not None:
                    test_handle = mon_sess.run(self._test_iterator.string_handle())
                else:
                    test_handle = mon_sess.run(self._valid_iterator.string_handle())
                # Operations run at client nodes during training session
                if self._device_type == "client_node":
                    for epoch in range(self._n_epochs):
                        mon_sess.run(self._train_iterator.initializer)
                        while True:
                            try:
                                mon_sess.run(self._net_model.train_op, feed_dict={self._handle: train_handle})
                            except tf.errors.OutOfRangeError:
                                break
                        self._measure_accuracy(epoch + 1, mon_sess, test_handle)
                        # Signaling upper layer node for the upcoming aggregation step
                        self._event_handler.publish("training/aggregate/{}".format(self._parent_id), 1, False)
                        self._event_handler.socket_server_start.wait()
                        federated_hook.send_and_update_local_weight(mon_sess.run(tf.trainable_variables()), mon_sess)
                        self._event_handler.socket_server_start.clear()
                        self._event_handler.start_new_epoch.wait()
                        self._event_handler.start_new_epoch.clear()
                else:
                    epoch = 0
                    while True:
                        # Continuously wait for an aggregation step until all clients have finished the training
                        if self._event_handler.aggregation_start.isSet():
                            agg_step += 1
                            if self._event_handler.aggregation_after_epoch_end.isSet():
                                self._event_handler.epoch_completion_signal.wait()
                                epoch += 1
                                self._event_handler.publish(
                                    "server_socket/start/{}_{}".format(self._device_type, self._device_index), b'',
                                    False)
                                federated_hook.combine_nets(mon_sess, self._parent_id, agg_step, 1)
                                self._measure_accuracy(epoch, mon_sess, test_handle)
                                self._event_handler.epoch_completion_signal.clear()
                                if self._device_type == "edge_node":
                                    self._event_handler.publish(
                                        "training/start_new_epoch/{}_{}".format(self._device_type, self._device_index),
                                        b'', False)
                            else:
                                federated_hook.combine_nets(mon_sess, self._parent_id, agg_step, 0)
                        # Check if all clients are done with training. If yes, exit the loop and perform final accuracy
                        # measurement for the last epoch if it was not done.
                        if self._event_handler.training_exit.isSet():
                            if epoch < self._n_epochs:
                                epoch += 1
                                self._measure_accuracy(epoch, mon_sess, test_handle)
                            break
                        else:
                            continue
        if self._device_type == "client_node":
            self._event_handler.publish("training/exit/{}".format(self._parent_id), b'', False)

    def test_net(self):
        self._logger.log_info("--- Begin Evaluation ---")
        # Reset graph to clear any ops stored in other devices
        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(self._checkpoint_path + '.meta', clear_devices=True)
            saver.restore(sess, self._checkpoint_path)
            self._logger.log_info('Model restored \n')
            graph = tf.get_default_graph()
            handle = graph.get_tensor_by_name('dataset/handle:0')
            accuracy = graph.get_tensor_by_name('accuracy/accuracy_metric:0')
            if "overall_final_testing" not in self._experiment:
                for data_class in range(0, 10):
                    ds_test = tf.data.Dataset.from_tensor_slices(
                        (self._test_ds[2]['x_class_{}'.format(data_class)][:].astype(
                            "float32") / 255, self._test_ds[2]['y_class_{}'.format(data_class)][:].astype("int8")))
                    ds_test = ds_test.shuffle(1, reshuffle_each_iteration=True).batch(
                        self._test_ds[2]['x_class_{}'.format(data_class)].shape[0])
                    # Make new test iterator
                    iterator = ds_test.make_initializable_iterator()
                    sess.run(iterator.initializer)
                    test_handle = sess.run(iterator.string_handle())
                    acc = sess.run(accuracy, feed_dict={handle: test_handle})
                    self._logger.log_test_acc(data_class, acc)
            else:
                ds_test = tf.data.Dataset.from_tensor_slices(
                    (self._test_ds[0].astype("float32") / 255, self._test_ds[1].astype("int8")))
                ds_test = ds_test.shuffle(1, reshuffle_each_iteration=True).batch(self._test_ds[0].shape[0])
                # Make new test iterator
                iterator = ds_test.make_initializable_iterator()
                sess.run(iterator.initializer)
                test_handle = sess.run(iterator.string_handle())
                acc = sess.run(accuracy, feed_dict={handle: test_handle})
                self._logger.log_test_acc("all", acc)
