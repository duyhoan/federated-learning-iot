import paho.mqtt.client as mqtt
import json
import threading


class EventHandler:

    def __init__(self, device_type, broker_ip, broker_port, device_index, device_ip_addr, published_port, parent_id=None, n_child_nodes=None):
        """
        :param device_type: the node type [client_node / edge_node / mid_node / cloud]
        :param broker_ip: IP address of MQTT broker
        :param broker_port: Port of MQTT broker
        :param device_index: the device index of node within its layer
        :param device_ip_addr: the socket IP address of device
        :param published_port: the published (exposed) port of device
        :param parent_id: the unique ID of device's parent node, in the form of {device-type_device-index}. For example,
         if the parent is the first edge node, then parent_id=edge_node_0. This param is NOT REQUIRED for cloud
        :param n_child_nodes: the number of children node of this device. This param is NOT REQUIRED for client nodes
        """
        self._device_type = device_type
        self._device_index = device_index
        self._parent_id = parent_id
        self._broker_ip = broker_ip
        self._broker_port = broker_port
        self._published_port = published_port
        self._device_ip_addr = device_ip_addr
        # this variable is used to count the number of aggregation signals, which come from it's child nodes.
        self._received_aggregate_signals = 0
        # this variable is used to count the number of child nodes, which have just passed an epoch in their training process
        self._epoch_pass = 0
        # this signal variable is used for signaling the main thread whenever the training session should start
        self.training_sess_start = threading.Event()
        if device_type != "client_node":
            # this signal variable is used for signaling the main thread whenever the parent node starts it's socket server.
            # The connection channel between a node and it's parent is established only after the socket server is started
            # This variable is NOT REQUIRED for cloud
            if device_type != "cloud":
                self.socket_server_start = threading.Event()
            # this signal variable is used for signaling the main thread whenever ALL of the child nodes have passed
            # one epoch. This variable is NOT REQUIRED for client node, since at the client node the passing of an epoch
            # can be computed obviously during the training process
            self.epoch_completion_signal = threading.Event()
            self.aggregation_after_epoch_end = threading.Event()
            # this signal variable is used for signaling the main thread whenever it's child node(s) trigger an
            # aggregation step
            self.aggregation_start = threading.Event()
            self._n_child_nodes = n_child_nodes
            self._n_connected_child_nodes = 0
            self.training_exit = threading.Event()
        else:
            self.start_new_epoch = threading.Event()
            self.socket_server_start = threading.Event()
        self._mqtt_client = None

    def _on_client_connect(self, mqtt_client, obj, flags, rc):
        # Subscribe to training start message
        self._mqtt_client.subscribe("training/start/{}".format(self._parent_id), qos=1)
        # Subscribe to a server socket start message to know whenever the socket server already started
        self._mqtt_client.subscribe("server_socket/start/{}".format(self._parent_id), qos=1)
        self._mqtt_client.subscribe("training/start_new_epoch/{}".format(self._parent_id), qos=1)

    def _on_higher_level_node_connect(self, mqtt_client, obj, flags, rc):
        # Subscribe for joining message to know whenever a node at the lower layer joins
        self._mqtt_client.subscribe("client/join/{}_{}".format(self._device_type, self._device_index), qos=2)
        # Subscribe for the training aggregation message to know whenever an aggregation step has started
        self._mqtt_client.subscribe("training/aggregate/{}_{}".format(self._device_type, self._device_index), qos=2)
        if self._device_type != "cloud":
            # Subscribe to training start message
            self._mqtt_client.subscribe("training/start/{}".format(self._parent_id), qos=1)
            # Subscribe to a server socket start message to know whenever the socket server already started
            self._mqtt_client.subscribe("server_socket/start/{}".format(self._parent_id), qos=1)
        # Subscribe to training exit message
        self._mqtt_client.subscribe("training/exit/{}_{}".format(self._device_type, self._device_index), qos=2)

    def _on_client_message(self, mqtt_client, obj, msg):
        if msg.topic == "training/start/{}".format(self._parent_id):
            if not self.training_sess_start.isSet():
                self.parent_sock_info = json.loads(msg.payload.decode())
                self.training_sess_start.set()
        elif msg.topic == "server_socket/start/{}".format(self._parent_id):
            if not self.socket_server_start.isSet():
                self.socket_server_start.set()
        elif msg.topic == "training/start_new_epoch/{}".format(self._parent_id):
            if not self.start_new_epoch.isSet():
                self.start_new_epoch.set()

    def _on_higher_level_node_message(self, mqtt_client, obj, msg):
        # Check if message is joining message coming from node at the lower level
        if msg.topic == "client/join/{}_{}".format(self._device_type, self._device_index):
            self._n_connected_child_nodes += 1
            if self._n_connected_child_nodes == self._n_child_nodes:
                if self._device_type != "cloud":
                    self.publish("client/join/{}".format(self._parent_id), b'', False)
                else:
                    self.publish("training/start/{}_{}".format(self._device_type, self._device_index),
                                 json.dumps({"sock_addr": self._device_ip_addr, "sock_port": self._published_port}),
                                 False)
                    self.training_sess_start.set()
        # Check if message is training start message
        elif msg.topic == "training/start/{}".format(self._parent_id):
            if not self.training_sess_start.isSet():
                self.publish("training/start/{}_{}".format(self._device_type, self._device_index),
                             json.dumps({"sock_addr": self._device_ip_addr, "sock_port": self._published_port}), False)
                self.parent_sock_info = json.loads(msg.payload.decode())
                self.training_sess_start.set()
        # Check if message is training aggregation message coming from node at the lower level. After receiving massage,
        # start the aggregation process.
        elif msg.topic == "training/aggregate/{}_{}".format(self._device_type, self._device_index):
            self._received_aggregate_signals += 1
            if int(msg.payload.decode()) == 1:
                self._epoch_pass += 1
                if self._epoch_pass == 1:
                    self.aggregation_after_epoch_end.set()
                if self._epoch_pass == self._n_child_nodes:
                    if not self.epoch_completion_signal.isSet():
                        self.epoch_completion_signal.set()
                    self._epoch_pass = 0
                    self.aggregation_after_epoch_end.clear()
            # Implement logic to ensure the aggregation is triggered only one at a single step
            if not self.aggregation_start.isSet():
                if self._received_aggregate_signals == 1:
                    self.aggregation_start.set()
            if self._received_aggregate_signals == self._n_child_nodes:
                self._received_aggregate_signals = 0
        elif msg.topic == "training/exit/{}_{}".format(self._device_type, self._device_index):
            if not self.training_exit.isSet():
                if self._device_type != "cloud":
                    self.publish("training/exit/{}".format(self._parent_id), b'', False)
                self.training_exit.set()
        elif msg.topic == "server_socket/start/{}".format(self._parent_id):
            if not self.socket_server_start.isSet():
                self.socket_server_start.set()

    def publish(self, channel, message, retain):
        try:
            self._mqtt_client.publish(topic=channel, payload=message, qos=2, retain=retain)
        except Exception as e:
            print(e)

    def initialize(self):
        try:
            self._mqtt_client = mqtt.Client(self._device_type+"_"+str(self._device_index))
            if self._device_type == "client_node":
                self._mqtt_client.on_message = self._on_client_message
                self._mqtt_client.on_connect = self._on_client_connect
            else:
                self._mqtt_client.on_message = self._on_higher_level_node_message
                self._mqtt_client.on_connect = self._on_higher_level_node_connect
        except TypeError:
            print('Connect to mqtt broker error')
            return

    def run(self):
        try:
            self._mqtt_client.connect(self._broker_ip, self._broker_port)
            self._mqtt_client.loop_start()
        except Exception as e:
            print('Error occurred in event handler: {}'.format(e))
