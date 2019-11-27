# Created by duyhoan at 9/15/19

# Utility module

# Helper libraries
import hashlib
import hmac
import json
import os
import pickle
import socket
import ssl
import numpy as np
import dns.reversename, dns.resolver
import tensorflow as tf

key = b'4C5jwen4wpNEjBeq1YmdBayIQ1oD'
hash_function = hashlib.sha1
hash_size = int(160 / 8)
error = b'err'
recv = b'rec'
signal = b'train'
buffer = 2048 * 2

key_path = '/usr/src/app/certs/server1.key'
cert_path = '/usr/src/app/certs/server1.pem'


class DatasetUtil:
    """
    Utility functions for partitioning dataset
    """

    @staticmethod
    def _parse_train_set_fashion_mnist(dataset_dist_config_dict, device_index, device_type):
        """
        Parsing training dataset configuration (applied for fashion-mnist dataset)
        :param dataset_dist_config_dict:
        :param device_index:
        :return:
        """
        train_set_dist = []
        train_set_config = dataset_dist_config_dict['train'][device_type].split(",")[device_index]
        for idx, class_config in enumerate(train_set_config.split(":")):
            class_config_arr = class_config.split("-")
            train_set_dist.append((class_config_arr[0], int(class_config_arr[1]), class_config_arr[2]))
        return train_set_dist

    @staticmethod
    def _parse_validate_set_fashion_mnist(dataset_dist_config_dict, device_index, device_type):
        """
        Parsing validation dataset configuration (applied for fashion-mnist dataset)
        :param dataset_dist_config_dict:
        :param device_index:
        :return:
        """
        validate_set_dist = []
        validate_set_config = dataset_dist_config_dict['validate'][device_type].split(",")[device_index]
        if validate_set_config != "all":
            for idx, class_config in enumerate(validate_set_config.split(":")):
                validate_set_dist.append(class_config)
        else:
            for idx, class_config in enumerate(range(0, 10)):
                validate_set_dist.append(class_config)
        return validate_set_dist

    def parse_fashion_mnist_dataset_dist(self, device_type, device_index, config_file):
        """
        Parsing data partitioning configuration for fashion-mnist dataset
        :param device_type:the node type [client_node / edge_node / mid_node / cloud]
        :param device_index:the device unique index, with regard to it's belonging layer
        :param config_file:the json file, which stores the configuration of class distribution at each node and each
        layer
        :return:
        """
        dataset_map = {}
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', 'config/{}'.format(config_file))),
                  'r') as f:
            dataset_dist_config_dict = json.load(f)
        if device_type == "client_node":
            dataset_map['train'] = self._parse_train_set_fashion_mnist(dataset_dist_config_dict, device_index,
                                                                       device_type)
        dataset_map['validate'] = self._parse_validate_set_fashion_mnist(dataset_dist_config_dict, device_index,
                                                                         device_type)
        return dataset_map

    @staticmethod
    def load_fashion_mnist_dataset(experiment, train_ds, device_type, node_index, n_nodes, dataset_map=None):
        """
        Function to initialize training and validation dataset
        :param experiment: experiment type
        :param train_ds: fashion-mnist training dataset
        :param device_type: device type [client_node/mid_node/edge_node/cloud]
        :param node_index: node's unique index
        :param n_nodes: total number of nodes
        :param dataset_map: dataset mapping configuration
        :return:
        """
        X_train = np.empty((0, 28, 28))
        y_train = np.empty((0,))
        X_valid = np.empty((0, 28, 28))
        y_valid = np.empty((0,))
        if device_type == "client_node":
            if "case_a" in experiment or experiment == "case_b" or "paper" in experiment:
                for idx, class_config in enumerate(dataset_map['train']):
                    start_index = class_config[1]
                    if class_config[2] != "end":
                        end_index = int(class_config[2])
                        X_train = np.concatenate(
                            (X_train, train_ds["x_class_{}".format(class_config[0])][start_index:end_index]),
                            axis=0)
                        y_train = np.concatenate(
                            (y_train, train_ds["y_class_{}".format(class_config[0])][start_index:end_index]),
                            axis=0)
                    else:
                        X_train = np.concatenate(
                            (X_train, train_ds["x_class_{}".format(class_config[0])][start_index:]), axis=0)
                        y_train = np.concatenate(
                            (y_train, train_ds["y_class_{}".format(class_config[0])][start_index:]), axis=0)
            elif "case_c" in experiment:
                X_train = np.array_split(train_ds[0], n_nodes)[node_index]
                y_train = np.array_split(train_ds[1], n_nodes)[node_index]
            if "case_a" in experiment or "case_c" in experiment:
                X_valid = X_train
                y_valid = y_train
            elif experiment == "case_b":
                X_valid = np.empty((0, 28, 28))
                y_valid = np.empty((0,))
                for idx, class_config in enumerate(dataset_map['validate']):
                    X_valid = np.concatenate(
                        (X_valid,
                         np.array_split(train_ds["x_class_{}".format(class_config)][:], n_nodes)[node_index]),
                        axis=0)
                    y_valid = np.concatenate(
                        (y_valid,
                         np.array_split(train_ds["y_class_{}".format(class_config)][:], n_nodes)[node_index]),
                        axis=0)
            return X_train, y_train, X_valid, y_valid
        else:
            if "case_a" in experiment:
                for idx, class_config in enumerate(dataset_map['validate']):
                    X_valid = np.concatenate((X_valid, train_ds["x_class_{}".format(class_config)][:]), axis=0)
                    y_valid = np.concatenate((y_valid, train_ds["y_class_{}".format(class_config)][:]), axis=0)
            elif experiment == "case_b":
                for idx, class_config in enumerate(dataset_map['validate']):
                    X_valid = np.concatenate(
                        (X_valid,
                         np.array_split(train_ds["x_class_{}".format(class_config)][:], n_nodes)[node_index]),
                        axis=0)
                    y_valid = np.concatenate(
                        (y_valid,
                         np.array_split(train_ds["y_class_{}".format(class_config)][:], n_nodes)[node_index]),
                        axis=0)
            elif "case_c" in experiment:
                X_valid = np.array_split(train_ds[0], n_nodes)[node_index]
                y_valid = np.array_split(train_ds[1], n_nodes)[node_index]
            return X_valid, y_valid


class DeviceUtil:
    """
    Utility functions for device related operations (e.g. get device id)
    """

    @staticmethod
    def get_device_id(device_type, device_ip_addr, client_group_start_index: int, n_higher_nodes: int):
        """
        1. Get the device unique index, with regard to it's belonging layer. For instance, the first edge node will have
        device index 0 (with regard to the edge layer). Similarly the first client node also has the device index 0
        regarding to client layer
        2. Get the node unique index, with regard to the whole system. For instance, if we have a simple architecture
        with 1 cloud, 2 edge node and 5 client, then the cloud at the top will be the starting point of indexing scheme,
        i.e. node index = 0. The first edge node at the edge layer will then have it's node index = 1, since it's
        counted right after the cloud. The indexing scheme will be counted following the top-down, left-to-right
        direction.
        :param device_type: the node type [client_node / edge_node / mid_node / cloud]
        :param device_ip_addr: the device's public socket IP address
        :param client_group_start_index: the starting index of client group, to which the client node belongs to. For
        example, the client group 0 will have the starting index 0. While the client group 1 will have the starting
        index 2, since the group 0 has 2 client nodes.
        :param n_higher_nodes: total number of higher level nodes. For example, with regard to the "simple" configuration,
        n_higher_nodes for each edge nodes will be 1 (since we have 1 cloud node above). Similarly, n_higher_nodes for
        each client nodes will be 3 (since we have 1 cloud nodes + 2 edge nodes above).
        :return: node index, device index
        """
        host_name = str(dns.resolver.query(dns.reversename.from_address(device_ip_addr), "PTR")[0])
        container_name = host_name.split(".")[0]
        if device_type == "cloud":
            return 0, 0
        elif device_type == "client_node":
            node_index = n_higher_nodes + client_group_start_index + int(container_name.split("_")[-1]) - 1
            device_index = client_group_start_index + int(container_name.split("_")[-1]) - 1
            return node_index, device_index
        else:
            node_index = n_higher_nodes + int(container_name.split("_")[-2])
            device_index = int(container_name.split("_")[-2])
            return node_index, device_index


class NetUtil:
    """
    Utility functions for network related operations (opening socket clients / servers)
    """

    @staticmethod
    def start_socket_client(public_ip, public_port):
        """ Creates a socket with ssl protection that will act as client.
            Returns:
                client_socket (socket): ssl secured socket.
         """
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ssl_client_socket = ssl.wrap_socket(client_socket,
                                            certfile=cert_path,
                                            keyfile=key_path,
                                            ssl_version=ssl.PROTOCOL_TLSv1)
        ssl_client_socket.connect((public_ip, public_port))
        return ssl_client_socket

    @staticmethod
    def start_socket_server(private_ip, private_port):
        """ Creates a socket with ssl protection that will act as server.
            Returns:
              sever_socket (socket): ssl secured socket.
         """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((private_ip, private_port))
        server_socket.listen()
        return server_socket

    @staticmethod
    def get_sock_ip_address():
        """
        Get the public socket IP address of device
        :return:
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ipv4_address = sock.getsockname()[0]
        sock.close()
        return ipv4_address


class FederatedLearningUtil:
    """
    Utility functions for performing federated learning
    """

    @staticmethod
    def assign_vars(placeholders, local_vars):
        """
        Utility to refresh local variables.
        :param placeholders: Placeholder tf variable
        :param local_vars: List of local variables
        :return: refresh_ops: The ops to assign value of global vars to local vars.
        """
        reassign_ops = []
        for var, fvar in zip(local_vars, placeholders):
            reassign_ops.append(tf.assign(var, fvar))
        return tf.group(*reassign_ops)

    def get_np_array(self, connection_socket):
        """
        Routine to receive a list of numpy arrays.
        :param connection_socket: connection_socket (socket): a socket with a connection already established.
        :return:
        """
        message = self._receiving_subroutine(connection_socket)
        received_weights = pickle.loads(message)
        return received_weights

    @staticmethod
    def send_np_array(arrays_to_send, connection_socket):
        """
        Routine to send a list of numpy arrays. It sends it as many time as necessary
        :param arrays_to_send: local updated mode weight as a list of numpy arrays
        :param connection_socket:a socket with a connection already established.
        :return:
        """
        serialized = pickle.dumps(arrays_to_send)
        signature = hmac.new(key, serialized, hash_function).digest()
        assert len(signature) == hash_size
        message = signature + serialized
        connection_socket.settimeout(240)
        try:
            connection_socket.sendall(message)
            while True:
                connection_socket.settimeout(240)
                try:
                    check = connection_socket.recv(len(error))
                    if check == error:
                        connection_socket.sendall(message)
                    elif check == recv:
                        connection_socket.settimeout(120)
                        break
                except socket.timeout:
                    break
            pass
        except socket.timeout:
            pass

    @staticmethod
    def _receiving_subroutine(connection_socket):
        """
        Subroutine inside get_np_array to receive a list of numpy arrays.
        If the sending was not correctly received it sends back an error message
        to the sender in order to try it again.
        :param connection_socket: a socket with a connection already established.
        :return:
        """
        timeout = 0.5
        while True:
            ultimate_buffer = b''
            connection_socket.settimeout(240)
            first_round = True
            while True:
                try:
                    receiving_buffer = connection_socket.recv(buffer)
                except socket.timeout:
                    break
                if first_round:
                    connection_socket.settimeout(timeout)
                    first_round = False
                if not receiving_buffer:
                    break
                ultimate_buffer += receiving_buffer

            pos_signature = hash_size
            signature = ultimate_buffer[:pos_signature]
            message = ultimate_buffer[pos_signature:]
            good_signature = hmac.new(key, message, hash_function).digest()

            if signature != good_signature:
                connection_socket.send(error)
                timeout += 0.5
                continue
            else:
                connection_socket.send(recv)
                connection_socket.settimeout(120)
                return message

    def receive_initialized_weight(self, client_socket, logger):
        logger.log_info('Receiving initialized weights from upper layer node \n')
        received_weight_array = self.get_np_array(client_socket)
        logger.log_info('Received initialized weights from upper layer node successfully \n')
        return received_weight_array

    def send_initialized_weight(self, server_socket, n_child_nodes, local_initialized_weight, logger):
        clients = []
        for _ in range(n_child_nodes):
            try:
                server_socket.settimeout(30)
                sock, address = server_socket.accept()
                connection_socket = ssl.wrap_socket(
                    sock,
                    server_side=True,
                    certfile=cert_path,
                    keyfile=key_path,
                    ssl_version=ssl.PROTOCOL_TLSv1)
                # logger.log_info('Connected: ' + address[0] + ':' + str(address[1]))
                clients.append((connection_socket, address))
            except socket.timeout:
                logger.log_info('Some workers could not connect while sending initalized weight')
                continue
            try:
                # logger.log_info('SENDING initialized weights to worker')
                self.send_np_array(local_initialized_weight, connection_socket)
                logger.log_info('SENT initialized weights to worker successfully')
            except (ConnectionResetError, BrokenPipeError):
                logger.log_info('Could not send to : ' + address[0] + ':' + str(address[1]) + ', fallen worker')
                connection_socket.close()
                clients.remove((connection_socket, address))

    def receive_weight_list_from_children(self, server_socket, n_child_nodes, logger):
        """
        Called at the beginning of each aggregation step.
        Edge node / Mid node / Cloud will try to receive the local updated weights from its children.
        :param logger: logger object
        :param server_socket: server socket object reference
        :param n_child_nodes: number of child nodes
        :return:
        """
        server_socket.listen(n_child_nodes)
        gathered_weights = []
        clients = []
        for _ in range(n_child_nodes):
            try:
                server_socket.settimeout(30)
                sock, address = server_socket.accept()
                connection_socket = ssl.wrap_socket(
                    sock,
                    server_side=True,
                    certfile=cert_path,
                    keyfile=key_path,
                    ssl_version=ssl.PROTOCOL_TLSv1)
                # logger.log_info('Connected: ' + address[0] + ':' + str(address[1]))
                clients.append((connection_socket, address))
            except socket.timeout:
                logger.log_info('Some workers could not connect while receiving local weights')
                continue
            try:
                received_weight = self.get_np_array(connection_socket)
                gathered_weights.append(received_weight)
                logger.log_info('Received from ' + address[0] + ':' + str(address[1]))
            except (ConnectionResetError, BrokenPipeError):
                logger.log_info('Could not receive from : '
                                + address[0] + ':' + str(address[1])
                                + ', fallen worker')
        return clients, gathered_weights

    def send_averaged_weight(self, clients, averaged_weight, logger):
        """
        Send averaged weight to children
        :param averaged_weight: averaged model weight to send
        :param logger: logger object
        :param clients: list of connected child node
        :return:
        """
        # Send weights to all trainable clients in its responsible group
        for client in clients:
            try:
                self.send_np_array(averaged_weight, client[0])
                client[0].close()
            except (ConnectionResetError, BrokenPipeError):
                logger.log_info('Fallen Worker: ' + client[1][0] + ':' + str(client[1][1]))
                try:
                    client[0].close()
                except socket.timeout:
                    logger.log_info("Socket timeout when sending weights to child node")

    @staticmethod
    def rearrange_and_average(weight_array):
        """
        In the received weight array, each list represents the weights of each worker. We want to gather in each list the
        weights of a single layer so that makes it easier to average them afterwards
        :param weight_array:
        :return:
        """
        rearranged_weights = []
        for i in range(len(weight_array[0])):
            rearranged_weights.append([elem[i] for elem in weight_array])
        for i, elem in enumerate(rearranged_weights):
            rearranged_weights[i] = np.mean(elem, axis=0)
        return rearranged_weights


class FileSystemUtil:
    @staticmethod
    def make_dirs(dir_path):
        """
        Utility function to make directory if it doesn't exist
        :param dir_path:
        :return:
        """
        try:
            os.makedirs(dir_path)
        except Exception as _:
            pass
