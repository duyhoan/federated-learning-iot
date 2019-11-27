# Part of the source code is based on the code base developed by coMind in 2018
# GitHub link: https://github.com/coMindOrg/federated-averaging-tutorials
# ==============================================================================

# Helper libraries
import argparse
from library.util import DeviceUtil, NetUtil, DatasetUtil, FileSystemUtil
from event_handler import EventHandler
from training import TrainingSession
from logger import Logger
import json
from library.datasets import *

# Disable usage of GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Disable tensorflow warnings and log information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'


def _get_parser():
    """
    Get the arguments passed to python command, which is executed when a node instance starts. The python command
    with it's arguments will be generated automatically along with the docker-compose file
    :return:the argument parser object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--device_type', type=str, required=True,
                        help='Device type [client_node/edge_node/mid_node/cloud]')

    parser.add_argument('--log_dir', type=str, required=True, help='Log directory path')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory path')
    parser.add_argument('--dataset', type=str, required=True, help="Dataset [mnist/fashion-mnist]")
    parser.add_argument('--n_nodes', type=int, required=True, help="Total number of nodes")
    parser.add_argument('--net_config', type=str, required=True,
                        help="Deep Neural Network to be used for classification [simple/inception]")

    parser.add_argument('--n_child_nodes', type=int,
                        help="Number of child nodes [sum(edge_nodes)/sum(mid_nodes)/sum(client_nodes)]. Only required for edge nodes / mid nodes / cloud")
    parser.add_argument('--client_group_start_index', type=int,
                        help="Start index number for indexing the clients in the group. Only required for client nodes")
    parser.add_argument('--n_higher_nodes', type=int,
                        help="Number of higher level nodes [sum(edge_nodes, mid_nodes, cloud)]")

    parser.add_argument('--dataset_config_file', type=str, required=True,
                        help="JSON file name to be processed. This file is used for any nodes reading the training / testing dataset configuration")

    parser.add_argument('--architecture_setting', required=True, type=str,
                        help="Complexity of architecture used to run. [simple/complex]")
    parser.add_argument('--experiment', required=True, type=str,
                        help="Type of experiment to run. [case_a/case_b/case_c]")

    parser.add_argument('--lr', type=float, required=True, help="Learning rate")
    parser.add_argument('--epochs', type=int, required=True, help="Epochs")
    parser.add_argument('--batch_size', type=int, required=True, help="Mini-batch size")

    parser.add_argument('--published_port', type=int,
                        help="Node's published port, used for opening socket server. Only required for edge/mid node and cloud")

    parser.add_argument('--parent', type=str,
                        help="ID of the higher level node, to which the current node connect. Not required for cloud")

    parser.add_argument('--mqtt_broker_ip', type=str, required=True,
                        help="IP of the MQTT broker, to which all nodes in the network connect. The MQTT broker is used for simple message transmission between nodes")
    parser.add_argument('--mqtt_broker_port', type=int, required=True,
                        help="Port of the MQTT broker")

    parser.add_argument('--interval_steps', type=int, default=100,
                        help="How many mini-batches should be processed before performing aggregation step. Not required for cloud")

    return parser


def _mkdirs(dir_path):
    """
    Utility function to make directory if it doesn't exist
    :param dir_path:
    :return:
    """
    try:
        os.makedirs(dir_path)
    except Exception as _:
        pass


def _init_dataset(dataset, experiment, device_type, dataset_config_file, data_dir, node_index, device_index, n_nodes):
    """
    Initialize the dataset for both training and testing
    :param dataset: the dataset chosen for experiment [fashion-mnist]
    :param experiment: the type of experiment to run. [case_a/case_b/random]
    :param device_type: device type [client_node/mid_node/edge_node/cloud]
    :param dataset_config_file: the name of json config file, which defines the dataset distribution configuration
    :param data_dir: the device's data directory. This directory is used to store the
    :param node_index: the node unique index, with regard to the whole system
    :param device_index: the device unique index, with regard to it's belonging layer
    :param n_nodes: total number of nodes in the system, i.e. sum of client nodes, edge nodes, mid nodes and a single cloud
    :return:
    """
    dataset_util = DatasetUtil()
    dataset_map = None
    if dataset == "fashion-mnist":
        fashion_mnist = FashionMNIST(data_dir, experiment, device_type, device_index)
        train_ds =fashion_mnist.build_train_dataset()
        test_ds = fashion_mnist.build_test_dataset()
        if "case_c" not in experiment:
            print("Parsing dataset distribution configuration \n")
            dataset_map = dataset_util.parse_fashion_mnist_dataset_dist(device_type=device_type,
                                                                        device_index=device_index,
                                                                        config_file=dataset_config_file)
        if device_type == "client_node":
            X_train, y_train, X_valid, y_valid = dataset_util.load_fashion_mnist_dataset(experiment=experiment,
                                                                                         train_ds=train_ds,
                                                                                         device_type=device_type,
                                                                                         node_index=node_index,
                                                                                         n_nodes=n_nodes,
                                                                                         dataset_map=dataset_map)
            return X_train, y_train, X_valid, y_valid, test_ds
        else:
            X_valid, y_valid = dataset_util.load_fashion_mnist_dataset(experiment=experiment, train_ds=train_ds,
                                                                       device_type=device_type, node_index=node_index,
                                                                       n_nodes=n_nodes, dataset_map=dataset_map)
            return None, None, X_valid, y_valid, test_ds


def _init_event_handler(args, device_index, device_ip_addr):
    """
    Initialize the event handler for communication to other nodes (parent nodes and child nodes)
    :param args:
    :param device_index:
    :param device_ip_addr:
    :return:
    """
    event_handler = EventHandler(args.device_type, args.mqtt_broker_ip, args.mqtt_broker_port, device_index,
                                 device_ip_addr, args.published_port,
                                 args.parent, args.n_child_nodes)
    event_handler.initialize()
    event_handler.run()
    return event_handler


def run_exp(mode="train"):
    """
    Main entry point when a node instance (a docker container) starts
    :return:
    """
    parser = _get_parser()
    # Getting the argument dictionary from parser object
    args = parser.parse_args()

    if args.n_higher_nodes is None:
        parser.error("--n_higher_nodes is required.")
    if args.device_type != "client_node" and (args.published_port is None or args.n_child_nodes is None):
        parser.error("--published_port and --n_child_nodes are required.")
    if args.device_type == "client_node" and args.client_group_start_index is None:
        parser.error("--client_group_start_index is required.")
    if args.device_type != "cloud" and (args.parent is None or args.interval_steps is None):
        parser.error("--parent and --interval_steps and --n_higher_nodes are required")

    info_log_dir = os.path.join(args.log_dir, "info", args.architecture_setting, args.experiment)
    FileSystemUtil().make_dirs(info_log_dir)
    train_log_dir = os.path.join(args.log_dir, "train_log", args.architecture_setting, args.experiment)
    FileSystemUtil().make_dirs(train_log_dir)
    test_log_dir = os.path.join(args.log_dir, "test_log", args.architecture_setting, args.experiment)
    FileSystemUtil().make_dirs(test_log_dir)

    print("Get device IP address \n")
    device_ip_addr = NetUtil().get_sock_ip_address()

    print("Get the device unique indexes \n")
    node_index, device_index = DeviceUtil().get_device_id(device_type=args.device_type,
                                                          device_ip_addr=device_ip_addr,
                                                          client_group_start_index=args.client_group_start_index,
                                                          n_higher_nodes=args.n_higher_nodes)

    # Dump experiment arguments before running
    with open(os.path.join(args.log_dir,
                           'experiment_arguments_{}_{}.json'.format(args.architecture_setting, args.experiment)),
              'w') as f:
        json.dump(str(args), f)

    print("Initialize logger \n")
    logger = Logger(args.experiment, args.architecture_setting, args.device_type, device_index, info_log_dir,
                    train_log_dir, test_log_dir)

    logger.log_info("Initialize event handler \n")
    event_handler = _init_event_handler(args, device_index, device_ip_addr)

    logger.log_info("Load dataset \n")
    X_train, y_train, X_valid, y_valid, test_ds = _init_dataset(dataset=args.dataset, experiment=args.experiment,
                                                                device_type=args.device_type,
                                                                dataset_config_file=args.dataset_config_file,
                                                                node_index=node_index, device_index=device_index,
                                                                data_dir=args.data_dir, n_nodes=args.n_nodes)

    train_sess = TrainingSession(args.net_config, args.lr, args.device_type, device_index, event_handler, args.parent,
                                 args.epochs, args.batch_size,
                                 X_train, y_train, X_valid, y_valid, test_ds, args.interval_steps, device_ip_addr,
                                 args.published_port, args.n_child_nodes, args.experiment, logger)
    if mode == "train" or mode == "train_from_checkpoint":
        train_sess.train_net(mode)
    train_sess.test_net()


if __name__ == '__main__':
    run_exp()
