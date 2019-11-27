import argparse
import json
import random
import yaml
from fractions import Fraction as frac


def gen_config_files(meta_config_file, dataset_config_file_name):
    with open("{}.json".format(meta_config_file), 'r') as f:
        config_dict = json.load(f)
    for exp_config in config_dict['experiment']:
        experiment_name = exp_config['name']
        architecture_config = exp_config['architecture']
        compose_ouput = []
        list_dataset_config_file = []
        for dataset_config in config_dict['class_distribution'][architecture_config]:
            list_dataset_config_file.append(
                _generate_data_distribution_file(experiment_name, exp_config['num_classes_per_client'],
                                                 config_dict['architecture'], config_dict['parameters']['dataset'],
                                                 architecture_config, dataset_config, dataset_config_file_name,
                                                 config_dict['version']))
        _generate_compose_file(exp_config, config_dict['parameters'], compose_ouput, config_dict['architecture'], architecture_config,
                               list_dataset_config_file, exp_config['mqtt_broker_ip'], exp_config['mqtt_broker_port'],
                               config_dict['version'])


def _randomly_distribute_class(experiment_name, num_classes_per_client, client_group_index, distributed_classes,
                               n_training_records_per_class, num_clients):
    clients_training_class_list = []
    clients_validate_class_list = []
    n_distributed_classes = len(distributed_classes)
    for client_idx, _ in enumerate(range(num_clients)):
        client_training_class_id_str = ""
        if "case_a" in experiment_name:
            client_validating_class_id_str = ""
        else:
            client_validating_class_id_str = "all"
        start = client_idx * (n_training_records_per_class // num_clients)
        if n_distributed_classes == num_classes_per_client:
            if client_idx < num_clients - 1:
                end = start + n_training_records_per_class // num_clients
            else:
                end = "end"
            for class_idx, class_id in enumerate(distributed_classes):
                if class_idx < n_distributed_classes - 1:
                    client_training_class_id_str += "{}-{}-{}".format(class_id, start, end) + ":"
                    if experiment_name == "case_a":
                        client_validating_class_id_str += class_id + ":"
                else:
                    client_training_class_id_str += "{}-{}-{}".format(class_id, start, end)
                    if experiment_name == "case_a":
                        client_validating_class_id_str += class_id
        elif n_distributed_classes > num_classes_per_client:
            if n_distributed_classes > num_clients:
                if n_distributed_classes % num_clients == 0:
                    random_sampled_class_ids = random.sample(distributed_classes, int(num_classes_per_client))
                    for class_idx, class_id in enumerate(random_sampled_class_ids):
                        if class_idx < len(random_sampled_class_ids) - 1:
                            client_training_class_id_str += "{}-{}-{}".format(class_id, 0, "end") + ":"
                            if experiment_name == "case_a":
                                client_validating_class_id_str += class_id + ":"
                        else:
                            client_training_class_id_str += "{}-{}-{}".format(class_id, 0, "end")
                            if experiment_name == "case_a":
                                client_validating_class_id_str += class_id
                        distributed_classes.remove(class_id)
            elif n_distributed_classes == num_clients:
                if "paper" in experiment_name:
                    class_id = distributed_classes[client_idx]
                    if client_group_index == 0:
                        client_training_class_id_str += "{}-{}-{}".format(class_id, 0, 3000)
                    else:
                        client_training_class_id_str += "{}-{}-{}".format(class_id, 3000, "end")
                    client_validating_class_id_str += class_id
        clients_training_class_list.append(client_training_class_id_str)
        clients_validate_class_list.append(client_validating_class_id_str)
    return clients_training_class_list, clients_validate_class_list


def _generate_data_distribution_file(experiment_name, num_classes_per_client, architecture_config, dataset,
                                     architecture_setting, dataset_config, dataset_config_file_name, config_version):
    if dataset == "fashion-mnist":
        classes = ":".join(str(class_id) for class_id in range(0,10))
        n_training_records_per_class = 6000
    data_dist = {"train": {"cloud": [], "mid_node": [], "edge_node": [], "client_node": []},
                 "validate": {"cloud": [], "mid_node": [], "edge_node": [], "client_node": []}}
    for client_group_index, client_group in enumerate(architecture_config[architecture_setting]['edge_node']):
        distributed_classes = dataset_config['edge_node'][client_group_index].split(",")
        num_clients = int(client_group.split(":")[1]) - int(client_group.split(":")[0]) + 1
        clients_training_class_list, clients_validate_class_list = _randomly_distribute_class(experiment_name,
                                                                                              num_classes_per_client,
                                                                                              client_group_index,
                                                                                              distributed_classes,
                                                                                              n_training_records_per_class,
                                                                                              num_clients)
        data_dist['train']['client_node'].extend(clients_training_class_list)
        data_dist['validate']['client_node'].extend(clients_validate_class_list)
        if "case_a" in experiment_name:
            data_dist['validate']['edge_node'].append(dataset_config['edge_node'][client_group_index].replace(",",":"))
        else:
            data_dist['validate']['edge_node'].append("all")
    if "mid_node" in architecture_config[architecture_setting].keys():
        for mid_node_dict in architecture_config[architecture_setting]['mid_node']:
            for mid_node_index, child_node_group in mid_node_dict.items():
                for child_node in child_node_group.split(":"):
                    node_index = child_node.split("_")[-1]
                    if experiment_name == "case_a":
                        data_dist['validate']['mid_node'].append(data_dist['validate']['edge_node'][int(node_index)])
                if experiment_name == "case_b":
                    data_dist['validate']['mid_node'].append("all")
    data_dist['train']['client_node'] = ",".join(data_dist['train']['client_node'])
    data_dist['validate']['client_node'] = ",".join(data_dist['validate']['client_node'])
    data_dist['validate']['edge_node'] = ",".join(data_dist['validate']['edge_node'])
    data_dist['validate']['mid_node'] = ",".join(data_dist['validate']['mid_node'])
    data_dist['validate']['cloud'] = classes
    dataset_config_file = "{}_{}_{}_{}.json".format(dataset_config_file_name, architecture_setting,
                                                    experiment_name, config_version)
    with open("config/{}".format(dataset_config_file), 'w') as f:
        json.dump(data_dist, f)
    return dataset_config_file


def _generate_compose_file(exp_config, parameters, compose_outputs, architecture_config, architecture_setting,
                           list_dataset_config_file, mqtt_broker_ip, mqtt_broker_port, config_version):
    for _ in range(len(list_dataset_config_file)):
        compose_outputs.append({"version": "2.4", "services":{}, "networks": {"brokernet":{"external": True}}})
    based_image = "duyhoan/python-ml:tensorflow-py3"
    volume_mapping = ["../app:/usr/src/app", "../config:/usr/src/config", "/etc/passwd:/etc/passwd:ro"]
    total_num_nodes = _calculate_total_num_nodes(architecture_config, architecture_setting)
    compose_outputs = _load_client_node_configs(exp_config, parameters, architecture_config, architecture_setting, compose_outputs,
                                                based_image, volume_mapping, total_num_nodes, list_dataset_config_file,
                                                mqtt_broker_ip, mqtt_broker_port)
    if "mid_node" in architecture_config[architecture_setting].keys():
        compose_outputs = _load_mid_node_configs(exp_config, parameters, architecture_config, architecture_setting, compose_outputs,
                                                based_image, volume_mapping, total_num_nodes, list_dataset_config_file, mqtt_broker_ip, mqtt_broker_port)
    compose_outputs = _load_cloud_configs(exp_config, parameters, architecture_config, architecture_setting, compose_outputs,
                                         based_image, volume_mapping, total_num_nodes, list_dataset_config_file, mqtt_broker_ip, mqtt_broker_port)
    for idx, compose_output in enumerate(compose_outputs):
        with open("config/docker-compose-{}-{}-dataset-config-{}-{}.yml".format(architecture_setting, config_version,
                                                                                exp_config['name'], idx), 'w') as f:
            yaml.Dumper.ignore_aliases = lambda *args: True
            yaml.dump(compose_output, f)


def _calculate_total_num_nodes(architecture_config, architecture_setting):
    if 'mid_node' in architecture_config[architecture_setting]:
        n_mid_nodes = sum(len(level) for level in (architecture_config[architecture_setting]['mid_node']))
    else:
        n_mid_nodes = 0
    total_num_nodes = len(architecture_config[architecture_setting]['edge_node']) + n_mid_nodes + 1
    for client_group in architecture_config[architecture_setting]['edge_node']:
        clients = client_group.split(":")
        total_num_nodes += int(clients[1]) - int(clients[0]) + 1
    return total_num_nodes


def _load_cloud_configs(exp_config, parameters, architecture_config, architecture_setting, compose_output, based_image,
                        volume_mapping, total_num_nodes, list_dataset_config_file, mqtt_broker_ip, mqtt_broker_port):
    n_child_nodes_cloud = len(architecture_config[architecture_setting]['cloud'].split(":"))
    for cloud_index, node_group in enumerate(architecture_config[architecture_setting]['cloud'].split(":")):
        for node in node_group.split(":"):
            node_index = node.split("_")[-1]
            list_command = []
            if 'mid_node' in node:
                for mid_node_dict in architecture_config[architecture_setting]['mid_node']:
                    if node_index in mid_node_dict.keys():
                        n_child_nodes = len(mid_node_dict[node_index].split(":"))
                        break
                network = "cluster_mid_{}".format(node_index)
                mid_interval_steps = int(exp_config['cloud_interval_steps'] * frac(exp_config[
                                                                                       'interval_ratio_mid_{}'.format(
                                                                                           len(architecture_config[
                                                                                                   architecture_setting][
                                                                                                   'mid_node']) + 1)]))
                for idx, dataset_config_file in enumerate(list_dataset_config_file):
                    list_command.append(
                        _generate_mid_node_run_command(parameters, exp_config['name'], architecture_setting,
                                                       total_num_nodes, "cloud_0", 1, mid_interval_steps, n_child_nodes,
                                                       dataset_config_file, mqtt_broker_ip, mqtt_broker_port))
            else:
                n_child_nodes = int(architecture_config[architecture_setting]['edge_node'][int(node_index)].split(":")[
                                        1]) - \
                                int(architecture_config[architecture_setting]['edge_node'][int(node_index)].split(":")[
                                        0]) + 1
                network = "cluster_{}".format(node_index)
                edge_interval_steps = int(exp_config['cloud_interval_steps'] * frac(exp_config['interval_ratio_edge']))
                for idx, dataset_config_file in enumerate(list_dataset_config_file):
                    list_command.append(
                        _generate_edge_node_run_command(parameters, exp_config['name'], architecture_setting,
                                                        total_num_nodes, "cloud_0", 1, edge_interval_steps,
                                                        n_child_nodes, dataset_config_file, mqtt_broker_ip,
                                                        mqtt_broker_port))
            for idx, command in enumerate(list_command):
                compose_output[idx]['services'][node] = {
                    "image": based_image,
                    "volumes": volume_mapping,
                    "restart": "on-failure",
                    "user": "$UID",
                    "ports": [8000],
                    "command": command,
                    "networks": [
                        "cloud_net",
                        network,
                        "brokernet"
                    ],
                    "depends_on": [
                        "cloud"
                    ]
                }
                if "cloud_net" not in compose_output[idx]["networks"].keys():
                    compose_output[idx]["networks"]["cloud_net"] = {
                        "driver": "bridge",
                        "ipam": {
                            "driver": "default"
                        }
                    }
                if network not in compose_output[idx]["networks"].keys():
                    compose_output[idx]["networks"][network] = {
                        "driver": "bridge",
                        "ipam": {
                            "driver": "default"
                        }
                    }
    for idx, dataset_config_file in enumerate(list_dataset_config_file):
        compose_output[idx]['services']["cloud"] = {
            "image": based_image,
            "volumes": volume_mapping,
            "restart": "on-failure",
            "user": "$UID",
            "ports": [8000],
            "command": _generate_cloud_run_command(parameters, exp_config['name'], architecture_setting,
                                                   total_num_nodes, 0, n_child_nodes_cloud, dataset_config_file,
                                                   mqtt_broker_ip, mqtt_broker_port),
            "networks": [
                "cloud_net",
                "brokernet"
            ]
        }
        return compose_output


def _load_mid_node_configs(exp_config, parameters, architecture_config, architecture_setting, compose_output,
                           based_image, volume_mapping, total_num_nodes, list_dataset_config_file, mqtt_broker_ip,
                           mqtt_broker_port):
    n_mid_nodes = sum(len(level) for level in (architecture_config[architecture_setting]['mid_node']))
    for mid_level, mid_node_dict in enumerate(architecture_config[architecture_setting]['mid_node']):
        for mid_node_index, child_node_group in mid_node_dict.items():
            for child_node in child_node_group.split(":"):
                node_index = child_node.split("_")[-1]
                list_command = []
                if "edge_node" in child_node:
                    edge_interval_steps = int(
                        exp_config['cloud_interval_steps'] * frac(exp_config['interval_ratio_edge']))
                    n_child_nodes = int(
                        architecture_config[architecture_setting]['edge_node'][int(node_index)].split(":")[1]) - int(
                        architecture_config[architecture_setting]['edge_node'][int(node_index)].split(":")[0]) + 1
                    for idx, dataset_config_file in enumerate(list_dataset_config_file):
                        list_command.append(
                            _generate_edge_node_run_command(parameters, exp_config['name'], architecture_setting, total_num_nodes,
                                                            mid_node_index, n_mid_nodes + 1, edge_interval_steps,
                                                            n_child_nodes, dataset_config_file, mqtt_broker_ip, mqtt_broker_port))
                    network1 = "cluster_{}".format(node_index)
                    network2 = "cluster_mid_{}".format(mid_node_index)
                else:
                    mid_interval_steps = int(exp_config['cloud_interval_steps'] * frac(
                        exp_config['interval_ratio_mid_{}'.format(mid_level + 1)]))
                    for mid_node_dict in architecture_config[architecture_setting]['mid_node']:
                        if node_index in mid_node_dict.keys():
                            n_child_nodes = len(mid_node_dict[node_index].split(":"))
                            break
                    for idx, dataset_config_file in enumerate(list_dataset_config_file):
                        list_command.append(
                            _generate_mid_node_run_command(parameters, exp_config['name'], architecture_setting, total_num_nodes,
                                                           mid_node_index, 1, mid_interval_steps, n_child_nodes,
                                                           dataset_config_file, mqtt_broker_ip, mqtt_broker_port))
                    network1 = "cluster_mid_{}".format(node_index)
                    network2 = "cluster_mid_{}".format(mid_node_index)
                for idx, command in enumerate(list_command):
                    compose_output[idx]['services'][child_node] = {
                        "image": based_image,
                        "volumes": volume_mapping,
                        "restart": "on-failure",
                        "user": "$UID",
                        "ports": [8000],
                        "command": command,
                        "networks": [
                            network1,
                            network2,
                            "brokernet"
                        ],
                        "depends_on": [
                            "mid_node_{}".format(mid_node_index)
                        ]
                    }
                    if network1 not in compose_output[idx]["networks"].keys():
                        compose_output[idx]["networks"][network1] = {
                            "driver": "bridge",
                            "ipam": {
                                "driver": "default"
                            }
                        }
                    if network2 not in compose_output[idx]["networks"].keys():
                        compose_output[idx]["networks"][network2] = {
                            "driver": "bridge",
                            "ipam": {
                                "driver": "default"
                            }
                        }
    return compose_output


def _load_client_node_configs(exp_config, parameters, architecture_config, architecture_setting, compose_output,
                              based_image, volume_mapping, total_num_nodes, list_dataset_config_file, mqtt_broker_ip,
                              mqtt_broker_port):
    client_group_start_index = 0
    if 'mid_node' in architecture_config[architecture_setting]:
        n_mid_nodes = sum(len(level) for level in (architecture_config[architecture_setting]['mid_node']))
    else:
        n_mid_nodes = 0
    client_interval_steps = int(int(exp_config['cloud_interval_steps']) * frac(exp_config['interval_ratio_client']))
    n_higher_nodes = len(architecture_config[architecture_setting]['edge_node']) + n_mid_nodes + 1
    for edge_node_index, client_group in enumerate(architecture_config[architecture_setting]['edge_node']):
        client_group = client_group.split(":")
        for idx, dataset_config_file in enumerate(list_dataset_config_file):
            compose_output[idx]['services']['client_group_' + str(edge_node_index)] = {
                "image": based_image,
                "volumes": volume_mapping,
                "restart": "on-failure",
                "user": "$UID",
                "ports": [8000],
                "scale": int(client_group[1]) - int(client_group[0]) + 1,
                "command": _generate_client_run_command(parameters, exp_config['name'], architecture_setting, total_num_nodes,
                                                        edge_node_index, client_group_start_index, n_higher_nodes,
                                                        client_interval_steps, dataset_config_file, mqtt_broker_ip, mqtt_broker_port),
                "networks": [
                    "cluster_{}".format(edge_node_index),
                    "brokernet"
                ],
                "depends_on": [
                    "edge_node_{}".format(edge_node_index)
                ]
            }
            if "cluster_{}".format(edge_node_index) not in compose_output[idx]["networks"].keys():
                compose_output[idx]["networks"]["cluster_{}".format(edge_node_index)] = {
                    "driver": "bridge",
                    "ipam": {
                        "driver": "default"
                    }
                }
        client_group_start_index += int(client_group[1]) - int(client_group[0]) + 1
    return compose_output


def _generate_client_run_command(parameters, experiment_name, architecture_setting, total_num_nodes, parent_node_index,
                                 client_group_start_index, n_higher_nodes, edge_interval_steps, dataset_config_file,
                                 mqtt_broker_ip, mqtt_broker_port):
    return "python3 experiment.py --device_type \"client_node\" --log_dir \"logs/fashion_mnist_test\" --data_dir \"device/data\" --dataset \"" + parameters['dataset'] + "\" --n_nodes " + str(total_num_nodes) + " --net_config \"" + parameters['net_config'] + "\" --dataset_config_file \"" + dataset_config_file + "\" --architecture_setting \"" + architecture_setting + "\" --experiment \"" + experiment_name + "\" --lr " + str(parameters['lr']) + " --epochs " + str(parameters['num_epochs']) + " --batch_size " +str(parameters['batch_size']) + " --published_port 8000 --parent \"edge_node_" + str(parent_node_index) + "\" --n_higher_nodes " + str(n_higher_nodes) + " --client_group_start_index " + str(client_group_start_index) + " --mqtt_broker_ip " + mqtt_broker_ip + " --mqtt_broker_port " + str(mqtt_broker_port) + " --interval_steps " + str(edge_interval_steps) + ""


def _generate_edge_node_run_command(parameters, experiment_name, architecture_setting, total_num_nodes, parent_node_index,
                                    n_higher_nodes, mid_interval_steps, n_child_nodes, dataset_config_file,
                                    mqtt_broker_ip, mqtt_broker_port):
    if parent_node_index != "cloud_0":
        parent_node_index = "mid_node_" + parent_node_index
    return "python3 experiment.py --device_type \"edge_node\" --log_dir \"logs/fashion_mnist_test\" --data_dir \"device/data\" --dataset \"" + parameters['dataset'] + "\" --n_nodes " + str(total_num_nodes) + " --n_child_nodes " + str(n_child_nodes) + " --net_config \"" + parameters['net_config'] + "\" --dataset_config_file \"" + dataset_config_file + "\" --architecture_setting \"" + architecture_setting + "\" --experiment \"" + experiment_name + "\" --lr " + str(parameters['lr']) + " --epochs " + str(parameters['num_epochs']) + " --batch_size " + str(parameters['batch_size']) + " --published_port 8000 --parent \"" + str(parent_node_index) + "\" --n_higher_nodes " + str(n_higher_nodes) + " --mqtt_broker_ip " + mqtt_broker_ip + " --mqtt_broker_port " + str(mqtt_broker_port) + " --interval_steps " + str(mid_interval_steps) + ""


def _generate_mid_node_run_command(parameters, experiment_name, architecture_setting, total_num_nodes, parent_node_index,
                                   n_higher_nodes, mid_interval_steps, n_child_nodes, dataset_config_file,
                                   mqtt_broker_ip, mqtt_broker_port):
    if parent_node_index != "cloud_0":
        parent_node_index = "mid_node_" + parent_node_index
    return "python3 experiment.py --device_type \"mid_node\" --log_dir \"logs/fashion_mnist_test\" --data_dir \"device/data\" --dataset \"" + parameters['dataset'] + "\" --n_nodes " + str(total_num_nodes) + " --n_child_nodes " + str(n_child_nodes) + " --net_config \"" + parameters['net_config'] + "\" --dataset_config_file \"" + dataset_config_file + "\" --architecture_setting \"" + architecture_setting + "\" --experiment \"" + experiment_name + "\" --lr " + str(parameters['lr']) + " --epochs " + str(parameters['num_epochs']) + " --batch_size " + str(parameters['batch_size']) + " --published_port 8000 --parent \"" + str(parent_node_index) + "\" --n_higher_nodes " + str(n_higher_nodes) + " --mqtt_broker_ip " + mqtt_broker_ip + " --mqtt_broker_port " + str(mqtt_broker_port) + " --interval_steps " + str(mid_interval_steps) + ""


def _generate_cloud_run_command(parameters, experiment_name, architecture_setting, total_num_nodes, n_higher_nodes,
                                n_child_nodes, dataset_config_file, mqtt_broker_ip, mqtt_broker_port):
    return "python3 experiment.py --device_type \"cloud\" --log_dir \"logs/fashion_mnist_test\" --data_dir \"device/data\" --dataset \"" + parameters['dataset'] + "\" --n_nodes " + str(total_num_nodes) + " --n_child_nodes " + str(n_child_nodes) + " --net_config \"" + parameters['net_config'] + "\" --dataset_config_file \"" + dataset_config_file + "\" --architecture_setting \"" + architecture_setting + "\" --experiment \"" + experiment_name + "\" --lr " + str(parameters['lr']) + " --epochs " + str(parameters['num_epochs']) + " --batch_size " + str(parameters['batch_size']) + " --published_port 8000 --n_higher_nodes " + str(n_higher_nodes) + " --mqtt_broker_ip " + mqtt_broker_ip + " --mqtt_broker_port " + str(mqtt_broker_port) + ""


def _get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--meta_config_file', type=str, required=True,
                        help='JSON file name, used to generate different configuration sets')

    parser.add_argument('--dataset_config_file', type=str, required=True,
                        help='JSON file name, used to store the data distribution configuration')
    return parser


if __name__ == '__main__':
    parser = _get_parser()
    # Getting the argument dictionary from parser object
    args = parser.parse_args()
    gen_config_files(args.meta_config_file, args.dataset_config_file)
