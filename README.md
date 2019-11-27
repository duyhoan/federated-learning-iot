## Multilevel Federated Learning


This is the code accompanying for the Master thesis "IoT architecture for big data processsing in Smart City"

#### Requirements to run the code:
---

1. Python 3.6
2. TensorFlow 1.8.0
3. Numpy

#### Before running main program:
---
  * You should have docker and docker-compose installed in your local machine.
  * Follow the following steps to create an isolated network for running the MQTT broker
    1. Create the virtual docker network: open Command Prompt and run `docker network create --drive=bridge --subnet=172.32.0.0/16 brokernet`
    2. Create the VerneMQ MQTT broker: run the command `docker run -d -p 1883:1883 -e "DOCKER_VERNEMQ_ALLOW_ANONYMOUS=on" --name mqttbroker --network=brokernet --ip=172.32.0.100 erlio/docker-vernemq`

#### Important source files:
---

1. `experiment.py`: Main entrypoint to the program. Used to run all experiments
2. `training.py`: Contains the main training code for multilevel federated learning
3. `event_handler.py`: Contains the MQTT message processing for communication and signaling between isolated nodes (docker containers)

#### Sample Commands:
---

1. Fashion-MNIST Simple configuration for client node:

`python3 experiment.py --device_type "client_node" --log_dir "logs/fashion_mnist_test"
      --data_dir "device/data" --dataset "fashion-mnist" --n_nodes 8 --net_config
      "simple" --dataset_config_file "config/dataset_dist.json" --architecture_complexity
      "simple" --experiment "case_a" --lr 0.001 --epochs 500 --batch_size 32 --published_port
      8000 --parent "edge_node_0" --n_higher_nodes 3 --client_group_start_index 0
      --mqtt_broker_ip 172.32.0.100 --mqtt_broker_port 1883 --interval_steps 25`

2. Fashion-MNIST Simple configuration for edge node:
    
`python3 experiment.py --device_type "mid_node" --log_dir "logs/fashion_mnist_test"
      --data_dir "device/data" --dataset "fashion-mnist" --n_nodes 8 --n_child_nodes
      2 --net_config "simple" --dataset_config_file "config/dataset_dist.json" --architecture_complexity
      "simple" --experiment "case_a" --lr 0.001 --epochs 500 --batch_size 32 --published_port
      8000 --parent "mid_node_cloud_0" --n_higher_nodes 1 --mqtt_broker_ip 172.32.0.100
      --mqtt_broker_port 1883 --interval_steps 20`


#### Important arguments:
---


The following arguments to the PFNM file control the important parameters of the experiment

1. `net_config`: Defines the local network architecture
2. `device_type`: Defines the device type [client_node/edge_node/mid_node/cloud]
3. `partition`: Kind of data partition. Values: homo, hetero-dir
4. `experiment`: Defines which experiments will be executed
5. `dataset_config_file`: Defines the JSON file name for loading dataset distribution configuration
6. `interval_steps`: Defines how many local updates (mini-batches) are performed before performing aggregation step


#### Output:
---

Some of the output is printed on the terminal. However, majority of the information is logged to a log file in the specified log folder. Each node has a separate log file.
