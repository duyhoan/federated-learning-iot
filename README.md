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
  ---
  **__IMPORTANTS__**
  
  If you tend to run multiple experiments at a time, you need to create multiple MQTT brokers. Each broker is dedicated for nodes of one experiment. It means that one MQTT broker CAN NOT be used for multiple experiments.
  
  ---
  
#### Important source files:
---

1. `gen_config.py`: Python script used to generating different configurations (docker-compose yaml files and data distribution json files) for running experiment 
2. `config_metadata.json`: The meta configuration json file, which consists of information for configuring the system architecture and class distribution on simulated client nodes (end-user devices) 
3. `experiment.py`: Main entrypoint to the program. Used to run all experiments
4. `training.py`: Contains the main training code for multilevel federated learning
5. `event_handler.py`: Contains the MQTT message processing for communication and signaling between isolated nodes (docker containers)

#### How to run experiments:
---

1. Run the `gen_config.py` script to generate configuration file(s) for running experiment:
   ```bash
   python3 gen_config.py --meta_config_file "config_metadata" --dataset_config_file "dataset_dist"
   ```
   The `gen_config.py` file will take the `config_metadata.json` file as input. Then it will read the configurations of system architecture and class distribution defined in that file, processes information and finally exports the docker-compose yaml file and class distribution file.
   
   > **Some important notes**: 
     > * The generated docker-compose file and class distribution file will be in the following format:
     > * **docker-compose-{architecture setting}-{configuration meta file version}-dataset-config-{experiment name}.yml**
     > * **dataset_dist_{architecture setting}_{experiment name}_{configuration meta file version}.json**
     > * There can be multiple docker-compose and class distribution files generated at a time, depends on how many experiments you defined in the `config_metadata.json` file. Each pair of (docker-compose, class distribution) files is dedicated to 1 experiment.

2. After running the above command, all the generated configuration files (docker compose and class distribution) will be located on `config` folder. Now just go to the `config` folder and run docker compose command:
   ```console
   foo@bar:~$ cd config
   foo@bar:~$ docker-compose -f [docker-compose yaml file] -p [experiment name] up
   ```
   The docker compose will initialize a set of networked docker containers, each container corresponds to one node in the system. After all nodes have been initialized, each of them will run the entrypoint command for starting the experiment. This entrypoint is the following file:
   `app/experiment.py`

#### Important arguments:
---

The following are some of arguments which are passed as important parameters of the experiment. 
> Note that this is just for your information. In order to change the experiment arguments, you can go to the `config_metadata.json` file and modify them. After modifying them, you need to re-run the `gen_config.py` script to generate configuration files again with new parameters. 

1. `net_config`: Defines the local network architecture
2. `device_type`: Defines the device type [client_node/edge_node/mid_node/cloud]
3. `partition`: Kind of data partition. Values: homo, hetero-dir
4. `experiment`: Defines which experiments will be executed
5. `dataset_config_file`: Defines the JSON file name for loading dataset distribution configuration
6. `interval_steps`: Defines how many local updates (mini-batches) are done before performing aggregation step


#### Output:
---

Some of the output is printed on the terminal. However, majority of the information is logged to a log file in the specified log folder. Each node has a separate log file.
