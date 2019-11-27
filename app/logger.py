import os
import numpy as np
import logging


class Logger:
    """
    Logger class, which is responsible for all logging activities while running
    """
    def __init__(self, experiment, architecture_setting, device_type, device_index, info_log_dir, train_log_dir, test_log_dir):
        """
        :param experiment: the type of experiment to run. [case_a/case_b/random]
        :param architecture_setting: the type of architecture used to run [simple/complex]
        :param device_type: the node type [client_node / edge_node / mid_node / cloud]
        :param device_index: the device unique index, with regard to it's belonging layer
        :param log_dir: target directory for storing log files
        """
        self._experiment = experiment
        self._architecture_setting = architecture_setting
        self._device_type = device_type
        self._device_index = device_index
        self._info_log_dir = info_log_dir
        self._train_log_dir = train_log_dir
        self._test_log_dir = test_log_dir
        logging.basicConfig(
            filename=os.path.join(info_log_dir, 'experiment_log_{}_{}_{}.log'.format(device_type, device_index, experiment)),
            format='%(asctime)s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    def log_training_acc(self, epoch, acc):
        try:
            with open(os.path.join(self._train_log_dir, "{}_training_accuracy_{}_{}_{}.csv".format(self._architecture_setting,
                                                                                                 self._device_type,
                                                                                                 self._device_index,
                                                                                             self._experiment)),
                                   'ab') as f:
                np.savetxt(f, [[epoch, acc]], fmt='%.4f')
        except Exception as e:
            print(e)

    def log_test_acc_on_training(self, epoch, acc):
        try:
            with open(os.path.join(self._train_log_dir, "{}_testing_accuracy_{}_{}_{}.csv".format(self._architecture_setting,
                                                                                                 self._device_type,
                                                                                                 self._device_index,
                                                                                             self._experiment)),
                                   'ab') as f:
                np.savetxt(f, [[epoch, acc]], fmt='%.4f')
        except Exception as e:
            print(e)

    def log_test_acc(self, data_class, acc):
        try:
            np.savetxt("{}/{}_testing_accuracy_class_{}_{}_{}_{}.csv".format(self._test_log_dir, self._architecture_setting,
                                                                             str(data_class), self._device_type,
                                                                             self._device_index, self._experiment),
                       np.array([[0], [acc]]), fmt='%.4f')
        except Exception as e:
            print(e)

    def log_info(self, msg):
        logging.info(msg)
