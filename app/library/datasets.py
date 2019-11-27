import os

import h5py
from tensorflow import keras
import numpy as np


class FashionMNIST:
    """
    Fashion-MNIST dataset loader
    """
    def __init__(self, data_dir, experiment, device_type, device_index):
        self._data_dir = data_dir
        self._experiment = experiment
        self._device_type = device_type
        self._device_index = device_index
        fashion_mnist = keras.datasets.fashion_mnist
        (self._train_images, self._train_labels), (self._test_images, self._test_labels) = fashion_mnist.load_data()

    def build_train_dataset(self):
        if "case_c" not in self._experiment:
            del self._train_images, self._train_labels
            return h5py.File(os.path.join(self._data_dir, "train.hdf5"), 'r')
        else:
            return [self._train_images, self._train_labels]

    def build_test_dataset(self):
        if "overall_final_testing" not in self._experiment:
            del self._test_images, self._test_labels
            if "paper" not in self._experiment:
                return h5py.File(os.path.join(self._data_dir, "test.hdf5"), 'r')
            else:
                if self._device_type == "client_node":
                    if self._device_index < 10:
                        with h5py.File(os.path.join(self._data_dir, "test_0.hdf5"), 'r') as file:
                            return [file["x_class_{}".format(self._device_index)][:],
                                    file["y_class_{}".format(self._device_index)][:],
                                    h5py.File(os.path.join(self._data_dir, "test.hdf5"), 'r')]
                    else:
                        with h5py.File(os.path.join(self._data_dir, "test_1.hdf5"), 'r') as file:
                            return [file["x_class_{}".format(self._device_index % 10)][:],
                                    file["y_class_{}".format(self._device_index % 10)][:],
                                    h5py.File(os.path.join(self._data_dir, "test.hdf5"), 'r')]
                elif self._device_type == "edge_node":
                    test_images = np.empty((0, 28, 28))
                    test_labels = np.empty((0,))
                    with h5py.File(os.path.join(self._data_dir, "test_{}.hdf5".format(self._device_index)),
                                   'r') as file:
                        for class_id in range(10):
                            test_images = np.concatenate((test_images, file["x_class_{}".format(class_id)][:]),
                                                         axis=0)
                            test_labels = np.concatenate((test_labels, file["y_class_{}".format(class_id)][:]),
                                                         axis=0)
                        return [test_images, test_labels, h5py.File(os.path.join(self._data_dir, "test.hdf5"), 'r')]
                else:
                    test_images = np.empty((0, 28, 28))
                    test_labels = np.empty((0,))
                    with h5py.File(os.path.join(self._data_dir, "test.hdf5"), 'r') as file:
                        for class_id in range(10):
                            test_images = np.concatenate((test_images, file["x_class_{}".format(class_id)][:]),
                                                         axis=0)
                            test_labels = np.concatenate((test_labels, file["y_class_{}".format(class_id)][:]),
                                                         axis=0)
                        return [test_images, test_labels, h5py.File(os.path.join(self._data_dir, "test.hdf5"), 'r')]
        else:
            return [self._test_images, self._test_labels]
