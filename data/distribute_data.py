# Created by duyhoan at 7/27/19

# Feature:  Data loading, shuffling and distribution. Used to distribute the dataset to simulated devices

# Importing libraries:
import csv
import os
import random
import math
from collections import Counter
import h5py
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import scipy
from tensorflow import keras


class DistributeData:
    _base_device_data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data'))  # Device's dataset dir
    _class_mapping = {}
    _class_images = {}

    def __init__(self, num_devices):
        self._num_devices = num_devices

    def distribute_dataset(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        self._num_classes = len(np.unique(train_labels))
        X_train = {}
        y_train = {}
        X_test = {}
        y_test = {}
        for idx, img in enumerate(train_images):
            if train_labels[idx] in X_train.keys():
                X_train[train_labels[idx]].append(img)
                y_train[train_labels[idx]].append(train_labels[idx])
            else:
                X_train[train_labels[idx]] = [img]
                y_train[train_labels[idx]] = [train_labels[idx]]
        for idx, img in enumerate(test_images):
            if test_labels[idx] in X_test.keys():
                X_test[test_labels[idx]].append(img)
                y_test[test_labels[idx]].append(test_labels[idx])
            else:
                X_test[test_labels[idx]] = [img]
                y_test[test_labels[idx]] = [test_labels[idx]]
        with h5py.File(os.path.join(self._base_device_data_dir, 'train.hdf5'), 'w') as file:
            for key in X_train.keys():
                file.create_dataset('x_class_{}'.format(key), data=X_train[key], dtype=np.float32)
                file.create_dataset('y_class_{}'.format(key), data=y_train[key], dtype=np.int8)
        with h5py.File(os.path.join(self._base_device_data_dir, 'test.hdf5'), 'w') as file:
            for key in X_test.keys():
                file.create_dataset('x_class_{}'.format(key), data=X_test[key], dtype=np.float32)
                file.create_dataset('y_class_{}'.format(key), data=y_test[key], dtype=np.int8)

    @staticmethod
    def _mk_dirs(dir_path):
        try:
            os.makedirs(dir_path)
        except Exception as _:
            pass

    def split_test_data(self):
        with h5py.File(os.path.join(self._base_device_data_dir, 'test.hdf5'), 'r') as file_test:
            for class_id in range(self._num_classes):
                with h5py.File(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'app/device/data', 'test_0.hdf5')), 'a') as file:
                    file.create_dataset('x_class_{}'.format(class_id), data=file_test["x_class_{}".format(class_id)][:500], dtype=np.float32)
                    file.create_dataset('y_class_{}'.format(class_id), data=file_test["y_class_{}".format(class_id)][:500], dtype=np.int8)
                with h5py.File(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'app/device/data', 'test_1.hdf5')), 'a') as file:
                    file.create_dataset('x_class_{}'.format(class_id), data=file_test["x_class_{}".format(class_id)][500:], dtype=np.float32)
                    file.create_dataset('y_class_{}'.format(class_id), data=file_test["y_class_{}".format(class_id)][500:], dtype=np.int8)


if __name__ == "__main__":
    distributed_data = DistributeData(20)
    distributed_data.distribute_dataset()
    distributed_data.split_test_data()
    del distributed_data
