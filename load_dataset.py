
# -*- coding = utf-8 -*-
# @Time: 2020/6/9 11:10
# @Author: Yudong Zhong
# @File: load_dataset.py
# @Software: PyCharm

import numpy as np
# from keras.datasets import mnist

# (x_train, y_train),(x_test, y_test) = mnist.load_data()


def load_data(path='mnist.npz'):
    """
    Loads the MNIST dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    path = get_file(path,
                    origin='https://s3.amazonaws.com/img-datasets/mnist.npz',
                    file_hash='8a61469f7ea1b51cbae51d4f78837e45')
    """
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)