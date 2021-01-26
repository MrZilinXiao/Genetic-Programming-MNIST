from sklearn.datasets import fetch_mldata
import numpy as np


def get_mnist():
    mnist = fetch_mldata('MNIST original', data_home='./')
    n_train, n_test = 60000, 10000
    train_idx = np.arange(0, n_train)
    test_idx = np.arange(n_train + 1, n_train + n_test)
    X_train, Y_train = mnist.data[train_idx], mnist.target[train_idx]
    X_test, Y_test = mnist.data[test_idx], mnist.target[test_idx]
    return X_train, Y_train, X_test, Y_test
