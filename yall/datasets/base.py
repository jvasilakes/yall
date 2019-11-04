import os
import numpy as np
from collections import namedtuple


# TODO: Maybe just use sklearn.utils.Bunch
Bunch = namedtuple("Bunch", ["data", "target", "filenames"])

# TODO: Write unit tests for data loading functions
def load_dexter():
    module_path = os.path.dirname(__file__)
    data_path = os.path.join(module_path, "data/dexter")

    train_data_file = os.path.join(data_path, "train.data")
    valid_data_file = os.path.join(data_path, "valid.data")
    train_labels_file = os.path.join(data_path, "train.labels")
    valid_labels_file = os.path.join(data_path, "valid.labels")

    with open(train_data_file, 'r') as inF:
        train_data = [l.strip().split() for l in inF]
    with open(valid_data_file, 'r') as inF:
        valid_data = [l.strip().split() for l in inF]
    with open(train_labels_file, 'r') as inF:
        train_labels = [l.strip() for l in inF]
    with open(valid_labels_file, 'r') as inF:
        valid_labels = [l.strip() for l in inF]

    D = 20000  # Dimension of the features
    in_data = train_data + valid_data
    target = np.array(train_labels + valid_labels, dtype=int)
    out_data = np.zeros(shape=(len(in_data), D), dtype=int)
    for i in range(len(in_data)):
        # Array of [index, frequency]
        x = np.array([iv.split(':') for iv in in_data[i]], dtype=int)
        out_data[i, x[:, 0]] = x[:, 1]   # Assign values to indices for row i

    return Bunch(data=out_data, target=target,
                 filenames=[train_data_file, valid_data_file,
                            train_labels_file, valid_labels_file])


def load_spect():
    module_path = os.path.dirname(__file__)
    data_path = os.path.join(module_path, "data", "SPECT")

    train_data_file = os.path.join(data_path, "SPECT.train")
    test_data_file = os.path.join(data_path, "SPECT.test")

    train_data = np.loadtxt(train_data_file, delimiter=',', dtype=int)
    test_data = np.loadtxt(test_data_file, delimiter=',', dtype=int)
    train_X = train_data[:, 1:]  # First column is target
    test_X = test_data[:, 1:]
    train_y = train_data[:, 0]
    test_y = test_data[:, 0]
    data = np.vstack([train_X, test_X])
    target = np.concatenate([train_y, test_y])

    return Bunch(data=data, target=target,
                 filenames=[train_data_file, test_data_file])


def load_spectf():
    module_path = os.path.dirname(__file__)
    data_path = os.path.join(module_path, "data", "SPECT")

    train_data_file = os.path.join(data_path, "SPECTF.train")
    test_data_file = os.path.join(data_path, "SPECTF.test")

    train_data = np.loadtxt(train_data_file, delimiter=',', dtype=int)
    test_data = np.loadtxt(test_data_file, delimiter=',', dtype=int)
    train_X = train_data[:, 1:]  # First column is target
    test_X = test_data[:, 1:]
    train_y = train_data[:, 0]
    test_y = test_data[:, 0]
    data = np.vstack([train_X, test_X])
    target = np.concatenate([train_y, test_y])

    return Bunch(data=data, target=target,
                 filenames=[train_data_file, test_data_file])
