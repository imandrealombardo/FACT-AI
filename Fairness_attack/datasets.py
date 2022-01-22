from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np

# Local running
DATA_FOLDER = './data'
OUTPUT_FOLDER = 'output'

def safe_makedirs(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

def check_orig_data(X_train, Y_train, X_test, Y_test):
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_test.shape[0] == Y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert np.max(Y_train) == 1, 'max of Y_train was %s' % np.max(Y_train)
    assert np.min(Y_train) == -1
    print(set(Y_train))
    assert len(set(Y_train)) == 2
    assert set(Y_train) == set(Y_test)

def load_german():
    dataset_path = os.path.join(DATA_FOLDER)
    print(os.path.join(dataset_path, "german_data.npz"))
    f = np.load(os.path.join(dataset_path, "german_data.npz"))

    X_train = f['X_train']
    Y_train = f['Y_train'].reshape(-1)
    Y_train[Y_train == 0] = -1
    X_test = f['X_test']
    Y_test = f['Y_test'].reshape(-1)
    Y_test[Y_test == 0] = -1
    

    check_orig_data(X_train, Y_train, X_test, Y_test)
    return X_train, Y_train, X_test, Y_test

def load_compas():
    dataset_path = os.path.join(DATA_FOLDER)
    print(os.path.join(dataset_path, "compas_data.npz"))
    f = np.load(os.path.join(dataset_path, "compas_data.npz"))

    X_train = f['X_train']
    Y_train = f['Y_train'].reshape(-1)
    Y_train[Y_train == 0] = -1
    X_test = f['X_test']
    Y_test = f['Y_test'].reshape(-1)
    Y_test[Y_test == 0] = -1
    

    check_orig_data(X_train, Y_train, X_test, Y_test)
    return X_train, Y_train, X_test, Y_test

def load_drug():
    dataset_path = os.path.join(DATA_FOLDER)
    print(os.path.join(dataset_path, "drug2_data.npz"))
    f = np.load(os.path.join(dataset_path, "drug2_data.npz"))

    X_train = f['X_train']
    Y_train = f['Y_train'].reshape(-1)
    Y_train[Y_train == 0] = -1
    X_test = f['X_test']
    Y_test = f['Y_test'].reshape(-1)
    Y_test[Y_test == 0] = -1
    

    check_orig_data(X_train, Y_train, X_test, Y_test)
    return X_train, Y_train, X_test, Y_test

def load_dataset(dataset_name):
    if dataset_name == 'german':
        return load_german()
    elif dataset_name == 'compas':
        return load_compas()
    elif dataset_name == 'drug':
        return load_drug()