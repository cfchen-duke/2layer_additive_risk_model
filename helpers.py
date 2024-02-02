# helpers.py
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

import os

def list_of_distances(X, Y):
    '''
    Given a list of vectors, X = [x_1, ..., x_n], and another list of vectors,
    Y = [y_1, ... , y_m], this function returns a list of vectors
            [[d(x_1, y_1), d(x_1, y_2), ... , d(x_1, y_m)],
             ...
             [d(x_n, y_1), d(x_n, y_2), ... , d(x_n, y_m)]],
    where the distance metric used is the sqared euclidean distance.
    The computation is achieved through a clever use of broadcasting.
    '''
    return tf.reduce_sum((
               tf.expand_dims(X, 2) \
             - tf.expand_dims(tf.transpose(Y), 0))**2, 1)

def list_of_norms(X):
    '''
    Given a list of vectors X = [x_1, ..., x_n], this function returns
        [||x_1||_2^2, ||x_2||_2^2, ... , ||x_n||_2^2].
    '''
    return tf.reduce_sum(tf.pow(X, 2), axis=1)

def split_train_test(data, test_size):
    positive_dat = data[data[:,0]==1]
    negative_dat = data[data[:,0]==0]
    train_positive, test_positive = train_test_split(positive_dat, test_size=test_size)
    train_negative, test_negative = train_test_split(negative_dat, test_size=test_size)
    train = np.concatenate([train_positive, train_negative], axis=0)
    test = np.concatenate([test_positive, test_negative], axis=0)
    # permutate rows
    train_indices = np.random.permutation(train.shape[0])
    test_indices = np.random.permutation(test.shape[0])
    train = train[train_indices,:]
    test = test[test_indices,:]
    
    return train, test

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)
        
def print_and_write(str, file):
    print(str)
    file.write(str + '\n')
