import numpy as np
import os
import tensorflow as tf
import math 
import h5py

def conv_to_one_hot(Y,N):
    """
    Y -- tensor or array assumed to be of the shape (1,m) where is the # of examples.
    N -- # of rows/columns in the output.
    
    returns Y -- tesnor with value 1 at the corresponding label.
    """
    
    Y = np.eye(N)[Y.reshape(-1)].T
    
    return Y

def load_dataset():
    """
    Returns the distribution of the data set in two parts i.e. Training and testing input and desired output.
    In the following manner:
    train_set_X_orig , train_set_Y_orig (trainig data).
    test_set_X_orig, test_set_Y_orig (testing data).
    
    """
    
    train_dataset = h5py.File('train_signs.h5', "r")
    train_set_X_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_Y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    
    test_dataset = h5py.File('test_signs.h5', "r")
    test_set_X_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_Y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    
    train_set_Y_orig = train_set_Y_orig.reshape((1, train_set_Y_orig.shape[0]))
    test_set_Y_orig = test_set_Y_orig.reshape((1, test_set_Y_orig.shape[0]))
    
    return train_set_X_orig, test_set_X_orig, train_set_Y_orig, test_set_Y_orig

