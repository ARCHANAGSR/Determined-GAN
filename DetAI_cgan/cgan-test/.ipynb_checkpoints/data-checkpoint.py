import tensorflow as tf
from keras.datasets import mnist
import numpy as np


def get_train_dataset(worker_rank: int):
    
    (X_train,Y_train),(X_test,Y_test) = mnist.load_data()
    
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32")
    X_train = (X_train-127.5) / 127.5
    #X_train = (X_train - 127.5) / 127.5  # Normalize the images to [-1, 1]
    #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    #X_train = np.expand_dims(X_train, axis=3)
   
    
    Y_train = Y_train.reshape(-1, 1)

    return X_train,Y_train

    

def get_validation_dataset(worker_rank: int):
    
    (temp_train,tempy_train) ,(X_test,Y_test) = mnist.load_data()
    
    #X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    #X_test = np.expand_dims(X_test, axis=3)
    
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32")
    X_test = (X_test-127.5) / 127.5
    
    Y_test = Y_test.reshape(-1, 1)
    
    return X_test,Y_test


