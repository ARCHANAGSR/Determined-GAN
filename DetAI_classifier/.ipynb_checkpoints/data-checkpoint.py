from keras.datasets import mnist
import numpy as np
import zipfile




def load_training_data():
    
    

    print("With 60K MNIST DATA")
    (X_train,Y_train), (x_test, y_test) = mnist.load_data()
    
    #X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)
    
    Y_train = Y_train.reshape(-1, 1)

    return X_train,Y_train

    

def load_validation_data():
    
    (temp_train,tempy_train) ,(X_test,Y_test) = mnist.load_data()
    
    #X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = np.expand_dims(X_test, axis=3)
    
    Y_test = Y_test.reshape(-1, 1)
    
    return X_test,Y_test


