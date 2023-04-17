from keras.datasets import mnist
import numpy as np
import zipfile

def load_training_data():
    (x_train1,y_train1), (x_test, y_test) = mnist.load_data()
    
    data=30000
    #reshaping the training and testing data
    x_train_mnist = x_train1[:(data)].reshape((data, x_train1.shape[1], x_train1.shape[2], 1))
    y_train_mnist = y_train1[:(data)].reshape((data, 1))
    x_test = x_test.reshape(x_test.shape[0],28,28,1)
    y_test = y_test.reshape(y_test.shape[0],1)

    path = "./mnistlikedataset.npz"
    with np.load(path) as data:
        X_train = data['DataX']
        Y_train = data['DataY']
    Y_train = Y_train.reshape((Y_train.shape[0], 1))
    x_train = np.concatenate([x_train_mnist,X_train])
    y_train = np.concatenate([y_train_mnist,Y_train])
    print("MNIST DATA",X_train.shape,Y_train.shape)

    return x_train,y_train

    

def load_validation_data():
    
    (temp_train,tempy_train) ,(X_test,Y_test) = mnist.load_data()
    X_test = X_test.reshape(X_test.shape[0],28,28,1)
    Y_test = Y_test.reshape(Y_test.shape[0],1)
    
    return X_test,Y_test


