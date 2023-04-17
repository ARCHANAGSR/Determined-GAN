"""
This example shows how to use Determined to implement an image
classification model to validate the accuracy of a model with GAN generated images.
"""
#********************** IMPORTANT ****************************#
#Uncomment the following line to import mnist data alone
#from  data1 import load_training_data,load_validation_data
#Comment the above line and umcomment the below line for mnist + GAN generated imges
from  data import load_training_data,load_validation_data

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
import numpy as np
from keras.layers import Activation, Dense, Flatten, Dropout, Conv2D, Conv2DTranspose, AveragePooling2D, MaxPool2D, UpSampling2D, Input, Reshape, Lambda, Concatenate, Subtract, Reshape, multiply,LeakyReLU
from tensorflow.keras.optimizers import Adam

from determined.keras import InputData, TFKerasTrial, TFKerasTrialContext

def classifier(img_shape):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=img_shape))
    model.add(MaxPool2D((2,2)))
    model.add(Conv2D(48, (3,3), activation='relu'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    return model


class DigitMNISTTrial(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext) -> None:
        self.context = context
        self.img_shape=(28,28,1)

    def build_model(self):
        model = classifier(self.img_shape)

        # Wrap the model.
        model = self.context.wrap_model(model)

        # Create and wrap the optimizer.
        optimizer = tf.keras.optimizers.Adam()
        optimizer = self.context.wrap_optimizer(optimizer)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        return model

    def build_training_data_loader(self) -> InputData:
        train_images, train_labels = load_training_data()
        train_images = train_images / 255.0

        return train_images, train_labels

    def build_validation_data_loader(self) -> InputData:
        test_images, test_labels = load_validation_data()
        test_images = test_images / 255.0

        return test_images, test_labels
