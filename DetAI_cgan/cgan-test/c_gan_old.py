"""
Implement DCGan model based on: https://www.tensorflow.org/tutorials/generative/dcgan.
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import itertools
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import cv2
from PIL import Image
from sklearn.metrics import confusion_matrix
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, AveragePooling2D, MaxPooling2D, UpSampling2D, Input, Reshape, Lambda, Concatenate, Subtract, Reshape, multiply
from keras.layers import Embedding, ZeroPadding2D
from keras.regularizers import l2
from keras.layers import LeakyReLU
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.metrics import categorical_accuracy, binary_accuracy
from keras.callbacks import Callback, History
import tensorflow as tf
import pandas as pd
import glob
from typing import Tuple
import PIL
from sys import getsizeof
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.ticker import MaxNLocator
from sklearn.utils import class_weight
from keras.datasets import mnist
import model_def

import tensorflow as tf
from tensorflow.keras import layers
num_classes=10
img_shape=(28,28,1)
import data




def define_discriminator():
    model = Sequential()
    model.add(Dense(512, input_dim=np.prod(img_shape)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=img_shape)
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Flatten()(Embedding(num_classes, np.prod(img_shape))(label))
    flat_img = Flatten()(img)

    model_input = multiply([flat_img, label_embedding])

    validity = model(model_input)

    return Model([img, label], validity)
    
def disc_loss(real_output, fake_output):
    
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    
    return total_loss


def define_generator(latent_dim, n_classes=10):
    
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    model.summary()

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))

    model_input = multiply([noise, label_embedding])
    img = model(model_input)
    return Model([noise, label], img)

def loss(fake_output, gen_output, target, lambda_=100):
    
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (lambda_ * l1_loss)

    return total_gen_loss, gan_loss, l1_loss
    


def load_real_samples():
    (trainX, trainy), (_, _) = load_data()
    X = expand_dims(trainX, axis=-1)
    X = X.astype('float32')
    X = (X - 127.5) / 127.5
    return [X, trainy]
 

def generate_real_samples(dataset, n_samples):
    
    images, labels = dataset
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = ones((n_samples, 1))
    
    return [X, labels]


def generate_latent_points(latent_dim, n_samples, n_classes=10):
    
    x_input = randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = randint(0, n_classes, n_samples)
    
    return [z_input, labels]
 

def generate_fake_samples(generator, latent_dim, n_samples):
    
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    images = generator.predict([z_input, labels_input])
    y = zeros((n_samples, 1))
    
    return [images, labels_input], y



def define_gan(generator, discriminator,latent_dim):
    
    discriminator.trainable = False
    
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,))
    
    img = generator([noise, label])
    
    valid = discriminator([img, label])
    
    combined = Model([noise, label], valid)
    
    return combined
    

class CGan(tf.keras.Model):
    def __init__(self, batch_size, noise_dim):
        super(CGan, self).__init__()
                

        self.batch_size = batch_size
        self.noise_dim = noise_dim
        
        self.generator = define_generator(self.noise_dim,num_classes)
               
        self.discriminator = define_discriminator()
        
        self.gan=define_gan(self.generator,self.discriminator,self.noise_dim)


        

    def compile(self,discriminator_optimizer, generator_optimizer,cgan_optimizer):
        super(CGan, self).compile()
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.cgan_optimizer=cgan_optimizer



    def train_step(self,batch, verbose=True):
        #(x_train, y_train) in enumerate(batch)
        print("BATCH.........",batch)
        #print("x.........",x_train)
        #print("y.........",y_train)
        #if isinstance(batch, tuple): real_images,labels = batch[0]
        #images,labels = tf.compat.v1.data.make_one_shot_iterator(batch.batch(len(batch))).get_next()
        #print("Here..................", real_images)
        for step, (x_train, y_train) in enumerate(batch):
            images,labels=x_batch_train, y_batch_train
        print(labels)
        #tf.print(batch)
        X_train,Y_train=batch
        if verbose:
            print(f"The type of batch is ",type(batch))
            print(f"Shape of input_images in train_step is {X_train,X_train.shape}")
            print(f"Shape of real_images in train_step is  {Y_train,Y_train.shape}")
         
        #X_train,Y_train=data.get_train_dataset()
        
        #print(X_train)
        shape=60000
        
        #idx = np.random.randint(0, X_train.shape[0], self.batch_size)
        idx = np.random.randint(0,shape, self.batch_size)
        imgs, labels = X_train[idx], Y_train[idx]
    
        noise = np.random.normal(0, 1, (self.batch_size, 100))
        
        X_real, y_real = self.generator(noise, labels)
        
        d_loss1, _ = d_model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
        X_gan = generate_latent_points(self.noise_dim,self.batch_size)
        y_gan = ones((self.batch_size, 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        return g_loss,0.5 * np.add(d_loss1, d_loss2)

    def test_step(self, batch, verbose=True):
        input_images, real_images = batch
        if verbose:
            print(f"Shape of input_images in train_step is {input_images.shape}")
            print(f"Shape of real_images in train_step is  {real_images.shape}")
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        generated_images = self.generator(noise, training=False)

        real_output = self.discriminator(images, training=False)
        fake_output = self.discriminator(generated_images, training=False)

        gen_loss = self.generator_loss(fake_output)
        disc_loss = self.discriminator_loss(real_output, fake_output)
        return {"d_loss": disc_loss, "g_loss": gen_loss}
