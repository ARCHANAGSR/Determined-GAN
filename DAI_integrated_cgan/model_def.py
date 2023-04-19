"""
This example demonstrates how to train a GAN with Determined's TF Keras API.

The Determined TF Keras API support using a subclassed `tf.keras.Model` which
defines a custom `train_step()` and `test_step()`.
"""

import tensorflow as tf
from data import get_train_dataset, get_validation_dataset
from c_gan import ConditionalGAN

from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


from determined.keras import InputData, TFKerasTrial, TFKerasTrialContext


class ConditionalGANTrial(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext) -> None:
        self.context = context     

    def build_model(self) ->   tf.keras.models.Model:
        model = ConditionalGAN(
            latent_dim=self.context.get_hparam("noise_dim"),            
            batch_size=self.context.get_per_slot_batch_size(), 
            #generator_in_channels = self.context.get_hparam("noise_dim") + 10,
            #discriminator_in_channels = 1+ 10,
            discriminator=keras.Sequential(
            [
                keras.layers.InputLayer((28, 28, 11)),
                layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.GlobalMaxPooling2D(),
                layers.Dense(1),
            ],
            name="discriminator",
            ),
            
            generator=keras.Sequential(
            [
                keras.layers.InputLayer((128+10)),
                # We want to generate 128 + num_classes coefficients to reshape into a
                # 7x7x(128 + num_classes) map.
                layers.Dense(7 * 7 * (128+10)),
                layers.LeakyReLU(alpha=0.2),
                layers.Reshape((7, 7,(128+10))),
                layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
            ],
            name="generator",
            ),
            
        
        )

        # Wrap the model.
        model = self.context.wrap_model(model)

        # Create and wrap the optimizers.
        g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.context.get_hparam("generator_lr")
        )
        g_optimizer = self.context.wrap_optimizer(g_optimizer)

        d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.context.get_hparam("discriminator_lr")
        )
        d_optimizer = self.context.wrap_optimizer(d_optimizer)

        model.compile(
            d_optimizer=d_optimizer,
            g_optimizer=g_optimizer,
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
        )

        return model

    def build_training_data_loader(self) -> InputData:
        ds = get_train_dataset(self.context.distributed.get_rank())
        print("ds====")
        print(ds)
        # Wrap the training dataset.
        ds = self.context.wrap_dataset(ds)
        #ds = ds.batch(self.context.get_per_slot_batch_size())
        print(ds)
        return ds

    def build_validation_data_loader(self) -> InputData:
        ds = get_validation_dataset(self.context.distributed.get_rank())

        # Wrap the validation dataset.
        ds = self.context.wrap_dataset(ds)
        ds = ds.batch(self.context.get_per_slot_batch_size())
        return ds
