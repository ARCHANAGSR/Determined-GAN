"""
This example demonstrates how to train a GAN with Determined's TF Keras API.

The Determined TF Keras API support using a subclassed `tf.keras.Model` which
defines a custom `train_step()` and `test_step()`.
"""

import tensorflow as tf
from data import get_train_dataset, get_validation_dataset
from c_gan import CGan
from typing import Any, Dict, Tuple
import numpy as np

from determined.keras import InputData, TFKerasTrial, TFKerasTrialContext


class CGanTrial(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext) -> None:
        self.context = context

    def build_model(self) -> tf.keras.models.Model:
        model = CGan(

            batch_size=self.context.get_per_slot_batch_size(),
            noise_dim=self.context.get_hparam("noise_dim"),
        )

        # Wrap the model.
        model = self.context.wrap_model(model)

        # Create and wrap the optimizers.
        g_optimizer = tf.keras.optimizers.Adam(    learning_rate=self.context.get_hparam("generator_lr")     )
        g_optimizer = self.context.wrap_optimizer(g_optimizer)
        
        d_optimizer = tf.keras.optimizers.Adam(    learning_rate=self.context.get_hparam("discriminator_lr")     )
        d_optimizer = self.context.wrap_optimizer(d_optimizer)
        
               
        gan_optimizer = tf.keras.optimizers.Adam(    learning_rate=self.context.get_hparam("discriminator_lr")     )
        gan_optimizer = self.context.wrap_optimizer(gan_optimizer)
       
         
        model.compile(
            discriminator_optimizer=d_optimizer,
            generator_optimizer=g_optimizer,
            cgan_optimizer=gan_optimizer
        )
        
        

        return model

    
        
   # def build_training_data_loader(self)->InputData:
        
       # X_train,Y_train = get_train_dataset()

             
      #  train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
       # train_label = tf.data.Dataset.from_tensor_slices(Y_train)
       # ds = tf.data.Dataset.zip((train_dataset,train_label))
       # ds_train = self.context.wrap_dataset(ds)
        
        
        #return ds_train
        
    def build_training_data_loader(self) -> InputData:
        train_dataset = get_train_dataset(self.context.distributed.get_rank())
        #train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        #train_dataset = train_dataset.shuffle(buffer_size=1024).batch(128)
        #t = (X_train,Y_train)
        #train_dataset = self.context.wrap_dataset(train_dataset)
        #return (train_dataset)

        ds = self.context.wrap_dataset(train_dataset)
        return ds
    
    def build_validation_data_loader(self) -> InputData:
        test_dataset = get_validation_dataset(self.context.distributed.get_rank())
        # Prepare the validation dataset.
        #val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        #val_dataset = val_dataset.batch(128)
        #val_dataset = self.context.wrap_dataset(val_dataset) 
        #v = (X_test,Y_test)
        #return (val_dataset)

        ds = self.context.wrap_dataset(test_dataset)
        return ds
    