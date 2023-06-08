"""
Implement CGan model based on: https://www.tensorflow.org/tutorials/generative/dcgan.
"""

import os
import time
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import layers


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import imageio
from torchvision.utils import save_image
from tensorflow.python.ops.numpy_ops import np_config
from PIL import Image


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim, batch_size):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile(run_eagerly=True)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        tf.compat.v1.enable_eager_execution()
        real_images, one_hot_labels = data
        image_size = 28
        num_classes = 10
        print(f"one_hot_labels = {one_hot_labels.shape}")
        print(f"real_images = {real_images.shape}")
        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[image_size * image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, image_size, image_size, num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_images)[0]
        print(f"batch_size={batch_size}")
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        print(f"random_latent_vectors={random_latent_vectors.shape}")
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        #fake_images *= 255.0
        #converted_images = fake_images.astype(np.uint8)
        #converted_images = tf.image.resize(converted_images, (96, 96)).numpy().astype(np.uint8)
        #for idx, img in enumerate(converted_images):
        #
        #imageio.mimsave("/tmp/cganDiv/images/generated_image.png", fake_images)
        #save_image(fake_images, 'img1.png')

        #sess = tf.compat.v1.Session()
        #writer = tf.io.write_file('image1.jpg', fake_images)
        #sess.run(writer)
        #imageio.mimsave("animation.gif", fake_images, fps=1)
        #embed.embed_file("animation.gif")

        np_config.enable_numpy_behavior()
        num_interpolation = 9
        start_class = 1  # @param {type:"slider", min:0, max:9, step:1}
        end_class = 5  # @param {type:"slider", min:0, max:9, step:1}


        fake_images = self.interpolate_class(start_class, end_class)

        #with tf.compat.v1.Session() as sess:
            # Run the computation of the tensor within the session
            #image_array = sess.run(fake_images)
            # Convert the image array to a PIL Image object
            #image = Image.fromarray(image_array)
            # Save the image
            #image.save('output_image.jpg')

        #fake_images *= 255.0
        #converted_images = fake_images.astype(np.uint8)
        #converted_images = tf.image.resize(converted_images, (96, 96)).numpy().astype(np.uint8)
        #imageio.mimsave("animation.gif", converted_images, fps=1)
        #embed.embed_file("animation.gif")

        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }


    def interpolate_class(self, first_number, second_number):
        num_interpolation = 9
        num_classes = 10

        interpolation_noise = tf.random.normal(shape=(1, self.latent_dim))
        interpolation_noise = tf.repeat(interpolation_noise, repeats=num_interpolation)
        interpolation_noise = tf.reshape(interpolation_noise, (num_interpolation, self.latent_dim))

        # Convert the start and end labels to one-hot encoded vectors.
        first_label = keras.utils.to_categorical([first_number], num_classes)
        second_label = keras.utils.to_categorical([second_number], num_classes)
        first_label = tf.cast(first_label, tf.float32)
        second_label = tf.cast(second_label, tf.float32)

        # Calculate the interpolation vector between the two labels.
        percent_second_label = tf.linspace(0, 1, num_interpolation)[:, None]
        percent_second_label = tf.cast(percent_second_label, tf.float32)
        interpolation_labels = (
            first_label * (1 - percent_second_label) + second_label * percent_second_label
        )

        # Combine the noise and the labels and run inference with the generator.
        noise_and_labels = tf.concat([interpolation_noise, interpolation_labels], 1)
        fake = self.generator(noise_and_labels)
        return fake

    def test_step(self, data):

        test_images, one_hot_labels = data
        print(f"Vikrant label {one_hot_labels}")

        batch_size = tf.shape(test_images)[0]
        #tf.compat.v1.disable_eager_execution() # need to disable eager in TF2.x
        num_interpolation = 9
        start_class = 1  # @param {type:"slider", min:0, max:9, step:1}
        end_class = 5  # @param {type:"slider", min:0, max:9, step:1}
        image_size = 28
        num_classes = 10

        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[image_size * image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, image_size, image_size, num_classes)
        )

        # Sample random points in the latent space.
        print(f"Vikrant {batch_size} {self.latent_dim}")
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
             [random_latent_vectors, one_hot_labels], axis=1
        )


        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        generated_images = self.generator(random_vector_labels, training=False)
            # fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)

        generated_images_eval = generated_images.numpy()

        #generated_images = self.generator(noise, training=False)
        #generated_images_eval = generated_images.numpy()
        print("generated_images_eval")

        plt.figure(figsize=(15, 15))

        print("Vikrant shapes")
        print(test_images.shape)
        print(generated_images_eval.shape)
        print(one_hot_labels)

        display_list = [test_images, generated_images_eval]
        title = ["Predicted Image"]


        for index in range(64):
            for i in range(len(display_list)):
                 plt.subplot(1, 2, i + 1)
                 #plt.title(title[i])
                 # Getting the pixel values in the [0, 1] range to plot.
                 timestr = time.strftime("%Y%m%d-%H%M%S")
                 plt.imshow(display_list[i][index] * 0.5 + 0.5)
                 plt.axis("off")
                 # plt.show()
            plt.savefig("images/generated_img_" + timestr + ".png")

        cwd = os.getcwd()
        files = os.listdir(cwd)
        print("Files in %r: %s" % (cwd, files))

        print(generated_images_eval.shape)
        fake_image_and_labels = tf.concat([generated_images_eval, image_one_hot_labels], -1)
        test_image_and_labels = tf.concat([test_images, image_one_hot_labels], -1)


        real_output = self.discriminator(test_image_and_labels, training=False)
        fake_output = self.discriminator(fake_image_and_labels, training=False)

        gen_loss = self.generator_loss(fake_output)
        disc_loss = self.discriminator_loss(real_output, fake_output)
        return {"d_loss": disc_loss, "g_loss": gen_loss}

