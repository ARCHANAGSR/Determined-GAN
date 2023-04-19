import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
def get_train_dataset(worker_rank: int):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    all_digits = np.concatenate([x_train, x_test])
    all_labels = np.concatenate([y_train, y_test])

    # Scale the pixel values to [0, 1] range, add a channel dimension to
    # the images, and one-hot encode the labels.
    all_digits = all_digits.astype("float32") / 255.0
    all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
    all_labels = keras.utils.to_categorical(all_labels, 10)

    # Create tf.data.Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(64)
    return dataset


def get_validation_dataset(worker_rank: int):
    (_, _), (test_images, _) = tf.keras.datasets.mnist.load_data(path=f"mnist-{worker_rank}.npz")

    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype("float32")
    test_images = (test_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(50000)
    return train_dataset
