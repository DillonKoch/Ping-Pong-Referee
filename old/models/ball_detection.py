# ==============================================================================
# File: ball_detection.py
# Project: models
# File Created: Saturday, 6th March 2021 8:52:41 pm
# Author: Dillon Koch
# -----
# Last Modified: Saturday, 6th March 2021 9:00:57 pm
# Modified By: Dillon Koch
# -----
# Collins Aerospace
#
# -----
# global and local ball detection
# ==============================================================================


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.load_data import Load_Video

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)

x_train = x_train / 255
x_test = x_test / 255


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs['accuracy'] > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True
        # if logs['loss'] < 0.01:
        #     print("Loss less than 0.01, so stopping training!")
        #     self.model.stop_training = True


callbacks = myCallback()
model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 320, 1)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(2, activation='relu')])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
