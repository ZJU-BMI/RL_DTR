import tensorflow as tf
import numpy as np
import os
from tensorflow_core.python.keras.models import Model

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Discriminator(Model):
    def __init__(self, hidden_size, feature_dims, predicted_visit):
        super().__init__(name='discriminator')
        self.hidden_size = hidden_size
        self.feature_dims = feature_dims
        self.predicted_visit = predicted_visit
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation=tf.keras.activations.sigmoid)
        self.dense2 = tf.keras.layers.Dense(hidden_size, activation=tf.keras.activations.sigmoid)
        self.dense3 = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)

    def call(self, data):
        states, rewards, actions = data
        batch = tf.shape(states)[0]
        discriminator_probability = tf.zeros(shape=[batch, 0, 1])
        for step in range(self.predicted_visit):
            state = states[:, step, :]
            action = actions[:, step, :]
            reward = rewards[:, step, :]
            data_input = tf.concat((tf.reshape(state, [batch, self.feature_dims]), tf.reshape(action, [batch, 1]), tf.reshape(reward, [batch, 1])), axis=1)
            hidden = self.dense1(data_input)
            hidden = self.dense2(hidden)
            prob = self.dense3(hidden)
            discriminator_probability = tf.concat((discriminator_probability, tf.reshape(prob, [batch, -1, 1])), axis=1)
        return discriminator_probability

