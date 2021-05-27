from utilis import *
import tensorflow as tf
from tensorflow_core.python.keras.models import Model
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Discriminator(Model):
    def __init__(self, hidden_size):
        super().__init__(name='discriminator_net')
        self.hidden_size = hidden_size
        self.dense1 = tf.keras.layers.Dense(self.hidden_size)
        self.dense2 = tf.keras.layers.Dense(self.hidden_size)
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, data):
        states, rewards, actions = data
        states = tf.cast(states, tf.float32)
        rewards = tf.cast(rewards, tf.float32)
        actions = tf.cast(actions, tf.float32)
        # 每一步鉴别的概率
        discriminator_probability = tf.zeros(shape=[tf.shape(states)[0], 0])
        hidden_representation = tf.zeros(shape=[tf.shape(states)[0], 0, self.hidden_size])
        for step in range(tf.shape(states)[1]):
            state = states[:, step, :]
            # reward = rewards[:, step, :]
            # action = actions[:, step, :]
            # features = tf.concat((state, tf.reshape(reward, [-1, 1]), tf.reshape(action, [-1, 1])), axis=1)
            features = state
            hidden_1 = self.dense1(features)
            hidden_2 = self.dense2(hidden_1)
            hidden_3 = self.dense3(hidden_2)
            discriminator_probability = tf.concat((discriminator_probability, tf.reshape(hidden_3, [-1, 1])), axis=1)
            hidden_representation = tf.concat((hidden_representation, tf.reshape(hidden_2, [-1, 1, self.hidden_size])), axis=1)
        result_probs = np.zeros_like(discriminator_probability)
        # 累计每一步的概率
        probs = tf.cumsum(discriminator_probability, axis=1)
        # 计算sigmoid后累计平均概率
        for step in range(tf.shape(probs)[1]):
            result_probs[:, step] = tf.keras.activations.sigmoid(probs[:, step]/(step+1.0))
        return hidden_representation, result_probs



