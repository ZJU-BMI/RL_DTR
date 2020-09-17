import tensorflow as tf
import numpy as np
from Environment import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data import *
from utilis import *
from tensorflow_core.python.keras.models import Model
from bayes_opt import BayesianOptimization
import os
from discrinmintor import Discriminator
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class PolicyGradientNN(Model):
    def __init__(self, actor_size, hidden_size):
        super().__init__(name='policy_gradient')
        self.actor_size = actor_size
        self.hidden_size = hidden_size
        self.dense1 = tf.keras.layers.Dense(units=self.hidden_size, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=self.actor_size, activation=tf.nn.relu)
        # self.dense3 = tf.keras.layers.Dense(units=self.actor_size)

    def call(self, state):
        hidden_1 = self.dense1(state)
        hidden_2 = self.dense2(hidden_1)
        actor_prob = tf.nn.softmax(hidden_2)  # 默认在最后一个维度上进行softmax
        return actor_prob


class Agent(object):
    def __init__(self, actor_size, hidden_size, gamma, learning_rate):
        self.name = 'agent'
        self.actor_size = actor_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.model = PolicyGradientNN(actor_size=actor_size, hidden_size=hidden_size)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    def act(self, state):
        prob = self.model(state)
        dist = tf.compat.v1.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        action = tf.reshape(action, [-1, 1])
        return action

    def loss(self, prob, action, reward):
        dist = tf.compat.v1.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(tf.reshape(action, [-1, ]))
        # log_prob = tf.clip_by_value(log_prob, [1e-5, 1])
        loss = -(log_prob*tf.reshape(reward, [-1, ]))
        return loss

    #  计算每条路径的return[batch, 1]
    def discont_reward(self, states, rewards):
        sum_reward = 0
        batch = tf.shape(states)[0]
        discnt_reward = tf.zeros(shape=[batch, 0, 1])
        rewards = tf.reverse(rewards, axis=[1])

        for r_index in range(tf.shape(rewards)[1]):
            r = rewards[:, r_index, :]
            sum_reward = r + self.gamma * sum_reward
            discnt_reward = tf.concat((discnt_reward, tf.reshape(sum_reward, [batch, -1, 1])), axis=1)
        discnt_reward = tf.reverse(discnt_reward, axis=[1])
        return discnt_reward

    def train(self, states, rewards, actions):
        loss = 0
        discnt_reward = self.discont_reward(states=states, rewards=rewards)
        for step in range(tf.shape(states)[1]):
            with tf.GradientTape() as tape:
                state = states[:, step, :]
                action = actions[:, step, :]
                reward = discnt_reward[:, step, :]
                p = self.model(state)
                loss += self.loss(p, action, reward)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


def pre_train_policy(hidden_size, learning_rate):
    actor_size = 65
    train_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA.npy')[:, :, 1:]
    test_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\validate_PLA.npy')[:, :, 1:]
    train_set = DataSet(train_set)

    # hidden_size = 2 ** int(hidden_size)
    # learning_rate = 10 ** learning_rate
    print('learning_rate----{}----hidden_size---{}-----'.format(learning_rate, hidden_size))

    policy_net = PolicyGradientNN(actor_size=actor_size, hidden_size=hidden_size)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    epochs = 20
    batch_size = 64
    predicted_visit = 7
    previous_visit = 3
    logged = set()

    while train_set.epoch_completed < epochs:
        input_x_train = train_set.next_batch(batch_size=batch_size)
        action_list = [0, 0.0357142873108387, 0.0625, 0.071428571, 0.125, 0.1875, 0.25, 0.2857142984867096, 0.3125, 0.321428571,
                       0.375, 0.4375, 0.446428571, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.875, 1,
                       1.125, 1.25, 1.375, 1.5, 1.55, 1.625, 1.75, 1.875, 2, 2.125,
                       2.25, 2.5, 2.75, 3, 3.125, 3.25, 3.375, 3.5, 3.75, 4,
                       4.25, 4.5, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 7,
                       7.5, 7.75, 8, 8.25, 8.5, 9, 9.625, 10, 11, 13.5,
                       14.5, 15, 21, 22.25, 25.625]
        batch = input_x_train.shape[0]
        action = input_x_train[:, previous_visit:previous_visit+predicted_visit-1, 0]
        action_labels = np.zeros_like(action)
        for i in range(action.shape[0]):
            for j in range(action.shape[1]):
                action_labels[i, j] = action_list.index(action[i, j])
        with tf.GradientTape() as tape:
            action_probs = tf.zeros(shape=[batch, 0, actor_size])
            for step in range(predicted_visit-1):
                state = input_x_train[:, step+previous_visit, 1:]
                action_prob = policy_net(state)
                action_probs = tf.concat((action_probs, tf.reshape(action_prob, [batch, -1, actor_size])), axis=1)
            loss = tf.keras.losses.sparse_categorical_crossentropy(tf.reshape(action_labels, [-1,]), tf.reshape(action_probs, [-1,actor_size]), from_logits=False)

            gradient = tape.gradient(loss, policy_net.trainable_variables)
            optimizer.apply_gradients(zip(gradient, policy_net.trainable_variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)
                batch_test = test_set.shape[0]
                predicted_probs = tf.zeros(shape=[batch_test, 0, actor_size])
                predicted_actions = test_set[:, previous_visit:previous_visit+predicted_visit-1, 0]
                predicted_lables = np.zeros_like(predicted_actions)

                for i in range(predicted_actions.shape[0]):
                    for j in range(predicted_actions.shape[1]):
                        predicted_lables[i, j] = action_list.index(predicted_actions[i, j])
                for step in range(predicted_visit-1):
                    state_predict = test_set[:, step+previous_visit, 1:]
                    action_pred = policy_net(state_predict)
                    predicted_probs = tf.concat((predicted_probs, tf.reshape(action_pred, [batch_test, -1, actor_size])), axis=1)

                predicted_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.reshape(predicted_lables, [-1, ]), tf.reshape(predicted_probs, [-1, actor_size]), from_logits=False)
                print('epoch---{}----train_loss---{}---predicted_loss---{}'.format(train_set.epoch_completed,
                                                                                   np.mean(loss),
                                                                                   np.mean(predicted_loss)))
                if np.mean(predicted_loss) < 1.800:
                    policy_net.save_weights('policy_bet_9_17.h5')

    tf.compat.v1.reset_default_graph()
    # return np.mean(predicted_loss)


# 对policy网络进行预训练
if __name__ == '__main__':
    test_test('9_17_预训练actor网络.txt')
    # Agent_BO = BayesianOptimization(
    #     pre_train_policy, {
    #         'hidden_size': (5, 8),
    #         'learning_rate': (-5, -1),
    #     }
    # )
    # Agent_BO.maximize()
    # print(Agent_BO.max)
    for i in range(50):
        pre_train_policy(hidden_size=128, learning_rate=0.001103128653571744)
















