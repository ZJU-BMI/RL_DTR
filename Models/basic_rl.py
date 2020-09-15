import tensorflow as tf
import numpy as np
from Environment import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data import *
from utilis import *
from utilis import *
from bayes_opt import BayesianOptimization
import os
from discrinmintor import Discriminator
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class PolicyGradientNN(tf.keras.layers.Layer):
    def __init__(self, actor_size, hidden_size):
        super().__init__(name='policy_gradient')
        self.actor_size = actor_size
        self.hidden_size = hidden_size
        self.dense1 = tf.keras.layers.Dense(units=self.hidden_size)
        self.dense2 = tf.keras.layers.Dense(units=self.hidden_size)
        self.dense3 = tf.keras.layers.Dense(units=self.actor_size, activation=tf.nn.sigmoid)

    def call(self, state):
        hidden_1 = self.dense1(state)
        hidden_2 = self.dense2(hidden_1)
        actor_prob = tf.nn.softmax(self.dense3(hidden_2))
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
        log_prob = dist.log_prob(tf.reshape(action, [-1,]))
        loss = -tf.reduce_sum(log_prob*tf.reshape(reward, [-1,]))
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


# 训练agent 单个病人的单次记录训练
def train():
    actor_size = 65
    hidden_size = 128
    previous_visit = 3
    gamma = 0.01
    predicted_visit = 7
    imbalance_1 = 1
    imbalance_2 = 10
    learning_rate = 0.01

    data_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA.npy')[:, :, 1:]  # 去掉当前天数
    construct_environment = ReconstructEnvir(hidden_size=32,
                                             feature_dims=39,
                                             previous_visit=3)
    construct_environment(tf.ones(shape=[3, 4, 40]))
    construct_environment.load_weights('environment_3_7_39.h5')
    batch = data_set.shape[0]

    rewards_all_episodes = []
    for patient in range(10000):
        agent = Agent(actor_size=actor_size, hidden_size=hidden_size, gamma=gamma, learning_rate=learning_rate)
        rewards = []
        states = []
        actions = []
        total_reward = 0
        for step in range(predicted_visit):
            if step == 0:
                state = data_set[0, step+previous_visit-1, 1:]  # s_3 39个变量
                action = agent.act(tf.reshape(state, [1, -1]))
                state_to_now = data_set[0, :step+previous_visit-1, :]  # (s_1, a_1)..(s_2, a_2)
                state_action = np.concatenate((np.array(action).reshape(1, -1), state.reshape(1, -1)), axis=1)
                state_to_now = np.concatenate((state_to_now, state_action)).reshape(1, -1, state_to_now.shape[1])
                next_state = construct_environment(state_to_now)
                reward = (next_state[0, 0] - state[1]) * imbalance_1
                rewards.append(reward)
                states.append(state)
                actions.append(action)
                total_reward += reward

            else:
                state = next_state.numpy()
                action = agent.act(state)
                state_action = np.concatenate((np.array(action).reshape(1, -1), state.reshape(1, -1)), axis=1)
                state_to_now_ = data_set[0, :previous_visit+step-1, :]  # (s_1, a_1),(s_2, a_2)
                states_array = np.zeros(shape=[0, 39])
                actions_array = np.zeros(shape=[0, 1])
                for i in range(len(states)):
                    states_array = np.concatenate((states_array, np.array(states[i]).reshape(1, -1)), axis=0)
                    actions_array = np.concatenate((actions_array, np.array(actions[i]).reshape(1, -1)), axis=0)
                state_actions = np.concatenate((actions_array, states_array), axis=1)
                state_to_now = np.concatenate((state_to_now_, state_actions, state_action), axis=0)
                next_state = construct_environment(state_to_now.reshape(1, -1, state_to_now.shape[1]))
                reward = (next_state[0, 0] - state[0, 1]) * imbalance_1
                rewards.append(reward)
                states.append(state)
                actions.append(action)
                total_reward += reward

                if step == predicted_visit-1:
                    total_reward += (next_state[0, 3] - state[0, 3]) * imbalance_2

        agent.train(states, rewards, actions)
        print('total_reward after  {} step is {}'.format(patient, total_reward))
        rewards_all_episodes.append(total_reward)
    plot(rewards_all_episodes)
    # tf.compat.v1.reset_default_graph()


def train_batch(hidden_size, learning_rate):
    train_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA.npy')[:, :, 1:]  # 去掉当前天数这个变量---->40个变量
    test_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\test_PLA.npy')[:, :, 1:]
    batch_size = 128
    feature_size = train_set.shape[2]-1
    train_set = DataSet(train_set)
    epochs = 10
    actor_size = 65
    previous_visit = 3
    predicted_visit = 7

    hidden_size = 2 ** int(hidden_size)
    learning_rate = 10 ** learning_rate
    # imbalance_1 = 10 ** int(imbalance_1)
    # imbalance_2 = 10 ** int(imbalance_2)
    gamma = 0.99
    imbalance_1 = 1
    imbalance_2 = 1
    action_list = [0, 0.035714286, 0.0625, 0.071428571, 0.125, 0.1875, 0.25, 0.285714286, 0.3125, 0.321428571,
                   0.375, 0.4375, 0.446428571, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.875, 1,
                   1.125, 1.25, 1.375, 1.5, 1.55, 1.625, 1.75, 1.875, 2, 2.125,
                   2.25, 2.5, 2.75, 3, 3.125, 3.25, 3.375, 3.5, 3.75, 4,
                   4.25, 4.5, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 7,
                   7.5, 7.75, 8, 8.25, 8.5, 9, 9.625, 10, 11, 13.5,
                   14.5, 15, 21, 22.25, 25.625]
    print('hidden_size---{}---gamma---{}---imbalance_1---{}---imbalance_2----{}---learning_rate---{}'
          .format(hidden_size, gamma, imbalance_1, imbalance_2, learning_rate))

    construct_environment = ReconstructEnvir(hidden_size=32,
                                             feature_dims=39,
                                             previous_visit=3)
    construct_environment(tf.ones(shape=[3, 4, 40]))
    construct_environment.load_weights('environment_3_7_39.h5')

    agent = Agent(actor_size=actor_size, hidden_size=hidden_size, gamma=gamma, learning_rate=learning_rate)

    logged = set()

    while train_set.epoch_completed < epochs:
        input_x_train = train_set.next_batch(batch_size=batch_size)
        batch = input_x_train.shape[0]
        rewards = tf.zeros(shape=[batch, 0, 1])
        states = tf.zeros(shape=[batch, 0, feature_size])
        actions = tf.zeros(shape=[batch, 0, 1])
        # total_rewards = tf.zeros(shape=[batch, 1])

        for step in range(predicted_visit):
            if step == 0:
                # state = initial_state  # 39个变量
                initial_state = input_x_train[:, step + previous_visit, 1:]
                state_to_now_ = input_x_train[:, :step + previous_visit, :]
                action_index = agent.act(initial_state)
                action_value = np.zeros_like(action_index)
                for i in range(tf.shape(action_index)[0]):
                    for j in range(tf.shape(action_index)[1]):
                        action_value[i, j] = action_list[action_index[i, j]]
                state_action = tf.concat((initial_state.reshape(batch, 1, -1), tf.reshape(action_value, [batch, -1, 1])), axis=2)
                state_to_now = tf.concat((state_to_now_, state_action), axis=1)
                next_state = construct_environment(state_to_now)

                states = tf.concat((states, initial_state.reshape(batch, -1, feature_size)), axis=1)
                actions = tf.concat((actions, tf.reshape(action_value, [batch, -1, 1])), axis=1)

                reward = (next_state[:, 0] - next_state[:, 1]) * imbalance_1  # 新状态的出入量差值
                rewards = tf.concat((rewards, tf.reshape(reward, [batch, -1, 1])), axis=1)
                # total_rewards += tf.reshape(reward, [batch, -1])

            else:
                initial_state = input_x_train[:, previous_visit, 1:]
                state_to_now_ = input_x_train[:, :previous_visit, :]
                state = next_state
                action_index = agent.act(state)
                action_value = np.zeros_like(action_index)
                for i in range(tf.shape(action_index)[0]):
                    for j in range(tf.shape(action_index)[1]):
                        action_value[i, j] = action_list[action_index[i, j]]
                state_action = tf.concat((tf.reshape(state, [batch, -1, feature_size]), tf.reshape(action_value, [batch, -1, 1])), axis=2)
                state_to_now__ = tf.concat((states, actions), axis=2)
                state_to_now = tf.concat((state_to_now_, state_to_now__, state_action), axis=1)
                next_state = construct_environment(state_to_now)

                states = tf.concat((states, tf.reshape(state, [batch, -1, feature_size])), axis=1)
                actions = tf.concat((actions, tf.reshape(action_value, [batch, -1, 1])), axis=1)
                reward = (next_state[:, 0] - next_state[:, 1]) * imbalance_1

                if step == predicted_visit-1:
                    reward += (initial_state[:, 3] - next_state[:, 3]) * imbalance_2

                rewards = tf.concat((rewards, tf.reshape(reward, [batch, -1, 1])), axis=1)
                # total_rewards += tf.reshape(reward, [batch, -1])
        loss = agent.train(states=states, actions=actions, rewards=rewards)
        discont_rewards = agent.discont_reward(states=states, rewards=rewards)

        if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
            logged.add(train_set.epoch_completed)
            batch_test = test_set.shape[0]
            rewards_test = tf.zeros(shape=[batch_test, 0, 1])
            states_test = tf.zeros(shape=[batch_test, 0, feature_size])
            actions_test = tf.zeros(shape=[batch_test, 0, 1])
            # total_rewards_test = tf.zeros(shape=[batch_test, 1])

            for step in range(predicted_visit):
                initial_state_test = test_set[:, step+previous_visit, 1:]
                state_to_now_test_ = test_set[:, :step + previous_visit, :]
                if step == 0:
                    state_test = tf.cast(initial_state_test, tf.float32)
                    action_test_index = agent.act(state_test)
                    action_test_value = np.zeros_like(action_test_index)
                    for i in range(tf.shape(action_test_index)[0]):
                        for j in range(tf.shape(action_test_index)[1]):
                            action_test_value[i, j] = action_list[action_test_index[i, j]]
                    state_action_test = tf.concat((tf.reshape(state_test, [batch_test, -1, feature_size]), tf.reshape(action_test_value, [batch_test, -1, 1])), axis=2)

                    state_to_now_test = tf.concat((state_to_now_test_, state_action_test), axis=1)
                    next_state_test = construct_environment(state_to_now_test)
                    reward_test = (next_state_test[:, 0] - next_state_test[:, 1]) * imbalance_1

                    rewards_test = tf.concat((rewards_test, tf.reshape(reward_test, [batch_test, 1, 1])), axis=1)
                    states_test = tf.concat((states_test, tf.reshape(state_test, [batch_test, -1, feature_size])), axis=1)
                    actions_test = tf.concat((actions_test, tf.reshape(action_test_value, [batch_test, -1, 1])), axis=1)
                    # total_rewards_test += tf.reshape(reward_test, [batch_test, -1])

                else:
                    state_test = next_state_test
                    action_test_index = agent.act(state_test)
                    action_test_value = np.zeros_like(action_test_index)
                    for i in range(tf.shape(action_test_index)[0]):
                        for j in range(tf.shape(action_test_index)[1]):
                            action_test_value[i, j] = action_list[action_test_index[i, j]]
                    state_action_test = tf.concat((tf.reshape(state_test, [batch_test, -1, feature_size]), tf.reshape(action_test_value, [batch_test, -1, 1])), axis=2)
                    state_to_now_test__ = tf.concat((states_test, actions_test), axis=2)

                    state_to_now_test = tf.concat((state_to_now_test_, state_to_now_test__, state_action_test), axis=1)
                    next_state_test = construct_environment(state_to_now_test)
                    reward_test = (next_state_test[:, 0] - next_state_test[:, 1]) * imbalance_1

                    if step == predicted_visit-1:
                        reward_test += (initial_state_test[:, 3] - next_state_test[:, 3]) * imbalance_2

                    rewards_test = tf.concat((rewards_test, tf.reshape(reward_test, [batch_test, -1, 1])), axis=1)
                    states_test = tf.concat((states_test, tf.reshape(state_test, [batch_test, -1, feature_size])), axis=1)
                    actions_test = tf.concat((actions_test, tf.reshape(action_test_value, [batch_test, -1, 1])), axis=1)
                    # total_rewards_test += tf.reshape(reward_test, [batch_test, -1])
            discont_rewards_test = agent.discont_reward(states=states_test, rewards=rewards_test)
            print('epoch {}    train_total_reward {}   train_loss {}    test_total_reward {}'
                  .format(train_set.epoch_completed, np.mean(discont_rewards), loss, np.mean(discont_rewards_test)))
    tf.compat.v1.reset_default_graph()
    return np.mean(discont_rewards_test)


if __name__ == '__main__':
    test_test('9_14_训练agent_3_7_train_修改action数值.txt')
    Agent_BO = BayesianOptimization(
        train_batch, {
            'hidden_size': (5, 8),
            'learning_rate': (-5, -1),
        }
    )
    Agent_BO.maximize()
    print(Agent_BO.max)
    # train_batch(hidden_size=64, gamma=0.99, imbalance_1=0.1, imbalance_2=0.5, learning_rate=0.01)
















