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


class PolicyGradientNN(tf.keras.layers.Layer):
    def __init__(self, actor_size, hidden_size):
        super().__init__(name='policy_gradient')
        self.actor_size = actor_size
        self.hidden_size = hidden_size
        self.dense1 = tf.keras.layers.Dense(units=self.hidden_size, activation=tf.keras.activations.sigmoid)
        self.dense2 = tf.keras.layers.Dense(units=self.hidden_size, activation=tf.keras.activations.sigmoid)
        self.dense3 = tf.keras.layers.Dense(units=self.actor_size, activation=tf.nn.sigmoid)

    def actor_prob(self, state):
        hidden_1 = self.dense1(state)
        hidden_2 = self.dense2(hidden_1)
        actor_prob = self.dense3(hidden_2)
        return actor_prob

    def call(self, state):
        hidden_1 = self.dense1(state)
        hidden_2 = self.dense2(hidden_1)
        actor_prob = self.dense3(hidden_2)
        # prob = self.actor_prob(state)
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
        return action

    def loss(self, prob, action, reward):
        dist = tf.compat.v1.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(tf.reshape(action, [-1,]))
        loss = -tf.reduce_sum(log_prob*tf.reshape(reward, [-1,]))
        return loss

    def train(self, states, rewards, actions):
        sum_reward = 0
        batch = tf.shape(states)[0]
        discnt_reward = tf.zeros(shape=[batch, 0, 1])
        rewards = tf.reverse(rewards, axis=[1])
        for r_index in range(tf.shape(rewards)[1]):
            r = rewards[:, r_index, :]
            sum_reward = r + self.gamma * sum_reward
            discnt_reward = tf.concat((discnt_reward, tf.reshape(sum_reward, [batch, -1, 1])), axis=1)
        discnt_reward = tf.reverse(discnt_reward, axis=[1])

        # for state, reward, action in zip(states, discnt_reward, actions):
        #     with tf.GradientTape() as tape:
        #         p = self.model(np.array([state]).reshape(1, -1))
        #         loss = self.loss(p, action, reward)
        #     grads = tape.gradient(loss, self.model.trainable_variables)
        #     self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        for step in range(tf.shape(states)[1]):
            with tf.GradientTape() as tape:
                state = states[:, step, :]
                action = actions[:, step, :]
                reward = discnt_reward[:, step, :]
                p = self.model(state)
                loss = self.loss(p, action, reward)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


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


def plot(data):
    x = np.arange(len(data))
    fig = plt.plot(x, data, 'r--')
    plt.ylabel('cumulative reward')
    plt.xlabel('episode')
    plt.legend()
    plt.show()


def train_batch(hidden_size, gamma, imbalance_1, imbalance_2, learning_rate):
    train_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA.npy')[:, :, 1:]  # 去掉当前天数这个变量---->40个变量
    test_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\test_PLA.npy')[:, :, 1:]
    batch_size = 64
    feature_size = train_set.shape[2]-1
    train_set = DataSet(train_set)
    epochs = 50
    actor_size = 65
    previous_visit = 3
    predicted_visit = 7

    hidden_size = 2 ** int(hidden_size)
    learning_rate = 10 ** learning_rate
    imbalance_1 = 10 ** int(imbalance_1)
    imbalance_2 = 10 ** int(imbalance_2)
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
        total_rewards = tf.zeros(shape=[batch, 1])

        for step in range(predicted_visit):
            if step == 0:
                # state = initial_state  # 39个变量
                initial_state = input_x_train[:, step + previous_visit - 1, 1:]
                state_to_now_ = input_x_train[:, :step + previous_visit - 1, :]
                action = agent.act(initial_state)
                state_action = tf.concat((initial_state.reshape(batch, 1, -1), tf.reshape(action, [batch, -1, 1])), axis=2)
                state_to_now = tf.concat((state_to_now_, state_action), axis=1)
                next_state = construct_environment(state_to_now)

                states = tf.concat((states, initial_state.reshape(batch, -1, feature_size)), axis=1)
                actions = tf.concat((actions, tf.reshape(action, [batch, -1, 1])), axis=1)

                reward = (next_state[:, 0] - next_state[:, 1]) * imbalance_1  # 新状态的出入量差值
                rewards = tf.concat((rewards, tf.reshape(reward, [batch, -1, 1])), axis=1)
                total_rewards += tf.reshape(reward, [batch, -1])

            else:
                initial_state = input_x_train[:, previous_visit - 1, 1:]
                state_to_now_ = input_x_train[:, :previous_visit - 1, :]
                state = next_state
                action = agent.act(state)
                state_action = tf.concat((tf.reshape(state, [batch, -1, feature_size]), tf.reshape(action, [batch, -1, 1])), axis=2)
                state_to_now__ = tf.concat((states, actions), axis=2)
                state_to_now = tf.concat((state_to_now_, state_to_now__, state_action), axis=1)
                next_state = construct_environment(state_to_now)

                states = tf.concat((states, tf.reshape(state, [batch, -1, feature_size])), axis=1)
                actions = tf.concat((actions, tf.reshape(action, [batch, -1, 1])), axis=1)
                reward = (next_state[:, 0] - next_state[:, 1]) * imbalance_1

                if step == predicted_visit-1:
                    reward += (next_state[:, 3] - initial_state[:, 3]) * imbalance_2

                rewards = tf.concat((rewards, tf.reshape(reward, [batch, -1, 1])), axis=1)
                total_rewards += tf.reshape(reward, [batch, -1])

        agent.train(states=states, actions=actions, rewards=rewards)

        if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
            logged.add(train_set.epoch_completed)
            batch_test = test_set.shape[0]
            rewards_test = tf.zeros(shape=[batch_test, 0, 1])
            states_test = tf.zeros(shape=[batch_test, 0, feature_size])
            actions_test = tf.zeros(shape=[batch_test, 0, 1])
            total_rewards_test = tf.zeros(shape=[batch_test, 1])

            for step in range(predicted_visit):
                initial_state_test = test_set[:, step+previous_visit-1, 1:]
                state_to_now_test_ = test_set[:, :step + previous_visit - 1, :]
                if step == 0:
                    state_test = tf.cast(initial_state_test, tf.float32)
                    action_test = agent.act(state_test)
                    state_action_test = tf.concat((tf.reshape(state_test, [batch_test, -1, feature_size]), tf.reshape(action_test, [batch_test, -1, 1])), axis=2)

                    state_to_now_test = tf.concat((state_to_now_test_, state_action_test), axis=1)
                    next_state_test = construct_environment(state_to_now_test)
                    reward_test = (next_state_test[:, 0] - next_state_test[:, 1]) * imbalance_1

                    rewards_test = tf.concat((rewards_test, tf.reshape(reward_test, [batch_test, 1, 1])), axis=1)
                    states_test = tf.concat((states_test, tf.reshape(state_test, [batch_test, -1, feature_size])), axis=1)
                    actions_test = tf.concat((actions_test, tf.reshape(action_test, [batch_test, -1, 1])), axis=1)
                    total_rewards_test += tf.reshape(reward_test, [batch_test, -1])

                else:
                    state_test = next_state_test
                    action_test = agent.act(state_test)
                    state_action_test = tf.concat((tf.reshape(state_test, [batch_test, -1, feature_size]), tf.reshape(action_test, [batch_test, -1, 1])), axis=2)
                    state_to_now_test__ = tf.concat((states_test, actions_test), axis=2)

                    state_to_now_test = tf.concat((state_to_now_test_, state_to_now_test__, state_action_test), axis=1)
                    next_state_test = construct_environment(state_to_now_test)
                    reward_test = (next_state_test[:, 0] - next_state_test[:, 1]) * imbalance_1

                    if step == predicted_visit-1:
                        reward_test += (next_state_test[:, 3] - initial_state_test[:, 3]) * imbalance_2

                    rewards_test = tf.concat((rewards_test, tf.reshape(reward_test, [batch_test, -1, 1])), axis=1)
                    states_test = tf.concat((states_test, tf.reshape(state_test, [batch_test, -1, feature_size])), axis=1)
                    actions_test = tf.concat((actions_test, tf.reshape(action_test, [batch_test, -1, 1])), axis=1)
                    total_rewards_test += tf.reshape(reward_test, [batch_test, -1])
            print('epoch---{}----train_total_reward---{}---test_total--reward---{}'
                  .format(train_set.epoch_completed, np.mean(total_rewards), np.mean(total_rewards_test)))
    return np.mean(total_rewards_test)


if __name__ == '__main__':
    test_test('9_14_训练agent_3_7_train.txt')
    Agent_BO = BayesianOptimization(
        train_batch, {
            'hidden_size': (5, 8),
            'learning_rate': (-5, -1),
            'imbalance_1': (-3, -1),
            'imbalance_2': (-3, -1),
            'gamma': (0, 1),
        }
    )
    Agent_BO.maximize()
    print(Agent_BO.max)
    # train_batch(hidden_size=64, gamma=0.99, imbalance_1=0.1, imbalance_2=0.5, learning_rate=0.01)
















