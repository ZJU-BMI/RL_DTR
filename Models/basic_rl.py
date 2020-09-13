import tensorflow as tf
import numpy as np
from Environment import *


class PolicyGradientNN(tf.keras.layers.Layer):
    def __init__(self, actor_size, hidden_size):
        super().__init__(name='policy_gradient')
        self.actor_size = actor_size
        self.hidden_size = hidden_size
        self.dense1 = tf.keras.layers.Dense(self.hidden_size, activation=tf.keras.activations.relu)
        self.dense2 = tf.keras.layers.Dense(self.hidden_size, activation=tf.keras.activations.relu)
        self.dense3 = tf.keras.layers.Dense(self.actor_size, activation=None)

    def actor_prob(self, state):
        hidden_1 = self.dense1(state)
        hidden_2 = self.dense2(hidden_1)
        actor_prob = self.dense3(hidden_2)
        return actor_prob

    def call(self, state):
        prob = self.actor_prob(state)
        return prob


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
        return int(action.numpy()[0])

    def loss(self, prob, action, reward):
        dist = tf.compat.v1.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*reward
        return loss

    def train(self, states, rewards, actions):
        sum_reward = 0
        discnt_reward = []
        rewards.reverse()
        for r in rewards:
            sum_reward = r + self.gamma * sum_reward
            discnt_reward.append(sum_reward)
        discnt_reward.reverse()

        for state, reward, action in zip(states, discnt_reward, actions):
            with tf.GradientTape() as tape:
                p = self.model(np.array([state]).reshape(1, -1))
                loss = self.loss(p, action, reward)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


# 训练agent
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
    feature_dims = data_set.shape[2] - 2

    construct_environment = ReconstructEnvir(hidden_size=32,
                                             feature_dims=39,
                                             previous_visit=3)
    construct_environment(tf.ones(shape=[3, 4, 40]))
    construct_environment.load_weights('environment_3_7_39.h5')
    agent = Agent(actor_size=actor_size, hidden_size=hidden_size, gamma=gamma, learning_rate=learning_rate)
    batch = data_set.shape[0]

    for patient in range(batch):
        rewards = []
        states = []
        actions = []
        total_reward = 0
        for step in range(predicted_visit):
            if step == 0:
                state = data_set[patient, step+previous_visit-1, 1:]  # s_3 39个变量
                action = agent.act(tf.reshape(state, [1, -1]))
                print(action)
                state_to_now = data_set[patient, :step+previous_visit-1, :]  # (s_1, a_1)..(s_2, a_2)
                state_action = np.concatenate((np.array(action).reshape(1, -1), state.reshape(1, -1)), axis=1)
                state_to_now = np.concatenate((state_to_now, state_action)).reshape(1, -1, state_to_now.shape[1])
                next_state = construct_environment(state_to_now)
                reward = (next_state[0, 0] - state[1]) * imbalance_1 + (next_state[0, 3] - state[3])*imbalance_2
                rewards.append(reward)
                states.append(state)
                actions.append(action)
                total_reward += reward

            else:
                state = next_state.numpy()
                action = agent.act(state)
                state_action = np.concatenate((np.array(action).reshape(1, -1), state.reshape(1, -1)), axis=1)
                print(action)
                state_to_now_ = data_set[patient, :previous_visit+step-1, :]  # (s_1, a_1),(s_2, a_2)
                states_array = np.zeros(shape=[0, 39])
                actions_array = np.zeros(shape=[0, 1])
                for i in range(len(states)):
                    states_array = np.concatenate((states_array, np.array(states[i]).reshape(1, -1)), axis=0)
                    actions_array = np.concatenate((actions_array, np.array(actions[i]).reshape(1, -1)), axis=0)
                state_actions = np.concatenate((actions_array, states_array), axis=1)
                state_to_now = np.concatenate((state_to_now_, state_actions, state_action), axis=0)
                next_state = construct_environment(state_to_now.reshape(1, -1, state_to_now.shape[1]))
                reward = (next_state[0, 0] - state[0, 1]) * imbalance_1 + (next_state[0, 3] - state[0, 3]) * imbalance_2
                rewards.append(reward)
                states.append(state)
                actions.append(action)
                total_reward += reward

        agent.train(states, rewards, actions)
        print('total_reward after{} step is {}'.format(step, total_reward))


if __name__ == '__main__':
    train()















