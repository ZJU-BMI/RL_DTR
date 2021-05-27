import tensorflow as tf
from tensorflow_core.python.keras import Model
from utilis import test_test
from environment import Encode, Environment
from reward import Reward
from origin.data import DataSet
from death_model import DeathModel
import random
import numpy as np
import os
from discriminator import Discriminator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def discount_reward_1(rewards, gamma, lambda_imbalance):
    sum_reward = 0.0
    batch = tf.shape(rewards)[0]
    return_value = tf.zeros(shape=[batch, 0, 1])
    discount_reward = tf.zeros(shape=[batch, 0, 1])
    rewards = tf.reverse(rewards, axis=[1])
    for i in range(tf.shape(rewards)[1]):
        rewards_current = rewards[:, i:, :]
        for r_index in range(tf.shape(rewards_current)[1]):
            r = rewards_current[:, r_index, :]
            # sum_reward = r + gamma * sum_reward
            sum_reward = (1 + r * lambda_imbalance) + gamma * sum_reward
            discount_reward = tf.concat((discount_reward, tf.reshape(sum_reward, [batch, -1, 1])), axis=1)
            discount_reward = tf.reverse(discount_reward, axis=[1])
        return_value = tf.concat((return_value, discount_reward), axis=1)
    return return_value


def discount_reward(rewards, gamma, lambda_imbalance, dis_probs):
    sum_reward = 0.0
    batch = tf.shape(rewards)[0]
    return_value = tf.zeros(shape=[batch, 0, 1])
    discount_reward = tf.zeros(shape=[batch, 0, 1])
    rewards = tf.reverse(rewards, axis=[1])
    rewards_current = rewards[:, :, :]
    for r_index in range(tf.shape(rewards_current)[1]):
        r = rewards_current[:, r_index, :]
        # sum_reward = r + gamma * sum_reward
        sum_reward = (1 + r * lambda_imbalance) + gamma * sum_reward
        discount_reward = tf.concat((discount_reward, tf.reshape(sum_reward, [batch, -1, 1])), axis=1)

    discount_reward = tf.reverse(discount_reward, axis=[1])
    for i in range(tf.shape(rewards_current)[1]):
        prob = tf.reshape(dis_probs[:, i], [batch, -1, 1])
        return_value = tf.concat((return_value, tf.reshape(discount_reward[:, i, :], [batch, -1, 1]) * prob), axis=1)
    return return_value


def train_random_policy():
    test_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\test_PLA.npy')[:, :, 1:]
    test_set.astype(np.float32)

    batch_size = np.shape(test_set)[0]

    epochs = 200
    actor_size = 65
    previous_visit = 3
    predicted_visit = 7
    feature_size = 35

    action_list = [0, 0.0357142873108387, 0.0625, 0.071428571, 0.125, 0.1875, 0.25, 0.2857142984867096, 0.3125,
                   0.321428571,
                   0.375, 0.4375, 0.446428571, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.875, 1,
                   1.125, 1.25, 1.375, 1.5, 1.55, 1.625, 1.75, 1.875, 2, 2.125,
                   2.25, 2.5, 2.75, 3, 3.125, 3.25, 3.375, 3.5, 3.75, 4,
                   4.25, 4.5, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 7,
                   7.5, 7.75, 8, 8.25, 8.5, 9, 9.625, 10, 11, 13.5,
                   14.5, 15, 21, 22.25, 25.625]

    # 初始化所有的modules,并加载预训练模型
    encode_net = Encode(hidden_size=128)
    encode_net([tf.zeros(shape=[batch_size, 4, feature_size]), tf.zeros(shape=[batch_size, 4, 1])])
    encode_net.load_weights('encode_net_11_30.h5')

    environment_net = Environment(hidden_size=128)
    environment_net([tf.zeros(shape=[batch_size, 128]), tf.zeros(shape=[batch_size, 1])])
    environment_net.load_weights('environment_net_11_30.h5')

    reward_net = Reward(hidden_size=32)
    reward_net([tf.zeros(shape=[batch_size, 128]), tf.zeros(shape=[batch_size, 1])])
    reward_net.load_weights('reward_net_11_30.h5')

    death_model = DeathModel(hidden_size=256)
    death_model(tf.zeros(shape=[batch_size, 4, 128]))
    death_model.load_weights('death_model_11_30.h5')

    discriminator = Discriminator(hidden_size=128)

    logged = set()
    epoch = 0
    rewards_offline = (test_set[:, previous_visit+1:previous_visit+predicted_visit, 1]
                       - test_set[:, previous_visit+1:previous_visit+predicted_visit, 2]).reshape(batch_size, -1, 1)
    actions_offline = test_set[:, previous_visit:previous_visit+predicted_visit-1, 0].reshape(batch_size, -1, 1)

    while epoch < epochs:
        states_online = tf.zeros(shape=[batch_size, 0, 128])
        rewards_online = tf.zeros(shape=[batch_size, 0, 1])
        actions_index_online = tf.zeros(shape=[batch_size, 0, 1])
        actions_online = tf.zeros(shape=[batch_size, 0, 1])
        for step in range(predicted_visit-1):
            if step == 0:
                features = test_set[:, :previous_visit+step, 5:]
                actions_ = tf.zeros(shape=[batch_size, 1, 1], dtype=tf.float64)
                actions = tf.reshape(test_set[:, :previous_visit+step-1, 0], [batch_size, -1, 1])
                actions = tf.concat((actions_, actions), axis=1)
                state = encode_net([features, actions])
            else:
                state = next_state

            states_online = tf.concat((states_online, tf.reshape(state, [batch_size, -1, 128])), axis=1)
            action_index = np.array([random.randint(0, 64) for _ in range(batch_size)])
            action_index = tf.convert_to_tensor(action_index.astype(np.float32).reshape(batch_size, -1, 1))
            actions_index_online = tf.concat((actions_index_online, action_index), axis=1)
            action_value = np.zeros_like(action_index)
            for patient in range(batch_size):
                for visit in range(tf.shape(action_index)[1]):
                    for m in range(tf.shape(action_index)[2]):
                        action_value[patient, visit, m] = action_list[action_index[patient, visit, m]]
            actions_online = tf.concat((actions_online, tf.reshape(action_value, [batch_size, -1, 1])), axis=1)

            reward = reward_net([state, action_value])
            rewards_online = tf.concat((rewards_online, tf.reshape(reward, [batch_size, -1, 1])), axis=1)
            next_state = environment_net([state, action_value])

        _, discriminator_probs_online = discriminator([states_online, rewards_online, actions_online])

        death_probs_online = death_model(states_online)
        death_estimated_online = np.zeros_like(death_probs_online)
        for patient in range(batch_size):
            for visit in range(predicted_visit-1):
                if death_probs_online[patient, visit, :] > 0.49651924:
                    death_estimated_online[patient, visit, :] = 1

        discont_rewards = discount_reward(rewards_online, 0.99, 0.5, discriminator_probs_online)
        print('epoch  {}  test_return  {}  test_death  {}'
              .format(epoch, tf.reduce_mean(discont_rewards),
                      np.sum(death_estimated_online)))
        epoch += 1

        if epoch == 199:
            np.save('11_30 最终版\\11_30_states_random_policy.npy', states_online)
            np.save('11_30 最终版\\11_30_death_random_policy.npy', death_estimated_online)
            np.save('11_30 最终版\\11_30_rewards_random_policy.npy', rewards_online)
            np.save('11_30 最终版\\11_30_actions_random_policy.npy', actions_online)
            np.save('11_30 最终版\\11_30_discount_reward_random_policy.npy', discont_rewards)

    tf.compat.v1.reset_default_graph()


if __name__ == '__main__':
    test_test('1_28_random_policy_保存数据_HF.txt')
    train_random_policy()



