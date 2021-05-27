import tensorflow as tf
import os
from data import DataSet
from reward_1 import Reward
from discriminator import Discriminator
from agent_1 import Agent
from environment_1 import Encode, Environment
from death_model_1 import DeathModel
import numpy as np
from utilis import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def discount_reward_1(rewards, gamma, lambda_imbalance, dis_probs):
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
        prob = tf.reshape(dis_probs[:, i], [batch, -1, 1])
        return_value = tf.concat((return_value, tf.reshape(discount_reward[:, 0, :], [batch, -1, 1]) * (1.0 - tf.abs(prob-0.5))), axis=1)
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


def reward_calculate(features, time_steps):
    rewards = np.zeros(shape=[features.shape[0], 0, 1])
    for time_index in range(time_steps-1):
        feature = features[:, time_index, :]
        next_feature = features[:, time_index + 1, :]
        part1_ = np.where(feature[:, 0] == next_feature[:, 0], 1, 0)
        part1__ = np.where(next_feature[:, 0] > 0, -0.025, 0)
        part1 = part1_ * part1__
        part2 = (next_feature[:, 0] - feature[:, 0]) * -0.125
        part3 = np.tanh((next_feature[:, 1] - feature[:, 1])) * -2
        one_step_reward = part1 + part2 + part3
        rewards = np.concatenate((rewards, tf.reshape(one_step_reward, [-1, 1, 1])), axis=1)
    rewards = tf.cast(rewards, tf.float32)
    return rewards


def train_batch():
    train_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\mimic_train.npy')[:4290, :, :]
    test_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\mimic_validate.npy')[:321, :, :]

    lambda_imbalance = 10

    test_set.astype(np.float32)
    batch_size = train_set.shape[0]
    feature_size = 45
    train_set = DataSet(train_set)
    epochs = 200
    actor_size = 25
    previous_visit = 3
    predicted_visit = 10

    gamma = 0.99

    iv_list = [0, 33, 102, 330, 1153]
    vas_list = [0, 0.045, 0.15, 0.32, 1.24]

    # 初始化所有的modules,并加载预训练模型
    encode_net = Encode(hidden_size=256)
    encode_net([tf.zeros(shape=[1, 4, feature_size]), tf.zeros(shape=[1, 4, 2])])
    encode_net.load_weights('encode_net_12_7_mimic.h5')

    environment_net = Environment(hidden_size=32)
    environment_net([tf.zeros(shape=[1, 256]), tf.zeros(shape=[1, 2])])
    environment_net.load_weights('environment_net_12_18_mimic.h5')

    reward_net = Reward(hidden_size=32)
    reward_net([tf.zeros(shape=[1, 256]), tf.zeros(shape=[1, 2])])
    reward_net.load_weights('reward_net_12_24_mimic.h5')

    agent = Agent(actor_size=actor_size, hidden_size=64, gamma=gamma, learning_rate=0.01, lambda_imbalance=lambda_imbalance)
    agent.act(tf.zeros(shape=[1, 256]))
    agent.model.load_weights('policy_net_12_21_mimic.h5')

    death_model = DeathModel(hidden_size=128)
    death_model(tf.zeros(shape=[1, 4, 256]))
    death_model.load_weights('death_model_12_27_mimic.h5')

    logged = set()

    while train_set.epoch_completed < epochs:
        with tf.GradientTape() as agent_tape, tf.GradientTape() as tape, tf.GradientTape() as dis_tape:
            input_x_train = train_set.next_batch(batch_size=64)
            batch = input_x_train.shape[0]
            input_x_train = input_x_train.astype(np.float32)
            rewards = tf.zeros(shape=[batch, 0, 1])
            rewards_value = tf.zeros(shape=[batch, 0, 2])
            states = tf.zeros(shape=[batch, 0, 256])
            actions_values = tf.zeros(shape=[batch, 0, 2])
            actions_index = tf.zeros(shape=[batch, 0, 1])

            offline_rewards_labels = input_x_train[:, previous_visit:previous_visit+predicted_visit, 1:3]  #
            offline_rewards = tf.zeros(shape=[batch, 0, 2])
            offline_actions_index_label = input_x_train[:, previous_visit:previous_visit+predicted_visit-1, 0]
            offline_actions_index = tf.zeros(shape=[batch, 0, 1])
            offline_actions_values_label = np.zeros(shape=[batch, predicted_visit-1, 2])
            offline_actions_values = tf.zeros(shape=[batch, 0, 2])
            offline_states_labels = tf.zeros(shape=[batch, 0, 256])
            offline_states = tf.zeros(shape=[batch, 0, 256])
            offline_actions_probs = tf.zeros(shape=[batch, 0, 25])

            # 获得offline action value
            for patient in range(batch):
                for visit in range(predicted_visit-1):
                    action_index = offline_actions_index_label[patient, visit]
                    iv_index = int(action_index / 5)
                    vas_index = int(action_index % 5)
                    offline_actions_values_label[patient, visit, 0] = iv_list[iv_index]
                    offline_actions_values_label[patient, visit, 1] = vas_list[vas_index]

            # online 数据
            for step in range(predicted_visit-1):
                if step == 0:
                    features = input_x_train[:, :step+previous_visit, 4:]
                    actions_index__ = tf.zeros(shape=[batch, 1, 1])
                    actions_index_ = tf.reshape(input_x_train[:, :step+previous_visit-1, 0], [batch, -1, 1])
                    actions_index_ = tf.concat((actions_index__, actions_index_), axis=1)
                    actions_index_ = actions_index_.numpy()
                    actions_values_ = np.zeros(shape=[batch, step+previous_visit, 2])

                    # get action values
                    for patient in range(batch):
                        for visit in range(step+previous_visit):
                            action_index = actions_index_[patient, visit, 0]
                            iv_index = int(action_index / 5)
                            vas_index = int(action_index % 5)
                            actions_values_[patient, visit, 0] = iv_list[iv_index]
                            actions_values_[patient, visit, 1] = vas_list[vas_index]
                    state = encode_net([features, actions_values_])
                else:
                    state = next_state

                states = tf.concat((states, tf.reshape(state, [batch, -1, 256])), axis=1)
                action_index = agent.act(state)
                # 保存选择的action index
                actions_index = tf.concat((actions_index, tf.reshape(action_index, [batch, -1, 1])), axis=1)
                action_value = np.zeros(shape=[batch, tf.shape(action_index)[1], 2])

                # get action values
                for patient in range(batch):
                    for visit in range(action_value.shape[1]):
                        action_index_ = action_index[patient, visit]
                        iv_index = int(action_index_ / 5)
                        vas_index = int(action_index_ % 5)
                        action_value[patient, visit, 0] = iv_list[iv_index]
                        action_value[patient, visit, 1] = vas_list[vas_index]
                action_value = tf.cast(action_value, tf.float32)

                # 保存选择的action value
                actions_values = tf.concat((actions_values, tf.reshape(action_value, [batch, -1, 2])), axis=1)
                reward_ = reward_net([state, action_value])
                rewards_value = tf.concat((rewards_value, tf.reshape(reward_, [batch, -1, 2])), axis=1)
                if step == 0:
                    reward__ = input_x_train[:, previous_visit:previous_visit+1, 1:3]

                else:
                    reward__ = rewards_value[:, step-1, :]  # 之前的lactate和SOFA
                reward_ = tf.concat((tf.reshape(reward__, [batch, -1, 2]), tf.reshape(reward_, [batch, -1, 2])), axis=1)
                reward_ = tf.cast(reward_, tf.float32)
                reward = reward_calculate(reward_, 2)
                rewards = tf.concat((rewards, tf.reshape(reward, [batch, -1, 1])), axis=1)  # 已经进行函数变换的
                next_state = environment_net([state, action_value])

            # offline 数据
            for step in range(predicted_visit-1):
                features = input_x_train[:, :step+previous_visit, 4:]
                actions_index__ = tf.zeros(shape=[batch, 1, 1])
                actions_index_ = tf.reshape(input_x_train[:, :step+previous_visit-1, 0], [batch, -1, 1])
                actions_index_ = tf.concat((actions_index__, actions_index_), axis=1)
                actions_value_ = np.zeros(shape=[batch, step+previous_visit, 2])

                # get action value
                for patient in range(batch):
                    for visit in range(step+previous_visit):
                        action_index = actions_index_[patient, visit, 0]
                        iv_index = int(action_index / 5)
                        vas_index = int(action_index % 5)
                        actions_value_[patient, visit, 0] = iv_list[iv_index]
                        actions_value_[patient, visit, 1] = vas_list[vas_index]

                state = encode_net([features, actions_value_])
                probs = agent.model(state)
                offline_actions_probs = tf.concat((offline_actions_probs, tf.reshape(probs, [batch, -1, 25])), axis=1)
                offline_states_labels = tf.concat((offline_states_labels, tf.reshape(state, [batch, -1, 256])), axis=1)

                action_index = agent.act(state)
                real_action_index = offline_actions_index_label[:, step]
                action_value = np.zeros(shape=[batch, 1, 2])
                # get action value
                for patient in range(batch):
                    index = real_action_index[patient]
                    iv_index = int(index / 5)
                    vas_index = int(index % 5)
                    action_value[patient, 0, 0] = iv_list[iv_index]
                    action_value[patient, 0, 1] = vas_list[vas_index]
                # 保存action_index, action_value
                action_value = tf.cast(action_value, tf.float32)
                offline_actions_index = tf.concat((offline_actions_index, tf.reshape(action_index, [batch, -1, 1])), axis=1) # 模型选择的action
                offline_actions_values = tf.concat((offline_actions_values, tf.reshape(action_value, [batch, -1, 2])), axis=1)
                reward = reward_net([state, action_value])
                offline_rewards = tf.concat((offline_rewards, tf.reshape(reward, [batch, -1, 2])), axis=1)

                next_state = environment_net([state, action_value])
                offline_states = tf.concat((offline_states, tf.reshape(next_state, [batch, -1, 256])), axis=1)

            features = input_x_train[:, :previous_visit+predicted_visit, 4:]
            action_index_ = tf.zeros(shape=[batch, 1, 1])
            action_index = input_x_train[:, :previous_visit+predicted_visit-1, 0]
            action_index = tf.concat((action_index_, tf.reshape(action_index, [batch, -1, 1])), axis=1)
            action_value = np.zeros(shape=[batch, previous_visit+predicted_visit, 2])
            for patient in range(batch):
                for visit in range(previous_visit+predicted_visit):
                    action = action_index[patient, visit, 0]
                    iv_index = int(action / 5)
                    vas_index = int(action % 5)
                    action_value[patient, visit, 0] = iv_list[iv_index]
                    action_value[patient, visit, 1] = vas_list[vas_index]
            state = encode_net([features, action_value])
            offline_states_labels = tf.concat((offline_states_labels, tf.reshape(state, [batch, -1, 256])), axis=1)

            death_prob_online = death_model(states)
            death_prob_offline = death_model(offline_states_labels)


        # 计算价值函数
        offline_rewards_labels_calculate = reward_calculate(offline_rewards_labels, predicted_visit)
        online_value = discount_reward(rewards, gamma, lambda_imbalance, tf.ones(shape=[batch, predicted_visit-1])*0.5)
        offline_value = discount_reward(offline_rewards_labels_calculate, gamma, lambda_imbalance, tf.ones(shape=[batch, predicted_visit-1])*0.5)

        if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
            logged.add(train_set.epoch_completed)
            batch_test = test_set.shape[0]

            rewards_test_online = tf.zeros(shape=[batch_test, 0, 2])
            states_test_online = tf.zeros(shape=[batch_test, 0, 256])
            actions_index_test_online = tf.zeros(shape=[batch_test, 0, 1])
            actions_values_test_online = tf.zeros(shape=[batch_test, 0, 2])

            # offline数据
            rewards_test_offline = test_set[:, previous_visit:previous_visit+predicted_visit, 1:3]
            rewards_test_offline_calculate = reward_calculate(rewards_test_offline, predicted_visit)
            states_test_offline = tf.zeros(shape=[batch_test, 0, 256])
            actions_index_test_offline = test_set[:, previous_visit:previous_visit+predicted_visit-1, 0]
            actions_values_test_offline = np.zeros(shape=[batch_test, predicted_visit-1, 2])

            # 得到offline的action values
            for patient in range(batch_test):
                for visit in range(predicted_visit-1):
                    action_index = actions_index_test_offline[patient, visit]
                    iv_index = int(action_index / 5)
                    vas_index = int(action_index % 5)
                    actions_values_test_offline[patient, visit, 0] = iv_list[iv_index]
                    actions_values_test_offline[patient, visit, 1] = vas_list[vas_index]

            # 得到offline的states
            for step in range(predicted_visit-1):
                features = test_set[:, :previous_visit+step, 4:]
                actions_index_ = tf.zeros(shape=[batch_test, 1, 1])
                actions_index = tf.reshape(test_set[:, :previous_visit+step-1, 0], [batch_test, -1, 1])
                actions_index = tf.cast(actions_index, tf.float32)
                actions_index = tf.concat((actions_index_, actions_index), axis=1)
                actions_value = np.zeros(shape=[batch_test, previous_visit+step, 2])
                # get action value
                for patient in range(batch_test):
                    for visit in range(previous_visit+step):
                        action_index = actions_index[patient, visit, 0]
                        iv_index = int(action_index / 5)
                        vas_index = int(action_index % 5)
                        actions_value[patient, visit, 0] = iv_list[iv_index]
                        actions_value[patient, visit, 1] = vas_list[vas_index]

                state = encode_net([features, actions_value])
                states_test_offline = tf.concat((states_test_offline, tf.reshape(state, [batch_test, -1, 256])), axis=1)

            for step in range(predicted_visit-1):
                if step == 0:
                    features = test_set[:, :previous_visit + step, 4:]
                    actions_index_ = tf.zeros(shape=[batch_test, 1, 1])
                    actions_index = tf.reshape(test_set[:, :previous_visit + step - 1, 0], [batch_test, -1, 1])
                    actions_index = tf.cast(actions_index, tf.float32)
                    actions_index = tf.concat((actions_index_, actions_index), axis=1)
                    actions_value = np.zeros(shape=[batch_test, previous_visit + step, 2])
                    # get action value
                    for patient in range(batch_test):
                        for visit in range(previous_visit + step):
                            action_index = actions_index[patient, visit, 0]
                            iv_index = int(action_index / 5)
                            vas_index = int(action_index % 5)
                            actions_value[patient, visit, 0] = iv_list[iv_index]
                            actions_value[patient, visit, 1] = vas_list[vas_index]
                    state_test = encode_net([features, actions_value])
                    rewards_test_online = tf.concat((rewards_test_online, test_set[:, previous_visit:previous_visit+1, 1:3]), axis=1)
                else:
                    state_test = next_state_test
                states_test_online = tf.concat((states_test_online, tf.reshape(state_test, [batch_test, -1, 256])), axis=1)

                action_index_test = agent.act(state_test)
                action_value = np.zeros(shape=[batch_test, 1, 2])
                for patient in range(batch_test):
                    index = action_index_test[patient, 0]
                    iv_index = int(index / 5)
                    vas_index = int(index % 5)
                    action_value[patient, 0, 0] = iv_list[iv_index]
                    action_value[patient, 0, 1] = vas_list[vas_index]
                action_value = tf.cast(action_value, tf.float32)
                actions_values_test_online = tf.concat((actions_values_test_online, tf.reshape(action_value, [batch_test, -1, 2])), axis=1)
                actions_index_test_online = tf.concat((actions_index_test_online, tf.reshape(action_index_test, [batch_test, -1, 1])), axis=1)

                # reward and next state
                reward_test = reward_net([state_test, action_value])
                next_state_test = environment_net([state_test, action_value])
                rewards_test_online = tf.concat((rewards_test_online, tf.reshape(reward_test, [batch_test, -1, 2])), axis=1)
            rewards_test_online_calculate = reward_calculate(rewards_test_online, predicted_visit)

            test_value_online = discount_reward(rewards_test_online_calculate, gamma, lambda_imbalance, tf.ones(shape=[batch_test, predicted_visit-1])*0.5)
            test_value_offline = discount_reward(rewards_test_offline_calculate, gamma, lambda_imbalance, tf.ones(shape=[batch_test, predicted_visit-1])*0.5)

            death_probs_online_test = death_model(states_test_online)
            death_probs_offline_test = death_model(states_test_offline)

            death_estimated_online_test = np.zeros_like(death_probs_online_test)
            death_estimated_offline_test = np.zeros_like(death_probs_offline_test)

            for patient in range(batch_test):
                for visit in range(predicted_visit-1):
                    if death_probs_online_test[patient, visit, :] >= 0.5767236:
                        death_estimated_online_test[patient, visit, :] = 1

                    if death_probs_offline_test[patient, visit, :] >= 0.5767236:
                        death_estimated_offline_test[patient, visit, :] = 1

            print('epoch {}  '
                  'train_value_online {}  train_value_offline {}  test_value_online {} test_value_offline  {}'
                  '  death_sum {}'
                  .format(train_set.epoch_completed,
                          tf.reduce_mean(online_value),
                          tf.reduce_mean(offline_value),
                          tf.reduce_mean(test_value_online),
                          tf.reduce_mean(test_value_offline),
                          np.sum(death_estimated_online_test)))

            if train_set.epoch_completed == 199:
                np.save('11_30 最终版\\1_28_states_bc_mimic.npy', states_test_online)
                np.save('11_30 最终版\\1_28_death_bc_mimic.npy', death_estimated_online_test)
                np.save('11_30 最终版\\1_28_rewards_bc_mimic.npy', rewards_test_online)
                np.save('11_30 最终版\\1_28_actions_bc_mimic.npy', actions_values_test_online)
                np.save('11_30 最终版\\1_28_discount_reward_bc_mimic.npy', test_value_online)

    tf.compat.v1.reset_default_graph()
    return tf.reduce_mean(test_value_online)


if __name__ == '__main__':
    test_test('BC__1_28_mimic_训练.txt')
    # TR_GAN = BayesianOptimization(
    #     train_batch, {
    #         'hidden_size': (2, 4),
    #         'learning_rate': (-5, -1),
    #         'l2_regularization': (-6, -1),
    #     }
    # )
    # TR_GAN.maximize()
    # print(TR_GAN.max)
    for i in range(1):
        test_online_value = train_batch()











