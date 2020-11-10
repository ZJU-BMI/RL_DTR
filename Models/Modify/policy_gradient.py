import tensorflow as tf
import os
from origin.data import *
from reward import Reward
from agent import Agent
from environment import Encode, Environment
import numpy as np
from utilis import *
from death_model import DeathModel
from discriminator import Discriminator

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
        return_value = tf.concat((return_value, discount_reward * (1.0 - tf.abs(prob-0.5))), axis=1)
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
        return_value = tf.concat((return_value, tf.reshape(discount_reward[:, i, :], [batch, -1, 1]) * (1.0 - tf.abs(prob-0.5))), axis=1)
    return return_value


def train_batch(hidden_size, learning_rate, l2_regularization):
    train_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA.npy')[:, :, 1:]
    test_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\test_PLA.npy')[:, :, 1:]

    hidden_size = 2 ** int(hidden_size)
    learning_rate = 10 ** learning_rate
    l2_regularization = 10 ** l2_regularization
    lambda_imbalance = 0.5
    print('hidden_size  {}  learning_rate  {}  l2_regularization  {}  lambda_imbalance  {}'
          .format(hidden_size, learning_rate, l2_regularization, lambda_imbalance))

    test_set.astype(np.float32)
    batch_size = 2018
    feature_size = 35
    train_set = DataSet(train_set)
    epochs = 200
    actor_size = 65
    previous_visit = 3
    predicted_visit = 7
    gamma = 0.99

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
    encode_net(tf.zeros(shape=[batch_size, 4, feature_size]))
    encode_net.load_weights('encode_net_10_27.h5')

    environment_net = Environment(hidden_size=32)
    environment_net([tf.zeros(shape=[batch_size, 128]), tf.zeros(shape=[batch_size, 1])])
    environment_net.load_weights('environment_net_10_27.h5')

    reward_net = Reward(hidden_size=128)
    reward_net([tf.zeros(shape=[batch_size, 128]), tf.zeros(shape=[batch_size, 1])])
    reward_net.load_weights('reward_net_10_27.h5')

    agent = Agent(actor_size=actor_size, hidden_size=128, gamma=gamma, learning_rate=learning_rate,
                  lambda_imbalance=lambda_imbalance)
    agent.act(tf.zeros(shape=[batch_size, 128]))
    agent.model.load_weights('policy_net_10_28.h5')

    discriminator = Discriminator(hidden_size=hidden_size)

    death_model = DeathModel(hidden_size=32)
    death_model(tf.zeros(shape=[batch_size, 4, 128]))
    death_model.load_weights('death_model_11_3.h5')

    agent_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    logged = set()

    while train_set.epoch_completed < epochs:
        with tf.GradientTape() as tape, tf.GradientTape() as agent_tape:
            input_x_train = train_set.next_batch(batch_size=batch_size)
            input_x_train.astype(np.float32)

            rewards = tf.zeros(shape=[batch_size, 0, 1])
            states = tf.zeros(shape=[batch_size, 0, 128])
            actions = tf.zeros(shape=[batch_size, 0, 1])
            actions_index = tf.zeros(shape=[batch_size, 0, 1])

            offline_rewards = tf.zeros(shape=[batch_size, 0, 1])
            offline_rewards_labels = (input_x_train[:, previous_visit+1:previous_visit+predicted_visit, 1]
                                      - input_x_train[:, previous_visit+1:previous_visit+predicted_visit, 2]).reshape(batch_size, -1, 1)
            offline_rewards_labels = tf.cast(offline_rewards_labels, tf.float32)
            offline_states = tf.zeros(shape=[batch_size, 0, 128])
            offline_states_labels = tf.zeros(shape=[batch_size, 0, 128])
            offline_actions = tf.zeros(shape=[batch_size, 0, 1])
            offline_actions_index = tf.zeros(shape=[batch_size, 0, 1])

            # online数据
            for step in range(predicted_visit-1):
                if step == 0 :
                    features = input_x_train[:, :step+previous_visit, 5:]
                    state = encode_net(features)
                else:
                    state = next_state
                states = tf.concat((states, tf.reshape(state, [batch_size, -1, 128])), axis=1)
                action_index = agent.act(state)
                actions_index = tf.concat((actions_index, tf.reshape(action_index, [batch_size, -1, 1])), axis=1)
                action_value = np.zeros_like(action_index)
                # 将选择的类别转换成真实数值
                for i in range(tf.shape(action_value)[0]):
                    for j in range(tf.shape(action_value)[1]):
                        action_value[i, j] = action_list[action_index[i, j]]
                actions = tf.concat((actions, tf.reshape(action_value, [batch_size, -1, 1])), axis=1)
                reward = reward_net([state, action_value])
                rewards = tf.concat((rewards, tf.reshape(reward, [batch_size, -1, 1])), axis=1)

                next_state = environment_net([state, action_value])

            # offline 数据
            for step in range(predicted_visit-1):
                features = input_x_train[:, :step+previous_visit, 5:]
                state = encode_net(features)
                offline_states_labels = tf.concat((offline_states_labels, tf.reshape(state, [batch_size, -1, 128])), axis=1)

                action_index = agent.act(state)
                offline_actions_index = tf.concat((offline_actions_index, tf.reshape(action_index, [batch_size, -1, 1])), axis=1)

                action_value = np.zeros_like(action_index)
                # 将选择的类别转换成真实数值
                for i in range(tf.shape(action_value)[0]):
                    for j in range(tf.shape(action_value)[1]):
                        action_value[i, j] = action_list[action_index[i, j]]
                offline_actions = tf.concat((offline_actions, tf.reshape(action_value, [batch_size, -1, 1])), axis=1)
                reward = reward_net([state, action_value])
                offline_rewards = tf.concat((offline_rewards, tf.reshape(reward, [batch_size, -1, 1])), axis=1)

                next_state = environment_net([state, action_value])
                offline_states = tf.concat((offline_states, tf.reshape(next_state, [batch_size, -1, 128])), axis=1)

            features = input_x_train[:, previous_visit:previous_visit+predicted_visit, 5:]
            state = encode_net(features)
            offline_states_labels = tf.concat((offline_states_labels, tf.reshape(state, [batch_size, -1, 128])), axis=1)

            _, discriminator_probs_online = discriminator([states, rewards, actions])
            _, discriminator_probs_offline = discriminator([offline_states_labels[:, :-1, :], offline_rewards_labels, offline_actions])

            loss = tf.zeros(shape=[batch_size, 1])
            for step in range(predicted_visit-1):
                discont_reward = agent.discount_reward(rewards[:, step:, :])
                agent_loss_online = agent.loss(discont_reward, actions_index[:, step:, :], states[:, step:, :], discriminator_probs_online[:, step:])
                loss += agent_loss_online
            # m_a的损失
            loss = tf.reduce_mean(loss)

            # m_x和m_r的损失，这里没法使用鉴别器，只能通过offline的数据进行控制
            loss_x = tf.reduce_mean(tf.keras.losses.mse(offline_states_labels[:, 1:, :], offline_states))
            loss_r = tf.reduce_mean(tf.keras.losses.mse(offline_rewards_labels, offline_rewards))
            loss_x_and_r = loss_x + loss_r

            agent_variables = [var for var in agent.model.trainable_variables]
            reward_variables = [var for var in reward_net.trainable_variables]
            environment_variables = [var for var in environment_net.trainable_variables]

            environment_and_reward_variables = environment_variables
            for var in reward_variables:
                environment_and_reward_variables.append(var)

        grads = agent_tape.gradient(loss, agent_variables)
        agent_optimizer.apply_gradients(zip(grads, agent_variables))

        grads = tape.gradient(loss_x_and_r, environment_and_reward_variables)
        agent_optimizer.apply_gradients(zip(grads, environment_and_reward_variables))

        # 计算价值函数
        online_value = discount_reward(rewards, gamma, lambda_imbalance, discriminator_probs_online)
        offline_value = discount_reward(offline_rewards_labels, gamma, lambda_imbalance, discriminator_probs_offline)

        if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
            logged.add(train_set.epoch_completed)

            # agent.model.load_weights('11_9_save_models\\policy_gradient\\11_9_policy_gradient_compare200_12.484636.h5')
            # reward_net.load_weights('11_9_save_models\\policy_gradient\\11_9_reward_compare200_12.484636.h5')
            # environment_net.load_weights('11_9_save_models\\policy_gradient\\11_9_environment_compare200_12.484636.h5')

            batch_test = test_set.shape[0]
            rewards_test_online = tf.zeros(shape=[batch_test, 0, 1])
            states_test_online = tf.zeros(shape=[batch_test, 0, 128])
            actions_test_online = tf.zeros(shape=[batch_test, 0, 1])
            actions_index_test_online = tf.zeros(shape=[batch_test, 0, 1])

            # offline数据
            rewards_test_offline = (test_set[:, previous_visit+1:previous_visit+predicted_visit, 1] -
                                    test_set[:, previous_visit+1:previous_visit+predicted_visit, 2]).reshape(batch_test, -1, 1)

            states_test_offline = tf.zeros(shape=[batch_test, 0, 128])

            actions_test_offline = test_set[:, previous_visit:previous_visit + predicted_visit - 1, 0].reshape(
                batch_test, -1, 1)

            # 获取真实的state标签
            for step in range(predicted_visit-1):
                features = test_set[:, :previous_visit+step, 5:]
                state = encode_net(features)
                states_test_offline = tf.concat((states_test_offline, tf.reshape(state, [batch_test, -1, 128])), axis=1)

            for step in range(predicted_visit-1):
                if step == 0:
                    features_test = test_set[:, :step+previous_visit, 5:]
                    state_test = encode_net(features_test)
                else:
                    state_test = next_state_test
                states_test_online = tf.concat((states_test_online, tf.reshape(state_test, [batch_test, -1, 128])), axis=1)

                action_index = agent.act(state_test)
                action_value = np.zeros_like(action_index)
                for i in range(batch_test):
                    for j in range(action_value.shape[1]):
                        action_value[i, j] = action_list[action_index[i, j]]

                actions_test_online = tf.concat((actions_test_online, tf.reshape(action_value, [batch_test, -1, 1])), axis=1)
                actions_index_test_online = tf.concat((actions_index_test_online, tf.reshape(action_index, [batch_test, -1, 1])), axis=1)

                reward_test = reward_net([state_test, action_value])
                rewards_test_online = tf.concat((rewards_test_online, tf.reshape(reward_test, [batch_test, -1, 1])), axis=1)

                next_state_test = environment_net([state_test, action_value])

            test_online_representation, discriminator_probs_online_test = discriminator([states_test_online, rewards_test_online, actions_test_online])
            test_offline_representation, discriminator_probs_offline_test = discriminator([states_test_offline, rewards_test_offline, actions_test_offline])
            test_online_value = discount_reward(rewards_test_online, gamma, lambda_imbalance, discriminator_probs_online_test)
            rewards_test_offline = tf.cast(rewards_test_offline, tf.float32)
            test_offline_value = discount_reward(rewards_test_offline, gamma, lambda_imbalance,  tf.ones_like(discriminator_probs_offline_test)*0.5)

            death_probs_online_test = death_model(states_test_online)
            death_estimated_online_test = np.zeros_like(death_probs_online_test)

            death_probs_offline_test = death_model(states_test_offline)
            death_estimated_offline_test = np.zeros_like(death_probs_offline_test)

            for patient in range(batch_test):
                for visit in range(predicted_visit-1):
                    if death_probs_online_test[patient, visit, :] > 0.4072313:
                        death_estimated_online_test[patient, visit, :] = 1
                    if death_probs_offline_test[patient, visit, :] > 0.4072313:
                        death_estimated_offline_test[patient, visit, :] = 1

            print(np.sum(death_estimated_offline_test), np.sum(death_estimated_online_test))

            print('epoch  {}  agent_loss  {}  train_value_online  {} train_value_offline  {} '
                  'test_value_online {} test_value_offline {}'
                  .format(train_set.epoch_completed,
                          tf.reduce_mean(loss),
                          tf.reduce_mean(online_value),
                          tf.reduce_mean(offline_value),
                          tf.reduce_mean(test_online_value),
                          tf.reduce_mean(test_offline_value)))
            #
            # np.save('11_9_test_onlie_value_policy_gradient.npy', test_online_value)
            #
            # np.save('11_9_states_policy_gradient.npy', states_test_online)
            # np.save('11_9_reward_policy_gradient.npy', rewards_test_online)
            # np.save('11_9_actions_policy_gradient.npy', actions_test_online)
            # np.save('11_9_death_policy_gradient.npy', death_estimated_online_test)
            #
            # np.save('11_9_test_online_representation_policy_gradient.npy', test_online_representation)
            # np.save('11_9_test_offline_representation_policy_gradient.npy', test_offline_representation)

            # if tf.reduce_mean(test_online_value) > 12.3 and train_set.epoch_completed > 100:
            #     i = train_set.epoch_completed
            #     j = np.mean(test_online_value.numpy())
            #     agent.model.save_weights('11_9_policy_gradient_compare' + str(i) + '_' + str(j) + '.h5')
            #     reward_net.save_weights('11_9_reward_compare' + str(i) + '_' + str(j) + '.h5')
            #     environment_net.save_weights('11_9_environment_compare' + str(i) + '_' + str(j) + '.h5')
            #     print('{}次迭代保存成功！'.format(i))
            #
            # if train_set.epoch_completed > 140 and tf.reduce_mean(test_online_value) < 12:
            #     break

    tf.compat.v1.reset_default_graph()
    return tf.reduce_mean(test_online_value)


if __name__ == '__main__':
    test_test('PG_train_11_10_修改loss函数.txt')
    PG = BayesianOptimization(
        train_batch, {
            'hidden_size': (5, 8),
            'learning_rate': (-6, -1),
            'l2_regularization': (-6, -1),
        }
    )
    PG.maximize()
    print(PG.max)
    # for i in range(50):
    #     test_online_value = train_batch(hidden_size=32, learning_rate=0.1, l2_regularization=0.0017796045909055304)







