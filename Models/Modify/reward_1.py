import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from utilis import *
from data import DataSet
import os
from environment_1 import Encode
from scipy import stats


class Reward(Model):
    def __init__(self, hidden_size):
        super().__init__(name='reward_net')
        self.hidden_size = hidden_size
        self.dense1 = tf.keras.layers.Dense(hidden_size)
        self.dense2 = tf.keras.layers.Dense(hidden_size)
        self.dense3 = tf.keras.layers.Dense(2, activation=tf.nn.sigmoid)

    def call(self, input_data):
        """
        :param input_data: s_(t-1), a_t
        :return: r_t
        """
        s_t_1, a_t = input_data
        batch = tf.shape(s_t_1)[0]
        inputs = tf.concat((s_t_1, tf.reshape(a_t, [batch, -1])), axis=1)
        output = self.dense1(inputs)
        output = self.dense2(output)
        output = self.dense3(output)
        return output


def pre_train_reward(hidden_size, learning_rate, l2_regularization):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpus = tf.config.experimental.list_physical_devices(device_type=True)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # hidden_size = 2 ** int(hidden_size)
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization
    print('hidden_size---{}---learning_rate---{}---l2_regularization---{}'
          .format(hidden_size, learning_rate, l2_regularization))

    # train_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA.npy')[:, :, 1:]
    # test_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\validate_PLA.npy')[:, :, 1:]

    train_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\mimic_train.npy')
    test_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\mimic_validate.npy')

    train_set.astype(np.float32)
    test_set.astype(np.float32)
    train_set = DataSet(train_set)
    batch_size = 64
    epochs = 30
    previous_visit = 3
    predicted_visit = 10

    encode_net = Encode(hidden_size=256)
    encode_net([tf.zeros(shape=[batch_size, 4, 45]), tf.zeros(shape=[batch_size, 4, 2])])
    reward_net = Reward(hidden_size=hidden_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    logged = set()

    iv_list = [0, 33, 102, 330, 1153]
    vas_list = [0, 0.045, 0.15, 0.32, 1.24]

    while train_set.epoch_completed < epochs:
        with tf.GradientTape() as tape:
            encode_net.load_weights('encode_net_12_7_mimic.h5')
            input_x_train = train_set.next_batch(batch_size=batch_size)
            input_x_train.astype(np.float32)
            # 净出量
            rewards_labels = input_x_train[:, previous_visit + 1:previous_visit+predicted_visit, 1:3]
            batch = input_x_train.shape[0]
            rewards_predicted = tf.zeros(shape=[batch, 0, 2])
            for i in range(predicted_visit-1):
                features = input_x_train[:, :previous_visit+i, 4:]
                actions_ = tf.zeros(shape=[batch, 1, 1], dtype=tf.float64)
                actions = tf.reshape(input_x_train[:, :previous_visit+i-1, 0], [batch, -1, 1])
                actions = tf.concat((actions_, actions), axis=1)
                actions_index = actions.numpy()
                actions_values = np.zeros(shape=[batch, i + previous_visit, 2])
                for patient in range(batch):
                    for visit in range(i+previous_visit):
                        action_index = actions_index[patient, visit, 0]
                        iv_index = int(action_index / 5)
                        vas_index = int(action_index % 5)
                        actions_values[patient, visit, 0] = iv_list[iv_index]
                        actions_values[patient, visit, 1] = vas_list[vas_index]

                state = encode_net([features, actions_values])  # s_t-1
                action = input_x_train[:, previous_visit+i-1, 0]  # a_t
                action_value_feed_reward = np.zeros(shape=[batch, 2])
                for patient in range(batch):
                    action_index = action[patient]
                    iv_index = int(action_index / 5)
                    vas_index = int(action_index % 5)
                    action_value_feed_reward[patient, 0] = iv_list[iv_index]
                    action_value_feed_reward[patient, 1] = vas_list[vas_index]

                reward = reward_net([state, action_value_feed_reward])
                # reward 包含SOFA和lactate两个部分
                rewards_predicted = tf.concat((rewards_predicted, tf.reshape(reward, [batch, -1, 2])), axis=1)

            rewards_labels_reshape = tf.reshape(rewards_labels, [batch, -1, 2])  # 得到所有的标签
            mse_loss = tf.reduce_mean(tf.keras.losses.mse(rewards_labels_reshape, rewards_predicted))
            variables = [var for var in reward_net.trainable_variables]
            for weight in reward_net.trainable_variables:
                mse_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            gradient = tape.gradient(mse_loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)
                batch_test = test_set.shape[0]
                rewards_predicted_test = tf.zeros(shape=[batch_test, 0, 2])

                rewards_labels_test = test_set[:, previous_visit+1:previous_visit+predicted_visit, 1:3]
                for i in range(predicted_visit-1):
                    features_test = test_set[:, :previous_visit+i, 4:]
                    actions_test_ = tf.zeros(shape=[batch_test, 1, 1], dtype=tf.float64)
                    actions_test = tf.reshape(test_set[:, :previous_visit+i-1, 0], [batch_test, -1, 1])
                    actions_test = tf.concat((actions_test_, actions_test), axis=1)
                    actions_test_index = actions_test.numpy()
                    actions_test_values = np.zeros(shape=[batch_test, previous_visit+i, 2])
                    for patient in range(batch_test):
                        for visit in range(previous_visit+i):
                            action_index = actions_test_index[patient, visit, 0]
                            iv_index = int(action_index / 5)
                            vas_index = int(action_index % 5)
                            actions_test_values[patient, visit, 0] = iv_list[iv_index]
                            actions_test_values[patient, visit, 1] = vas_list[vas_index]

                    state_test = encode_net([features_test, actions_test_values])
                    action_test = test_set[:, previous_visit+i-1, 0]
                    action_test_reward = np.zeros(shape=[batch_test, 2])
                    for patient in range(batch_test):
                        action_index = action_test[patient]
                        iv_index = int(action_index / 5)
                        vas_index = int(action_index % 5)
                        action_test_reward[patient, 0] = iv_list[iv_index]
                        action_test_reward[patient, 1] = vas_list[vas_index]
                    reward_test = reward_net([state_test, action_test_reward])
                    rewards_predicted_test = tf.concat((rewards_predicted_test, tf.reshape(reward_test, [batch_test, -1, 2])), axis=1)

                rewards_only_output_test = tf.reshape(rewards_labels_test, [batch_test, -1, 2])
                mse_test = tf.reduce_mean(tf.keras.losses.mse(rewards_only_output_test, rewards_predicted_test))
                mae_test = tf.reduce_mean(tf.keras.losses.mae(rewards_only_output_test, rewards_predicted_test))

                # r_value_all = []
                # for patient in range(batch_test):
                #     real_data = tf.reshape(rewards_only_output_test[patient, :, :], [-1, ])
                #     predicted_data = tf.reshape(rewards_predicted_test[patient, :, :], [-1, ])
                #     r_value = stats.pearsonr(real_data, predicted_data)
                #     r_value_all.append(r_value[0])
                r_value_all = stats.pearsonr(tf.reshape(rewards_only_output_test, [-1,]), tf.reshape(rewards_predicted_test, [-1,]))[0]
                print('epoch---{}---mse_train---{}---mse_test---{}---mae_test---{}---r_value---{}'
                      .format(train_set.epoch_completed,
                              mse_loss,
                              mse_test,
                              mae_test,
                              r_value_all))

                # if mse_test < 0.004715 and np.abs(r_value_all) > 0.10:
                #     reward_net.save_weights('reward_net_11_30.h5')
    tf.compat.v1.reset_default_graph()
    return -1 * mse_test


if __name__ == '__main__':
    test_test('12_22_reward_mimic_train.txt')
    # reward_train = BayesianOptimization(
    #     pre_train_reward, {
    #         'hidden_size': (5, 8),
    #         'learning_rate': (-6, 0),
    #         'l2_regularization': (-6, 0),
    #     }
    # )
    # reward_train.maximize()
    # print(reward_train.max)
    for i in range(50):
        mse_test = pre_train_reward(hidden_size=32, learning_rate=1.0, l2_regularization=1e-06)

