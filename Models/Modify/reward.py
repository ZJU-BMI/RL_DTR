import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.models import Model
from utilis import *
from origin.data import DataSet
import os
from environment import Encode
from scipy import stats


class Reward(Model):
    def __init__(self, hidden_size):
        super().__init__(name='reward_net')
        self.hidden_size = hidden_size
        self.dense1 = tf.keras.layers.Dense(hidden_size)
        self.dense2 = tf.keras.layers.Dense(hidden_size)
        self.dense3 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

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

    train_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA.npy')[:, :, 1:]
    test_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\validate_PLA.npy')[:, :, 1:]
    train_set.astype(np.float32)
    test_set.astype(np.float32)
    train_set = DataSet(train_set)
    batch_size = 64
    epochs = 30
    previous_visit = 3
    predicted_visit = 7

    encode_net = Encode(hidden_size=128)
    encode_net(tf.zeros(shape=[batch_size, 5, 35]))
    reward_net = Reward(hidden_size=hidden_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    logged = set()
    while train_set.epoch_completed < epochs:
        with tf.GradientTape() as tape:
            encode_net.load_weights('encode_net_10_27.h5')
            input_x_train = train_set.next_batch(batch_size=batch_size)
            input_x_train.astype(np.float32)
            # 净出量
            rewards_only_output = input_x_train[:, previous_visit + 1:previous_visit + predicted_visit, 1] - input_x_train[:, previous_visit + 1:previous_visit + predicted_visit, 2]
            batch = input_x_train.shape[0]
            rewards_predicted = tf.zeros(shape=[batch, 0, 1])
            for i in range(predicted_visit-1):
                features = input_x_train[:, :previous_visit+i, 5:]
                state = encode_net(features)
                action = input_x_train[:, previous_visit+i-1, 0]
                reward = reward_net([state, action])
                rewards_predicted = tf.concat((rewards_predicted, tf.reshape(reward, [batch, -1, 1])), axis=1)

            rewards_only_output = tf.reshape(rewards_only_output, [batch, -1, 1])
            mse_loss = tf.reduce_mean(tf.keras.losses.mse(rewards_only_output, rewards_predicted))
            variables = [var for var in reward_net.trainable_variables]
            for weight in reward_net.trainable_variables:
                mse_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            gradient = tape.gradient(mse_loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)
                batch_test = test_set.shape[0]
                rewards_predicted_test = tf.zeros(shape=[batch_test, 0, 1])

                rewards_only_output_test = test_set[:, previous_visit+1:previous_visit+predicted_visit, 1] - test_set[:, previous_visit+1:previous_visit+predicted_visit, 2]
                for i in range(predicted_visit-1):
                    features_test = test_set[:, :previous_visit+i, 5:]
                    state_test = encode_net(features_test)
                    action_test = test_set[:, previous_visit+i-1, 0]
                    reward_test = reward_net([state_test, action_test])
                    rewards_predicted_test = tf.concat((rewards_predicted_test, tf.reshape(reward_test, [batch_test, -1, 1])), axis=1)

                rewards_only_output_test = tf.reshape(rewards_only_output_test, [batch_test, -1, 1])
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

                if mse_test < 0.00475 and np.abs(r_value_all) > 0.14:
                    reward_net.save_weights('reward_net_10_27.h5')
    tf.compat.v1.reset_default_graph()
    return -1 *mse_test


if __name__ == '__main__':
    test_test('11_5_reward_net_重新训练_保存参数.txt')
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
        mse_test = pre_train_reward(hidden_size=128, learning_rate=0.038456099491477, l2_regularization=8.777769094726096e-06)

