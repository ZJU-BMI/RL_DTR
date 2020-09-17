import tensorflow as tf
import numpy as np
import os
from tensorflow_core.python.keras.models import Model
from utilis import *
from data import DataSet
from bayes_opt import BayesianOptimization
from scipy import stats

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# 环境重建  输入前m天state, 输入第m+1天state,需要在解码中嵌套编码的过程
class ReconstructEnvir(Model):
    def __init__(self, hidden_size, feature_dims, previous_visit):
        super().__init__(name='environment_construct')
        self.hidden_size = hidden_size
        self.previous_visit = previous_visit
        self.encode = Encode(hidden_size)
        self.decode = Decode(hidden_size, feature_dims)

    def call(self, input_data):
        time_len = input_data.shape[1]
        batch = input_data.shape[0]
        for predicted_time in range(time_len-self.previous_visit+1):
            if predicted_time == 0:
                decode_c = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
                decode_h = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
            sequence_time = input_data[:, self.previous_visit+predicted_time-1, :]
            for previous_time in range(self.previous_visit+predicted_time):
                encode_sequence = input_data[:, previous_time, :]
                if previous_time == 0:
                    encode_c = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
                    encode_h = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
                encode_c, encode_h = self.encode([encode_sequence, encode_c, encode_h])
            decode_input = tf.concat((sequence_time, encode_h), axis=1)
            sequence_time_next, decode_c, decode_h = self.decode([decode_input, decode_c, decode_h])
        return sequence_time_next


# 环境编码器
class Encode(tf.keras.layers.Layer):
    def __init__(self, hidden_size):
        super().__init__(name='encode')
        self.hidden_size = hidden_size
        self.LSTM_Cell_encode = tf.keras.layers.LSTMCell(hidden_size)

    def call(self, input_data):
        sequence_time, encode_c, encode_h = input_data
        encode_state = [encode_c, encode_h]
        output, state = self.LSTM_Cell_encode(sequence_time, encode_state)
        return state[0], state[1]


# 环境解码器
class Decode(tf.keras.layers.Layer):
    def __init__(self, hidden_size, feature_dims):
        super().__init__(name='decode')
        self.hidden_size = hidden_size
        self.feature_dims = feature_dims
        self.LSTM_Cell_decode = tf.keras.layers.LSTMCell(hidden_size)
        self.dense1 = tf.keras.layers.Dense(feature_dims, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(feature_dims, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(feature_dims, activation=tf.nn.relu)

    def call(self, input_data):
        sequence_time, decode_c, decode_h = input_data
        decode_state = [decode_c, decode_h]
        output, state = self.LSTM_Cell_decode(sequence_time, decode_state)
        y_1 = self.dense1(output)
        y_2 = self.dense2(y_1)
        y_3 = self.dense3(y_2)
        return y_3, state[0], state[1]


#  环境预训练
def train_environment(hidden_size, learning_rate, l2_regularization):
    train_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA.npy')[:, :, 1:]
    # test_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\test_PLA.npy')
    test_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\validate_PLA.npy')[:, :, 1:]
    train_set.astype(np.float32)
    test_set.astype(np.float32)

    previous_visit = 3
    predicted_visit = 7
    feature_dims = 35

    train_set = DataSet(train_set)
    batch_size = 64
    epochs = 30

    hidden_size = 2**(int(hidden_size))
    learning_rate = 10 ** learning_rate
    l2_regularization = 10 ** l2_regularization

    print('feature_size---{}---previous_visit---{}----'
          'predicted_visit---{}---hidden_size---{}---'
          'learning_rate---{}---l2_regularization----{}'.format(feature_dims,
                                                                previous_visit,
                                                                predicted_visit,
                                                                hidden_size,
                                                                learning_rate,
                                                                l2_regularization))
    # 环境初始化
    construct_environment = ReconstructEnvir(hidden_size=hidden_size,
                                             feature_dims=feature_dims,
                                             previous_visit=previous_visit)
    batch_test = test_set.shape[0]
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    logged = set()
    while train_set.epoch_completed < epochs:
        input_x_train = train_set.next_batch(batch_size=batch_size)
        batch = input_x_train.shape[0]
        with tf.GradientTape() as tape:
            predicted_trajectory = np.zeros(shape=[batch, 0, feature_dims])

            for predicted_visit_ in range(predicted_visit-1):
                input_feature = input_x_train[:, :previous_visit+predicted_visit_, 5:]  # 将state和action共同输入至模型中(36个)
                input_action = input_x_train[:, :previous_visit+predicted_visit_, 0]
                input_data = tf.concat((input_feature, tf.reshape(input_action, [batch, -1, 1])), axis=2)

                feature_next = construct_environment(input_data)  # 仅仅预测下一个时间点的state(35个)
                predicted_trajectory = tf.concat((predicted_trajectory, tf.reshape(feature_next, [-1, 1, feature_dims])), axis=1)

            mse_loss = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, previous_visit+1:previous_visit+predicted_visit, 5:], predicted_trajectory))
            variables = [var for var in construct_environment.trainable_variables]
            for weight in construct_environment.trainable_variables:
                mse_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            gradient = tape.gradient(mse_loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)

                predicted_trajectory_test = tf.zeros(shape=[batch_test, 0, feature_dims])
                for predicted_visit_ in range(predicted_visit-1):
                    input_feature_test = test_set[:, :previous_visit, 5:]  # 缺少action
                    input_action_test = test_set[:, :previous_visit, 0]
                    input_data_test = tf.concat((input_feature_test, tf.reshape(input_action_test, [batch_test, -1, 1])), axis=2)
                    input_data_test = tf.cast(input_data_test, tf.float32)

                    # 将action定为真实的action
                    if predicted_trajectory_test.shape[1] != 0:
                        action = tf.cast(test_set[:, previous_visit:previous_visit+predicted_visit_, 1], tf.float32)
                        predicted_trajectory_test_ = tf.concat((predicted_trajectory_test, tf.reshape(action, [batch_test, predicted_visit_, 1])), axis=2)
                        input_data_test = tf.concat((input_data_test, predicted_trajectory_test_), axis=1)
                        input_data_test = tf.cast(input_data_test, tf.float32)
                    else:
                        input_data_test = input_data_test

                    feature_next_test = construct_environment(input_data_test)
                    predicted_trajectory_test = tf.concat((predicted_trajectory_test, tf.reshape(feature_next_test, [-1, 1, feature_dims])), axis=1)

                mse_loss_test = tf.reduce_mean(tf.keras.losses.mse(test_set[:, previous_visit+1:previous_visit+predicted_visit,5:], predicted_trajectory_test))
                mae_loss_test = tf.reduce_mean(tf.keras.losses.mae(test_set[:, previous_visit+1:previous_visit+predicted_visit,5:], predicted_trajectory_test))

                r_value_all = []
                for visit in range(predicted_trajectory_test.shape[0]):
                    for visit_day in range(predicted_trajectory_test.shape[1]):
                        real_data = test_set[visit, visit_day, 5:]
                        predicted_data = predicted_trajectory_test[visit, visit_day, :]
                        r_value_ = stats.pearsonr(real_data, predicted_data)
                        r_value_all.append(r_value_[0])

                print('epoch---{}----train_mse---{}---test_mse---{}---test_mae---{}--test_r--{}'
                      .format(train_set.epoch_completed,
                              mse_loss, mse_loss_test,
                              mae_loss_test,
                              np.mean(r_value_all)))
                # if mse_loss_test < 0.0076:
                #     construct_environment.save_weights('environment_3_7_39_9_15.h5')

    tf.compat.v1.reset_default_graph()
    # return mse_loss_test, mae_loss_test, np.mean(r_value_all)
    return -1*mse_loss_test


if __name__ == '__main__':
    test_test('environment_s2s_simulate_保存参数_.txt')
    Encode_Decode_Time_BO = BayesianOptimization(
        train_environment, {
            'hidden_size': (5, 8),
            'learning_rate': (-5, -1),
            'l2_regularization': (-5, -1),
        }
    )
    Encode_Decode_Time_BO.maximize()
    print(Encode_Decode_Time_BO.max)

    # mse_all = []
    # mae_all = []
    # r_value_all = []
    # for i in range(50):
    #     mse, mae, r = train_environment(hidden_size=64,
    #                                     learning_rate=0.0036370379238332626,
    #                                     l2_regularization=6.0823681740674096e-05)
    #     mse_all.append(mse)
    #     mae_all.append(mae)
    #     r_value_all.append(r)
    #     print('epoch---{}--mse_ave---{}---mae_ave---r_value_ave---{}---mse_std---{}---mae_std---{}---r_value_std---{}'
    #           .format(i,
    #                   np.mean(mse_all),
    #                   np.mean(mae_all),
    #                   np.mean(r_value_all),
    #                   np.std(mse_all),
    #                   np.std(mae_all),
    #                   np.std(r_value_all)))

















