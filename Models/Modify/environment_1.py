import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.models import Model
from utilis import *
from data import DataSet
import os
from scipy import stats

gpus = tf.config.experimental.list_physical_devices(device_type=True)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# These modules are used for training MIMIC-III dataset
class Environment(Model):
    def __init__(self, hidden_size):
        super().__init__(name='environment')
        self.hidden_size = hidden_size
        self.dense1 = tf.keras.layers.Dense(hidden_size)
        self.dense2 = tf.keras.layers.Dense(hidden_size)
        self.dense3 = tf.keras.layers.Dense(256, activation=tf.nn.relu)

    def call(self, input_data):
        """
        :param input_data: [s_(t-1), a_t]
        :return: sequence_time_next: s_t
        """
        s_t_1, a_t = input_data
        batch = tf.shape(s_t_1)[0]
        inputs = tf.concat((s_t_1, tf.reshape(a_t, [batch, -1])), axis=1)
        output = self.dense1(inputs)
        output = self.dense2(output)
        output = self.dense3(output)
        return output


class Encode(Model):
    def __init__(self, hidden_size):
        super().__init__(name='encode_net')
        self.hidden_size = hidden_size
        self.encode = tf.keras.layers.LSTMCell(hidden_size)

    def call(self, input_data):
        """
        :param input_data: (x_1, x_2, ...x_t), (a_1, a_2, ...a_t_1)
        :return: s_t
        """
        features, actions = input_data
        data_input = tf.concat((features, actions), axis=2)
        global state, encode_c, encode_h
        time_len = tf.shape(data_input)[1]
        batch = tf.shape(data_input)[0]
        for time_ in range(time_len):
            if time_ == 0:
                encode_c = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
                encode_h = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
            sequence_time = data_input[:, time_, :]
            encode_state = [encode_h, encode_c]
            output, state = self.encode(sequence_time, encode_state)
        return state[0]


class Decode(Model):
    def __init__(self, feature_size, hidden_size):
        super().__init__(name='decode_net')
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.decode = tf.keras.layers.LSTMCell(hidden_size)
        self.dense1 = tf.keras.layers.Dense(feature_size, activation=tf.nn.relu)

    def call(self, input_data):
        """
        :param input_data: s_0, s_1,s_2...s_t
        :return: x_t
        """
        global state, decode_c, decode_h

        time_len = input_data.shape[1]
        batch = input_data.shape[0]
        for time_ in range(time_len):
            if time_ == 0:
                decode_c = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
                decode_h = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
            sequence_time = input_data[:, time_, :]
            decode_state = [decode_h, decode_c]
            output, [decode_h, decode_c] = self.decode(sequence_time, decode_state)
        reconstruct_x = self.dense1(decode_h)
        return reconstruct_x


def pre_train_autoencoder(hidden_size, learning_rate, l2_regularization):

    # train_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA.npy')[:, :, 1:]
    # test_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\validate_PLA.npy')[:, :, 1:]
    train_set = np.load('mimic_train.npy')
    test_set = np.load('mimic_validate.npy')

    train_set.astype(np.float32)
    test_set.astype(np.float32)
    train_set = DataSet(train_set)
    batch_size = 64
    epochs = 30
    feature_size = 45
    previous_visit = 3
    predicted_visit = 10
    
    iv_list = [0, 33, 102, 330, 1153]
    vas_list = [0, 0.045, 0.15, 0.32, 1.24]
    
    # hidden_size = 2 ** int(hidden_size)
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization

    print('hidden_size---{}---learning_rate---{}---l2_regularization---{}'
          .format(hidden_size, learning_rate, l2_regularization))

    encode_net = Encode(hidden_size=hidden_size)
    decode_net = Decode(feature_size=feature_size, hidden_size=hidden_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    logged = set()
    while train_set.epoch_completed < epochs:
        with tf.GradientTape() as tape:
            input_x_train = train_set.next_batch(batch_size=batch_size)
            input_x_train.astype(np.float32)
            batch = input_x_train.shape[0]
            reconstruct_x = tf.zeros(shape=[batch, 0, feature_size])
            encode_save = tf.zeros(shape=[batch, 0, hidden_size])
            for i in range(predicted_visit):
                features = input_x_train[:, :previous_visit+i, 4:]
                actions_index_ = tf.zeros(shape=[batch, 1, 1])*1.0
                actions_index_ = tf.cast(actions_index_, tf.float64)
                actions_index = tf.reshape(input_x_train[:, :previous_visit+i-1, 0], [batch, -1, 1])
                actions_index = tf.concat((actions_index_, actions_index), axis=1).numpy()
                actions_value = np.zeros(shape=[batch, previous_visit+i, 2], dtype=np.float32)
                for patient in range(batch):
                    for j in range(previous_visit+i):
                        action_index = actions_index[patient, j, 0]
                        action_index_1 = int(action_index / 5)
                        action_index_2 = int(action_index % 5)
                        actions_value[patient, j, 0] = iv_list[action_index_1] 
                        actions_value[patient, j, 1] = vas_list[action_index_2]
                
                feature_encode = encode_net([features, actions_value])
                encode_save = tf.concat((encode_save, tf.reshape(feature_encode, [batch, -1, hidden_size])), axis=1)
                feature_decode = decode_net(encode_save)
                reconstruct_x = tf.concat((reconstruct_x, tf.reshape(feature_decode, [batch, -1, feature_size])),  axis=1)
            mse_loss = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit+predicted_visit, 4:], reconstruct_x))
            variables = [var for var in encode_net.trainable_variables]
            for weight in encode_net.trainable_variables:
                mse_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            for weight in decode_net.trainable_variables:
                mse_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

            gradient = tape.gradient(mse_loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)
                batch_test = test_set.shape[0]

                predicted_reconstruct_features = tf.zeros(shape=[batch_test, 0, feature_size])
                encode_test_save = tf.zeros(shape=[batch_test, 0, hidden_size])

                for i in range(predicted_visit):
                    # features_test = test_set[:, :previous_visit+i, 5:]
                    features_test = test_set[:, :previous_visit+i, 4:]
                    actions_test_index_ = tf.zeros(shape=[batch_test, 1, 1])*1.0
                    actions_test_index_ = tf.cast(actions_test_index_, tf.float64)
                    actions_test_index = tf.reshape(test_set[:, :previous_visit+i-1, 0], [batch_test, -1, 1])
                    actions_test_index = tf.concat((actions_test_index_, actions_test_index), axis=1)
                    actions_test_value = np.zeros(shape=[batch_test, previous_visit+i, 2], dtype=np.float32)
                    for patient in range(batch_test):
                        for j in range(previous_visit+i):
                            action_index = actions_test_index[patient, j, 0]
                            action_index_1 = int(action_index/5)
                            action_index_2 = int(action_index%5)
                            actions_test_value[patient, j, 0] = iv_list[action_index_1] 
                            actions_test_value[patient, j, 1] = vas_list[action_index_2]
                    
                    features_test_encode = encode_net([features_test, actions_test_value])
                    encode_test_save = tf.concat((encode_test_save, tf.reshape(features_test_encode, [batch_test, -1, hidden_size])), axis=1)
                    features_decode = decode_net(encode_test_save)
                    predicted_reconstruct_features = tf.concat((predicted_reconstruct_features, tf.reshape(features_decode, [batch_test, -1, feature_size])), axis=1)
                # PLA dataset
                # mse_loss_test = tf.reduce_mean(tf.keras.losses.mse(test_set[:, previous_visit:previous_visit+predicted_visit, 4:], predicted_reconstruct_features))
                # mae_loss_test = tf.reduce_mean(tf.keras.losses.mae(test_set[:, previous_visit:previous_visit+predicted_visit, 4:], predicted_reconstruct_features))

                # MIMIC-III dataset
                mse_loss_test = tf.reduce_mean(tf.keras.losses.mse(test_set[:, previous_visit:previous_visit + predicted_visit, 4:], predicted_reconstruct_features))
                mae_loss_test = tf.reduce_mean(tf.keras.losses.mae(test_set[:, previous_visit:previous_visit + predicted_visit, 4:], predicted_reconstruct_features))

                r_value_all = []
                for patient in range(predicted_reconstruct_features.shape[0]):
                    for visit in range(predicted_reconstruct_features.shape[1]):
                        # real_data = test_set[patient, visit, 5:]
                        real_data = test_set[patient, visit, 4:]
                        predicted_data = predicted_reconstruct_features[patient, visit, :]
                        r_value_ = stats.pearsonr(real_data, predicted_data)
                        r_value_all.append(r_value_[0])
                print('mse_loss---{}---mse_test---{}---mae_test---{}---r_value---{}'
                      .format(mse_loss, mse_loss_test, mae_loss_test, np.mean(r_value_all)))
                if mse_loss_test < 0.00363 and np.mean(r_value_all) > 0.975 and train_set.epoch_completed == 30:
                    encode_net.save_weights('encode_net_12_7_mimic.h5')
                    decode_net.save_weights('decode_net_12_7_mimic.h5')
                    break

    tf.compat.v1.reset_default_graph()
    return mse_loss_test, mae_loss_test, np.mean(r_value_all)
    # return -1 * mse_loss_test


def train_environment(hidden_size, learning_rate, l2_regularization):
    gpus = tf.config.experimental.list_physical_devices(device_type=True)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # hidden_size = 2 ** int(hidden_size)
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization
    print('hidden_size---{}---learning_rate---{}---l2_regularization---{}'
          .format(hidden_size, learning_rate, l2_regularization))

    previous_visit = 3
    predicted_visit = 10

    # train_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA.npy')[:, :, 1:]
    # test_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\validate_PLA.npy')[:, :, 1:]
    
    train_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\mimic_train.npy')
    test_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\mimic_validate.npy')

    iv_list = [0, 33, 102, 330, 1153]
    vas_list = [0, 0.045, 0.15, 0.32, 1.24]

    train_set.astype(np.float32)
    test_set.astype(np.float32)
    train_set = DataSet(train_set)
    epochs = 30
    batch_size = 64

    encode_net = Encode(hidden_size=256)
    environment_net = Environment(hidden_size=hidden_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    encode_net([tf.zeros(shape=[batch_size, 4, 45]), tf.zeros(shape=[batch_size, 4, 2])])

    logged = set()
    while train_set.epoch_completed < epochs:
        with tf.GradientTape() as tape:
            encode_net.load_weights('encode_net_12_7_mimic.h5')
            input_x_train = train_set.next_batch(batch_size=batch_size)
            batch = input_x_train.shape[0]
            state_labels = tf.zeros(shape=[batch, 0, 256])
            state_predict = tf.zeros(shape=[batch, 0, 256])
            for i in range(predicted_visit-1):
                features = input_x_train[:, :previous_visit + i, 4:]
                actions_ = tf.zeros(shape=[batch, 1, 1], dtype=tf.float64)
                actions = tf.reshape(input_x_train[:, :previous_visit+i-1, 0], [batch, -1, 1])
                actions = tf.concat((actions_, actions), axis=1)
                actions_index = actions.numpy()
                actions_values = np.zeros(shape=[batch, previous_visit+i, 2])
                for patient in range(batch):
                    for visit in range(previous_visit+i):
                        action_index = actions_index[patient, visit, 0]
                        iv_index = int(action_index / 5)
                        vas_index = int(action_index % 5)
                        actions_values[patient, visit, 0] = iv_list[iv_index]
                        actions_values[patient, visit, 1] = vas_list[vas_index]
                encode_feature = encode_net([features, actions_values])  # s_t
                action = input_x_train[:, previous_visit + i - 1, 0].reshape(-1, 1, 1)
                action_value = np.zeros(shape=[batch, 1, 2])
                for patient in range(batch):
                    for visit in range(action.shape[1]):
                        action_index = action[patient, visit, 0]
                        iv_index = int(action_index / 5)
                        vas_index = int(action_index % 5)
                        action_value[patient, visit, 0] = iv_list[iv_index]
                        action_value[patient, visit, 1] = vas_list[vas_index]
                next_encode_feature_predicted = environment_net([encode_feature, action_value])
                state_predict = tf.concat((state_predict, tf.reshape(next_encode_feature_predicted, [batch, -1, 256])), axis=1)  # 获取预测得到的结果

                features_next = input_x_train[:, :previous_visit+i+1, 4:]
                actions_next_ = tf.zeros(shape=[batch, 1, 1], dtype=tf.float64)
                actions_next = tf.reshape(input_x_train[:, :previous_visit+i, 0], [batch, -1, 1])
                actions_next = tf.concat((actions_next_, actions_next), axis=1)
                actions_next_values = np.zeros(shape=[batch, previous_visit+i+1, 2])
                actions_next_index = actions_next.numpy()
                for patient in range(batch):
                    for visit in range(previous_visit+i+1):
                        action_next_index = actions_next_index[patient, visit, 0]
                        iv_index = int(action_next_index / 5)
                        vas_index = int(action_next_index % 5)
                        actions_next_values[patient, visit, 0] = (iv_list[iv_index])
                        actions_next_values[patient, visit, 1] = (vas_list[vas_index])

                next_encode_feature_label = encode_net([features_next, actions_next_values])
                state_labels = tf.concat((state_labels, tf.reshape(next_encode_feature_label, [batch, -1, 256])), axis=1)  # 获取label

            mse_loss = tf.reduce_mean(tf.keras.losses.mse(state_labels, state_predict))
            variables = [var for var in environment_net.trainable_variables]
            for weight in environment_net.trainable_variables:
                mse_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            gradient = tape.gradient(mse_loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)
                batch_test = test_set.shape[0]
                state_labels_test = tf.zeros(shape=[batch_test, 0, 64])
                state_predicted_test = tf.zeros(shape=[batch_test, 0, 64])
                for i in range(predicted_visit-1):
                    features_test = test_set[:, :previous_visit+i, 4:]
                    actions_test_ = tf.zeros(shape=[batch_test, 1, 1], dtype=tf.float64)
                    actions_test = tf.reshape(test_set[:, :previous_visit+i-1, 0], [batch_test, -1, 1])
                    actions_test = tf.concat((actions_test_, actions_test), axis=1)
                    actions_test_index = actions_test.numpy()
                    actions_test_values = np.zeros(shape=[batch_test, previous_visit+i, 2])
                    for patient in  range(batch_test):
                        for visit in range(previous_visit+i):
                            action_index = actions_test_index[patient, visit, 0]
                            iv_index = int(action_index / 5)
                            vas_index = int(action_index % 5)
                            actions_test_values[patient, visit, 0] = iv_list[iv_index]
                            actions_test_values[patient, visit, 1] = vas_list[vas_index]

                    encode_feature_test = encode_net([features_test, actions_test_values])

                    action_test = test_set[:, previous_visit+i-1, 0].reshape(-1, 1, 1)
                    action_test_value = np.zeros(shape=[batch_test, 1, 2])
                    for patient in range(batch_test):
                        for visit in range(action_test.shape[1]):
                            action_test_index = action_test[patient, visit, 0]
                            iv_index = int(action_test_index / 5)
                            vas_index = int(action_test_index % 5)

                            action_test_value[patient, visit, 0] = iv_list[iv_index]
                            action_test_value[patient, visit, 1] = vas_list[vas_index]

                    next_encode_feature_predicted_test = environment_net([encode_feature_test, action_test_value])
                    state_predicted_test = tf.concat((state_predicted_test, tf.reshape(next_encode_feature_predicted_test, [batch_test, -1, 64])), axis=1)  # predicted next state

                    next_features_test = test_set[:, :previous_visit+i+1, 4:]
                    next_actions_test_ = tf.zeros(shape=[batch_test, 1, 1], dtype=tf.float64)
                    next_actions_test = tf.reshape(test_set[:, :previous_visit+i, 0], [batch_test, -1, 1])
                    next_actions_test = tf.concat((next_actions_test_, next_actions_test), axis=1)
                    next_actions_test_index = next_actions_test.numpy()
                    next_actions_test_values = np.zeros(shape=[batch_test, previous_visit+i+1, 2])
                    for patient in range(batch_test):
                        for visit in range(next_actions_test_values.shape[1]):
                            action_index = next_actions_test_index[patient, visit, 0]
                            iv_index = int(action_index / 5)
                            vas_index = int(action_index % 5)
                            next_actions_test_values[patient, visit, 0] = iv_list[iv_index]
                            next_actions_test_values[patient, visit, 1] = vas_list[vas_index]

                    next_encode_feature_label_test = encode_net([next_features_test, next_actions_test_values])
                    state_labels_test = tf.concat((state_labels_test, tf.reshape(next_encode_feature_label_test, [batch_test, -1, 64])), axis=1)  # predicted state label

                mse_loss_test = tf.reduce_mean(tf.keras.losses.mse(state_labels_test, state_predicted_test))
                mae_loss_test = tf.reduce_mean(tf.keras.losses.mae(state_labels_test, state_predicted_test))

                r_value_all = []
                for patient in range(batch_test):
                    for visit in range(predicted_visit-1):
                        real_data = state_labels_test[patient, visit, :]
                        predicted_data = state_predicted_test[patient, visit, :]
                        r_value = stats.pearsonr(real_data, predicted_data)
                        r_value_all.append(r_value[0])
                print('epoch---  {}---mse_train---  {}--mse_test---   {}--- mae_test---   {}---r_value---  {}'
                      .format(train_set.epoch_completed, mse_loss, mse_loss_test, mae_loss_test, np.mean(r_value_all)))

                # if mse_loss_test < 0.0074 and np.mean(r_value_all) > 0.89:
                    # environment_net.save_weights('environment_net_11_30.h5')

    tf.compat.v1.reset_default_graph()
    # return mse_loss_test, mae_loss_test, np.mean(r_value_all)
    return -1*mse_loss_test


if __name__ == '__main__':
    test_test('12_14_environment_train.txt')
    mse_all = []
    mae_all = []
    # environment_train = BayesianOptimization(
    #     train_environment, {
    #         'hidden_size': (5, 8),
    #         'learning_rate': (-6, 0),
    #         'l2_regularization': (-6, 0),
    #     }
    # )
    # environment_train.maximize()
    # print(environment_train.max)
    for i in range(50):
        mse, mae_, r = train_environment(hidden_size=256,
                                         learning_rate=0.01,
                                         l2_regularization=1e-06)






