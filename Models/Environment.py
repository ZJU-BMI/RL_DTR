import tensorflow as tf
import numpy as np
import os
from tensorflow_core.python.keras.models import Model

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


# 仅为测试以上代码是否运行正确
def train():
    hidden_size = 128
    previous_visit = 3
    predicted_visit = 3
    l2_regularization = 0.0001
    learning_rate = 0.001
    input_data = np.load('../../RL_DTR/resource/HF_train_.npy').reshape(-1, 6, 30)[:, :, 1:6]
    input_data_test = np.load('../../RL_DTR/resource/HF_test_.npy').reshape(-1, 6, 30)[:, :, 1:6]
    feature_dims = input_data.shape[2]
    reconstruct_environment = ReconstructEnvir(hidden_size=hidden_size, feature_dims=feature_dims,
                                               previous_visit=previous_visit)
    batch_test = input_data_test.shape[0]
    # 训练
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    for epoch in range(10):
        batch = input_data.shape[0]
        with tf.GradientTape() as tape:
            predicted_trajectory = tf.zeros(shape=[batch, 0, feature_dims])
            for predicted_visit_ in range(predicted_visit):
                input_feature = input_data[:, :previous_visit+predicted_visit_, :]
                feature_next = reconstruct_environment(input_feature)
                predicted_trajectory = tf.concat((predicted_trajectory, tf.reshape(feature_next, [-1, 1, feature_dims])), axis=1)

            mse_loss = tf.reduce_mean(tf.keras.losses.mse(input_data[:, previous_visit:previous_visit+predicted_visit,:], predicted_trajectory))

            variables = [var for var in reconstruct_environment.trainable_variables]
            for weight in reconstruct_environment.trainable_variables:
                mse_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            gradient = tape.gradient(mse_loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))

            predicted_trajectory_test = np.zeros(shape=[batch_test, 0, feature_dims])
            for predicted_visit_ in range(predicted_visit):
                input_feature_test = tf.concat((input_data_test[:, :previous_visit, :], predicted_trajectory_test), axis=1)
                feature_next_test = reconstruct_environment(input_feature_test)
                predicted_trajectory_test = tf.concat((predicted_trajectory_test, tf.reshape(feature_next_test, [-1, 1, feature_dims])), axis=1)
            mse_loss_test = tf.reduce_mean(tf.keras.losses.mse(input_data_test[:, previous_visit:previous_visit+predicted_visit, :], predicted_trajectory_test))
            print('epoch----{}----mse_loss_train----{}---mse_loss_test---{}'.format(epoch, mse_loss, mse_loss_test))
            tf.compat.v1.reset_default_graph()


if __name__ == '__main__':
    train()

















