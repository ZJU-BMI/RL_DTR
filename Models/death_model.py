import tensorflow as tf
import numpy as np
from data import DataSet
from tensorflow_core.python.keras.models import Model
import os
from utilis import *
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from imblearn.over_sampling import SMOTE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# 死亡模型
class DeathModel(Model):
    def __init__(self, hidden_size):
        super().__init__(name='death_model')
        self.hidden_size = hidden_size
        self.LSTM_Cell_encode = tf.keras.layers.LSTMCell(hidden_size)
        self.dense1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, input_data):
        time_len = input_data.shape[1]
        batch = input_data.shape[0]
        for predicted_time in range(time_len):
            sequence_time = input_data[:, predicted_time, :]
            if predicted_time == 0:
                encode_c = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
                encode_h = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
            state = [encode_c, encode_h]
            output, state = self.LSTM_Cell_encode(sequence_time, state)
        death_probability = self.dense1(output)
        return tf.reshape(death_probability, [batch, -1])


def imbalance_preprocess(feature, label):
    method = SMOTE()
    x_res, y_res = method.fit_sample(feature.reshape([-1, feature.shape[2]]),
                                     label.reshape([-1, 1]))
    x_size = int(x_res.shape[0]/feature.shape[1]) * feature.shape[1]
    feature_new = x_res[0:x_size, :].reshape(-1, feature.shape[1], feature.shape[2])
    label_new = y_res[0:x_size].reshape(-1, feature.shape[1])
    return feature_new, label_new


def train_death_model(hidden_size, learning_rate, l2_regularization):
    train_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA.npy')[:, :, 1:]
    train_set_label = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA_label.npy')
    train_set_label = train_set_label.reshape([train_set_label.shape[0], train_set_label.shape[1], -1])
    train_set = np.concatenate((train_set[:, :, 5:], train_set_label), axis=2)

    test_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\test_PLA.npy')[:, :, 1:]
    test_set_label = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\test_PLA_label.npy')
    test_set_label = test_set_label.reshape([test_set_label.shape[0], test_set_label.shape[1], -1])
    test_set = np.concatenate((test_set[:, :, 5:], test_set_label), axis=2)

    train_set.astype(np.float32)
    test_set.astype(np.float32)
    train_set = DataSet(train_set)
    batch_size = 256
    epochs = 30
    time_steps = test_set.shape[1]

    # hidden_size = 2 ** int(hidden_size)
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization
    print('hidden_size---{}---learning_rate---{}---l2_regularization---{}'
          .format(hidden_size, learning_rate, l2_regularization))

    death_model = DeathModel(hidden_size=hidden_size)
    batch_test = test_set.shape[0]
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    logged = set()
    while train_set.epoch_completed < epochs:
        input_x_data = train_set.next_batch(batch_size=batch_size)
        input_x_feature = input_x_data[:, :, :-1]
        input_x_label = input_x_data[:, :, -1]
        input_x_feature, input_x_label = imbalance_preprocess(input_x_feature, input_x_label)
        batch = input_x_feature.shape[0]
        with tf.GradientTape() as tape:
            predicted_death = np.zeros(shape=[batch, 0])
            for predicted_time_ in range(time_steps):  # 当次状态预测当次标签
                sequence_time = input_x_feature[:, :predicted_time_+1, :]
                death_ = death_model(sequence_time)
                predicted_death = tf.concat((predicted_death, death_), axis=1)

            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=input_x_label, y_pred=predicted_death))
            variables = [var for var in death_model.trainable_variables]
            for weight in death_model.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            gradient = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                death_model.load_weights('death_model_0_10_10_13_train_test.h5')
                logged.add(train_set.epoch_completed)
                predicted_death_test = tf.zeros(shape=[batch_test, 0])
                death_test_label = test_set[:, :, -1]
                for predicted_time_ in range(time_steps):
                    sequence_time_test = test_set[:, :predicted_time_+1, :-1]
                    death_test_ = death_model(sequence_time_test)
                    predicted_death_test = tf.concat((predicted_death_test, death_test_), axis=1)

                test_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=death_test_label, y_pred=predicted_death_test))
                auc = roc_auc_score(death_test_label.reshape([-1,]), predicted_death_test.numpy().reshape([-1,]))
                fpr, tpr, thread = roc_curve(death_test_label.reshape([-1,]), predicted_death_test.numpy().reshape([-1,]), pos_label=1)
                threshold = thread[np.argmax(tpr - fpr)]
                y_pre_label = (predicted_death_test.numpy().reshape([-1,]) >= threshold) * 1
                print(max(y_pre_label))
                acc = accuracy_score(death_test_label.reshape([-1,]), y_pre_label)
                print(threshold)

                # test_auc = tf.keras.metrics.AUC(labels=death_test_label, predictions=predicted_death_test)
                print('epoch---{}---train_loss--{}----test_loss---{}--auc---{}---acc---{}-'.format(
                    train_set.epoch_completed, loss, test_loss, auc, acc))

                # if test_loss < 0.565 and auc > 0.70 and acc > 0.736:
                #     death_model.save_weights('death_model_0_10_10_13_train_test.h5')

    tf.compat.v1.reset_default_graph()
    return -1 * test_loss


if __name__ == "__main__":
    # test_test('death_model_10_12_train_test_set.txt')
    # Encode_Decode_Time_BO = BayesianOptimization(
    #     train_death_model, {
    #         'hidden_size': (5, 8),
    #         'learning_rate': (-5, 0),
    #         'l2_regularization': (-5, 0),
    #     }
    # )
    # Encode_Decode_Time_BO.maximize()
    # print(Encode_Decode_Time_BO.max)

    test_loss_all = []
    for i in range(50):
        test_loss = train_death_model(hidden_size=128,
                                      learning_rate=0.008637291855531266,
                                      l2_regularization=0.00020277674620112292)
        test_loss_all.append(test_loss)
        print(i, np.mean(test_loss_all))













