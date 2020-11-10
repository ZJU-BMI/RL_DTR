import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.models import Model
from utilis import *
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from origin.data import DataSet
from environment import Encode

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class DeathModel(Model):
    def __init__(self, hidden_size):
        super().__init__(name='death_model')
        self.dense1 = tf.keras.layers.Dense(hidden_size)
        self.dense2 = tf.keras.layers.Dense(hidden_size)
        self.dense3 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, states):
        batch = tf.shape(states)[0]
        death_probs = tf.zeros(shape=[batch, 0, 1])
        for step in range(tf.shape(states)[1]):
            state = states[:, step, :]
            hidden = self.dense1(state)
            hidden = self.dense2(hidden)
            output = self.dense3(hidden)
            death_probs = tf.concat((death_probs, tf.reshape(output, [batch, -1, 1])), axis=1)
        return death_probs


def imbalance_preprocess(feature, label):
    method = SMOTE()
    x_res, y_res = method.fit_sample(feature.reshape([-1, feature.shape[2]]),
                                     label.reshape([-1, 1]))
    x_size = int(x_res.shape[0]/feature.shape[1]) * feature.shape[1]
    feature_new = x_res[0:x_size, :].reshape(-1, feature.shape[1], feature.shape[2])
    label_new = y_res[0:x_size].reshape(-1, feature.shape[1])
    return feature_new, label_new


def pre_train_death_model(hidden_size, learning_rate, l2_regularization):
    # hidden_size = 2 ** int(hidden_size)
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization
    print('hidden_size---{}---learning_rate---{}---l2_regularization---{}'
          .format(hidden_size, learning_rate, l2_regularization))

    train_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA.npy')[:, :, 1:]
    train_set_label = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA_label.npy')
    train_set_label = train_set_label.reshape([train_set_label.shape[0], train_set_label.shape[1], -1])
    train_set = np.concatenate((train_set[:, :, 5:], train_set_label), axis=2) # 将标签和特征存到一起

    test_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\test_PLA.npy')[:, :, 1:]
    test_set_label = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\test_PLA_label.npy')
    test_set_label = test_set_label.reshape([test_set_label.shape[0], test_set_label.shape[1], -1])
    test_set = np.concatenate((test_set[:, :, 5:], test_set_label), axis=2)

    train_set.astype(np.float32)
    test_set.astype(np.float32)
    train_set = DataSet(train_set)
    batch_size = 256
    epochs = 30
    predicted_visit = 7
    previous_visit = 3

    encode_net = Encode(hidden_size=128)
    encode_net(tf.zeros(shape=[batch_size, 4, 35]))
    encode_net.load_weights('encode_net_10_27.h5')

    death_model = DeathModel(hidden_size=hidden_size)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    batch_test = test_set.shape[0]
    logged = set()

    while train_set.epoch_completed < epochs:
        input_x_train = train_set.next_batch(batch_size=batch_size)
        input_x_features = input_x_train[:, :, :-1]
        batch = input_x_train.shape[0]
        input_x_labels = input_x_train[:, previous_visit:, -1].reshape(batch, -1, 1)
        states = tf.zeros(shape=[batch, 0, 128])
        with tf.GradientTape() as tape:
            for step in range(predicted_visit):
                features = input_x_features[:, :previous_visit+step, :]
                state = encode_net(features)
                states = tf.concat((states, tf.reshape(state, [batch, -1, 128])), axis=1)
            predicted_labels = death_model(states)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=input_x_labels, y_pred=predicted_labels))
            variables = [var for var in death_model.trainable_variables]
            for weight in death_model.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            gradient = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)
                death_model.load_weights('death_model_11_3.h5')

                test_labels = test_set[:, previous_visit:, -1].reshape(batch_test, -1, 1)
                test_states = tf.zeros(shape=[batch_test, 0, 128])
                for step in range(predicted_visit):
                    test_features = test_set[:, :previous_visit+step, :-1]
                    state = encode_net(test_features)
                    test_states = tf.concat((test_states, tf.reshape(state, [batch_test, -1, 128])), axis=1)

                test_predicted_labels = death_model(test_states)
                test_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=test_labels, y_pred=test_predicted_labels))
                auc = roc_auc_score(test_labels.reshape([-1, ]), test_predicted_labels.numpy().reshape([-1, ]))
                fpr, tpr, thread = roc_curve(test_labels.reshape([-1, ]), test_predicted_labels.numpy().reshape([-1,]))
                threshold = thread[np.argmax(tpr-fpr)]
                y_pre_label = (test_predicted_labels.numpy().reshape([-1,]) >= threshold) * 1
                # print(max(y_pre_label))
                acc = accuracy_score(test_labels.reshape([-1, ]), y_pre_label)
                print(threshold)  # 0.4072313

                # if test_loss < 0.70 and (auc > 0.70 and acc >= 0.765):
                #     death_model.save_weights('death_model_11_3.h5')

                print('epoch---{}---train_loss--{}----test_loss---{}--auc---{}---acc---{}-'.format(
                    train_set.epoch_completed, loss, test_loss, auc, acc))
    tf.compat.v1.reset_default_graph()
    return auc


if __name__ == '__main__':
    # test_test('death_model_11_3_train_test_set_保存.txt')
    test_test('打印门限.txt')
    # Encode_Decode_Time_BO = BayesianOptimization(
    #     pre_train_death_model, {
    #         'hidden_size': (5, 8),
    #         'learning_rate': (-5, 0),
    #         'l2_regularization': (-5, 0),
    #     }
    # )
    # Encode_Decode_Time_BO.maximize()
    # print(Encode_Decode_Time_BO.max)
    for i in range(50):
        auc = pre_train_death_model(hidden_size=32, learning_rate=0.008458542958160842, l2_regularization=1e-5)
