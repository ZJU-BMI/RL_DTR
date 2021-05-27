import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.models import Model
from utilis import *
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score
from imblearn.over_sampling import SMOTE
from data import DataSet
from environment_1 import Encode

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# death model for mimic-iii data-set
class DeathModel(Model):
    def __init__(self, hidden_size):
        super().__init__(name='death_model')
        self.dense1 = tf.keras.layers.Dense(hidden_size)
        self.dense2 = tf.keras.layers.Dense(hidden_size)
        self.dense3 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, states):
        # hidden = self.dense1(states)
        # hidden = self.dense2(hidden)
        # death_probs = self.dense3(hidden)
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
    label_new = y_res[0:x_size].reshape(-1, feature.shape[1], 1)
    return feature_new, label_new


def pre_train_death_model(hidden_size, learning_rate, l2_regularization):
    # hidden_size = 2 ** int(hidden_size)
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization
    print('hidden_size---{}---learning_rate---{}---l2_regularization---{}'
          .format(hidden_size, learning_rate, l2_regularization))

    train_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\mimic_train.npy')
    test_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\mimic_validate.npy')

    train_set.astype(np.float32)
    test_set.astype(np.float32)
    train_set = DataSet(train_set)
    batch_size = 74
    epochs = 30
    previous_visit = 3
    predicted_visit = 10

    encode_net = Encode(hidden_size=256)
    encode_net([tf.zeros(shape=[batch_size, 4, 45]), tf.zeros(shape=[batch_size, 4, 2])])
    encode_net.load_weights('encode_net_12_7_mimic.h5')

    death_model = DeathModel(hidden_size=hidden_size)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    batch_test = test_set.shape[0]
    logged = set()
    iv_list = [0, 33, 102, 330, 1153]
    vas_list = [0, 0.045, 0.15, 0.32, 1.24]

    while train_set.epoch_completed < epochs:
        input_x_train = train_set.next_batch(batch_size=batch_size)
        input_x_features = input_x_train
        batch = input_x_train.shape[0]
        input_x_labels = input_x_train[:, previous_visit:, 3].reshape(batch, -1, 1)
        states = tf.zeros(shape=[batch, 0, 256])
        with tf.GradientTape() as tape:
            for step in range(predicted_visit):
                features = input_x_features[:, :previous_visit+step, 4:]
                actions_ = tf.zeros(shape=[batch, 1, 1], dtype=tf.float64)
                actions = tf.reshape(input_x_train[:, :previous_visit+step-1, 0], [batch, -1, 1])
                actions = tf.concat((actions_, actions), axis=1)
                actions_index = actions.numpy()
                actions_values = np.zeros(shape=[batch, step+previous_visit, 2])
                for patient in range(batch):
                    for visit in range(step+previous_visit):
                        action_index = actions_index[patient, visit, 0]
                        iv_index = int(action_index / 5)
                        vas_index = int(action_index % 5)
                        actions_values[patient, visit, 0] = iv_list[iv_index]
                        actions_values[patient, visit, 1] = vas_list[vas_index]
                state = encode_net([features, actions_values])
                states = tf.concat((states, tf.reshape(state, [batch, -1, 256])), axis=1)
            states_res, label_res = imbalance_preprocess(states.numpy(), input_x_labels)
            predicted_labels = death_model(states_res)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=label_res, y_pred=predicted_labels))
            variables = [var for var in death_model.trainable_variables]
            for weight in death_model.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            gradient = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)
                test_labels = test_set[:, previous_visit:, 3].reshape(batch_test, -1, 1)  # 获取death 标签
                test_states = tf.zeros(shape=[batch_test, 0, 256])
                for step in range(predicted_visit):
                    test_features = test_set[:, :previous_visit+step, 4:]
                    test_actions_ = tf.zeros(shape=[batch_test, 1, 1], dtype=tf.float64)
                    test_actions = tf.reshape(test_set[:, :previous_visit+step-1, 0], [batch_test, -1, 1])
                    test_actions = tf.concat((test_actions_, test_actions), axis=1)
                    test_actions_index = test_actions.numpy()
                    test_actions_values = np.zeros(shape=[batch_test, previous_visit+step, 2])
                    for patient in range(batch_test):
                        for visit in range(previous_visit+step):
                            action_index = test_actions_index[patient, visit, 0]
                            iv_index = int(action_index / 5)
                            vas_index = int(action_index % 5)
                            test_actions_values[patient, visit, 0] = iv_list[iv_index]
                            test_actions_values[patient, visit, 1] = vas_list[vas_index]

                    state = encode_net([test_features, test_actions_values])
                    test_states = tf.concat((test_states, tf.reshape(state, [batch_test, -1, 256])), axis=1)

                test_predicted_labels = death_model(test_states)
                test_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=test_labels, y_pred=test_predicted_labels))
                auc = roc_auc_score(test_labels.reshape([-1, ]), test_predicted_labels.numpy().reshape([-1, ]))
                fpr, tpr, thread = roc_curve(test_labels.reshape([-1, ]), test_predicted_labels.numpy().reshape([-1,]))
                threshold = thread[np.argmax(tpr-fpr)]
                y_pre_label = (test_predicted_labels.numpy().reshape([-1, ]) >= threshold) * 1
                # print(max(y_pre_label))
                acc = accuracy_score(test_labels.reshape([-1, ]), y_pre_label)
                f1 = f1_score(test_labels.reshape([-1, ]), y_pre_label)
                print(threshold)  # 0.49651924

                print('epoch---{}---train_loss--{}----test_loss---{}--auc---{}---acc---{}- f1---{}'.format(
                    train_set.epoch_completed, loss, test_loss, auc, acc, f1))

                # if test_loss < 0.67 and (auc > 0.775 and acc >= 0.729):
                #     death_model.save_weights('death_model_11_30.h5')
                #     print('模型保存成功！')
    tf.compat.v1.reset_default_graph()
    return auc


if __name__ == '__main__':
    test_test('12_24_death_model_train_mimic.txt')
    Encode_Decode_Time_BO = BayesianOptimization(
        pre_train_death_model, {
            'hidden_size': (5, 8),
            'learning_rate': (-5, 0),
            'l2_regularization': (-5, 0),
        }
    )
    Encode_Decode_Time_BO.maximize()
    print(Encode_Decode_Time_BO.max)
    # for i in range(50):
    #     auc = pre_train_death_model(hidden_size=256, learning_rate=1e-5, l2_regularization=0.000403887329375655)
