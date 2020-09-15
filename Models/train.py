from scipy import stats
from data import *
from Environment import *
from utilis import *
from bayes_opt import BayesianOptimization
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def train_environment(hidden_size, learning_rate, l2_regularization):
    train_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA.npy')
    # test_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\test_PLA.npy')
    test_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\validate_PLA.npy')

    previous_visit = 3
    predicted_visit = 7
    feature_dims = train_set.shape[2]-2

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
            for predicted_visit_ in range(predicted_visit):
                input_feature = input_x_train[:, :previous_visit+predicted_visit_, 1:]  # 将state和action共同输入至模型中(40个)
                feature_next = construct_environment(input_feature)  # 仅仅预测下一个时间点的state(39个)
                predicted_trajectory = tf.concat((predicted_trajectory, tf.reshape(feature_next, [-1, 1, feature_dims])), axis=1)

            mse_loss = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit+predicted_visit, 2:], predicted_trajectory))
            variables = [var for var in construct_environment.trainable_variables]
            for weight in construct_environment.trainable_variables:
                mse_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            gradient = tape.gradient(mse_loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)

                predicted_trajectory_test = np.zeros(shape=[batch_test, 0, feature_dims])
                for predicted_visit_ in range(predicted_visit):
                    input_feature_test = test_set[:, :previous_visit, 1:]  # 缺少action
                    # 将action定为真实的action
                    if predicted_trajectory_test.shape[1] != 0:
                        action = tf.cast(test_set[:, previous_visit:previous_visit+predicted_visit_, 1], tf.float32)
                        predicted_trajectory_test_ = tf.concat((tf.reshape(action, [-1, predicted_visit_, 1]), predicted_trajectory_test), axis=2)
                        input_feature_test = tf.concat((input_feature_test, predicted_trajectory_test_), axis=1)
                    else:
                        input_feature_test = input_feature_test

                    feature_next_test = construct_environment(input_feature_test)
                    predicted_trajectory_test = tf.concat((predicted_trajectory_test, tf.reshape(feature_next_test, [-1, 1, feature_dims])), axis=1)

                mse_loss_test = tf.reduce_mean(tf.keras.losses.mse(test_set[:, previous_visit:previous_visit+predicted_visit,2:], predicted_trajectory_test))
                mae_loss_test = tf.reduce_mean(tf.keras.losses.mae(test_set[:, previous_visit:previous_visit+predicted_visit,2:], predicted_trajectory_test))

                r_value_all = []
                for visit in range(predicted_trajectory_test.shape[0]):
                    for visit_day in range(predicted_trajectory_test.shape[1]):
                        real_data = test_set[visit, visit_day, 2:]
                        predicted_data = predicted_trajectory_test[visit, visit_day, :]
                        r_value_ = stats.pearsonr(real_data, predicted_data)
                        r_value_all.append(r_value_[0])

                print('epoch---{}----train_mse---{}---test_mse---{}---test_mae---{}--test_r--{}'
                      .format(train_set.epoch_completed,
                              mse_loss, mse_loss_test,
                              mae_loss_test,
                              np.mean(r_value_all)))
                # if mse_loss_test < 0.0058:
                #     construct_environment.save_weights('environment_3_7_39.h5')

    tf.compat.v1.reset_default_graph()
    # return mse_loss_test, mae_loss_test, np.mean(r_value_all)
    return -1*mse_loss_test


if __name__ == '__main__':
    test_test('environment_s2s_simulate_train_.txt')
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
    #     mse, mae, r = train_environment(hidden_size=32,
    #                                     learning_rate=0.0016882124627026714,
    #                                     l2_regularization=1.0009925374329186e-05)
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






