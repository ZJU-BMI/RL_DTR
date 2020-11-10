from tensorflow_core.python.keras.models import Model
from origin.data import *
from utilis import *

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Reward(Model):
    def __init__(self, hidden_size, output_size):
        super().__init__(name='reward_net')
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dense1 = tf.keras.layers.Dense(hidden_size)
        self.dense2 = tf.keras.layers.Dense(hidden_size)
        self.dense3 = tf.keras.layers.Dense(output_size)

    def call(self, input_data):
        state, action = input_data
        batch = tf.shape(state)[0]
        inputs_data = tf.concat((state, tf.reshape(action, [batch, -1])), axis=1)
        hidden_1 = self.dense1(inputs_data)
        hidden_2 = self.dense2(hidden_1)
        return self.dense3(hidden_2)


def pre_train_reward_net(hidden_size, learning_rate, l2_regularization):
    train_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA.npy')[:, :, 1:]
    test_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\validate_PLA.npy')[:, :, 1:]
    train_set = DataSet(train_set)

    output_size = 2

    # hidden_size = 2 ** int(hidden_size)
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization
    epochs = 50
    batch_size = 64
    previous_visit = 3
    predicted_visit = 7
    print('hidden_size-----{}---learning_rate---{}---l2_regularization---{}'.format(hidden_size, learning_rate, l2_regularization))
    reward_net = Reward(hidden_size=hidden_size, output_size=output_size)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    logged = set()
    loss = 0
    count = 0
    while train_set.epoch_completed < epochs:
        loss_pre = loss

        input_x_train = train_set.next_batch(batch_size=batch_size)
        batch = input_x_train.shape[0]
        rewards_pre = tf.zeros(shape=[batch, 0, output_size])
        rewards_only_output = input_x_train[:, previous_visit+1:previous_visit+predicted_visit, 1] - input_x_train[:, previous_visit+1:previous_visit+predicted_visit, 2]
        rewards_bnp = input_x_train[:, previous_visit+1:previous_visit+predicted_visit, 4]
        rewards_label = tf.concat((tf.reshape(rewards_only_output, [batch, -1, 1]), tf.reshape(rewards_bnp, [batch, -1, 1])), axis=2)
        with tf.GradientTape() as tape:
            for step in range(predicted_visit-1):
                state_current = input_x_train[:, previous_visit+step, 5:]  # 35个变量当做患者的state
                action_current = input_x_train[:, previous_visit+step, 0]  # 药物剂量

                reward = reward_net([state_current, action_current])
                rewards_pre = tf.concat((rewards_pre, tf.reshape(reward, [batch, -1, output_size])), axis=1)

            loss = tf.reduce_mean(tf.keras.losses.mse(rewards_label, rewards_pre))

            for weight in reward_net.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            gradient = tape.gradient(loss, reward_net.trainable_variables)
            optimizer.apply_gradients(zip(gradient, reward_net.trainable_variables))

            loss_diff = loss_pre - loss

            if loss > 0.01:
                count = 0
            else:
                if loss_diff > 0.0001:
                    count = 0
                else:
                    count += 1
            if count > 9:
                break

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)

                batch_test = test_set.shape[0]
                pred_rewards_test = tf.zeros(shape=[batch_test, 0, output_size])

                rewards_test_only_output = test_set[:, previous_visit+1:previous_visit+predicted_visit, 1] - test_set[:, previous_visit+1:previous_visit+predicted_visit, 2]  # 净出量
                rewards_test_bnp = test_set[:, previous_visit+1:previous_visit+predicted_visit, 4] # BNP
                rewards_label_test = tf.concat((tf.reshape(rewards_test_only_output, [batch_test, -1, 1]), tf.reshape(rewards_test_bnp, [batch_test, -1, 1])), axis=2)

                for step in range(predicted_visit-1):
                    state_current_test = test_set[:, previous_visit+step, 5:]
                    action_current_test = test_set[:, previous_visit+step, 0]

                    reward_test = reward_net([state_current_test, action_current_test])
                    pred_rewards_test = tf.concat((pred_rewards_test, tf.reshape(reward_test, [batch_test, -1, output_size])), axis=1)

                test_loss = tf.reduce_mean(tf.keras.losses.mse(rewards_label_test, pred_rewards_test))
                print('epoch---{}---train_loss--{}---test_loss---{}---count---{}'.format(train_set.epoch_completed,
                                                                                         loss, test_loss,
                                                                                         count))

                if test_loss < 0.01200:
                    reward_net.save_weights('reward_9_17.h5')
    tf.compat.v1.reset_default_graph()
    # return -test_loss


if __name__ == '__main__':
    test_test('Reward_net预训练_9_17.txt')
    # Reward_net_BO = BayesianOptimization(
    #     pre_train_reward_net, {
    #         'hidden_size': (5, 8),
    #         'learning_rate': (-5, -1),
    #         'l2_regularization': (-5, -1),
    #     }
    # )
    # Reward_net_BO.maximize()
    # print(Reward_net_BO.max)

    for i in range(50):
        pre_train_reward_net(hidden_size=32, learning_rate=0.0002361746704730307, l2_regularization=0.0002509883922487462)






