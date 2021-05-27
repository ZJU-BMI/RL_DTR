from reward_1 import *
from data import *
from utilis import *
from tensorflow_core.python.keras.models import Model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class PolicyGradientNN(Model):
    def __init__(self, hidden_size, actor_size):
        super().__init__(name='policy_gradient')
        self.actor_size = actor_size
        self.hidden_size = hidden_size
        self.dense1 = tf.keras.layers.Dense(self.hidden_size)
        self.dense2 = tf.keras.layers.Dense(self.hidden_size)
        self.dense3 = tf.keras.layers.Dense(self.actor_size)

    def call(self, state):
        """
        :param state: s_(t-1)
        :return: p(a_t|s_(t-1))
        """
        hidden_1 = self.dense1(state)
        hidden_2 = self.dense2(hidden_1)
        hidden_3 = self.dense3(hidden_2)
        actor_prob = tf.nn.softmax(hidden_3)
        return actor_prob


class Agent(object):
    def __init__(self, actor_size, hidden_size, gamma, learning_rate, lambda_imbalance):
        self.name = 'agent'
        self.actor_size = actor_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.lambda_imbalance = lambda_imbalance
        self.learning_rate = learning_rate
        self.model = PolicyGradientNN(hidden_size=hidden_size, actor_size=actor_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def act(self, state):
        prob = self.model(state)
        dist = tf.compat.v1.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        action_index = tf.reshape(action, [-1, 1])
        return action_index

    def loss(self, discount_rewards, action_index_all, states, discriminator_probs):
        loss = tf.zeros(shape=[tf.shape(discount_rewards)[0], ])
        for step in range(tf.shape(states)[1]):
            action_index = action_index_all[:, step, :]
            state = states[:, step, :]
            action_probs = self.model(state)
            neg_log_likelihood = tf.keras.losses.sparse_categorical_crossentropy(
                tf.reshape(action_index, [-1, ]),
                tf.reshape(action_probs, [-1, self.actor_size]))
            # p = 1.0 - tf.math.abs(discriminator_probs[:, step]-0.5)
            p = discriminator_probs[:, step]
            loss += neg_log_likelihood * tf.reshape(discount_rewards[:, step, :], [-1, ]) * p
        return loss

    def discount_reward(self, rewards):
        sum_reward = 0.0
        batch = tf.shape(rewards)[0]
        discount_reward = tf.zeros(shape=[batch, 0, 1])
        rewards = tf.reverse(rewards, axis=[1])
        for r_index in range(tf.shape(rewards)[1]):
            r = rewards[:, r_index, :]
            sum_reward = (1 + r*self.lambda_imbalance) + self.gamma * sum_reward
            # sum_reward = r + self.gamma * sum_reward
            discount_reward = tf.concat((discount_reward, tf.reshape(sum_reward, [batch, -1, 1])), axis=1)
        discount_reward = tf.reverse(discount_reward, axis=[1])
        return discount_reward

    def loss_policy_gradient(self, discount_rewards, action_index_all, states):
        loss = tf.zeros(shape=[tf.shape(discount_rewards)[0], ])
        for step in range(tf.shape(states)[1]):
            action_index = action_index_all[:, step, :]
            state = states[:, step, :]
            action_probs = self.model(state)
            neg_log_likelihood = tf.keras.losses.sparse_categorical_crossentropy(
                tf.reshape(action_index, [-1, ]),
                tf.reshape(action_probs, [-1, self.actor_size]))
            loss += neg_log_likelihood * tf.reshape(discount_rewards[:, step, :], [-1, ])
        return loss


def pre_train_policy(hidden_size, learning_rate, l2_regularization):
    hidden_size = 2 ** int(hidden_size)
    learning_rate = 10 ** learning_rate
    l2_regularization = 10 ** l2_regularization
    print('hidden_size---{}---learning_rate---{}---l2_regularization---{}'
          .format(hidden_size, learning_rate, l2_regularization))

    actor_size = 25
    # train_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA.npy')[:, :, 1:]
    # test_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\validate_PLA.npy')[:, :, 1:]
    train_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\mimic_train.npy')
    test_set = np.load('..\\..\\..\\RL_DTR\\Resource\\preprocess\\mimic_validate.npy')

    train_set.astype(np.float32)
    test_set.astype(np.float32)

    train_set = DataSet(train_set)
    epochs = 30
    batch_size = 64
    logged = set()
    previous_visit = 3
    predicted_visit = 10

    policy_net = PolicyGradientNN(hidden_size=hidden_size, actor_size=actor_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    encode_net = Encode(hidden_size=256)
    encode_net([tf.zeros(shape=[batch_size, 4, 45]), tf.zeros(shape=[batch_size, 4, 2])])
    iv_list = [0, 33, 102, 330, 1153]
    vas_list = [0, 0.045, 0.15, 0.32, 1.24]

    while train_set.epoch_completed < epochs:
        with tf.GradientTape() as tape:
            encode_net.load_weights('encode_net_12_7_mimic.h5')
            input_x_train = train_set.next_batch(batch_size=batch_size)
            input_x_train.astype(np.float32)
            action_labels = input_x_train[:, previous_visit:previous_visit+predicted_visit, 0]  # 直接获取所有的label

            batch = input_x_train.shape[0]
            action_probs = tf.zeros(shape=[batch, 0, actor_size])
            # 将原始x转换为hidden representation
            for step in range(predicted_visit):
                features = input_x_train[:, :step+previous_visit, 4:]
                actions_ = tf.zeros(shape=[batch, 1, 1], dtype=tf.float64)  # 将最开始的action当作0

                actions = tf.reshape(input_x_train[:, :step+previous_visit-1, 0], [batch, -1, 1])
                actions = tf.concat((actions_, actions), axis=1)
                actions_index = actions.numpy()  # 获得之前数据的action_label_index,将其转换为真实的剂量，得到state
                actions_values = np.zeros(shape=[batch, step+previous_visit, 2])
                for patient in range(batch):
                    for visit in range(step+previous_visit):
                        action_index = actions_index[patient, visit, 0]
                        iv_index = int(action_index /5)
                        vas_index = int(action_index % 5)
                        actions_values[patient, visit, 0] = iv_list[iv_index]
                        actions_values[patient, visit, 1] = vas_list[vas_index]

                state = encode_net([features, actions_values])
                action_prob = policy_net(state)
                action_probs = tf.concat((action_probs, tf.reshape(action_prob, [batch, -1, actor_size])), axis=1)

            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.reshape(action_labels, [-1,]), tf.reshape(action_probs, [-1, actor_size])))

            gradient = tape.gradient(loss, policy_net.trainable_variables)
            optimizer.apply_gradients(zip(gradient, policy_net.trainable_variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)

                batch_test = test_set.shape[0]
                predicted_probs = tf.zeros(shape=[batch_test, 0, actor_size])
                predicted_actions_labels = test_set[:, previous_visit:previous_visit+predicted_visit, 0] # 测试集上的actinn_label_index

                for step in range(predicted_visit):
                    features_test = test_set[:, :previous_visit+step, 4:]
                    actions_test_ = tf.zeros(shape=[batch_test, 1, 1], dtype=tf.float64)
                    actions_test = tf.reshape(test_set[:, :previous_visit+step-1, 0], [batch_test, -1, 1])
                    actions_test = tf.concat((actions_test_, actions_test), axis=1)
                    actions_test_index = actions_test.numpy()

                    actions_test_values = np.zeros(shape=[batch_test, previous_visit+step, 2]) # 将index转换成values 输入到encode中
                    for patient in range(batch_test):
                        for visit in range(previous_visit+step):
                            action_index = actions_test_index[patient, visit, 0]
                            iv_index = int(action_index / 5)
                            vas_index = int(action_index % 5)
                            actions_test_values[patient, visit, 0] = iv_list[iv_index]
                            actions_test_values[patient, visit, 1] = vas_list[vas_index]

                    state_test = encode_net([features_test, actions_test_values])
                    action_prob_test = policy_net(state_test)
                    predicted_probs = tf.concat((predicted_probs, tf.reshape(action_prob_test, [batch_test, -1, actor_size])), axis=1)

                predicted_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.reshape(predicted_actions_labels, [-1,]), tf.reshape(predicted_probs, [-1, actor_size])))
                print('epoch---{}----train_loss---{}---test_loss---{}'
                      .format(train_set.epoch_completed, loss, predicted_loss))

                # if predicted_loss < 1.69:
                #     policy_net.save_weights('policy_net_11_30.h5')

    tf.compat.v1.reset_default_graph()
    return -1 * predicted_loss


if __name__ == '__main__':
    test_test('12_19_agent_训练——mimic.txt')
    agent_bo = BayesianOptimization(
        pre_train_policy, {
            'hidden_size': (5, 8),
            'learning_rate': (-6, 0),
            'l2_regularization': (-6, 0),
        }
    )
    agent_bo.maximize()
    print(agent_bo.max)

    # for i in range(50):
    #     loss = pre_train_policy(hidden_size=128, learning_rate=9.475285020707197e-05, l2_regularization=5.478971706578164e-05)




