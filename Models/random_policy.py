from data import *
from Environment import *
from Reward import *
from utilis import *
from bayes_opt import BayesianOptimization
from basic_rl import Agent, Discriminator
from death_model import DeathModel
import os
import random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def train_batch_random_policy(hidden_size, learning_rate, l2_regularization):
    train_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA.npy')[:, :, 1:]  # 去掉当前天数这个变量---->40个变量
    test_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\test_PLA.npy')[:, :, 1:]
    test_set_label = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\test_PLA_label.npy')
    test_set.astype(np.float32)
    batch_size = 504
    # batch_size = 64
    feature_size = 35
    train_set = DataSet(test_set)
    epochs = 200
    actor_size = 65
    previous_visit = 3
    predicted_visit = 7

    hidden_size = 2 ** int(hidden_size)
    learning_rate = 10 ** learning_rate
    l2_regularization = 10 ** l2_regularization
    # imbalance_1 = 10 ** int(imbalance_1)
    # imbalance_2 = 10 ** int(imbalance_2)
    gamma = 0.99
    imbalance_1 = 10
    imbalance_2 = 20
    action_list = [0, 0.0357142873108387, 0.0625, 0.071428571, 0.125, 0.1875, 0.25, 0.2857142984867096, 0.3125, 0.321428571,
                   0.375, 0.4375, 0.446428571, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.875, 1,
                   1.125, 1.25, 1.375, 1.5, 1.55, 1.625, 1.75, 1.875, 2, 2.125,
                   2.25, 2.5, 2.75, 3, 3.125, 3.25, 3.375, 3.5, 3.75, 4,
                   4.25, 4.5, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 7,
                   7.5, 7.75, 8, 8.25, 8.5, 9, 9.625, 10, 11, 13.5,
                   14.5, 15, 21, 22.25, 25.625]
    print('hidden_size---{}---gamma---{}---imbalance_1---{}---imbalance_2----{}---learning_rate---{}'
          .format(hidden_size, gamma, imbalance_1, imbalance_2, learning_rate))

    construct_environment = ReconstructEnvir(hidden_size=64,
                                             feature_dims=35,
                                             previous_visit=3)
    construct_environment(tf.ones(shape=[3, 4, feature_size+1]))
    construct_environment.load_weights('environment_3_7_39_9_17.h5')

    reward_net = Reward(hidden_size=32, output_size=2)
    reward_net([tf.ones(shape=[3, feature_size]), tf.ones(shape=[3, 1])])
    reward_net.load_weights('reward_9_17.h5')

    death_model = DeathModel(hidden_size=32)
    death_model(tf.ones(shape=[3, 1, 35]))
    death_model.load_weights('death_model_0_10_10_12.h5')

    logged = set()
    while train_set.epoch_completed < epochs:
        input_x_train = train_set.next_batch(batch_size=batch_size)
        input_x_train = input_x_train.astype(np.float32)
        batch = input_x_train.shape[0]
        rewards = tf.zeros(shape=[batch, 0, 1])
        states = tf.zeros(shape=[batch, 0, feature_size])
        actions = tf.zeros(shape=[batch, 0, 1])
        actions_index = tf.zeros(shape=[batch, 0, 1])

        real_states = input_x_train[:, previous_visit:previous_visit+predicted_visit, 5:]
        real_actions = input_x_train[:, previous_visit:previous_visit+predicted_visit, 0].reshape(batch, -1, 1)
        real_rewards = tf.zeros(shape=[batch, 0, 1])
        estimated_death = tf.zeros(shape=[batch, 0])

        for step in range(predicted_visit-1):
            real_reward = (input_x_train[:, step+previous_visit+1, 1] - input_x_train[:, step+previous_visit+1, 2])*imbalance_1  # 净出量
            real_rewards = tf.concat((real_rewards, tf.reshape(real_reward, [batch, -1, 1])), axis=1)

            if step == 0:
                initial_state = input_x_train[:, step + previous_visit, 5:]
                state_to_now_feature = input_x_train[:, :step + previous_visit, 5:]
                state_to_now_action = input_x_train[:, :step+previous_visit, 0]
                state_to_now_ = tf.concat((state_to_now_feature, tf.reshape(state_to_now_action, [batch, -1, 1])), axis=2)  # state和action进行拼接
                action_index = np.array([random.randint(0, 64) for _ in range(batch)])
                action_index = tf.convert_to_tensor(action_index.astype(np.float32).reshape(batch, -1))

                actions_index = tf.concat((actions_index, tf.reshape(action_index, [batch, -1, 1])), axis=1)
                action_value = np.zeros_like(action_index)
                for i in range(tf.shape(action_index)[0]):
                    for j in range(tf.shape(action_index)[1]):
                        action_value[i, j] = action_list[action_index[i, j]]
                state_action = tf.concat((initial_state.reshape(batch, 1, -1), tf.reshape(action_value, [batch, -1, 1])), axis=2)  # 当前时刻的state action
                state_to_now = tf.concat((state_to_now_, state_action), axis=1)
                next_state = construct_environment(state_to_now)

                states = tf.concat((states, initial_state.reshape(batch, -1, feature_size)), axis=1)
                actions = tf.concat((actions, tf.reshape(action_value, [batch, -1, 1])), axis=1)

                reward = reward_net([initial_state, action_value])[:, 0] * imbalance_1  # 新状态的出入量差值
                rewards = tf.concat((rewards, tf.reshape(reward, [batch, -1, 1])), axis=1)

                death_model_input = tf.concat(
                    (state_to_now[:, :, :-1], tf.reshape(next_state, [batch, -1, feature_size])), axis=1)
                death_test = death_model(death_model_input)
                estimated_death = tf.concat((estimated_death, death_test), axis=1)

            else:
                initial_state = initial_state
                state_to_now_feature = input_x_train[:, :previous_visit, 5:]
                state_to_now_action = input_x_train[:, :previous_visit, 0]
                state_to_now_ = tf.concat((state_to_now_feature, tf.reshape(state_to_now_action, [batch, -1, 1])), axis=2)  # 最开始的初始状态
                state = next_state
                action_index = np.array([random.randint(0, 64) for _ in range(batch)])
                action_index = tf.convert_to_tensor(action_index.astype(np.float32).reshape(batch, -1))
                actions_index = tf.concat((actions_index, tf.reshape(action_index, [batch, -1, 1])), axis=1)
                action_value = np.zeros_like(action_index)
                for i in range(tf.shape(action_index)[0]):
                    for j in range(tf.shape(action_index)[1]):
                        action_value[i, j] = action_list[action_index[i, j]]
                state_action = tf.concat((tf.reshape(state, [batch, -1, feature_size]), tf.reshape(action_value, [batch, -1, 1])), axis=2) # 当前时刻的state action
                state_to_now__ = tf.concat((states, actions), axis=2)  # 已经产生过的state action
                state_to_now = tf.concat((state_to_now_, state_to_now__, state_action), axis=1)
                next_state = construct_environment(state_to_now)

                states = tf.concat((states, tf.reshape(state, [batch, -1, feature_size])), axis=1)
                actions = tf.concat((actions, tf.reshape(action_value, [batch, -1, 1])), axis=1)
                reward = reward_net([state, action_value])[:, 0] * imbalance_1
                rewards = tf.concat((rewards, tf.reshape(reward, [batch, -1, 1])), axis=1)

                death_model_input = tf.concat(
                    (state_to_now[:, :, :-1], tf.reshape(next_state, [batch, -1, feature_size])), axis=1)
                death_test = death_model(death_model_input)
                estimated_death = tf.concat((estimated_death, death_test), axis=1)

        pro_death = np.zeros_like(estimated_death)
        for patient in range(batch):
            for visit in range(predicted_visit-1):
                if estimated_death[patient, visit] >= 0.048136085:
                    pro_death[patient, visit] = 1

        discont_rewards = discont_reward(states=states, rewards=rewards)
        discont_rewards_real = discont_reward(states=real_states, rewards=real_rewards)
        if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
            logged.add(train_set.epoch_completed)
            print('epoch   {}   random_train_reward   {}   train_reward_real   {}   death_sum  {}'
                  .format(train_set.epoch_completed,
                          np.mean(discont_rewards),
                          np.mean(discont_rewards_real),
                          np.sum(pro_death)))
        np.save('death_random_policy.npy', pro_death)
        np.save('death_random_discont_reward.npy', discont_rewards)
        np.save('doctor_discont_reward.npy', discont_rewards_real)
    return np.mean(discont_rewards)


if __name__ == '__main__':
    test_test('10_15_随机action_return_保存return.txt')
    Agent_BO = BayesianOptimization(
        train_batch_random_policy, {
            'hidden_size': (5, 8),
            'learning_rate': (-5, -1),
            'l2_regularization': (-6, -1),
        }
    )
    Agent_BO.maximize()
    print(Agent_BO.max)