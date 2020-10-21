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


# 训练agent 单个病人的单次记录训练
def train():
    actor_size = 65
    hidden_size = 128
    previous_visit = 3
    gamma = 0.01
    predicted_visit = 7
    imbalance_1 = 1
    imbalance_2 = 10
    learning_rate = 0.01

    data_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA.npy')[:, :, 1:]  # 去掉当前天数
    construct_environment = ReconstructEnvir(hidden_size=32,
                                             feature_dims=39,
                                             previous_visit=3)
    construct_environment(tf.ones(shape=[3, 4, 40]))
    construct_environment.load_weights('environment_3_7_39_9_15.h5')
    batch = data_set.shape[0]

    rewards_all_episodes = []
    for patient in range(10000):
        agent = Agent(actor_size=actor_size, hidden_size=hidden_size, gamma=gamma, learning_rate=learning_rate)
        rewards = []
        states = []
        actions = []
        total_reward = 0
        for step in range(predicted_visit):
            if step == 0:
                state = data_set[0, step+previous_visit-1, 1:]  # s_3 39个变量
                action = agent.act(tf.reshape(state, [1, -1]))
                state_to_now = data_set[0, :step+previous_visit-1, :]  # (s_1, a_1)..(s_2, a_2)
                state_action = np.concatenate((np.array(action).reshape(1, -1), state.reshape(1, -1)), axis=1)
                state_to_now = np.concatenate((state_to_now, state_action)).reshape(1, -1, state_to_now.shape[1])
                next_state = construct_environment(state_to_now)
                reward = (next_state[0, 0] - state[1]) * imbalance_1
                rewards.append(reward)
                states.append(state)
                actions.append(action)
                total_reward += reward

            else:
                state = next_state.numpy()
                action = agent.act(state)
                state_action = np.concatenate((np.array(action).reshape(1, -1), state.reshape(1, -1)), axis=1)
                state_to_now_ = data_set[0, :previous_visit+step-1, :]  # (s_1, a_1),(s_2, a_2)
                states_array = np.zeros(shape=[0, 39])
                actions_array = np.zeros(shape=[0, 1])
                for i in range(len(states)):
                    states_array = np.concatenate((states_array, np.array(states[i]).reshape(1, -1)), axis=0)
                    actions_array = np.concatenate((actions_array, np.array(actions[i]).reshape(1, -1)), axis=0)
                state_actions = np.concatenate((actions_array, states_array), axis=1)
                state_to_now = np.concatenate((state_to_now_, state_actions, state_action), axis=0)
                next_state = construct_environment(state_to_now.reshape(1, -1, state_to_now.shape[1]))
                reward = (next_state[0, 0] - state[0, 1]) * imbalance_1
                rewards.append(reward)
                states.append(state)
                actions.append(action)
                total_reward += reward

                if step == predicted_visit-1:
                    total_reward += (next_state[0, 3] - state[0, 3]) * imbalance_2

        agent.train(states, rewards, actions)
        print('total_reward after  {} step is {}'.format(patient, total_reward))
        rewards_all_episodes.append(total_reward)
    plot(rewards_all_episodes)
    # tf.compat.v1.reset_default_graph()


def train_batch(hidden_size, learning_rate, l2_regularization):
    train_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\train_PLA.npy')[:, :, 1:]  # 去掉当前天数这个变量---->40个变量
    test_set = np.load('..\\..\\RL_DTR\\Resource\\preprocess\\test_PLA.npy')[:, :, 1:]
    test_set.astype(np.float32)
    batch_size = 2018
    # batch_size = 64
    feature_size = 35
    train_set = DataSet(train_set)
    epochs = 200
    actor_size = 65
    previous_visit = 3
    predicted_visit = 7

    # hidden_size = 2 ** int(hidden_size)
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization
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
    print('hidden_size---{}---gamma---{}---imbalance_1---{}---imbalance_2----{}-'
          '--learning_rate---{}  l2_regularization---{}'
          .format(hidden_size, gamma, imbalance_1, imbalance_2, learning_rate, l2_regularization))

    construct_environment = ReconstructEnvir(hidden_size=64,
                                             feature_dims=35,
                                             previous_visit=3)
    construct_environment(tf.ones(shape=[3, 4, feature_size+1]))
    construct_environment.load_weights('environment_3_7_39_9_17.h5')

    reward_net = Reward(hidden_size=32, output_size=2)
    reward_net([tf.ones(shape=[3, feature_size]), tf.ones(shape=[3, 1])])
    reward_net.load_weights('reward_9_17.h5')

    agent = Agent(actor_size=actor_size, hidden_size=32, gamma=gamma, learning_rate=learning_rate)
    agent.act(tf.ones(shape=[3, feature_size]))
    agent.model.load_weights('policy_net_9_18.h5')

    death_model = DeathModel(hidden_size=32)
    death_model(tf.ones(shape=[3, 1, 35]))
    death_model.load_weights('death_model_0_10_10_12.h5')

    logged = set()
    while train_set.epoch_completed < epochs:
        input_x_train = train_set.next_batch(batch_size=batch_size)
        input_x_train = input_x_train.astype(np.float32)
        batch = input_x_train.shape[0]
        rewards = tf.zeros(shape=[batch, 0, 1])
        states = tf.zeros(shape=[batch, 0, feature_size]) # 保存已经生成的state(从previous visit时刻开始)
        actions = tf.zeros(shape=[batch, 0, 1])  # 保存已经选择的action(从previous visit开始)
        actions_index = tf.zeros(shape=[batch, 0, 1])  # 保存已经获得的reward(从previous visit开始)
        # 从previous_visit到最后一步的state
        real_states = input_x_train[:, previous_visit:previous_visit+predicted_visit-1, 5:]
        # 从previous_visit到最后一步的action
        real_actions = input_x_train[:, previous_visit:previous_visit+predicted_visit-1, 0].reshape(batch, -1, 1)
        real_rewards = tf.zeros(shape=[batch, 0, 1])

        for step in range(predicted_visit-1):
            real_reward = (input_x_train[:, step+previous_visit+1, 1] -
                           input_x_train[:, step+previous_visit+1, 2])*imbalance_1  # 净出量
            if step == 0:
                real_rewards = tf.concat((real_rewards, tf.reshape(real_reward, [batch, -1, 1])), axis=1)
                initial_state = input_x_train[:, step + previous_visit, 5:]
                state_to_now_feature = input_x_train[:, :step + previous_visit, 5:]
                state_to_now_action = input_x_train[:, :step+previous_visit, 0]
                state_to_now_ = tf.concat((state_to_now_feature, tf.reshape(state_to_now_action, [batch, -1, 1])), axis=2)  # 到现在时刻为止的所有state和action进行拼接
                action_index = tf.reshape(agent.act(initial_state), [batch, -1])  # 选择一个action
                actions_index = tf.concat((actions_index, tf.reshape(action_index, [batch, -1, 1])), axis=1)
                action_value = np.zeros_like(action_index)
                for i in range(tf.shape(action_index)[0]):
                    for j in range(tf.shape(action_index)[1]):
                        action_value[i, j] = action_list[action_index[i, j]]
                state_action = tf.concat((initial_state.reshape(batch, 1, -1), tf.reshape(action_value, [batch, -1, 1])), axis=2)  # 当前时刻选择的state action
                state_to_now = tf.concat((state_to_now_, state_action), axis=1)
                next_state = construct_environment(state_to_now)

                states = tf.concat((states, initial_state.reshape(batch, -1, feature_size)), axis=1)
                actions = tf.concat((actions, tf.reshape(action_value, [batch, -1, 1])), axis=1)

                reward = reward_net([initial_state, action_value])[:, 0] * imbalance_1  # 新状态的净出量
                rewards = tf.concat((rewards, tf.reshape(reward, [batch, -1, 1])), axis=1)

            else:
                initial_state = initial_state
                state_to_now_feature = input_x_train[:, :previous_visit, 5:]
                state_to_now_action = input_x_train[:, :previous_visit, 0]
                state_to_now_ = tf.concat((state_to_now_feature, tf.reshape(state_to_now_action, [batch, -1, 1])), axis=2)  # 最开始到previous visit的state action
                state = next_state
                action_index = agent.act(state)
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
                # if step == predicted_visit-2:
                #     reward += (initial_state[:, 3] - next_state[:, 3]) * imbalance_2
                #     real_reward += (initital_state_real[:, 3]-next_state_real[:, 3]) * imbalance_2  # 仅仅考虑出入量，未考虑BNP
                rewards = tf.concat((rewards, tf.reshape(reward, [batch, -1, 1])), axis=1)
            real_rewards = tf.concat((real_rewards, tf.reshape(real_reward, [batch, -1, 1])), axis=1)
        loss = agent.train(states=states, actions=actions_index, rewards=rewards)
        discont_rewards = agent.discont_reward(states=states, rewards=rewards)
        discont_rewards_real = agent.discont_reward(states=real_states, rewards=real_rewards)

        if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
            agent.model.load_weights('policy_gradient_net.h5')
            logged.add(train_set.epoch_completed)
            batch_test = test_set.shape[0]
            rewards_test = tf.zeros(shape=[batch_test, 0, 1])
            states_test = tf.zeros(shape=[batch_test, 0, feature_size])
            actions_test = tf.zeros(shape=[batch_test, 0, 1])
            estimated_death_test = tf.zeros(shape=[batch_test, 0])

            states_test_real = test_set[:, previous_visit:previous_visit+predicted_visit-1, 5:]
            rewards_test_real = tf.reshape((test_set[:, previous_visit+1:previous_visit+predicted_visit, 1] - test_set[:, previous_visit+1:previous_visit+predicted_visit, 2]) * imbalance_1, [batch_test, -1, 1])
            rewards_test_real = tf.cast(rewards_test_real, tf.float32)

            for step in range(predicted_visit):
                initial_state_test_feature = test_set[:, step+previous_visit, 5:]

                state_to_now_test_feature = test_set[:, :previous_visit, 5:]
                state_to_now_test_action = test_set[:, :previous_visit, 0]
                state_to_now_test_ = tf.concat((state_to_now_test_feature, tf.reshape(state_to_now_test_action, [batch_test, -1, 1])), axis=2)  # 之前时刻的state action
                state_to_now_test_ = tf.cast(state_to_now_test_, tf.float32)
                if step == 0:
                    state_test = tf.cast(initial_state_test_feature, tf.float32)
                    action_test_index = agent.act(state_test)
                    action_test_value = np.zeros_like(action_test_index)
                    for i in range(tf.shape(action_test_index)[0]):
                        for j in range(tf.shape(action_test_index)[1]):
                            action_test_value[i, j] = action_list[action_test_index[i, j]]
                    state_action_test = tf.concat((tf.reshape(state_test, [batch_test, -1, feature_size]), tf.reshape(action_test_value, [batch_test, -1, 1])), axis=2)  # 当前时刻选择的state action 拼接

                    state_to_now_test = tf.concat((state_to_now_test_, state_action_test), axis=1) #从0时刻到现在时刻的state action
                    next_state_test = construct_environment(state_to_now_test)
                    reward_test = reward_net([state_test, action_test_value])[:, 0] * imbalance_1

                    rewards_test = tf.concat((rewards_test, tf.reshape(reward_test, [batch_test, 1, 1])), axis=1)
                    states_test = tf.concat((states_test, tf.reshape(state_test, [batch_test, -1, feature_size])), axis=1)
                    actions_test = tf.concat((actions_test, tf.reshape(action_test_value, [batch_test, -1, 1])), axis=1)

                    death_model_input = tf.concat((state_to_now_test[:, :, :-1], tf.reshape(next_state_test, [batch_test, -1, feature_size])), axis=1)
                    death_test = death_model(death_model_input)
                    estimated_death_test = tf.concat((estimated_death_test, death_test), axis=1)

                else:
                    state_test = next_state_test   # 当前时刻的state
                    action_test_index = agent.act(state_test)  # 选择新的action
                    action_test_value = np.zeros_like(action_test_index)
                    for i in range(tf.shape(action_test_index)[0]):
                        for j in range(tf.shape(action_test_index)[1]):
                            action_test_value[i, j] = action_list[action_test_index[i, j]]
                    state_action_test = tf.concat((tf.reshape(state_test, [batch_test, -1, feature_size]), tf.reshape(action_test_value, [batch_test, -1, 1])), axis=2)  # 当前时刻的state action
                    state_to_now_test__ = tf.concat((states_test, actions_test), axis=2)  # 已经保存的state action

                    state_to_now_test = tf.concat((state_to_now_test_, state_to_now_test__, state_action_test), axis=1)
                    next_state_test = construct_environment(state_to_now_test)
                    reward_test = (reward_net([state_test, action_test_value])[:, 0]) * imbalance_1
                    # if step == predicted_visit-1:
                    #     reward_test += (initial_state_test[:, 3] - next_state_test[:, 3]) * imbalance_2
                    rewards_test = tf.concat((rewards_test, tf.reshape(reward_test, [batch_test, -1, 1])), axis=1)
                    states_test = tf.concat((states_test, tf.reshape(state_test, [batch_test, -1, feature_size])), axis=1)
                    actions_test = tf.concat((actions_test, tf.reshape(action_test_value, [batch_test, -1, 1])), axis=1)

                    death_model_input = tf.concat((state_to_now_test[:, :, :-1], tf.reshape(next_state_test, [batch_test, -1, feature_size])), axis=1)
                    death_test = death_model(death_model_input)
                    estimated_death_test = tf.concat((estimated_death_test, death_test), axis=1)
            pro_death_test = np.zeros_like(estimated_death_test)
            for patient in range(batch_test):
                for visit in range(predicted_visit):
                    if estimated_death_test[patient, visit] >= 0.048136085:
                        pro_death_test[patient, visit] = 1

            discont_rewards_test = agent.discont_reward(states=states_test, rewards=rewards_test)
            discont_rewards_test_real = agent.discont_reward(states=states_test_real, rewards=rewards_test_real)
            print('epoch {}    train_total_reward {}  train_rewards_real {}  test_total_reward {}  test_rewards_real {}'
                  'train_loss {}  --test_death_rate---{}'
                  .format(train_set.epoch_completed, np.mean(discont_rewards), np.mean(discont_rewards_real),
                          np.mean(discont_rewards_test), np.mean(discont_rewards_test_real),
                          np.mean(loss),
                          np.sum(pro_death_test)
                          ))
            np.save('death_pre_policy_gradient.npy', pro_death_test)
            np.save('discont_rewards_test_policy_gradient.npy', discont_rewards_test)
            # if np.mean(discont_rewards_test) < 3 and train_set.epoch_completed > 20:
            #     break

            # if np.mean(discont_rewards_test) > 4 and np.sum(pro_death_test) > 30:
            #     print('开始保存模型！')
            #     agent.model.save_weights('policy_gradient_net.h5')
    tf.compat.v1.reset_default_graph()
    return np.mean(discont_rewards_test)


if __name__ == '__main__':
    test_test('10_13_训练agent_3_7_pp.txt')
    # Agent_BO = BayesianOptimization(
    #     train_batch, {
    #         'hidden_size': (5, 8),
    #         'learning_rate': (-5, -1),
    #         'l2_regularization': (-6, -1),
    #     }
    # )
    # Agent_BO.maximize()
    # print(Agent_BO.max)
    for i in range(100):
        discount_reward = train_batch(hidden_size=128,
                                      learning_rate=0.09525829073800425,
                                      l2_regularization=10 ** (-5.992370919158627))








