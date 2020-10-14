import os
import sys
import tensorflow as tf


def test_test(name):
    class Logger(object):
        def __init__(self, filename="Default.log"):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger(name)

    print(path)
    print(os.path.dirname(__file__))
    print('------------------')


def discont_reward(states, rewards, gamma=0.99):
    sum_reward = 0
    batch = tf.shape(states)[0]
    discnt_reward = tf.zeros(shape=[batch, 0, 1])
    rewards = tf.reverse(rewards, axis=[1])

    for r_index in range(tf.shape(rewards)[1]):
        r = rewards[:, r_index, :]
        sum_reward = r + gamma * sum_reward
        discnt_reward = tf.concat((discnt_reward, tf.reshape(sum_reward, [batch, -1, 1])), axis=1)
    discnt_reward = tf.reverse(discnt_reward, axis=[1])
    return discnt_reward

def plot(data):
    x = np.arange(len(data))
    fig = plt.plot(x, data, 'r--')
    plt.ylabel('cumulative reward')
    plt.xlabel('episode')
    plt.legend()
    plt.show()