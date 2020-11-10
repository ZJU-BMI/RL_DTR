import os
import sys
import tensorflow as tf
from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
import datetime
import zipfile
import nibabel as nib
import skimage.io as io
import numpy as np
import imageio
from nibabel import nifti1
import pandas as pd
from nibabel.viewers import OrthoSlicer3D


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


def log_prob_calculation():
    probs = [0.1, 0.5, 0.4]
    m = tf.compat.v1.distributions.Categorical(probs)
    action = m.sample()
    print(action)
    print(m.log_prob(action))

    neg_log_likelihood = tf.keras.losses.sparse_categorical_crossentropy(
        tf.reshape(action, [-1, ]),
        tf.reshape(probs, [-1, 3]))
    print(neg_log_likelihood)


def cal_time_intervals():
    d1 = datetime.datetime(2006, 4, 8)  # 第一个日期
    d2 = datetime.datetime(2006, 11, 2)  # 第二个日期
    interval = d2 - d1  # 两日期差距
    print(interval.days)


def read_zip_file_name():
    z = zipfile.ZipFile(r'G:\ADNI\11_3\tbm jacobian Maps MDT-SC\TBM_Jacobian_Maps_MDT-SC.zip', 'r')
    names = z.namelist()
    list_all = []
    for i in names:
        name = i.split('/')
        print(name[1])
        list_all.append(name[1])
    np.savetxt('name_all.csv', list_all, delimiter=',', fmt='%s')


def read_nii_file():
    path = 'G:\ADNI\ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070802163833409_S32678_I64025.nii'
    img = nib.load(path)
    img_arr = img.get_fdata()
    img_arr = np.squeeze(img_arr)
    for i in range(img_arr.shape[2]):
        img_i = img_arr[:, :, i]
        imageio.imwrite(os.path.join('G:\ADNI', '{}.png'.format(i)), img_i)
    # io.imshow(img_arr)


def read_2():
    example_filename = 'G:\ADNI\ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070802163833409_S32678_I64025.nii'

    img = nib.load(example_filename)
    print(img)
    print(img.header['db_name'])  # 输出头信息

    width, height, queue = img.dataobj.shape

    OrthoSlicer3D(img.dataobj).show()

    num = 1
    for i in range(0, queue, 10):
        img_arr = img.dataobj[:, :, i]
        plt.subplot(5, 4, num)
        plt.imshow(img_arr, cmap='gray')
        num += 1


def read_name_list():
    name_all = pd.read_csv('name_all.csv', delimiter=',').values
    list_1 = name_all[:, 0]
    list_2 = name_all[:, 1]

    set_1 = set(list_1)
    set_2 = set(list_2)
    new_list = [item for item in set_2 if item not in set_1]
    print(new_list)


if __name__ == '__main__':
    read_name_list()