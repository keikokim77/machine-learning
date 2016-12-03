# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:17:49 2016

@author: yonezawakeiko
MNIST
Evaluate filter effects with MNIST
Chap 4-1
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

np.random.seed(20161104)
mnist = input_data.read_data_sets("/Users/yonezawakeiko/work/ae/try1/MNIST_data/", one_hot=True)

images, labels = mnist.train.next_batch(40)

def edge_filter():
    filter0 = np.array([[2,1,0,-1,-2],
                        [3,2,0,-2,-3],
                        [4,3,0,-3,-4],
                        [3,2,0,-2,-3],
                        [2,1,0,-1,-2]])/23.0
    filter1 = np.array([[2,3,4,3,2],
                        [1,2,3,2,1],
                        [0,0,0,0,0],
                        [-1,-2,-3,-2,-1],
                        [-2,-3,-4,-3,-2]])/23.0
    filter_array = np.zeros([5, 5, 1, 2])
    filter_array[:,:,0,0] = filter0
    filter_array[:,:,0,1] = filter1

    return tf.constant(filter_array, dtype=tf.float32)      

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv = edge_filter()
h_conv = tf.abs(tf.nn.conv2d(x_image, W_conv, strides=[1, 1, 1, 1], padding='SAME'))
h_conv_cutoff = tf.nn.relu(h_conv - 0.2)
h_pool = tf.nn.max_pool(h_conv_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')

sess = tf.Session()
sess.run(tf.initialize_all_variables())

filter_vals, conv_vals = sess.run([W_conv, h_conv_cutoff], feed_dict={x:images[:9]})
pool_vals = sess.run(h_pool, feed_dict={x:images[:9]})

fig = plt.figure(figsize=(10, 5))
for i in range(2):
    subplot = fig.add_subplot(5, 10, 10*(i+1) + 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(filter_vals[:,:,0,i],cmap=plt.cm.gray_r, interpolation="nearest")

v_max = np.max(conv_vals)
v_max2 = np.max(pool_vals)


for i in range(9):
    subplot = fig.add_subplot(5, 10, i+2)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[i]))
    subplot.imshow(images[i].reshape((28, 28)),vmin=0, vmax=1, cmap=plt.cm.gray_r, interpolation="nearest")

    subplot = fig.add_subplot(5, 10, 10 + i + 2)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(conv_vals[i, :, :, 0],vmin=0, vmax=v_max, cmap=plt.cm.gray_r, interpolation="nearest")

    subplot = fig.add_subplot(5, 10, 20 + i + 2)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(conv_vals[i, :, :, 1],vmin=0, vmax=v_max, cmap=plt.cm.gray_r, interpolation="nearest")

    subplot = fig.add_subplot(5, 10, 30 + i + 2)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(pool_vals[i, :, :, 0],vmin=0, vmax=v_max2, cmap=plt.cm.gray_r, interpolation="nearest")

    subplot = fig.add_subplot(5, 10, 40 + i + 2)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(pool_vals[i, :, :, 1],vmin=0, vmax=v_max2, cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()

#fig = plt.figure(figsize=(10, 5))
#for i in range(40):
#    subplot = fig.add_subplot(4, 10, i + 1)
#    subplot.set_xticks([])
#    subplot.set_yticks([])
#    subplot.set_title('%d' % np.argmax(labels[i]))
#    subplot.imshow(images[i].reshape((28,28)), vmin=0, vmax=1, cmap=plt.cm.gray_r, interpolation="nearest")
#plt.show()
