# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:17:49 2016

@author: yonezawakeiko
MNIST
Learn filter coefficients
Chap 4-2
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

np.random.seed(20161109)
tf.set_random_seed(20161109)
mnist = input_data.read_data_sets("/Users/yonezawakeiko/work/ae/try1/MNIST_data/", one_hot=True)

#images, labels = mnist.train.next_batch(100)
images_test, labels_test = mnist.train.next_batch(100)

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv = tf.Variable(tf.truncated_normal([5, 5, 1, 2], stddev=0.1))
h_conv = tf.abs(tf.nn.conv2d(x_image, W_conv, strides=[1, 1, 1, 1], padding='SAME'))
h_conv_cutoff = tf.nn.relu(h_conv - 0.2)
h_pool = tf.nn.max_pool(h_conv_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')
h_pool_flat = tf.reshape(h_pool, [-1, 392])

num_units1 = 392
num_units2 = 2

w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.zeros([num_units2]))
hidden2 = tf.nn.tanh(tf.matmul(h_pool_flat, w2) + b2)

w0 = tf.Variable(tf.zeros([num_units2, 10]))
b0 = tf.Variable(tf.zeros([10]))
p = tf.nn.softmax(tf.matmul(hidden2, w0) + b0)

t = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdamOptimizer().minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

i = 0
for _ in range(1000):
    i += 1
    images, labels = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:images, t:labels})
    if i % 100 == 0:
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x:images, t:labels})
        print ('Step(train): %d, Loss: %f, Accuracy: %f' %(i, loss_val, acc_val))
        loss_val2, acc_val2 = sess.run([loss, accuracy], feed_dict={x:images_test, t:labels_test})
        print ('Step(test): %d, Loss: %f, Accuracy: %f' %(i, loss_val2, acc_val2))
   
filter_vals, conv_vals = sess.run([W_conv, h_conv_cutoff], feed_dict={x:images[:9]})
pool_vals = sess.run(h_pool, feed_dict={x:images[:9]})
   
"""     
hidden2_vals = sess.run(hidden2, feed_dict={x:images})

z1_val = [[],[],[],[],[],[],[],[],[],[]]
z2_val = [[],[],[],[],[],[],[],[],[],[]]

for hidden2_val, label in zip(hidden2_vals, labels):
    label_num = np.argmax(label)
    #print('label_num' + repr(label_num)+ repr(hidden2_val[0]))
    z1_val[label_num].append(hidden2_val[0])
    z2_val[label_num].append(hidden2_val[1])
    
fig = plt.figure(figsize=(5, 5))
markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3"]
subplot = fig.add_subplot(1, 1, 1)
subplot.scatter(z1_val[0], z2_val[0], s=200, marker=markers[0])
subplot.scatter(z1_val[1], z2_val[1], s=200, marker=markers[1])
subplot.scatter(z1_val[2], z2_val[2], s=200, marker=markers[2])
subplot.scatter(z1_val[3], z2_val[3], s=200, marker=markers[3])
subplot.scatter(z1_val[4], z2_val[4], s=200, marker=markers[4])
subplot.scatter(z1_val[5], z2_val[5], s=200, marker=markers[5])
subplot.scatter(z1_val[6], z2_val[6], s=200, marker=markers[6])
subplot.scatter(z1_val[7], z2_val[7], s=200, marker=markers[7])
subplot.scatter(z1_val[8], z2_val[8], s=200, marker=markers[8])
subplot.scatter(z1_val[9], z2_val[9], s=200, marker=markers[9]) 
plt.show()
"""

#filter_vals, conv_vals, pool_val = sess.run([W_conv, h_conv_cutoff, h_pool], feed_dict={x:images})

v_max = np.max(conv_vals)
v_max2 = np.max(pool_vals)

fig = plt.figure(figsize=(10, 5))
for i in range(2):
    subplot = fig.add_subplot(5, 10, 10*(i+1) + 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(filter_vals[:,:,0,i],cmap=plt.cm.gray_r, interpolation="nearest")

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
