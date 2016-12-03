# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 20:04:26 2016

@author: yonezawakeiko
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(20160703)
tf.set_random_seed(20160703)

mnist = input_data.read_data_sets("/Users/yonezawakeiko/work/ae/try1/MNIST_data/", one_hot=True)

class SingleCNN:
    def __init__(self, num_filters, num_units):
        with tf.Graph().as_default():
            self.prepare_model(num_filters, num_units)
            self.prepare_session()

    def prepare_model(self, num_filters, num_units):
        num_units1 = 14*14*num_filters
        num_units2 = num_units
        
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 784], name='input')
            x_image = tf.reshape(x, [-1,28,28,1])

        with tf.name_scope('convolution'):
            W_conv = tf.Variable(
                tf.truncated_normal([5,5,1,num_filters], stddev=0.1),
                name='conv-filter')
            h_conv = tf.nn.conv2d(
                x_image, W_conv, strides=[1,1,1,1], padding='SAME',
                name='filter-output')

        with tf.name_scope('pooling'):            
            h_pool =tf.nn.max_pool(h_conv, ksize=[1,2,2,1],
                                   strides=[1,2,2,1], padding='SAME',
                                   name='max-pool')
            h_pool_flat = tf.reshape(h_pool, [-1, 14*14*num_filters],
                                     name='pool-output')

        with tf.name_scope('fully-connected'):
            w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
            b2 = tf.Variable(tf.zeros([num_units2]))
            hidden2 = tf.nn.relu(tf.matmul(h_pool_flat, w2) + b2,
                                 name='fc-output')

        with tf.name_scope('softmax'):
            w0 = tf.Variable(tf.zeros([num_units2, 10]))
            b0 = tf.Variable(tf.zeros([10]))
            p = tf.nn.softmax(tf.matmul(hidden2, w0) + b0,
                              name='softmax-output')
            
        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, 10], name='labels')
            loss = -tf.reduce_sum(t * tf.log(p), name='loss')
            train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)
            
        with tf.name_scope('evaluator'):
            correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                              tf.float32), name='accuracy')
            
        tf.scalar_summary("loss", loss)
        tf.scalar_summary("accuracy", accuracy)
        tf.histogram_summary("convolution_filters", W_conv)
        
        self.x, self.t, self.p = x, t, p
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy
        
    def prepare_session(self):
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        summary = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("/tmp/mnist_df_logs", sess.graph)
        
        self.sess = sess
        self.summary = summary
        self.writer = writer
        
cnn = SingleCNN(16, 1024)

i = 0
for _ in range(4000):
    i += 1
    batch_xs, batch_ts = mnist.train.next_batch(100)
    cnn.sess.run(cnn.train_step, feed_dict={cnn.x:batch_xs, cnn.t:batch_ts})
    if i % 50 == 0:
        summary, loss_val, acc_val = cnn.sess.run(
            [cnn.summary, cnn.loss, cnn.accuracy],
            feed_dict={cnn.x:mnist.test.images, cnn.t:mnist.test.labels})
        print ('Step: %d, Loss: %f, Accuracy: %f'
               % (i, loss_val, acc_val))
        cnn.writer.add_summary(summary, i)       