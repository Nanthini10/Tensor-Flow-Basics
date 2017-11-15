# -*- coding: utf-8 -*-
"""
Spyder Editor
date: October 16, 2017
@author: nanthini
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#Single high and all low representation
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Placeholder define the variables in the equation y = Wx + b
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#Trains the model with a gradient descent to find the optimal solution by minimizing the cost or entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Create a new session s 'sess'
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#Start training the model with data in batches
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#Compare the predictions to the actual values and calculate the accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))