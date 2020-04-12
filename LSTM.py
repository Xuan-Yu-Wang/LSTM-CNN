#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Xuan-Yu Wang'

import tensorflow as tf

num_units = 128
n_classes = 2
time_steps = 28
n_input = 28
learning_rate = 0.001

out_weights = tf.Variable(tf.compat.v1.random_normal([num_units, n_classes]))
out_bias = tf.Variable(tf.compat.v1.random_normal([n_classes]))

def lstm_detector (x, y_):

    x = tf.compat.v1.placeholder(tf.float32, [None,time_steps, n_input])
    y_ = tf.compat.v1.placeholder(tf.int32, [None, n_classes])

    input = tf.unstack(x, time_steps, 1)

    lstm_layer = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units,forget_bias=1)
    outputs, _ = tf.compat.v1.nn.static_rnn(lstm_layer,input, dtype="float32")
    prediction = tf.matmul(outputs[-1], out_weights) + out_bias

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_))
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return loss, opt, accuracy


