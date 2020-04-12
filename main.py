#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Xuan-Yu Wang'

import tensorflow as tf
import numpy as np
from Functions import get_image_paths
from Functions import resize_img
from Functions import save_as_tfrecord
from Functions import variable_summaries
from Functions import variable_with_weight_loss
from Functions import read_and_decode
from Functions import input_pipeline
from LSTM import lstm_detector

log_dir = '../log_with_summaries'
batch_size = 200
max_step = 1000
img_size = 128
save_step = 100
disp_step = 5
epoch = 5
step = 0

with tf.name_scope("pretreament"):
    pos_filenames = get_image_paths("./data/dip")
    neg_filenames = get_image_paths("./data/hap")
    print("num of dip samples is %d" % len(pos_filenames))
    print("num of hap samples is %d" % len(neg_filenames))

    TRAIN_SEC, TEST_SEC = 0.8, 0.2
    pos_sample_num, neg_sample_num = len(pos_filenames), len(neg_filenames)
    np.random.shuffle(np.arange(len(pos_filenames)))
    np.random.shuffle(np.arange(len(neg_filenames)))
    pos_train, pos_test = pos_filenames[: int(pos_sample_num * TRAIN_SEC)], pos_filenames[int(pos_sample_num * TRAIN_SEC) :]
    neg_train, neg_test = neg_filenames[: int(neg_sample_num * TRAIN_SEC)], neg_filenames[int(neg_sample_num * TRAIN_SEC) :]

    print("dip sample : train num is %d, test num is %d" % (len(pos_train), len(pos_test)))
    print("hap sample : train num is %d, test num is %d" % (len(neg_train), len(neg_test)))

    all_train, all_test = [], []
    all_train_label, all_test_label = [], []
    all_train.extend(pos_train)
    all_train.extend(neg_train)
    all_test.extend(pos_test)
    all_test.extend(neg_test)
    pos_train_label, pos_test_label = np.ones(len(pos_train), dtype=np.int64), np.ones(len(pos_test), dtype=np.int64)
    neg_train_label, neg_test_label = np.zeros(len(neg_train), dtype=np.int64), np.zeros(len(neg_test), dtype=np.int64)
    all_train_label = np.hstack((pos_train_label, neg_train_label))
    all_test_label = np.hstack((pos_test_label, neg_test_label))
    print("train num is %d, test num is %d" % (len(all_train), len(all_test)))
    print("train_label num is %d, test_label num is %d" % (len(all_train_label), len(all_test_label)))

with tf.name_scope("trans_files"):
    save_as_tfrecord(all_test, all_test_label, "train.tfrecord")
    save_as_tfrecord(all_train, all_train_label, "test.tfrecord")

with tf.name_scope("input"):
    x = tf.compat.v1.add_check_numerics_ops(tf.float32, [batch_size, 24, 24, 3], name='img_input')
    y_ = tf.compat.v1.placeholder(tf.int32, [batch_size, 2], name='lab_input')
    tf.summary.image('input', x, 10)

with tf.name_scope("conv1"):
    with tf.name_scope('w_conv1'):
        weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
        variable_summaries(weight1)
    kernel1 = tf.nn.conv2d(x, weight1, [1, 1, 1, 1], padding='SAME')
    with tf.name_scope('b_conv1'):
        bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
        variable_summaries(bias1)
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    #vis_norm1 = tf.slice(norm1,(0,0,0,0),(200,24,24,3))
    #vis_norm1 = tf.reshape(vis_norm1,(200,24,24,3))
    tf.summary.histogram('norm1', norm1)
    #tf.summary.image('layer1', vis_norm1, 10)

with tf.name_scope("conv2"):
    with tf.name_scope('w_conv2'):
        weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
        variable_summaries(weight2)
    kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
    with tf.name_scope('b_conv2'):
        bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
        variable_summaries(bias2)
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME')
    tf.summary.histogram('pool2', pool2)
    
with tf.name_scope("full_conection1"):
    reshape = tf.reshape(pool2, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    with tf.name_scope('w_fc1'):
        weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
        variable_summaries(weight3)
    with tf.name_scope('b_fc1'):
        bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
        variable_summaries(bias3)
    local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)
    tf.summary.histogram('local3', local3)

with tf.name_scope("full_conection2"):
    with tf.name_scope('w_fc2'):
        weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
        variable_summaries(weight4)
    with tf.name_scope('b_fc2'):
        bias4 = tf.Variable(tf.constant(0.1, shape=[192]))  
        variable_summaries(bias4)
    local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)
    tf.summary.histogram('local4', local4)

with tf.name_scope("full_conection3"):
    with tf.name_scope('w_fc3'):
        weight5 = variable_with_weight_loss(shape=[192, 2], stddev=1/192.0, wl=0.0)
        variable_summaries(weight5)
    with tf.name_scope('b_fc3'):
        bias5 = tf.Variable(tf.constant(0.0, shape=[2]))
        variable_summaries(bias5)
    logits = tf.add(tf.matmul(local4, weight5), bias5)
    tf.summary.histogram('logits', logits)

with tf.name_scope("control_loss"):    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
    correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(logits,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    train_op = tf.compat.v1.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

with tf.name_scope("input_pipeline"):
    img_batch, label_batch = input_pipeline(["train.tfrecord"], batch_size)

saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    merged = tf.compat.v1.summary.merge_all()
    writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)
    coord = tf.train.Coordinator()
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())

    threads = tf.compat.v1.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() and step < max_step:
            step += 1
            imgs, labels = sess.run([img_batch, label_batch])
            """
            if step % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_op],feed_dict = {x : imgs, y_ : labels})
                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                train_writer.add_summary(summary, step)
                print('Adding run metadata for', step)
            """
            sess.run([merged, train_op],feed_dict = {x : imgs, y_ : labels})
            if epoch % disp_step == 0:
                summary, acc = sess.run([merged, accuracy], feed_dict = {x : imgs, y_ : labels})
                writer.add_summary(summary, step)
                print('step %s training accuracy is %.2f' % (step, acc))
            if step % save_step == 0:  
                save_path = saver.save(sess, 'graph.ckpt', global_step=step)
                print("save graph to %s" % save_path)
    except tf.errors.OutOfRangeError:
        print("reach epoch limit")
    finally:
        coord.request_stop()
    coord.join(threads)
    save_path = saver.save(sess, 'graph.ckpt', global_step=epoch)
   
    print("training is done")
    writer.close()

test_img_batch, test_label_batch = input_pipeline(["test.tfrecord"], 200)
with tf.compat.v1.Session as sess:
    saver.restore(sess, 'graph.ckpt-1000')
    coord_test = tf.train.Coordinator()
    threads_test = tf.compat.v1.train.start_queue_runners(coord=coord_test)
    test_imgs, test_labels = sess.run([test_img_batch, test_label_batch])
    acc = sess.run(accuracy, feed_dict = {x : test_imgs, y_ : test_labels})
    print("predict accuracy is %.2f" % acc)
    coord_test.request_stop()
    coord_test.join(threads_test)