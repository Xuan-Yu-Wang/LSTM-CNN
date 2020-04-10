#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Xuan-Yu Wang'

from __future__ import print_function
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import os

def get_image_paths (img_dir):
    filenames = os.listdir(img_dir)
    filenames = [os.path.join(img_dir, item) for item in filenames]
    return filenames

def resize_img(img_path, shape):
    im = Image.open(img_path)
    im = im.resize(shape)
    im = im.conver('RGB')
    return im

def save_as_tfrecord(samples, labels, bin_path):
    assert len(samples) == len(labels)
    writer = tf.io.TFRecordWriter.TFRcordWriter(bin_path)
    img_label = list(zip(samples, labels))
    np.random.shuffle(img_label)
    for img, label in img_label:
        im = resize_img(img, (24, 24))
        im_raw = im.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[im_raw]))
            }))
        writer.write(example.SerializaToString())
    writer.close()

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.random.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.compat.v1.add_to_collection('losses', weight_loss)
    return var

def read_and_decode(filename_queue):
        reader = tf.compat.v1.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)   
        features = tf.compat.v1.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.compat.v1.FixedLenFeature([], tf.int64),
                                               'img_raw' : tf.compat.v1.FixedLenFeature([], tf.string),
                                           })

        img = tf.compat.v1.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, [24, 24, 3])
        img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 # normalize
        label = tf.cast(features['label'], tf.int32)
        label = tf.compat.v1.sparse_to_dense(label, [2], 1, 0)

        return img, label

def input_pipeline(filenames, batch_size, num_epochs=None):
        filename_queue = tf.compat.v1.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
        example, label = read_and_decode(filename_queue)
        min_after_dequeue = 1000
        num_threads = 4
        capacity = min_after_dequeue + (num_threads + 3) * batch_size
        example_batch, label_batch = tf.compat.v1.train.shuffle_batch(
            [example, label], batch_size=batch_size, capacity=capacity, num_threads = num_threads,
            min_after_dequeue=min_after_dequeue)
        return example_batch, label_batch