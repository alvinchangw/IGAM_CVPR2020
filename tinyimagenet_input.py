"""
Utilities for importing the TinyImagenet dataset.
Each image in the dataset is a numpy array of shape (64, 64, 3), with the values
being unsigned integers (i.e., in the range 0,1,...,255).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import re
import matplotlib.pyplot as plt
import pathlib

version = sys.version_info


class TinyImagenetData(object):
    """
    Inputs to constructor
    =====================
        - path: path to the pickled dataset. The training data must be pickled
        into five files named data_batch_i for i = 1, ..., 5, containing 10,000
        examples each, the test data
        must be pickled into a single file called test_batch containing 10,000
        examples, and the 10 class names must be
        pickled into a file called batches.meta. The pickled examples should
        be stored as a tuple of two objects: an array of 10,000 32x32x3-shaped
        arrays, and an array of their 10,000 true labels.
    """

    def __init__(self, dataset_path="./datasets/tiny-imagenet/tiny-imagenet-200", init_shuffle=True):
        # Load training data
        tinyimagenet_train_dir = os.path.join(dataset_path, 'train')
        data_dir = pathlib.Path(tinyimagenet_train_dir)
        image_count = len(list(data_dir.glob('*/images/*.JPEG')))
        CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])

        train_images = []
        train_labels = []

        for class_ind, class_name in enumerate(CLASS_NAMES):
            image_list = list(data_dir.glob('{}/images/*.JPEG'.format(class_name)))
            image_count = len(image_list)
            for img_path in image_list:
                img_np = plt.imread(img_path)
                if len(img_np.shape) == 2: # Black and White image
                    img_nps = [img_np for i in range(3)]
                    img_np = np.stack(img_nps, axis=-1)
                img_np = np.expand_dims(img_np, axis=0)
                train_images.append(img_np)
            
            class_labels = np.full((image_count,), class_ind)
            train_labels.append(class_labels)
            
        train_images = np.concatenate(train_images, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        # Load val data
        tinyimagenet_val_dir = os.path.join(dataset_path, 'val')
        data_dir = pathlib.Path(tinyimagenet_val_dir)
        image_count = len(list(data_dir.glob('images/*/*.JPEG')))
        eval_images = []
        eval_labels = []

        for class_ind, class_name in enumerate(CLASS_NAMES):
            image_list = list(data_dir.glob('images/{}/*.JPEG'.format(class_name)))
            image_count = len(image_list)
            for img_path in image_list:
                img_np = plt.imread(img_path)
                if len(img_np.shape) == 2: # Black and White image
                    img_nps = [img_np for i in range(3)]
                    img_np = np.stack(img_nps, axis=-1)
                img_np = np.expand_dims(img_np, axis=0)
                eval_images.append(img_np)
            
            class_labels = np.full((image_count,), class_ind)
            eval_labels.append(class_labels)
            
        eval_images = np.concatenate(eval_images, axis=0)
        eval_labels = np.concatenate(eval_labels, axis=0)

        self.label_names = CLASS_NAMES

        self.train_data = DataSubset(train_images, train_labels, init_shuffle=init_shuffle)
        self.eval_data = DataSubset(eval_images, eval_labels, init_shuffle=False)

class AugmentedTinyImagenetData(object):
    """
    Data augmentation wrapper over a loaded dataset.
    Inputs to constructor
    =====================
        - raw_tinyimagenetdata: the loaded TinyImagenet dataset, via the TinyImagenetData class
        - sess: current tensorflow session
        - model: current model (needed for input tensor)
    """

    def __init__(self, raw_tinyimagenetdata, sess, model):
        assert isinstance(raw_tinyimagenetdata, TinyImagenetData)
        self.image_size = 64

        # create augmentation computational graph
        self.x_input_placeholder = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        padded = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
            img, self.image_size + 4, self.image_size + 4),
                           self.x_input_placeholder)
        cropped = tf.map_fn(lambda img: tf.random_crop(img, [self.image_size,
                                                             self.image_size,
                                                             3]), padded)
        flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped)
        self.augmented = flipped

        self.train_data = AugmentedDataSubset(raw_tinyimagenetdata.train_data, sess,
                                              self.x_input_placeholder,
                                              self.augmented)
        self.eval_data = AugmentedDataSubset(raw_tinyimagenetdata.eval_data, sess,
                                             self.x_input_placeholder,
                                             self.augmented)
        self.label_names = raw_tinyimagenetdata.label_names


class DataSubset(object):
    def __init__(self, xs, ys, init_shuffle=True):
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        if init_shuffle:
            self.cur_order = np.random.permutation(self.n)
        else:
            self.cur_order = np.arange(self.n)

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start: batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start: batch_end], ...]
            self.batch_start += actual_batch_size
            if actual_batch_size < batch_size:
                print('actual_batch_size < batch_size, padding with zeros')
                batch_xs_pad = np.zeros(shape=(batch_size - actual_batch_size, batch_xs.shape[1], batch_xs.shape[2], batch_xs.shape[3]), dtype=batch_xs.dtype)
                batch_ys_pad = np.zeros(batch_size - actual_batch_size, dtype=batch_ys.dtype)
                batch_xs = np.concatenate([batch_xs, batch_xs_pad], axis=0)
                batch_ys = np.concatenate([batch_ys, batch_ys_pad], axis=0)
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start: batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start: batch_end], ...]
        self.batch_start += actual_batch_size
        return batch_xs, batch_ys


class AugmentedDataSubset(object):
    def __init__(self, raw_datasubset, sess, x_input_placeholder,
                 augmented):
        self.sess = sess
        self.raw_datasubset = raw_datasubset
        self.x_input_placeholder = x_input_placeholder
        self.augmented = augmented

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        raw_batch = self.raw_datasubset.get_next_batch(batch_size, multiple_passes,
                                                       reshuffle_after_pass)
        images = raw_batch[0].astype(np.float32)
        return self.sess.run(self.augmented, feed_dict={self.x_input_placeholder:
                                                            raw_batch[0]}), raw_batch[1]
