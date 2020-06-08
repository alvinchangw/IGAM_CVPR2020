"""
Utilities for importing the CIFAR100 dataset.

Each image in the dataset is a numpy array of shape (32, 32, 3), with the values
being unsigned integers (i.e., in the range 0,1,...,255).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import tensorflow as tf
version = sys.version_info

import numpy as np

class CIFAR100Data(object):
    """
    Unpickles the CIFAR100 dataset from a specified folder containing a pickled
    version following the format of Krizhevsky which can be found
    [here](https://www.cs.toronto.edu/~kriz/cifar.html).

    Inputs to constructor
    =====================

        - path: path to the pickled dataset. The training data must be pickled
        into  one file name train containing 50,000, and 10,000
        examples each, the test data
        must be pickled into a single file called test containing 10,000
        examples, and the 100 fine or 20 coarse class names must be
        pickled into a file called meta. The pickled examples should
        be stored as a tuple of two objects: an array of 50,000 32x32x3-shaped
        arrays, and an array of their 50,000 true labels.

    """
    def __init__(self, path, init_shuffle=True):
        train_filename = 'train'
        eval_filename = 'test'#'test_batch'
        metadata_filename = 'meta'#'batches.meta'

        train_images = np.zeros((50000, 32, 32, 3), dtype='uint8')
        train_labels = np.zeros(50000, dtype='int32')
        train_images, train_labels = self._load_datafile(
            os.path.join(path, train_filename))
        eval_images, eval_labels = self._load_datafile(
            os.path.join(path, eval_filename))

        with open(os.path.join(path, metadata_filename), 'rb') as fo:
              if version.major == 3:
                  data_dict = pickle.load(fo, encoding='bytes')
              else:
                  data_dict = pickle.load(fo)

              self.label_names = data_dict[b'fine_label_names']
        for ii in range(len(self.label_names)):
            self.label_names[ii] = self.label_names[ii].decode('utf-8')

        self.train_data = DataSubset(train_images, train_labels, init_shuffle=init_shuffle)
        self.eval_data = DataSubset(eval_images, eval_labels, init_shuffle=init_shuffle)

    @staticmethod
    def _load_datafile(filename):
      with open(filename, 'rb') as fo:
          if version.major == 3:
              data_dict = pickle.load(fo, encoding='bytes')
          else:
              data_dict = pickle.load(fo)

          assert data_dict[b'data'].dtype == np.uint8
          image_data = data_dict[b'data']
          image_data = image_data.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
          return image_data, np.array(data_dict[b'fine_labels'])


class AugmentedCIFAR100Data(object):
    """
    Data augmentation wrapper over a loaded dataset.

    Inputs to constructor
    =====================
        - raw_cifar10data: the loaded CIFAR100 dataset, via the CIFAR100Data class
        - sess: current tensorflow session
        - model: current model (needed for input tensor)
    """
    def __init__(self, raw_cifar100data, sess, model):
        assert isinstance(raw_cifar100data, CIFAR100Data)
        self.image_size = 32

        # create augmentation computational graph
        self.x_input_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        padded = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
            img, self.image_size + 4, self.image_size + 4),
            self.x_input_placeholder)
        cropped = tf.map_fn(lambda img: tf.random_crop(img, [self.image_size,
                                                             self.image_size,
                                                             3]), padded)
        flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped)
        self.augmented = flipped

        self.train_data = AugmentedDataSubset(raw_cifar100data.train_data, sess,
                                             self.x_input_placeholder,
                                              self.augmented)
        self.eval_data = AugmentedDataSubset(raw_cifar100data.eval_data, sess,
                                             self.x_input_placeholder,
                                             self.augmented)
        self.label_names = raw_cifar100data.label_names


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

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True, return_indices=False):
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
            data_indices = self.cur_order[self.batch_start : batch_end]
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
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        data_indices = self.cur_order[self.batch_start : batch_end]
        self.batch_start += batch_size
        if return_indices:
            return batch_xs, batch_ys, data_indices
        else:
            return batch_xs, batch_ys


class AugmentedDataSubset(object):
    def __init__(self, raw_datasubset, sess, x_input_placeholder,
                 augmented):
        self.sess = sess
        self.raw_datasubset = raw_datasubset
        self.x_input_placeholder = x_input_placeholder
        self.augmented = augmented

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True, return_indices=False):
        if return_indices:
            raw_batch_xs, raw_batch_ys, data_indices = self.raw_datasubset.get_next_batch(batch_size, multiple_passes,
                                                        reshuffle_after_pass, return_indices=True)
            images = raw_batch_xs.astype(np.float32)
            return self.sess.run(self.augmented, feed_dict={self.x_input_placeholder:
                                                        raw_batch_xs}), raw_batch_ys, data_indices
        else:
            raw_batch_xs, raw_batch_ys = self.raw_datasubset.get_next_batch(batch_size, multiple_passes,
                                                        reshuffle_after_pass, return_indices=False)
            images = raw_batch_xs.astype(np.float32)
            return self.sess.run(self.augmented, feed_dict={self.x_input_placeholder:
                                                        raw_batch_xs}), raw_batch_ys
