#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: _defined
@Time:  2019/8/9 16:55
@Description: 
"""
import os
import tensorflow as tf
from settings import (config, DataMode)


class DataLoader(object):
    """
    data loader for train, test and validation
    """

    def __init__(self, mode):
        """
        :param mode: DataMode.Train, DataMode.Test or DataMode.Val
        """
        self.mode = mode
        self._size = 0

    @staticmethod
    def _parse_example(serial_example):
        features = tf.io.parse_single_example(
            serial_example,
            features={
                'label': tf.io.FixedLenFeature([], tf.string),
                'image': tf.io.FixedLenFeature([], tf.string),
            }
        )
        shape = (config.resize[0], config.resize[1], config.channel)
        image = tf.reshape(tf.io.decode_raw(features['image'], tf.float32), shape)
        label = tf.reshape(tf.io.decode_raw(features['label'], tf.int32), [config.max_seq_len, ])
        # label = tf.one_hot(tf.reshape(label, [config.max_seq_len, ]), self.n_class)
        return image, label

    def _load_dataset_from_tfrecords(self):
        """
        :return:
        """
        tfrecord_dir = './dataset/{}'.format(config.dataset)
        path = os.path.join(tfrecord_dir, "{}_{}.tfrecords".format(config.dataset, self.mode))
        if not os.path.exists(path):
            raise FileNotFoundError("No tfrecords file found. Please execute 'make_dataset.py' before your training")
        dataset = tf.data.TFRecordDataset(filenames=path).map(self._parse_example)
        self._size = len([_ for _ in dataset])
        return dataset

    def load_batch_from_tfrecords(self):
        """
        :return: tf.Dataset
        """
        min_after_dequeue = 2000
        dataset = self._load_dataset_from_tfrecords()
        dataset = dataset.shuffle(min_after_dequeue).batch(config.batch_size)
        return dataset

    def load_all_from_tfreocrds(self):
        """
        :return:
        """
        return self._load_dataset_from_tfrecords()

    @property
    def size(self):
        """
        :return: dataset size
        """
        return self._size


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    dataloader = DataLoader(DataMode.Test)
    la = list()
    dataset = dataloader.load_all_from_tfreocrds()
    print(dataloader.size)
    # print(dataloader.size())
    for i in range(2):
        for data in dataset:
            images, labels = data
            print(labels.shape)
            la.append(labels)
    print(len(la))
