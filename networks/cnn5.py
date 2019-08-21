#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: _defined
@Time:  2019/8/21 14:20
@Description:
    This script defines some CNN5 structures
"""
from tensorflow.python.keras import layers
from tensorflow.python.keras.regularizers import l2


def CNN5(input_tensor, kr=0.01):
    """
    CNN-5
    :param input_tensor:
    :param kr: kernel regularizer rate
    :return:
    """
    filters = [32, 64, 128, 128, 256]
    x = input_tensor
    for i in range(5):
        x = layers.Conv2D(filters[i], (3, 3), padding='same', kernel_regularizer=l2(kr))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        if i == 0:
            x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
        else:
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x
