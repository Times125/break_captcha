#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: _defined
@Time:  2019/8/6 16:38
@Description:
    This script defines some network structures,
    such as ResNet50, CNN5, BiLSTM and BiGRU
"""
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras.regularizers import l2

__all__ = ['ResNet50', 'CNN5', 'BiLSTM', 'BiGRU']


def _identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main file_list
        filters: list of integers, the filters of 3 conv layer at main file_list
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def _conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main file_list
        filters: list of integers, the filters of 3 conv layer at main file_list
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main file_list is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis,
                                         name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet50(input_tensor, pooling='avg'):
    img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2),
                      kernel_initializer='he_normal', name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = _conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = _identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = _identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = _conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = _identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = _identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = _identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = _conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = _conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = _identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = _identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    return x


def CNN5(input_tensor):
    """
    CNN-5
    :param input_tensor:
    :return:
    """
    filters = [32, 64, 128, 128, 256]
    x = input_tensor
    for i in range(5):
        x = layers.Conv2D(filters[i], (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        if i == 0:
            x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
        else:
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x


def BiGRU(input_tensor, units=64, use_gpu=True):
    """
    Bi-GRU
    :param input_tensor:
    :param units:
    :param use_gpu: if true, use CuDNNGRU to accelerate computing.
    :return:
    """
    if use_gpu:
        GRU = layers.CuDNNGRU
    else:
        GRU = layers.GRU
    gru1 = layers.Bidirectional(
        GRU(units, return_sequences=True, kernel_initializer='he_normal', name='gru1'),
        merge_mode='sum')(input_tensor)
    x = layers.Bidirectional(
        GRU(units, return_sequences=True, kernel_initializer='he_normal', name='gru2'),
        merge_mode='concat')(gru1)
    x = layers.TimeDistributed(layers.Dense(units=units * 2, activation='relu'), name='fc')(x)
    x = layers.TimeDistributed(layers.Dropout(0.3), name='dropout')(x)
    return x


def BiLSTM(input_tensor, units=64, use_gpu=False):
    """
    Bi-LSTM
    :param input_tensor:
    :param units:
    :param use_gpu: if true, use CuDNNGRU to accelerate computing.
    :return:
    """
    if use_gpu:
        LSTM = layers.CuDNNLSTM
    else:
        LSTM = layers.LSTM
    lstm1 = layers.Bidirectional(
        LSTM(units, return_sequences=True, kernel_initializer='he_normal', name='lstm1'),
        merge_mode='sum')(input_tensor)
    x = layers.Bidirectional(
        LSTM(units, return_sequences=True, kernel_initializer='he_normal', name='lstm2'),
        merge_mode='concat')(lstm1)
    x = layers.TimeDistributed(layers.Dense(units=units * 2, activation='relu'), name='fc')(x)
    x = layers.TimeDistributed(layers.Dropout(0.3), name='dropout')(x)
    return x
