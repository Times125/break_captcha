# ! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: _defined
@Time:  2019/8/6 16:38
@Description:
    This script defines some network structures,
    such as BiLSTM and BiGRU
"""
from tensorflow.python.keras import layers


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
    x = layers.Dense(units=units * 2, activation='relu', name='fc')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    return x


# BiLSTM
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
    x = layers.Dense(units=units * 2, activation='relu', name='fc')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    return x
