#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: _defined
@Time:  2019/8/6 16:38
@Description: 
"""
import os
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from networks import (
    ResNet50, CNN5, BiGRU, BiLSTM,
    DenseNet121, DenseNet169, DenseNet201

)
from settings import config
from ctc_ops import ctc_batch_cost

__all__ = ['build_model']


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN tend to be garbage
    y_pred = y_pred[:, :, :]
    return ctc_batch_cost(labels, y_pred, input_length, label_length)


def build_model():
    """
    build CNN-RNN model
    :return:
    """
    input_shape = (config.resize[0], config.resize[1], config.channel)
    inputs = Input(shape=input_shape)
    # CNN layers
    x = None

    if config.cnn_type == 'CNN5':
        x = CNN5(inputs, config.l2)
    elif config.cnn_type == 'ResNet50':
        x = ResNet50(inputs)
    elif config.cnn_type == 'DenseNet121':
        x = DenseNet121(inputs)
    elif config.cnn_type == 'DenseNet169':
        x = DenseNet169(inputs)
    elif config.cnn_type == 'DenseNet201':
        x = DenseNet201(inputs)
    conv_shape = x.get_shape()
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
    # concat Bi-RNN layers to encode and decode sequence
    x = BiLSTM(x, units=config.rnn_units, use_gpu=config.use_gpu) if config.rnn_type == 'BiLSTM' \
        else BiGRU(x, units=config.rnn_units, use_gpu=config.use_gpu)
    predictions = Dense(config.n_class, kernel_initializer='he_normal', activation='softmax')(x)
    base_model = Model(inputs=inputs, outputs=predictions)
    # CTC_loss
    labels = Input(name='the_labels', shape=[config.max_seq_len, ], dtype='float32')
    input_length = Input(name='input_length', shape=[1, ], dtype='int64')
    label_length = Input(name='label_length', shape=[1, ], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [predictions, labels, input_length, label_length])
    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=[loss_out])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr=config.lr, decay=config.decay))
    if not os.path.exists('./plotModel'):
        os.makedirs('./plotModel')
    plot_model(model, './plotModel/{}-{}_model.png'.format(config.cnn_type, config.rnn_type), show_shapes=True)
    plot_model(base_model, './plotModel/{}-{}_base_model.png'.format(config.cnn_type, config.rnn_type),
               show_shapes=True)
    return model, base_model, int(conv_shape[1])
