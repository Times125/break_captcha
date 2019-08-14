#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: _defined
@Time:  2019/8/12 19:02
@Description: 
"""
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.ctc_ops import (ctc_loss, ctc_beam_search_decoder)
from settings import (config, DataMode)
from DataLoader import DataLoader
from model import (build_model, test_model)
from logger import event_logger

SVAED_MODEL_DIR = './savedModel/{}'.format(config.dataset)
if not os.path.exists(SVAED_MODEL_DIR):
    os.makedirs(SVAED_MODEL_DIR)

CHECKPOINT_DIR = './checkpoints/{}'.format(config.dataset)
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)


def train():
    """
    train model
    :return:
    """
    model, base_model, seq_step_len = build_model()
    print('seq_step_len ', seq_step_len)
    train_dataset = DataLoader(DataMode.Train).load_batch_from_tfrecords()
    val_dataset = DataLoader(DataMode.Val).load_batch_from_tfrecords()

    latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    start_epoch = 0
    if latest_ckpt:
        start_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
        print('start epoch at ', start_epoch)
        model.load_weights(latest_ckpt)
        event_logger.info('model resumed from: {}, start at epoch: {}'.format(latest_ckpt, start_epoch))
    else:
        event_logger.info('passing resume since weights not there. training from scratch')

    def _validation():
        """
        validate the model's acc
        :return: acc
        """
        _val_losses = []
        _val_accuracy = []
        for _batch, _data in enumerate(val_dataset):
            _images, _labels = _data
            _input_length = np.array(np.ones(len(_images)) * int(seq_step_len))
            _label_length = np.array(np.ones(len(_images)) * config.max_seq_len)
            _loss = model.evaluate([_images, _labels, _input_length, _label_length], _labels, verbose=0)
            _acc = _compute_acc(_images, _labels, _input_length)
            _val_losses.append(_loss)
            _val_accuracy.append(_acc)
        return np.mean(_val_losses), np.mean(_val_accuracy)

    def _compute_acc(_images, _labels, _input_length):
        """
        :param _images: a batch of image, [samples, w, h, c]
        :param _labels:
        :param _input_length:
        :return: acc
        """
        _y_pred = base_model.predict_on_batch(x=_images)
        # print(_y_pred)  # (64, 9, 37)
        _decoded_dense, _ = tf.keras.backend.ctc_decode(_y_pred, _input_length, greedy=True,
                                                        beam_width=5, top_paths=1)
        _error_count = 0
        for pred, real in zip(_decoded_dense[0], _labels):
            str_real = ''.join([config.characters[x] for x in real if x != -1])
            str_pred = ''.join([config.characters[x] for x in pred if x != -1])
            # print(str_real, str_pred)
            if str_pred != str_real:
                _error_count += 1
        _acc = (len(_labels) - _error_count) / len(_labels)
        return _acc

    # start training progress
    for epoch in range(start_epoch, config.epochs):
        for batch, data in enumerate(train_dataset):
            images, labels = data
            input_length = np.array(np.ones(len(images)) * int(seq_step_len))
            label_length = np.array(np.ones(len(images)) * config.max_seq_len)
            train_loss = model.train_on_batch(x=[images, labels, input_length, label_length], y=labels)

            # logging result every 10-batch. (about 10 * batch_size images)
            if batch % 10 == 0:
                train_acc = _compute_acc(images, labels, input_length)
                val_loss, val_acc = _validation()
                print('Epoch: [{epoch}/{epochs}], iter: {batch}, train_loss: {train_loss}, train_acc: {train_acc}, '
                      'val_loss: {val_loss}, val_acc: {val_acc}'.format(epoch=epoch + 1, epochs=config.epochs,
                                                                        batch=batch, train_loss=train_loss,
                                                                        train_acc=train_acc, val_loss=val_loss,
                                                                        val_acc=val_acc))

        ckpt_path = os.path.join(CHECKPOINT_DIR, 'CRNNORC-{epoch}'.format(epoch=epoch + 1))
        model.save_weights(ckpt_path)
        base_model.save(os.path.join(SVAED_MODEL_DIR, '{}_model.h5'.format(config.dataset)))


if __name__ == '__main__':
    train()

