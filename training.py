#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: _defined
@Time:  2019/8/12 19:02
@Description: 
"""
import os
import time
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.ctc_ops import (ctc_loss, ctc_beam_search_decoder)
from settings import (config, DataMode)
from DataLoader import DataLoader
from model import build_model
from logger import event_logger
from ctc_ops import ctc_decode

SVAED_MODEL_DIR = './savedModel/{}'.format(config.dataset)
if not os.path.exists(SVAED_MODEL_DIR):
    os.makedirs(SVAED_MODEL_DIR)

CHECKPOINT_DIR = './checkpoints/{}'.format(config.dataset)
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

TENSORBOARD_DIR = './tensorboard/{}'.format(config.dataset)
if not os.path.exists(TENSORBOARD_DIR):
    os.makedirs(TENSORBOARD_DIR)


def train():
    """
    train model
    :return:
    """
    model, base_model, seq_step_len = build_model()
    print('input lengths ', seq_step_len, 'label length', config.max_seq_len)
    train_dataset = DataLoader(DataMode.Train).load_batch_from_tfrecords()
    val_dataset = DataLoader(DataMode.Val).load_batch_from_tfrecords()

    train_summary_writer = tf.summary.create_file_writer(os.path.join(TENSORBOARD_DIR, 'trainLogs'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(TENSORBOARD_DIR, 'valLogs'))

    latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    start_epoch = 0
    if latest_ckpt:
        start_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
        model.load_weights(latest_ckpt)
        event_logger.info('model resumed from: {}, start at epoch: {}'.format(latest_ckpt, start_epoch))
    else:
        event_logger.info('passing resume since weights not there. training from scratch')

    def _validation():
        """
        validate the model's acc
        :return: loss and acc
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
        :param _images: a batch of images, [samples, w, h, c]
        :param _labels:
        :param _input_length:
        :return: acc
        """
        _y_pred = base_model.predict_on_batch(x=_images)
        # print(_y_pred)  # (64, 9, 37)
        _decoded_dense, _ = ctc_decode(_y_pred, _input_length,
                                       greedy=config.ctc_greedy,
                                       beam_width=config.beam_width,
                                       top_paths=config.top_paths,
                                       merge_repeated=config.decode_merge_repeated)
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
        train_acc_avg = []
        train_loss_avg = []
        start = time.time()
        for batch, data in enumerate(train_dataset):
            images, labels = data
            input_length = np.array(np.ones(len(images)) * int(seq_step_len))
            label_length = np.array(np.ones(len(images)) * config.max_seq_len)
            train_loss = model.train_on_batch(x=[images, labels, input_length, label_length], y=labels)
            train_acc = _compute_acc(images, labels, input_length)
            train_acc_avg.append(train_acc)
            train_loss_avg.append(train_loss)
        train_loss = np.mean(train_loss_avg)
        train_acc = np.mean(train_acc_avg)
        val_loss, val_acc = _validation()
        # write train and val logs
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss, step=epoch)
            tf.summary.scalar('acc', train_acc, step=epoch)
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss, step=epoch)
            tf.summary.scalar('acc', val_acc, step=epoch)
        print('Epoch: [{epoch}/{epochs}], train_loss: {train_loss}, train_acc: {train_acc}, '
              'val_loss: {val_loss}, val_acc: {val_acc}, '
              'one epoch costs time: {time} s, learning rate: {lr}'.format(epoch=epoch + 1, epochs=config.epochs,
                                                                           train_loss=train_loss, train_acc=train_acc,
                                                                           val_loss=val_loss, val_acc=val_acc,
                                                                           time=time.time() - start, lr=config.lr))
        ckpt_path = os.path.join(CHECKPOINT_DIR, '{cnn}&{rnn}-{epoch}'.format(cnn=config.cnn_type,
                                                                              rnn=config.rnn_type,
                                                                              epoch=epoch + 1))
        model.save_weights(ckpt_path)
        if val_acc >= config.end_acc or val_loss <= config.end_cost:
            # tf.saved_model.save(base_model, os.path.join(SVAED_MODEL_DIR, '{name}_model.h5'.format(name=config.dataset)))
            base_model.save(os.path.join(SVAED_MODEL_DIR, '{name}_model.h5'.format(name=config.dataset)))
            break


def model_test():
    """
    test the model on test dataset
    :return:
    """
    test_dataloader = DataLoader(DataMode.Train)
    test_dataset = test_dataloader.load_all_from_tfreocrds()
    base_model = tf.keras.models.load_model(os.path.join(SVAED_MODEL_DIR, '{}_model.h5'.format(config.dataset)))
    error_text = []
    real_text = []
    _, _, time_step = build_model()
    error_count = 0
    for batch, data in enumerate(test_dataset):
        images, label = data
        # print(images.shape, label.shape)
        input_length = np.array(np.ones(1) * int(time_step))
        y_pred = base_model.predict(x=images[tf.newaxis, :, :, :])
        # print(y_pred.shape)  # (64, 9, 37)
        decoded_dense, _ = ctc_decode(y_pred, input_length,
                                      greedy=config.ctc_greedy,
                                      beam_width=config.beam_width,
                                      top_paths=config.top_paths,
                                      merge_repeated=config.decode_merge_repeated)

        str_real = ''.join([config.characters[x] for x in label if x != -1])
        str_pred = ''.join([config.characters[x] for x in decoded_dense[0][0] if x != -1])
        if str_pred != str_real:
            error_count += 1
            error_text.append(str_pred)
            real_text.append(str_real)

    test_accuracy = (test_dataloader.size - error_count) / test_dataloader.size
    print('test acc %f' % test_accuracy)
    for real, pred in zip(real_text, error_text):
        if len(pred) == 4:
            print('error pair: ', real, ' ', pred, )


if __name__ == '__main__':
    train()
    # model_test()
