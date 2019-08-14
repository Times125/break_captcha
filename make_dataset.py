#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: _defined
@Time:  2019/5/25 17:27
@Description:
    This script is use for converting your train, test and val dataset to tfrecords format.
"""
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from PIL import (Image, ImageFilter)
from settings import (config, DataMode)
from logger import event_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

_RANDOM_SEED = 0
TFRECORDS_DIR = './dataset/{}'.format(config.dataset)
# train,test,val 存放的路径
TRAINS_PATH = config.train_path
TEST_PATH = config.test_path
VAL_PATH = config.val_path

if not os.path.exists(TFRECORDS_DIR):
    os.makedirs(TFRECORDS_DIR)


def _per_image_standardization(image):
    """
    :param image: image numpy array
    :return:
    """
    num_compare = 1
    for dim in image.shape:
        num_compare = np.multiply(num_compare, dim)
    _standardization = (image - np.mean(image)) / max(np.std(image), 1 / num_compare)
    return _standardization


def _per_image_binaryzation(image, value):
    """
    :param image: image numpy array
    :param value: threshold
    :return:
    """
    ret, _binarization = cv2.threshold(image, value, 255, cv2.THRESH_BINARY)
    return _binarization


def _per_image_median_blur(image, value):
    """
    :param image: image numpy array
    :param value: ksize
    :return:
    """
    if not value:
        return image
    value = value + 1 if value % 2 == 0 else value
    _smooth = cv2.medianBlur(image, value)
    return _smooth


def _per_image_gaussian_blur(image, value):
    """
    :param image: image numpy array
    :param value: ksize
    :return:
    """
    if not value:
        return image
    value = value + 1 if value % 2 == 0 else value
    _blur = cv2.GaussianBlur(image, (value, value), 0)
    return _blur


def _process_image(image):
    mode = image.split()
    # process images with color channels greater than 3  or 'P' mode image
    if len(mode) >= 3 and config.replace_transparent or image.mode in ['p', 'P']:
        image = image.convert('RGB')
    # channel equals 1 means converting image mode to 'L'
    if config.channel == 1:
        image = image.convert("L")
    image = image.resize((config.resize[0], config.resize[1]), Image.LANCZOS)
    return image


def _image(path):
    images = Image.open(path)
    image = _process_image(images)
    im = np.array(image)

    if config.binaryzation > 0:
        im = _per_image_binaryzation(im, config.binaryzation)
    if config.smooth > 1:
        im = _per_image_median_blur(im, config.smooth)
    if config.blur > 1:
        im = _per_image_gaussian_blur(im, config.blur)
    if config.standardization:
        im = _per_image_standardization(im)
    im = im.swapaxes(0, 1)
    return np.array((im[:, :, np.newaxis] if config.channel == 1 else im[:, :]).astype(np.float32) / 255)


def _dataset_exists(dataset_dir):
    for split_name in ['train', 'test', 'val']:
        output_filename = os.path.join(dataset_dir, "{}_{}.tfrecords".format(config.dataset, split_name))
        if not tf.io.gfile.exists(output_filename):
            return False
    return True


def _one_hot_label(text):
    character_set = config.characters
    labels = []
    for char in text.lower():
        if char not in character_set:
            labels.append(character_set.index(''))
        else:
            labels.append(character_set.index(char))
    if len(labels) < config.max_seq_len:
        for i in range(config.max_seq_len - len(labels)):
            labels.append(character_set.index(''))
    return labels


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values.tostring()]))


def image_to_tfrecords(image_data, label):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'image': bytes_feature(image_data),
                'label': bytes_feature(label),
            }))


def _convert_dataset(path, mode):
    output_filename = os.path.join(TFRECORDS_DIR, "{}_{}.tfrecords".format(config.dataset, mode))
    with tf.io.TFRecordWriter(output_filename) as writer:
        all_files = os.listdir(path)
        np.random.shuffle(all_files)
        pbar = tqdm(all_files)
        for i, file_name in enumerate(pbar):
            try:
                pbar.set_description('Processing %s' % file_name)
                image_data = _image(os.path.join(path, file_name))
                # file name like "abcdef_md5value.jpg" or 'abcdef.jpg'
                text = file_name.split('_')[0] if '_' in file_name else file_name.split('.')[0]
                labels = _one_hot_label(text)
                example = image_to_tfrecords(image_data, np.array(labels).astype(np.int32))
                writer.write(example.SerializeToString())
                pbar.set_description('[Processing dataset %s] [filename: %s]' % (mode, os.path.join(path, file_name)))
            except IOError as e:
                print('could not read:', os.path.join(path, file_name))
                print('error:', e)
                print('skip it \n')


def run():
    if _dataset_exists(TFRECORDS_DIR):
        print('Exists!')
    else:
        _convert_dataset(TRAINS_PATH, DataMode.Train)
        _convert_dataset(TEST_PATH, DataMode.Test)
        _convert_dataset(VAL_PATH, DataMode.Val)
        event_logger.info("convert data to tfrecord. Done!")


if __name__ == '__main__':
    run()
    # path = TRAINS_PATH
    # all_files = os.listdir(path)
    # for i, file_name in enumerate(all_files):
    #     # file name like "abcdef_md5value.jpg" or 'abcdef.jpg'
    #     text = file_name.split('_')[0] if '_' in file_name else file_name.split('.')[0]
    #     labels = _one_hot_label(text)
    #     print(labels, ' ', text)
