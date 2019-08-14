#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: _defined
@Time:  2019/8/9 14:17
@Description: 
"""
import os
import yaml
import string

__all__ = ['config', 'DataMode']


class DataMode(object):
    """
    mark train, test and validation data
    """
    Train = 'train'
    Test = 'test'
    Val = 'val'


class Config(object):
    """
    project config
    """

    def __init__(self):
        self.__load_yaml()

    def reload_config(self):
        self.__load_yaml()

    def __load_yaml(self):
        yaml_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(yaml_path, encoding='utf-8') as f:
            yaml_cont = f.read()
        cf = yaml.load(yaml_cont, Loader=yaml.SafeLoader)
        self.__model_args = cf.get('model')
        self.cnn = self.__model_args.get('cnn', 'CNN5')
        self.rnn = self.__model_args.get('rnn', 'BiLSTM')
        self.use_gpu = self.__model_args.get('use_gpu', False)
        self.image_width = self.__model_args.get('image_width', 150)
        self.image_height = self.__model_args.get('image_height', 50)
        self.channel = self.__model_args.get('channel', 3)
        self.resize = self.__model_args.get('resize', [self.image_width, self.image_height])
        self.replace_transparent = self.__model_args.get('replace_transparent', True)
        self.standardization = self.__model_args.get('standardization', False)
        self.smooth = self.__model_args.get('smooth', -1)
        self.binaryzation = self.__model_args.get('binaryzation', -1)
        self.blur = self.__model_args.get('blur', -1)
        self.dataset = self.__model_args.get('dataset', 'dataset')
        self.train_path = self.__model_args.get('train_path', './dataset')
        self.test_path = self.__model_args.get('test_path', './dataset')
        self.val_path = self.__model_args.get('val_path', './dataset')
        self.n_class = self.__model_args.get('n_class', 37)
        self.max_seq_len = self.__model_args.get('max_seq_len', 4)
        self.epochs = self.__model_args.get('epochs', 500)
        self.batch_size = self.__model_args.get('batch_size', 64)
        self.end_acc = self.__model_args.get('end_acc', 50)
        self.characters = self.__model_args.get('characters', list(string.ascii_letters + string.digits) + [''])


config = Config()
