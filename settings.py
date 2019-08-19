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
        _cf = yaml.load(yaml_cont, Loader=yaml.SafeLoader)
        self.__model_args = _cf.get('model')
        _cnn = self.__model_args.get('cnn', 'CNN5')
        self.cnn_type = _cnn if _cnn in ['CNN5', 'ResNet50'] else 'CNN5'
        _rnn = self.__model_args.get('rnn', 'BiLSTM')
        self.rnn_type = _rnn if _rnn in ['BiGRU', 'BiLSTM'] else 'BiLSTM'
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
        self.end_acc = self.__model_args.get('end_acc', 0.5)
        self.end_cost = self.__model_args.get('end_cost', 1)
        self.lr = self.__model_args.get('learning_rate', 1e-4)
        self.l2 = self.__model_args.get('regularizer_l2', 0.01)
        self.rnn_units = self.__model_args.get('rnn_units', 64)
        self.ctc_greedy = self.__model_args.get('ctc_greedy', True)
        self.beam_width = self.__model_args.get('beam_width', 10)
        self.top_paths = self.__model_args.get('top_paths', 1)
        self.ctc_merge_repeated = self.__model_args.get('ctc_merge_repeated', True)
        self.time_major = self.__model_args.get('time_major', True)
        self.preprocess_collapse_repeated = self.__model_args.get('preprocess_collapse_repeated', False)
        self.decode_merge_repeated = self.__model_args.get('ctc_decode_merge_repeated', False)
        self.characters = self.__model_args.get('characters', list(string.ascii_letters + string.digits) + [''])


config = Config()
