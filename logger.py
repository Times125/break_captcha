#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:_defined
@Time:  2018/8/28 19:52
@Description: 
"""
import os
import logging
from logging import config as log_conf
from settings import config

__all__ = ["event_logger"]

abs_path = './LogDir/{}'.format(config.dataset)
if not os.path.exists(abs_path):
    os.makedirs(abs_path)
log_path = os.path.join(abs_path, '{}.log'.format(config.dataset))
log_config = {
    'version': 1.0,
    'formatters': {
        'detail': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': "%Y-%m-%d %H:%M:%S"
        },
        'simple': {
            'format': '%(name)s - %(levelname)s - %(message)s',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'detail'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'maxBytes': 1024 * 1024 * 5,
            'backupCount': 10,
            'filename': log_path,
            'level': 'INFO',
            'formatter': 'detail',
            'encoding': 'utf-8',
        },
    },
    'loggers': {
        'event_logger': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        },
        'other_logger': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
        },
        'db_logger': {
            'handlers': ['file'],
            'level': 'INFO',
        }
    }
}

log_conf.dictConfig(log_config)
event_logger = logging.getLogger("event_logger")
