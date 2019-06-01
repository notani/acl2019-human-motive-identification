#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import product
from os import path
import logging
import numpy as np

def init_logger(name='logger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)
    return logger


def print_instances(path_output, df, y_true, y_pred, labels):
    cols = ['text']
    for label in labels:
        cols.append('gold_' + label)
        cols.append('pred_' + label)

    write_header = False if path.isfile(path_output) else True
    with open(path_output, 'a') as f:
        if write_header:
            f.write('\t'.join(cols) + '\n')
        for i, (_, row) in enumerate(df.iterrows()):
            buff = [row['text']]
            for c, label in enumerate(labels):
                buff.append(str(y_true[i, c]))
                buff.append(str(y_pred[i, c]))
            f.write('\t'.join(buff) + '\n')


def expand_param_grid(param_grid):
    params = []
    for i, key in enumerate(param_grid.keys()):
        buff =[]
        for val in param_grid[key]:
            buff.append((key, val))
        params.append(buff)
    for comb in product(*params):
        yield {k: v for k, v in comb}


def calc_mean_and_std_score(scores, label=None):
    if not isinstance(scores, dict):
        raise ValueError
    for tag, values in scores.items():
        line = '' if label is None else label + ': '
        if tag != '_':
            line += '[{}] '.format(tag)
        line += '{:.3f} (+/-{:.3f})'.format(np.mean(values), np.std(values))
        yield line

    
