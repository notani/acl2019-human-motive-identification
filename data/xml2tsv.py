#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from itertools import chain
from os import path
from tqdm import tqdm
import argparse
import json
import logging
import xml.etree.ElementTree as ET


verbose = False
logger = None

label_set = set()


def init_logger(name='logger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)
    return logger


def read_sentences(root):
    global label_set
    for review in root.iter('Review'):
        rids, texts, labels = [], [], []
        for sentence in review.iter('sentence'):
            rids.append(sentence.get('id'))
            texts.append(sentence[0].text)
            labels.append([e.get('category') + '@' + e.get('polarity')
                           for e in sentence.iter('Opinion')])
            for label in labels[-1]:
                label_set.add(label)
        yield rids, texts, labels


def main(args):
    global verbose
    verbose = args.verbose

    if verbose:
        logger.info('Read ' + args.path_input)
    tree = ET.parse(args.path_input)
    root = tree.getroot()

    if verbose:
        logger.info('Write to ' + args.path_output)

    reviews = list(read_sentences(root))
    indexer = {label: i for i, label in enumerate(sorted(list(label_set)))}
    n_labels = len(indexer)

    with open(args.path_output, 'w') as f:
        cols = ['id', 'context1', 'text', 'context2']
        cols += [aspect for aspect, _
                 in sorted(indexer.items(), key=lambda t: t[1])]
        f.write('\t'.join(cols) + '\n')
        for review in reviews:
            rids, texts, labels = review
            for i, (rid, text, label) in enumerate(zip(rids, texts, labels)):
                if args.context_window >= 0:
                    context1 = ' '.join(texts[i-args.context_window:i])
                    context2 = ' '.join(texts[i+1:i+1+args.context_window])
                else:
                    context1 = ' '.join(texts[i-args.context_window:i])
                    context2 = ' '.join(texts[i+1:i+1+args.context_window])
                buff = [rid, context1, text, context2]
                label_vec = [0 for _ in range(n_labels)]
                for l in label:
                    label_vec[indexer[l]] = 1
                buff += [str(i) for i in label_vec]
                f.write('\t'.join(buff) + '\n')

    return 0


if __name__ == '__main__':
    logger = init_logger('xml2tsv')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to input file')
    parser.add_argument('-o', '--output', dest='path_output',
                        required=True,
                        help='path to output file')
    parser.add_argument('--window', dest='context_window', type=int, default=-1)
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
