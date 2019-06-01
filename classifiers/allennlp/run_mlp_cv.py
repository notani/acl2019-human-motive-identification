#!/usr/bin/env python
# -*- coding: utf-8 -*-

from allennlp.commands import train
from copy import copy
import argparse
import json
import os

from classifiers.allennlp.cv_runner import CVRunner
from classifiers.allennlp.cv_runner import default_param_grid


class MlpCVRunner(CVRunner):
    def get_overrides(self, param):
        n_hid = param['num_layers']
        act = param['activations']
        overrides = {
            'train_data_path': self.conf['train_data_path'],
            'validation_data_path': self.conf['validation_data_path'],
            'test_data_path': self.conf['test_data_path'],
            'dataset_reader': {
                'out_domain_weight': param.get('out_domain_weight', None)
            },
            'model': {
                'classifier_feedforward': {
                    'num_layers': n_hid + 1,
                    'hidden_dims': [param['hidden_dims'] for _ in range(n_hid)] + [6],
                    'activations': [act for _ in range(n_hid)] + ['linear'],
                    'dropout': param['dropout']
                },
            },
            'trainer': {'optimizer': {
                'weight_decay': param.get('weight_decay', 0)
            }}
        }
        return overrides


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('param_path',
                        type=str,
                        help='path to parameter file describing the model to be trained')
    parser.add_argument('-s', '--serialization-dir',
                        required=True,
                        type=str,
                        help='directory in which to save the model and its logs')

    parser.add_argument('-r', '--recover',
                        action='store_true',
                        default=False,
                        help='recover training from the state in serialization_dir')

    parser.add_argument('-f', '--force',
                        action='store_true',
                        required=False,
                        help='overwrite the output directory if it exists')

    parser.add_argument('-o', '--overrides',
                        type=str,
                        default="",
                        help='a JSON structure used to override the experiment configuration')

    parser.add_argument('--file-friendly-logging',
                        action='store_true',
                        default=False,
                        help='outputs tqdm status on separate lines and slows tqdm refresh rate')
    parser.add_argument('--include-package',
                        type=str,
                        action='append',
                        default=[],
                        help='additional packages to include')
    parser.set_defaults(func=train.train_model_from_file)

    args = parser.parse_args()

    param_grid = default_param_grid
    print('Param grid: {}'.format(json.dumps(param_grid, indent=4)))

    runner = MlpCVRunner(args, param_grid)
    runner.run()

    with open(os.path.join(args.serialization_dir, 'param_grid.json'), 'w') as f:
        f.write((json.dumps(param_grid, indent=4)) + '\n')
