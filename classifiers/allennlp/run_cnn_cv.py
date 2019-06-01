#!/usr/bin/env python
# -*- coding: utf-8 -*-

from allennlp.commands import train
from copy import copy
import argparse
import json
import os

from classifiers.allennlp.cv_runner import CVRunner
from classifiers.allennlp.cv_runner import default_param_grid


class CnnCVRunner(CVRunner):
    def get_overrides(self, param):
        out_enc = param['cnn_num_filters'] * len(param['cnn_ngram_filter_sizes'])
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
                'encoder': {
                    'num_filters': param['cnn_num_filters'],
                    'ngram_filter_sizes': param['cnn_ngram_filter_sizes']
                },
                'classifier_feedforward': {
                    'input_dim': out_enc,
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
        if self.multitask_learning:
            n_hid = param['num_layers_asp']
            fnn = {
                "input_dim": out_enc,
                "num_layers": n_hid + 1,
                "activations":  [act for _ in range(n_hid)] + ['linear'],
                "dropout": param['dropout']
            }
            out_dims = {'res': 12, 'lap': 81}
            target_domain = self.conf['dataset_reader']['target_domain']
            for dom in ['res', 'lap']:
                if self.transfer_learning or dom == target_domain:
                    fnn['hidden_dims'] = [param['hidden_dims_asp']
                                          for _ in range(n_hid)] + [out_dims[dom]]
                    overrides['model']['classifier_aspect_' + dom] = copy(fnn)
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
    param_grid['cnn_num_filters'] = [50, 100]
    param_grid['cnn_ngram_filter_sizes'] = [[3, 4, 5]]
    print('Param grid: {}'.format(json.dumps(param_grid, indent=4)))

    runner = CnnCVRunner(args, param_grid)
    runner.run()

    with open(os.path.join(args.serialization_dir, 'param_grid.json'), 'w') as f:
        f.write((json.dumps(param_grid, indent=4)) + '\n')
