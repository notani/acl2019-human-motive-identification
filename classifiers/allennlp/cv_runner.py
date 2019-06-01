#!/usr/bin/env python
# -*- coding: utf-8 -*-

from allennlp.common.util import import_submodules
import json
import numpy as np
import os
import time

from classifiers.utils import expand_param_grid

class CVRunner():
    def __init__(self, args, param_grid, n_fold=3):
        self.args = args
        self.param_path = args.param_path
        self.serialization_dir = args.serialization_dir
        self.param_grid = param_grid
        self.n_fold = 3

        # Read configurations
        with open(self.param_path) as f:
            self.conf = json.load(f)

        self.transfer_learning = '@@@' in self.conf['train_data_path']
        if self.transfer_learning:
            self.param_grid['out_domain_weight'] = [0.125, 0.25, 0.5]
            print('Conduct transfer learning: out_domain_weight={}'.format(
                self.param_grid['out_domain_weight']))
        self.multitask_learning = self.conf['dataset_reader'].get('multitask', False)
        if self.multitask_learning:
            self.param_grid['num_layers_asp'] = [1]
            self.param_grid['hidden_dims_asp'] = [50, 100, 200]

    def get_overrides(self, param):
        raise NotImplementedError

    def one_iteration(self, fold):
        args = self.args
        for split in ['train', 'validation', 'test']:
            col = split + '_data_path'
            self.conf[col] = self.conf[col].replace('/0/', '/{}/'.format(fold))

        serialization_dir = os.path.join(self.serialization_dir, str(fold))

        param_cands = list(expand_param_grid(self.param_grid))
        n = len(param_cands)
        print('Total: {}'.format(n))

        buff = []
        for progress, param in enumerate(param_cands, start=1):
            if progress % 5 == 0:
                print('Progress: {}/{}'.format(progress, n))
            overrides = self.get_overrides(param)
            overrides['trainer'] = {
                'num_epochs': 10,
                'patience': None,
                'num_serialized_models_to_keep': 1
            }
            result = args.func(self.param_path,
                               serialization_dir,
                               json.dumps(overrides),
                               args.file_friendly_logging,
                               args.recover,
                               args.force)
            # Retrieve scores
            with open(os.path.join(serialization_dir, 'metrics.json')) as f:
                metrics = json.load(f)
                score_validation = metrics['validation_macro_f1']
                score_train = metrics['training_macro_f1']
            buff.append((score_validation, score_train, param))

        # Save scores and parameters
        buff = sorted(buff, key=lambda t: (t[0], t[1]), reverse=True)
        filename = os.path.join(self.serialization_dir, 'grid_search-{}.tsv'.format(fold))
        with open(filename, 'w') as f:
            for item in buff:
                f.write('{}\t{}\t{}\n'.format(*item))

        # Best parameters
        param = buff[0][-1]
        overrides = self.get_overrides(param)
        overrides['train_data_path'] = self.conf['train_data_path'].replace(
            'train.tsv', 'train+valid.tsv')
        overrides['validation_data_path'] = self.conf['validation_data_path'].replace(
            'valid.tsv', 'test.tsv')
        overrides['test_data_path'] = None
        overrides['trainer'] = {'patience': None}
        result = args.func(self.param_path,
                           serialization_dir,
                           json.dumps(overrides),
                           args.file_friendly_logging,
                           args.recover,
                           args.force)
        print('Done')


    def run(self, n_folds=3):
        args = self.args
        start = time.time()

        for package_name in getattr(args, 'include_package', ()):
            import_submodules(package_name)

            for fold in range(n_folds):
                print('[Fold-{}]'.format(fold+1))
                self.one_iteration(fold=fold)
                print('Elapsed time: {}'.format(get_elepsed_time(start)))

            # Report final results
            prec, rec, f1 = [], [], []
            for fold in range(n_folds):
                filename = os.path.join(self.serialization_dir,
                                        str(fold) + '/metrics.json')
                with open(filename) as f:
                    metrics = json.load(f)
                    prec.append(metrics['validation__macro_precision'])
                    rec.append(metrics['validation__macro_recall'])
                    f1.append(metrics['validation_macro_f1'])

            with open(os.path.join(self.serialization_dir, 'result.csv'), 'w') as f:
                print('|P|R|F1|')
                f.write('P,R,F1\n')
                numbers = ['{:.3f}({:.3f})'.format(np.mean(l), np.std(l))
                           for l in [prec, rec, f1]]
                print('|' + '|'.join(numbers) + '|')
                numbers = ['{}({})'.format(np.mean(l), np.std(l))
                           for l in [prec, rec, f1]]
                f.write(','.join(numbers) + '\n')

        print('Elapsed time: {}'.format(get_elepsed_time(start)))

def get_elepsed_time(start):
    process_time = int(time.time() - start)
    h = process_time // (60 * 60)
    m = (process_time % (60 * 60)) // 60
    s = (process_time % 60)
    return '{}:{}:{}'.format(h, m, s)


default_param_grid = {
    'num_layers': [1],
    'hidden_dims': [50, 100, 200],
    'activations': ['relu', 'tanh'],
    'dropout': [0.5],
    'weight_decay': [0, 0.1]
}
