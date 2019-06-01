"""Downloaded from https://raw.githubusercontent.com/joelgrus/kaggle-toxic-allennlp/master/toxic/training/metrics/multilabel_f1.py"""

from allennlp.training.metrics.metric import Metric
from typing import Optional
import numpy as np
import torch


@Metric.register("multilabel-f1")
class MultiLabelF1Measure(Metric):
    """
    Computes multilabel F1. Assumes that predictions are 0 or 1.
    """
    def __init__(self, macro: bool = False) -> None:
        self.macro = macro
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0

    def __call__(self,
                 predictions: torch.LongTensor,
                 gold_labels: torch.LongTensor):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of 0 and 1 predictions of shape (batch_size, ..., num_labels).
        gold_labels : ``torch.Tensor``, required.
            A tensor of 0 and 1 predictions of shape (batch_size, ..., num_labels).
        """
        if self.macro:
            self._true_positives += (predictions * gold_labels).sum(dim=0).numpy()
            self._false_positives += (predictions * (1 - gold_labels)).sum(dim=0).numpy()
            self._true_negatives += ((1 - predictions) * (1 - gold_labels)).sum(dim=0).numpy()
            self._false_negatives += ((1 - predictions) * gold_labels).sum(dim=0).numpy()
        else:
            self._true_positives += (predictions * gold_labels).sum().item()
            self._false_positives += (predictions * (1 - gold_labels)).sum().item()
            self._true_negatives += ((1 - predictions) * (1 - gold_labels)).sum().item()
            self._false_negatives += ((1 - predictions) * gold_labels).sum().item()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        predicted_positives = self._true_positives + self._false_positives
        actual_positives = self._true_positives + self._false_negatives

        if self.macro and not isinstance(self._true_positives, float):
            precision = np.zeros_like(self._true_positives)
            recall = np.zeros_like(self._true_positives)
            f1_measure = np.zeros_like(self._true_positives)
            valid = []
            for i in range(self._true_positives.shape[0]):  # for each class
                if actual_positives[i] == 0:
                    precision[i] = recall[i] = f1_measure[i] = -1
                    continue
                if predicted_positives[i] > 0:
                    precision[i] = self._true_positives[i] / predicted_positives[i]
                recall[i] = self._true_positives[i] / actual_positives[i]
                if precision[i] + recall[i] > 0:
                    f1_measure[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
                valid.append(i)
            dump = {'precision': precision,
                    'recall': recall,
                    'f1': f1_measure}
            precision = precision[valid].mean() if len(valid) > 0 else 0
            recall = recall[valid].mean() if len(valid) > 0 else 0
            f1_measure = f1_measure[valid].mean() if len(valid) > 0 else 0
        else:
            precision = self._true_positives / predicted_positives if predicted_positives > 0 else 0
            recall = self._true_positives / actual_positives if actual_positives > 0 else 0

            if precision + recall > 0:
                f1_measure = 2 * precision * recall / (precision + recall)
            else:
                f1_measure = 0
            dump = None

        if reset:
            self.reset()

        return precision, recall, f1_measure, dump

    def reset(self):
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0
