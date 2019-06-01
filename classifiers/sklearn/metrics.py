#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import metrics

def f1_score(y_true, y_pred):
    if y_pred.sum() == 0:  # no prediction
        return 0.0
    return metrics.f1_score(y_true, y_pred)


def precision_score(y_true, y_pred):
    if y_pred.sum() == 0:  # no prediction
        return 0.0
    return metrics.precision_score(y_true, y_pred)


def recall_score(y_true, y_pred):
    if y_true.sum() == 0:  # no correct labels
        return 1.0
    return metrics.recall_score(y_true, y_pred)
