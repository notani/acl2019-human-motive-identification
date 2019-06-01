#!/usr/bin/env python
# -*- coding: utf-8 -*-

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from classifiers.allennlp.metrics import MultiLabelF1Measure
from classifiers.constants import MOTIVES
from classifiers.constants import NUM_MOTIVES


@Model.register('lstm_classifier')
class LstmClassifier(Model):
    def __init__(self,
                 word_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary,
                 classifier_feedforward: FeedForward,
                 classifier_aspect_res: FeedForward = None,
                 classifier_aspect_lap: FeedForward = None) -> None:
        super().__init__(vocab)
        self.word_embedder = word_embedder
        self.encoder = encoder
        self.classifier_feedforward = classifier_feedforward
        self.classifier_aspect_res = classifier_aspect_res
        self.classifier_aspect_lap = classifier_aspect_lap

        self._macro_f1 = MultiLabelF1Measure(macro=True)
        self._f1_aspect_res = MultiLabelF1Measure(macro=True)
        self._f1_aspect_lap = MultiLabelF1Measure(macro=True)
        self._loss_func = F.binary_cross_entropy_with_logits

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None,
                labels_aspect: torch.Tensor = None,
                domain: torch.Tensor = None,
                sample_weight: torch.Tensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        pos_weight = None
        if 'label_prior' in metadata[0]:
            # prior on labels
            pos_weight = torch.from_numpy(metadata[0]['label_prior']).float()

        if 'label_prior_aspect' in metadata[0]:
            # prior on labels 
            pos_weight_aspect = [None for _ in range(2)]
            for dom_idx in range(2):
                try:
                    idx = int((domain==dom_idx).nonzero()[0])
                except IndexError:
                    continue
                w = metadata[idx]['label_prior_aspect']
                pos_weight_aspect[dom_idx] = torch.from_numpy(w).float()

        labels = labels[:, :NUM_MOTIVES]  # Delete ``NULL'' column


        embeddings = self.word_embedder(sentence)

        mask = get_text_field_mask(sentence)
        encoder_out = self.encoder(embeddings, mask)

        label_logits = self.classifier_feedforward(encoder_out)
        output = {'label_logits': label_logits}

        if labels is not None:
            label_ids = (label_logits.sign().long() + 1) / 2
            self._macro_f1(label_ids, labels)
            output['loss'] = self._loss_func(label_logits, labels.float(),
                                             weight=sample_weight,
                                             pos_weight=pos_weight)

        if labels_aspect is not None:
            components = [
                (self.classifier_aspect_res, self._f1_aspect_res),
                (self.classifier_aspect_lap, self._f1_aspect_lap)
            ]
            for dom_idx, (clf, f1) in enumerate(components):
                h = encoder_out[domain == dom_idx]  # encoded sentences
                if h.shape[0] == 0:  # no instance
                    continue
                aspect_logits = clf(h)
                aspects_pred = (aspect_logits.sign().long() + 1) / 2
                aspects = labels_aspect[domain==dom_idx][:, :aspect_logits.shape[1]]
                f1(aspects_pred, aspects)
                output['loss'] += self._loss_func(
                    aspect_logits, aspects.float(),
                    weight=sample_weight[domain == dom_idx],
                    pos_weight=pos_weight_aspect[dom_idx])

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        suffix = ['', '_res', '_lap']
        ret = {}
        for i, scores in enumerate([self._macro_f1.get_metric(reset),
                                    self._f1_aspect_res.get_metric(reset),
                                    self._f1_aspect_lap.get_metric(reset)]):
            for j, cat in enumerate(['precision', 'recall', 'f1']):
                col = '_macro_{cat}{suffix}'.format(cat=cat, suffix=suffix[i])
                ret[col] = scores[j]
            if scores[-1] is None:
                continue
            for cat in ['precision', 'recall', 'f1']:
                for idx in range(len(scores[-1][cat])):
                    col = '_{cat}_{idx}{suffix}'.format(cat=cat, idx=idx, suffix=suffix[i])
                    ret[col] = scores[-1][cat][idx]
        ret['macro_f1'] = ret['_macro_f1']
        del ret['_macro_f1']
            
        return ret

