#!/usr/bin/env python
# -*- coding: utf-8 -*-

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import ArrayField
from allennlp.data.fields import LabelField
from allennlp.data.fields import MultiLabelField
from allennlp.data.fields import TextField
from allennlp.data.fields import MetadataField
from typing import Dict
from typing import Iterator
from typing import List
from typing import Sequence
from allennlp.data.tokenizers import Token
from csv import DictReader
import numpy as np

from classifiers.constants import DOMAIN
from classifiers.constants import MOTIVES
from classifiers.constants import NUM_MOTIVES

@DatasetReader.register('semeval2016')
class SemEval2016Reader(DatasetReader):
    """DatasetReader for SemEval 2016 datasets"""

    def __init__(self, target_domain: str,
                 out_domain_weight: float = 1.0,
                 multitask: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        # Set a token indexer
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.num_instances = None
        self.label_counts = None
        self.label_prior = None
        self.label_prior_aspect = None
        self.target_domain = target_domain
        self.out_domain_weight = out_domain_weight
        self.multitask = multitask

    def text_to_instance(self, tokens: List[Token],
                         labels: Sequence[int] = None,
                         labels_aspect: Sequence[int] = None,
                         domain: str = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)  # sentence and indexer
        fields = {'sentence': sentence_field}

        if domain != None:
            fields['domain'] = LabelField(label=DOMAIN.index(domain),
                                          label_namespace='domain-labels',
                                          skip_indexing=True)
            in_domain = domain == self.target_domain
            fields['sample_weight'] = ArrayField(
                np.array([1.0 if in_domain else self.out_domain_weight]))

        if labels:
            label_field = MultiLabelField(labels=labels,
                                          label_namespace='motive-labels',
                                          skip_indexing=True,
                                          num_labels=NUM_MOTIVES+1)
            fields['labels'] = label_field

        if self.multitask and labels_aspect:
            num_aspects = max([c.shape[0]
                               for c in self.label_counts_aspect.values()])
            label_field = MultiLabelField(labels=labels_aspect,
                                          label_namespace='aspect-labels',
                                          skip_indexing=True,
                                          num_labels=num_aspects)
            fields['labels_aspect'] = label_field

        fields['metadata'] = MetadataField({
            'label_prior': self.label_prior,
            'label_prior_aspect': self.label_prior_aspect[domain]
        })

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        if '@@@' in file_path:
            # Example: 'res:data/res.v2/0/train.tsv@@@lap:data/lap.v2/0/train.tsv'
            file_paths = {item.split(':', 1)[0]: item.split(':', 1)[1]
                         for item in file_path.split('@@@')}
        else:  # Single domain
            file_paths = {self.target_domain: file_path}

        self.num_instances = 0
        self.aspects = {}

        # Indexing aspects
        for domain, file_path in file_paths.items():
            with open(file_path) as f:
                header = f.readline().split('\t')
            aspects = [asp.split('@')[0]
                       for asp in header[4:header.index('words')]]
            aspects = sorted(list(set(aspects)))
            self.aspects[domain] = sorted(list(set(aspects)))

        # Class distribution
        self.label_counts = np.zeros(NUM_MOTIVES)
        self.label_counts_aspect = {}
        for domain, file_path in file_paths.items():
            aspects = self.aspects[domain]
            self.label_counts_aspect[domain] = np.zeros(len(aspects))
            with open(file_path) as f:
                for row in DictReader(f, delimiter='\t'):
                    self.num_instances += 1
                    for i, motive in enumerate(MOTIVES):
                        if row[motive] == '1':
                            self.label_counts[i] += 1
                    for i, aspect in enumerate(aspects):
                        for polarity in ['positive', 'neutral', 'negative']:
                            try:
                                if row[aspect + '@' + polarity] == '1':
                                    self.label_counts_aspect[domain][i] += 1
                                    break
                            except KeyError:
                                pass

        self.label_prior = (self.num_instances - self.label_counts) \
                           / self.label_counts
        self.label_prior_aspect = {}
        for domain, counts in self.label_counts_aspect.items():
            idx = np.nonzero(counts)
            self.label_prior_aspect[domain] = np.ones_like(counts)
            self.label_prior_aspect[domain][idx] \
                = (self.num_instances - counts[idx]) / counts[idx]

        # Convert instances
        for domain, file_path in file_paths.items():
            aspects = self.aspects[domain]
            with open(file_path) as f:
                for row in DictReader(f, delimiter='\t'):
                    sentence = row['text_tkn'].lower().split()

                    labels = [MOTIVES.index(label) for label in MOTIVES
                              if row[label] == '1']
                    if len(labels) == 0:
                        labels.append(NUM_MOTIVES)

                    labels_aspect = []
                    for i, aspect in enumerate(aspects):
                        for polarity in ['positive', 'neutral', 'negative']:
                            try:
                                if row[aspect + '@' + polarity] == '1':
                                    labels_aspect.append(i)
                                    break
                            except KeyError:
                                pass

                    if len(labels_aspect) == 0:
                        import pdb
                        pdb.set_trace()
                    yield self.text_to_instance(
                        [Token(word) for word in sentence],
                        labels, labels_aspect, domain)
