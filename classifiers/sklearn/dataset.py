#!/usr/bin/env python
# -*- coding: utf-8 -*-


from collections import defaultdict
from nltk.util import ngrams
from scipy import sparse
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd

from classifiers.utils import init_logger
from classifiers.constants import MOTIVES


MOTIVES_7 = [
    "Self-fulfill",
    "Appreciating Beauty",
    "Social Relaton",
    "Family",
    "Health",
    "Ambition & Ability",
    "Finance",
]

MOTIVES_6 = [
    "Self-fulfill",
    "Appreciating Beauty",
    "Social Relation",
    "Health",
    "Ambition & Ability",
    "Finance",
]


def count_words(texts, ngram=(1, 3)):
    """Count words"""
    counter = defaultdict(int)
    for text in texts:
        words = text.lower().split()
        for n in range(ngram[0], ngram[1] + 1):
            for item in ngrams(words, n):
                counter["@@@".join(item)] += 1
    return counter


def make_vocab(texts, ngram=(1, 3), min_freq=2, pad=False, unk=False):
    """Make vocabulary"""
    counter = count_words(texts, ngram=ngram)
    vocab = [
        w
        for w, v in reversed(sorted(counter.items(), key=lambda t: t[1]))
        if v >= min_freq
    ]
    indexer = {}
    if pad:
        indexer["<PAD>"] = len(indexer)
    if unk:
        indexer["<UNK>"] = len(indexer)
    for i, w in enumerate(vocab, start=len(indexer)):
        indexer[w] = i

    return indexer, counter


def text2ids(texts, indexer, ngram=(1, 3)):
    mat = lil_matrix((len(texts), len(indexer)), dtype=np.int8)
    for idx, text in enumerate(texts):
        words = text.lower().split()
        for n in range(ngram[0], ngram[1] + 1):
            for item in ngrams(words, n):
                try:
                    tid = indexer["@@@".join(item)]  # token ID
                except KeyError:  # OOV
                    if "<UNK>" not in indexer:
                        continue
                    tid = indexer["<UNK>"]
                mat[idx, tid] += 1
    return mat.tocsr()


def text2idseq(texts, indexer, maxlen=None):
    assert "<PAD>" in indexer.keys()
    assert "<UNK>" in indexer.keys()

    seqs = [[] for text in texts]
    for idx, text in enumerate(texts):
        for word in text.lower().split():
            seqs[idx].append(indexer.get(word, indexer["<UNK>"]))

    if maxlen is None:
        maxlen = max([len(seq) for seq in seqs])

    for idx, seq in enumerate(seqs):
        for _ in range(maxlen - len(seq)):
            seqs[idx].append(indexer["<PAD>"])

    return np.array(seqs)


class Dataset:
    def __init__(
        self, path_input, verbose=False, logger=init_logger("Dataset")
    ):
        self.verbose = verbose
        self.logger = logger

        # Read files
        self.df = {split: {} for split in path_input.keys()}
        for split, filename in path_input.items():
            if isinstance(filename, str):
                self.df[split]["_"] = pd.read_csv(filename, delimiter='\t')
            elif isinstance(filename, dict):
                for tag, filename_ in filename.items():
                    self.df[split][tag] = pd.read_csv(filename_, delimiter='\t')
            if self.verbose:
                self.logger.info(
                    "Read {} items from {}".format(
                        len(self.df[split]), filename
                    )
                )

    def ngram_encoder(
        self,
        ngram,
        tfidf=False,
        df_unlabeled=None,
        return_df=False,
        return_idx=True,
    ):
        """Iterate data"""
        indices = {}
        # Feature extraction
        df_train = pd.concat(
            self.df["train"].values(), ignore_index=True, sort=False
        )
        indexer, counter = make_vocab(df_train["text_tkn"].values, ngram=ngram)
        X, y = {}, {}
        for split in ["train", "valid"]:
            X[split] = {
                tag: text2ids(df_["text_tkn"].values, indexer, ngram=ngram)
                for tag, df_ in self.df[split].items()
            }
            y[split] = {
                tag: df_[MOTIVES].values.astype(int)
                for tag, df_ in self.df[split].items()
            }
            if "_" not in X[split]:  # combine
                if return_idx:
                    indices[split] = {}  # record indices
                    offset = 0
                    for tag in X[split].keys():
                        n = X[split][tag].shape[0]
                        indices[split][tag] = np.arange(offset, offset + n)
                        offset += n
                X[split]["_"] = sparse.vstack(X[split].values())
                y[split]["_"] = np.vstack(y[split].values())

        if df_unlabeled is not None:
            X_unlabeled = text2ids(
                df_unlabeled["text_tkn"].values, indexer, ngram=ngram
            )

        if tfidf:
            transformer = TfidfTransformer().fit(X["train"]["_"])
            for split in ["train", "valid"]:
                X[split] = {
                    tag: transformer.transform(X_)
                    for tag, X_ in X[split].items()
                }
            if df_unlabeled is not None:
                X_unlabeled = transformer.transform(X_unlabeled)

        # Feature extraction (train+valid)
        self.df["tv"] = {
            tag: pd.concat(
                (self.df["train"][tag], self.df["valid"][tag]),
                ignore_index=True,
            )
            for tag in self.df["train"].keys()
        }
        df_tv = pd.concat(
            self.df["tv"].values(), ignore_index=True, sort=False
        )
        indexer, counter = make_vocab(df_tv["text_tkn"].values, ngram=ngram)
        for split in ["tv", "test"]:
            X[split] = {
                tag: text2ids(df_["text_tkn"].values, indexer, ngram=ngram)
                for tag, df_ in self.df[split].items()
            }
            y[split] = {
                tag: df_[MOTIVES].values.astype(int)
                for tag, df_ in self.df[split].items()
            }
            if "_" not in X[split]:  # combine
                if return_idx:
                    indices[split] = {}  # record indices
                    offset = 0
                    for tag in X[split].keys():
                        n = X[split][tag].shape[0]
                        indices[split][tag] = np.arange(offset, offset + n)
                        offset += n
                X[split]["_"] = sparse.vstack(X[split].values())
                y[split]["_"] = np.vstack(y[split].values())

        if df_unlabeled is not None:
            X_unlabeled_tv = text2ids(
                df_unlabeled["text_tkn"].values, indexer, ngram=ngram
            )

        if tfidf:
            transformer = TfidfTransformer().fit(X["tv"]["_"])
            for split in ["tv", "test"]:
                X[split] = {
                    tag: transformer.transform(X_)
                    for tag, X_ in X[split].items()
                }
            if df_unlabeled is not None:
                X_unlabeled_tv = transformer.transform(X_unlabeled_tv)

        returns = {"X": X, "y": y}
        if return_df:
            returns["df"] = df
        if return_idx:
            returns["indices"] = indices
        return returns
