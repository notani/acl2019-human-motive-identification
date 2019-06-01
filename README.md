# Human Motive Identification

This repository contains code and data for reproducing experiments in our paper ``Toward Comprehensive Understanding of a Sentiment Based on Human Motives.''


# Setup

## Data

For setting up data, see `data/README.md`.

In addition, download [GloVe embeddings](https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz) and place it at `embeddings/GloVe/glove.6B.100d.txt.gz`.

## Code

You need to install `allennlp` (for an MLP classifier) and `sckit-learn` (for an SVM classifier) to run classifiers.

```shell
pip install allennlp
pip install scikit-learn
```


# Run experiments


## SVM-BoNG

```shell
PYTHONPATH=. python classifiers/sklearn/run_cv.py config_files/linsvm_1_res.json -v
PYTHONPATH=. python classifiers/sklearn/run_cv.py config_files/linsvm_1_lap.json -v
```

## SVM-BoNG-tfidf
```
PYTHONPATH=. python classifiers/sklearn/run_cv.py config_files/linsvm_2_res.json -v
PYTHONPATH=. python classifiers/sklearn/run_cv.py config_files/linsvm_2_lap.json -v
```


## MLP


```shell
# SWEM
PYTHONPATH=. python classifiers/allennlp/run_mlp_cv.py config_files/mlp_swem_1_res.json -s <result dir> --include-package classifiers.allennlp
PYTHONPATH=. python classifiers/allennlp/run_mlp_cv.py config_files/mlp_swem_1_lap.json -s <result dir> --include-package classifiers.allennlp

# CNN
PYTHONPATH=. python classifiers/allennlp/run_cnn_cv.py config_files/cnn_1_res.json -s <result dir> --include-package classifiers.allennlp
PYTHONPATH=. python classifiers/allennlp/run_cnn_cv.py config_files/cnn_1_lap.json -s <result dir> --include-package classifiers.allennlp

# BiLSTM
PYTHONPATH=. python classifiers/allennlp/run_lstm_cv.py config_files/lstm_1_res.json -s <result dir> --include-package classifiers.allennlp
PYTHONPATH=. python classifiers/allennlp/run_lstm_cv.py config_files/lstm_1_lap.json -s <result dir> --include-package classifiers.allennlp
```

## Transfer learning


```shell
# SVM-BoNG
PYTHONPATH=. python classifiers/sklearn/run_cv.py config_files/transfer/linsvm_1_res.json -v
PYTHONPATH=. python classifiers/sklearn/run_cv.py config_files/transfer/linsvm_1_lap.json -v

# SVM-BoNG-tfidf
PYTHONPATH=. python classifiers/sklearn/run_cv.py config_files/transfer/linsvm_2_res.json -v
PYTHONPATH=. python classifiers/sklearn/run_cv.py config_files/transfer/linsvm_2_lap.json -v

# SWEM
PYTHONPATH=. python classifiers/allennlp/run_mlp_cv.py config_files/transfer/mlp_swem_1_res.json -s <result dir> --include-package classifiers.allennlp
PYTHONPATH=. python classifiers/allennlp/run_mlp_cv.py config_files/transfer/mlp_swem_1_lap.json -s <result dir> --include-package classifiers.allennlp

# CNN
PYTHONPATH=. python classifiers/allennlp/run_cnn_cv.py config_files/transfer/cnn_1_res.json -s <result dir> --include-package classifiers.allennlp
PYTHONPATH=. python classifiers/allennlp/run_cnn_cv.py config_files/transfer/cnn_1_lap.json -s <result dir> --include-package classifiers.allennlp

# BiLSTM
PYTHONPATH=. python classifiers/allennlp/run_lstm_cv.py config_files/transfer/lstm_1_res.json -s <result dir> --include-package classifiers.allennlp
PYTHONPATH=. python classifiers/allennlp/run_lstm_cv.py config_files/transfer/lstm_1_lap.json -s <result dir> --include-package classifiers.allennlp
```


# Citation

```
Naoki Otani and Eduard Hovy. 2019. Toward Comprehensive Understanding of a Sentiment Based on Human Motives. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), July. Association for Computational Linguistics.
```
