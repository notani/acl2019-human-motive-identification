#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import random

from classifiers.sklearn.linsvm import run as run_svm
from classifiers.utils import init_logger

verbose = False
logger = None


def main(args):
    global verbose
    verbose = args.verbose

    conf = None
    with open(args.path_conf) as f:
        conf = json.load(f)

    random.seed(42)
    np.random.seed(42)

    if args.override:
        override = json.loads(args.override)
        for key, val in override.items():
            conf[key] = val

    args.target = conf.get("target", "_")
    args.ngram = conf["model"].get("ngram", (1, 3))
    args.class_weight = conf["model"].get("class_weight", True)
    args.tfidf = conf["model"].get("tfidf", False)

    prec, rec, f1 = [], [], []
    for path_input in conf["datasets"]:
        args.path_train = path_input["train_data_path"]
        args.path_valid = path_input["validation_data_path"]
        args.path_test = path_input["test_data_path"]
        if conf["model"]["type"] == "linsvm":
            scores = run_svm(args)
        if conf["model"]["type"] == "nb":
            scores = run_nb(args)
        prec.append(scores[0])
        rec.append(scores[1])
        f1.append(scores[2])

    if conf["average_score"]:
        print()
        print("P|R|F1")
        for tag in f1[0]["test"].keys():
            buff = []
            for metric in [prec, rec, f1]:
                l = [metric[i]["test"][tag][0] for i in range(len(metric))]
                buff.append("{:.3f} ({:.3f})".format(np.mean(l), np.std(l)))
            print(("" if tag == "_" else tag + "|") + "|".join(buff))

    return 0


if __name__ == "__main__":
    logger = init_logger("Run")
    parser = argparse.ArgumentParser()
    parser.add_argument("path_conf", help="path to config file")
    parser.add_argument("-o", "--override")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="verbose output",
    )
    args = parser.parse_args()
    main(args)
