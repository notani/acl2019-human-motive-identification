#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from sklearn.svm import LinearSVC
import argparse
import numpy as np

from classifiers.constants import NUM_MOTIVES
from classifiers.sklearn.dataset import Dataset
from classifiers.sklearn.metrics import f1_score
from classifiers.sklearn.metrics import precision_score
from classifiers.sklearn.metrics import recall_score
from classifiers.utils import calc_mean_and_std_score
from classifiers.utils import expand_param_grid
from classifiers.utils import init_logger

verbose = False


def run(args):
    global verbose
    verbose = args.verbose

    logger = args.logger if "logger" in args else init_logger("LinSVM")

    if verbose:
        logger.info(
            "Parameters\n"
            + "\n".join(
                "{}: {}".format(k, v)
                for k, v in sorted(dict(vars(args)).items())
            )
        )

    target = args.target if "target" in args else "_"

    path_input = {
        "train": args.path_train,
        "valid": args.path_valid,
        "test": args.path_test,
    }
    dataset = Dataset(path_input, verbose=verbose, logger=logger)

    data = dataset.ngram_encoder(
        args.ngram, tfidf=args.tfidf, return_df=False, return_idx=True
    )
    X, y, indices = data["X"], data["y"], data["indices"]

    # Parameter Tuning
    param_grid = {
        "penalty": ["l1", "l2"],
        "loss": ["hinge", "squared_hinge"],
        "C": [2**i for i in range(-3, 1)],
    }
    sample_weight = None
    if isinstance(args.path_train, dict):  # Multi-task learning
        param_grid["sample_weight"] = [0.125, 0.25, 0.5]

    f1 = {"train": defaultdict(list), "valid": defaultdict(list)}

    for params in expand_param_grid(param_grid):
        f1_ = {"train": defaultdict(list), "valid": defaultdict(list)}
        for c in range(NUM_MOTIVES):
            X_, y_ = X["train"]["_"], y["train"]["_"][:, c]
            # Shuffle
            shuf = np.random.permutation(np.arange(X_.shape[0]))
            X_, y_ = X_[shuf], y_[shuf]

            class_weight = None
            if args.class_weight:
                n_pos = float(y_.sum())
                w_pos = (y_.shape[0] - n_pos) / n_pos
                class_weight = {0: 1, 1: w_pos}

            if "sample_weight" in params:
                sample_weight = np.ones(y_.shape[0]) * params.pop(
                    "sample_weight"
                )
                sample_weight[indices["train"][target]] = 1
                sample_weight = sample_weight[shuf]

            try:
                clf = LinearSVC(class_weight=class_weight, **params)
                # Sample weighting
                if sample_weight is not None:
                    params["sample_weight"] = sample_weight.min()
                # Training
                clf.fit(X_, y_, sample_weight=sample_weight)
            except ValueError:  # Invalid combination of parameters
                continue

            for split in ["train", "valid"]:
                for tag, X_ in X[split].items():
                    f1_[split][tag].append(
                        f1_score(
                            y_true=y[split][tag][:, c], y_pred=clf.predict(X_)
                        )
                    )
        if len(f1_["train"]) == 0:
            continue
        for split in ["train", "valid"]:  # MEMO
            for tag, scores in f1_[split].items():
                f1[split][tag].append(
                    (params, np.mean(scores), np.std(scores))
                )

    best = sorted(
        zip(f1["valid"][target], f1["train"][target]),
        key=lambda t: (t[0][1], t[1][1]),
        reverse=True,
    )[0]
    best_params = best[0][0]
    print("Best params: {}".format(best_params))
    print("Train: {:.3f} (+/-{:.3f})".format(best[1][1], best[1][2]))
    print("Valid: {:.3f} (+/-{:.3f})".format(best[0][1], best[0][2]))

    # Evaluation
    f1_, prec_, rec_ = [
        {"tv": defaultdict(list), "test": defaultdict(list)} for _ in range(3)
    ]
    for c in range(NUM_MOTIVES):
        X_, y_ = X["tv"]["_"], y["tv"]["_"][:, c]
        # Shuffle
        shuf = np.random.permutation(np.arange(X_.shape[0]))
        X_, y_ = X_[shuf], y_[shuf]

        class_weight = None
        if args.class_weight:
            n_pos = float(y_.sum())
            w_pos = (y_.shape[0] - n_pos) / n_pos
            class_weight = {0: 1, 1: w_pos}

        if "sample_weight" in best_params:
            sample_weight = np.ones(y_.shape[0]) * best_params.pop(
                "sample_weight"
            )
            sample_weight[indices["tv"][target]] = 1
            sample_weight = sample_weight[shuf]

        clf = LinearSVC(class_weight=class_weight, **best_params)
        # Training
        clf.fit(X_, y_, sample_weight=sample_weight)

        for split in ["tv", "test"]:
            for tag, X_ in X[split].items():
                y_ = y[split][tag][:, c]
                pred = clf.predict(X_)
                f1_[split][tag].append(f1_score(y_true=y_, y_pred=pred))
                prec_[split][tag].append(
                    precision_score(y_true=y_, y_pred=pred)
                )
                rec_[split][tag].append(recall_score(y_true=y_, y_pred=pred))

    print("\n".join(calc_mean_and_std_score(f1_["tv"], label="Train+Valid")))
    print("\n".join(calc_mean_and_std_score(f1_["test"], label="Test")))

    prec, rec, f1 = {}, {}, {}
    for split in ["tv", "test"]:
        prec[split] = {
            tag: (np.mean(values), np.std(values))
            for tag, values in prec_[split].items()
        }
        rec[split] = {
            tag: (np.mean(values), np.std(values))
            for tag, values in rec_[split].items()
        }
        f1[split] = {
            tag: (np.mean(values), np.std(values))
            for tag, values in f1_[split].items()
        }

    print("||P|R|F1|")
    for tag in f1["test"].keys():
        buff = [tag]
        buff += [
            "{:.3f} ({:.3f})".format(*val)
            for val in [prec["test"][tag], rec["test"][tag], f1["test"][tag]]
        ]
        print("|".join(buff))

    return prec, rec, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        dest="path_train",
        required=True,
        help="path to training file",
    )
    parser.add_argument(
        "--valid",
        dest="path_valid",
        required=True,
        help="path to validation file",
    )
    parser.add_argument(
        "--test", dest="path_test", required=True, help="path to testing file"
    )
    parser.add_argument(
        "--ngram", nargs=2, type=int, default=(1, 3), help="range of ngram"
    )
    parser.add_argument(
        "--no-class-weight",
        dest="class_weight",
        action="store_false",
        help="ignore empirical class weight",
    )
    parser.add_argument(
        "--tfidf", action="store_true", help="use TFIDF transformer"
    )
    parser.add_argument(
        "-o", "--output", dest="path_output", help="path to output file"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="verbose output",
    )
    args = parser.parse_args()
    run(args)
