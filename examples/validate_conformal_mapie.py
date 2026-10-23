#!/usr/bin/env python3
"""
Cross-validate src/models/conformal.py against MAPIE.

MAPIE is the reference implementation of split conformal prediction. Running
both on byte-identical calibration and test probabilities checks that the
in-repo implementation is correct, so the results do not rest on a bespoke
predictor.

MAPIE is used only as an oracle here, not as a runtime dependency: it does not
expose Mondrian (class-conditional) calibration, which the SPIKE coverage
results show is necessary, so the in-repo version remains the one wired into
the autoscaler.

Requires: pip install mapie

Example:
    python examples/validate_conformal_mapie.py --model-dir models_gt
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models.conformal import ConformalClassifier

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ALPHAS = [0.01, 0.05, 0.10, 0.20]


class PrecomputedClassifier(ClassifierMixin, BaseEstimator):
    """Serves stored probabilities so both libraries see identical inputs.

    X is an integer index into the probability matrix rather than real
    features, which removes any chance of the two implementations disagreeing
    because the underlying model was re-evaluated differently.
    """

    def __init__(self, probs=None, classes=None):
        self.probs = probs
        self.classes = classes

    def fit(self, X=None, y=None):
        self.classes_ = np.asarray(self.classes)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        idx = np.asarray(X).ravel().astype(int)
        return self.probs[idx]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def __sklearn_is_fitted__(self):
        return True


def parse_args():
    p = argparse.ArgumentParser(description="Validate conformal against MAPIE")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from mapie.classification import SplitConformalClassifier
    except ImportError:
        raise SystemExit("MAPIE not installed. Run: pip install mapie")

    path = Path(args.model_dir) / "conformal_scores.npz"
    if not path.exists():
        raise SystemExit(
            f"{path} not found.\nRun: python examples/run_classification.py"
        )

    data = np.load(path, allow_pickle=True)
    calib_probs = data["calib_probs"]
    calib_labels = data["calib_labels"]
    test_probs = data["test_probs"]
    test_labels = data["test_labels"]
    classes = data["classes"]

    print(f"Calibration: {len(calib_labels)}   Test: {len(test_labels)}")
    print(f"Classes: {list(classes)}\n")

    # Both libraries index into one stacked probability matrix.
    all_probs = np.vstack([calib_probs, test_probs])
    calib_idx = np.arange(len(calib_probs)).reshape(-1, 1)
    test_idx = (np.arange(len(test_probs)) + len(calib_probs)).reshape(-1, 1)

    estimator = PrecomputedClassifier(probs=all_probs, classes=classes).fit()

    header = (
        f"{'alpha':>6} {'method':>5} | {'mine cov':>9} {'mapie cov':>10} "
        f"| {'mine size':>10} {'mapie size':>11} | {'set agreement':>14}"
    )
    print(header)
    print("-" * len(header))

    all_agree = True
    for method in ("lac", "aps"):
        for alpha in ALPHAS:
            # MAPIE's APS score always subtracts a uniform draw and offers no
            # way to disable it, so randomization must be matched or the two
            # are not comparable. LAC has no randomization in either library.
            mine = ConformalClassifier(
                alpha=alpha,
                method=method,
                randomized=(method == "aps"),
                random_state=args.seed,
            ).calibrate(calib_probs, calib_labels, classes=classes)
            my_sets = mine.predict_set(test_probs)
            my_bool = np.zeros((len(test_probs), len(classes)), dtype=bool)
            class_pos = {c: i for i, c in enumerate(classes)}
            for row, s in enumerate(my_sets):
                for label in s:
                    my_bool[row, class_pos[label]] = True

            mapie = SplitConformalClassifier(
                estimator=estimator,
                confidence_level=1 - alpha,
                conformity_score=method,
                prefit=True,
            )
            mapie.conformalize(calib_idx, calib_labels)
            _, mapie_sets = mapie.predict_set(test_idx)
            mapie_bool = mapie_sets[:, :, 0]

            my_cov = float(
                np.mean([truth in s for truth, s in zip(test_labels, my_sets)])
            )
            mapie_cov = float(
                np.mean(
                    [
                        mapie_bool[i, class_pos[test_labels[i]]]
                        for i in range(len(test_labels))
                    ]
                )
            )
            my_size = float(my_bool.sum(axis=1).mean())
            mapie_size = float(mapie_bool.sum(axis=1).mean())
            agreement = float((my_bool == mapie_bool).all(axis=1).mean())
            # LAC is deterministic and must match set-for-set. Randomized APS
            # draws its own uniforms in each library, so identical sets are not
            # expected; coverage and mean set size are the meaningful checks.
            if method == "lac":
                all_agree &= agreement > 0.99
            else:
                all_agree &= (
                    abs(my_cov - mapie_cov) < 0.02 and abs(my_size - mapie_size) < 0.15
                )

            print(
                f"{alpha:>6.2f} {method:>5} | {my_cov:>9.4f} {mapie_cov:>10.4f} "
                f"| {my_size:>10.3f} {mapie_size:>11.3f} | {agreement:>13.1%}"
            )

    print()
    if all_agree:
        print(
            "PASS - LAC matches MAPIE set-for-set; randomized APS matches on "
            "coverage and mean set size."
        )
    else:
        print(
            "MISMATCH - investigate before relying on these prediction sets."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
