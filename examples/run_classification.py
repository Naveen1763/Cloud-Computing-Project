#!/usr/bin/env python3
"""
Run the AAPA workload classification pipeline.

Steps:
1. Load workload traces (bundled KSWD dataset, or the raw Azure Functions 2019 CSVs)
2. Extract features over sliding windows
3. Apply weak supervision to label every window
4. Train a LightGBM archetype classifier
5. Evaluate and persist the model

Examples:
    python examples/run_classification.py                     # 200 KSWD workloads
    python examples/run_classification.py --max-workloads 1000
    python examples/run_classification.py --source azure --data-dir data/raw
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.data.kswd_loader import KSWDLoader
from src.data.loader import AzureFunctionsDataLoader
from src.data.windowing import (
    SlidingWindowProcessor,
    create_train_test_split,
    create_workload_split,
)
from src.models.classifier import WorkloadClassifierTrainer
from src.models.conformal import ConformalClassifier
from src.models.labeling import WindowedWeakSupervisionLabeler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="AAPA workload classification pipeline")
    parser.add_argument(
        "--source",
        choices=["kswd", "azure"],
        default="kswd",
        help="kswd = bundled dataset/ traces (default), azure = raw Azure CSVs",
    )
    parser.add_argument("--data-dir", default="data/raw", help="Azure raw CSV directory")
    parser.add_argument("--dataset-dir", default="dataset", help="KSWD dataset directory")
    parser.add_argument("--output-dir", default="models", help="Where to write the model")
    parser.add_argument("--max-workloads", type=int, default=200, help="Workloads to load")
    parser.add_argument("--window-size", type=int, default=60, help="Window size in minutes")
    parser.add_argument("--stride", type=int, default=10, help="Window stride in minutes")
    parser.add_argument(
        "--min-invocations",
        type=float,
        default=1000,
        help="Drop workloads with fewer total invocations",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for LightGBM")
    parser.add_argument(
        "--labels",
        choices=["weak", "kswd"],
        default="weak",
        help=(
            "weak = programmatic labeling functions (paper's setup); "
            "kswd = the dataset's ground-truth archetype per workload"
        ),
    )
    parser.add_argument(
        "--split",
        choices=["workload", "day"],
        default="day",
        help="workload = leak-free; day = the paper's protocol (default, reproduces it)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Conformal miscoverage level; 0.1 targets 90%% coverage",
    )
    parser.add_argument(
        "--conformal-method",
        choices=["aps", "lac"],
        default="aps",
        help="Nonconformity score: aps adapts set size, lac gives smaller sets",
    )
    parser.add_argument(
        "--mondrian",
        action="store_true",
        help="Calibrate per class for class-conditional coverage",
    )
    parser.add_argument(
        "--calib-fraction",
        type=float,
        default=0.5,
        help="Fraction of validation workloads held out for calibration",
    )
    return parser.parse_args()


def split_calibration(val_df, fraction, seed):
    """Split the validation frame into (early-stopping, calibration) parts.

    Splitting by workload rather than by row keeps windows from the same trace
    out of both halves. Adjacent sliding windows overlap heavily, so a random
    row split would leak and inflate the measured coverage.
    """
    workloads = np.sort(val_df["function_hash"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(workloads)

    n_calib = max(1, int(round(len(shuffled) * fraction)))
    if n_calib >= len(shuffled):  # keep at least one workload for early stopping
        n_calib = max(1, len(shuffled) - 1)

    calib_ids = set(shuffled[:n_calib].tolist())
    mask = val_df["function_hash"].isin(calib_ids)
    return val_df[~mask], val_df[mask]


def transform_split(trainer, df):
    """Apply the trainer's fitted scaler to an extra split."""
    X = df[trainer.feature_names].values
    return trainer.scalers["standard"].transform(X), df["label"].values


def predict_proba(trainer, X):
    """Class probabilities from the trained booster, in `classes_` order."""
    booster = trainer.models["lightgbm"]
    return booster.predict(X, num_iteration=booster.best_iteration)


def load_traces(args):
    """Load raw per-minute traces as {workload_id: np.ndarray}."""
    if args.source == "kswd":
        logger.info("Loading traces from the bundled KSWD dataset...")
        loader = KSWDLoader(args.dataset_dir)
        return loader.load_invocation_data(max_workloads=args.max_workloads)

    logger.info("Loading traces from the Azure Functions 2019 dataset...")
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(
            f"Azure data directory not found: {data_dir}\n"
            "Download it from Kaggle, or use the default --source kswd."
        )
    loader = AzureFunctionsDataLoader(data_dir)
    traces = loader.load_invocation_data(days=list(range(1, 15)))
    if args.max_workloads:
        traces = dict(list(traces.items())[: args.max_workloads])
    return traces


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting workload classification pipeline")

    # Step 1: load traces
    traces = load_traces(args)
    logger.info(f"Loaded {len(traces)} workloads")

    # Step 2: filter low-activity workloads
    active = {k: v for k, v in traces.items() if np.sum(v) >= args.min_invocations}
    if not active:
        raise SystemExit(
            f"No workload reached --min-invocations {args.min_invocations}. Lower the threshold."
        )
    logger.info(f"Kept {len(active)} workloads above the activity threshold")

    # Step 3: sliding windows + feature extraction
    logger.info("Extracting windowed features...")
    processor = SlidingWindowProcessor(window_size=args.window_size, stride=args.stride)
    features_df = processor.process_all_functions(active)
    if features_df.empty:
        raise SystemExit("Feature extraction produced no windows.")
    logger.info(f"Built {len(features_df)} windows")

    # Step 4: labels
    logger.info("Applying weak supervision labeling...")
    labeler = WindowedWeakSupervisionLabeler()
    labeled_df = labeler.label_windows(features_df)

    stats = labeler.get_labeling_statistics(labeled_df)
    logger.info(f"Labeling statistics: {json.dumps(stats, indent=2, default=str)}")

    if args.labels == "kswd":
        # The weak labels are derived from the same statistics the model sees,
        # so training on them measures how well the model reproduces the
        # labeling functions rather than how well it recovers the archetype.
        # Swap in the dataset's ground-truth archetype for an honest task.
        if args.source != "kswd":
            raise SystemExit("--labels kswd requires --source kswd")
        truth = KSWDLoader(args.dataset_dir).load_archetype_labels(
            sorted(labeled_df["function_hash"].unique())
        )
        weak = labeled_df["label"]
        labeled_df = labeled_df.copy()
        labeled_df["label"] = labeled_df["function_hash"].map(truth)
        labeled_df = labeled_df[labeled_df["label"].notna()]
        agreement = (weak.loc[labeled_df.index] == labeled_df["label"]).mean()
        logger.info(
            f"Using KSWD ground-truth archetypes; weak labels agree with them "
            f"on {agreement:.1%} of windows"
        )
        logger.info(
            "Ground-truth distribution:\n"
            + labeled_df["label"].value_counts().to_string()
        )

    # Step 5: split by day, then train
    if args.split == "workload":
        train_df, val_df, test_df = create_workload_split(labeled_df, seed=args.seed)
    else:
        train_df, val_df, test_df = create_train_test_split(labeled_df)
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise SystemExit(
            "Train/val/test split is empty - the traces are too short. "
            "Load more days or more workloads."
        )
    if train_df["label"].nunique() < 2:
        raise SystemExit(
            f"Weak supervision produced a single class ({train_df['label'].unique()}). "
            "Increase --max-workloads for a more diverse sample."
        )

    # Conformal calibration needs data the model never saw. The validation
    # days drive early stopping, so carve the calibration split out of them
    # rather than reusing the fitting or test data.
    val_df, calib_df = split_calibration(val_df, args.calib_fraction, args.seed)
    logger.info(
        f"Held out {len(calib_df)} windows for conformal calibration "
        f"({len(val_df)} left for early stopping)"
    )

    trainer = WorkloadClassifierTrainer(output_dir=str(output_dir))
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_data(
        train_df, val_df, test_df
    )
    X_calib, y_calib = transform_split(trainer, calib_df)

    logger.info("Training LightGBM classifier...")
    trainer.train_lightgbm(X_train, y_train, X_val, y_val, use_gpu=args.gpu)

    # Step 6: evaluate
    logger.info("Evaluating classifier...")
    results = trainer.evaluate_model("lightgbm", X_test, y_test)
    report = results["classification_report"]

    logger.info(f"Test accuracy: {report['accuracy']:.4f}")
    logger.info(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    logger.info(f"Mean prediction confidence: {results['avg_confidence']:.4f}")
    logger.info(f"Confusion matrix (labels={results['labels']}):\n{results['confusion_matrix']}")

    # Step 6b: conformal calibration
    logger.info("Calibrating conformal predictor...")
    classes = np.array(sorted(trainer.models["lightgbm_label_map"], key=lambda k: trainer.models["lightgbm_label_map"][k]))
    calib_probs = predict_proba(trainer, X_calib)
    test_probs = predict_proba(trainer, X_test)

    conformal = ConformalClassifier(
        alpha=args.alpha,
        method=args.conformal_method,
        mondrian=args.mondrian,
        random_state=args.seed,
    ).calibrate(calib_probs, y_calib, classes=classes)

    cp_results = conformal.evaluate(test_probs, y_test)
    logger.info(
        f"Conformal ({args.conformal_method.upper()}, alpha={args.alpha}"
        f"{', class-conditional' if args.mondrian else ''}): "
        f"target coverage {cp_results['target_coverage']:.3f}, "
        f"empirical {cp_results['empirical_coverage']:.4f}"
    )
    logger.info(
        f"  avg set size {cp_results['avg_set_size']:.3f}, "
        f"singleton rate {cp_results['singleton_rate']:.3f}, "
        f"set sizes {cp_results['set_size_distribution']}"
    )
    for label, stats_k in sorted(cp_results["per_class"].items()):
        logger.info(
            f"  {label:<18} n={stats_k['n']:<6} coverage={stats_k['coverage']:.4f} "
            f"avg_set_size={stats_k['avg_set_size']:.3f}"
        )

    # Step 7: persist artifacts
    trainer.save_models()
    trainer.plot_confusion_matrix(
        results, "lightgbm", save_path=str(output_dir / "confusion_matrix.png")
    )

    with open(output_dir / "label_map.json", "w", encoding="utf-8") as fh:
        json.dump(trainer.models["lightgbm_label_map"], fh, indent=2)

    with open(output_dir / "feature_names.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(trainer.feature_names) + "\n")

    # Artifacts consumed by src/core/simulator.py's AAPAAutoscaler: a native
    # LightGBM booster file plus a LabelEncoder whose class order matches the
    # booster's output columns, and the scaler used at training time.
    label_map = trainer.models["lightgbm_label_map"]
    trainer.models["lightgbm"].save_model(str(output_dir / "lightgbm_model.txt"))

    encoder = LabelEncoder()
    encoder.fit(sorted(label_map, key=label_map.get))
    with open(output_dir / "label_encoder.pkl", "wb") as fh:
        pickle.dump(encoder, fh)

    with open(output_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(trainer.scalers["standard"], fh)

    with open(output_dir / "conformal.pkl", "wb") as fh:
        pickle.dump(conformal, fh)

    with open(output_dir / "conformal_report.json", "w", encoding="utf-8") as fh:
        json.dump(cp_results, fh, indent=2, default=str)

    # Cache the score matrices so alpha/method can be swept without retraining.
    np.savez(
        output_dir / "conformal_scores.npz",
        calib_probs=calib_probs,
        calib_labels=np.asarray(y_calib, dtype=object),
        test_probs=test_probs,
        test_labels=np.asarray(y_test, dtype=object),
        classes=np.asarray(classes, dtype=object),
    )

    # Real confusion matrix for examples/create_visualizations.py. sklearn
    # builds it over the classes actually present in y_test/y_pred, sorted, so
    # derive the tick labels the same way instead of using the full label map.
    cm_labels = sorted(set(y_test) | set(results["predictions"]))
    np.savez(
        output_dir / "confusion_matrix.npz",
        cm=results["confusion_matrix"],
        labels=np.array(cm_labels, dtype=object),
    )
    logger.info(f"Saved booster, label encoder and scaler to {output_dir}")

    logger.info("Classification pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
