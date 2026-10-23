#!/usr/bin/env python3
"""
Head-to-head comparison of scaling controllers at the decision level.

Every controller sees the same observed window and is judged against the same
required capacity for the next window, so any difference in outcome is
attributable purely to decision logic - no simulator queueing noise, no
per-run variance. This isolates the contribution being tested.

Controllers:
    Reactive              HPA-style, fixed 70% CPU target, archetype-blind.
    AAPA_Beta             AAPA's real mechanism: archetype + beta-calibrated
                          scalar confidence + blanket margin (Algorithm 1).
    Conformal_TargetOnly  Provision for the most demanding archetype in the
                          conformal set (max over 1/target_cpu).
    Conformal_Envelope    As above, but enveloping every strategy parameter.

The headline split is clean (|C(x)| = 1) versus ambiguous (|C(x)| > 1)
windows. On clean windows the conformal controllers reduce to AAPA without
its margin; all the interesting behaviour is on the ambiguous ones.

Example:
    python examples/run_conformal_controllers.py --labels kswd --max-workloads 200
"""

import argparse
import json
import logging
import os
import sys
from math import ceil
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data.kswd_loader import KSWDLoader
from src.data.windowing import (
    SlidingWindowProcessor,
    create_train_test_split,
    create_workload_split,
)
from src.models.calibration import BetaCalibrator, expected_calibration_error
from src.models.classifier import WorkloadClassifierTrainer
from src.models.conformal import ConformalClassifier
from src.models.labeling import WindowedWeakSupervisionLabeler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# AAPA Table III / src/core/strategies.py
TARGET_CPU = {
    "SPIKE": 0.30,
    "PERIODIC": 0.75,
    "RAMP": 0.60,
    "STATIONARY_NOISY": 0.55,
}
REACTIVE_TARGET_CPU = 0.70          # Kubernetes HPA default
FALLBACK_ARCHETYPE = "STATIONARY_NOISY"
POD_CAPACITY_RPM = 60.0             # requests per minute one pod can absorb
MIN_REPLICAS = 1                    # a deployment never runs fractional pods

METADATA_COLS = [
    "function_hash", "window_idx", "window_start_minute",
    "window_end_minute", "day", "hour_of_day", "label", "confidence",
]


def parse_args():
    p = argparse.ArgumentParser(description="Controller comparison at decision level")
    p.add_argument("--dataset-dir", default="dataset")
    p.add_argument("--results-dir", default="results_controllers")
    p.add_argument("--max-workloads", type=int, default=200)
    p.add_argument("--min-invocations", type=float, default=100)
    p.add_argument("--window-size", type=int, default=60)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--labels", choices=["kswd", "weak"], default="kswd",
                   help="kswd = ground-truth archetypes (recommended)")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--conformal-method", choices=["aps", "lac"], default="aps")
    p.add_argument("--mondrian", action="store_true",
                   help="Class-conditional calibration (protects SPIKE)")
    p.add_argument("--calib-fraction", type=float, default=0.5)
    p.add_argument(
        "--min-activity",
        type=float,
        default=0.0,
        help=(
            "Minimum fraction of minutes a workload must be active. The "
            "windowing default of 0.1 deletes 98%% of SPIKE workloads, since "
            "bursty traffic is idle by definition. 0.0 keeps everything."
        ),
    )
    p.add_argument(
        "--load-multiplier",
        type=float,
        default=1.0,
        help=(
            "Scale trace load before sizing. The KSWD traces are mostly idle "
            "(median window peak 1 req/min), so at 1.0 about 95%% of windows "
            "need a single pod and there is no scaling decision to get right. "
            "60 puts ~20%% of windows in genuine scaling territory. Equivalent "
            "to lowering per-pod capacity; expressed as traffic scale because "
            "'a busier deployment' is the more natural reading."
        ),
    )
    p.add_argument("--split", choices=["workload", "day"], default="workload",
                   help="workload = leak-free (default); day = the paper's protocol, "
                        "which puts the same workload in train and test")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def provision(required_replicas: float, target_cpu: float) -> int:
    """Replicas needed to serve `required_replicas` worth of load at a target."""
    return int(ceil(required_replicas / target_cpu))


# --------------------------------------------------------------------------
# Controllers
# --------------------------------------------------------------------------

def decide_reactive(current: float, **_) -> int:
    return provision(current, REACTIVE_TARGET_CPU)


def decide_aapa_beta(current: float, archetype: str, confidence: float, **_) -> int:
    """AAPA Algorithm 1: archetype target, then a blanket confidence margin."""
    base = provision(current, TARGET_CPU.get(archetype, TARGET_CPU[FALLBACK_ARCHETYPE]))
    margin = 1.0 + 0.5 * (1.0 - confidence)
    return int(ceil(base * margin))


def decide_top1_no_margin(current: float, archetype: str, **_) -> int:
    """Ablation control: top-1 archetype target, no margin, no set.

    Separates two effects that AAPA_Beta and the conformal controllers
    otherwise confound: dropping the blanket confidence margin, and hedging
    over a prediction set. On singleton windows Conformal_TargetOnly reduces
    to exactly this, so any gap between the two is attributable to conformal
    hedging and nothing else.
    """
    return provision(current, TARGET_CPU.get(archetype, TARGET_CPU[FALLBACK_ARCHETYPE]))


def decide_conformal_target_only(current: float, prediction_set, **_) -> int:
    """Provision for the most demanding archetype still plausible."""
    members = [a for a in prediction_set if a in TARGET_CPU] or [FALLBACK_ARCHETYPE]
    return max(provision(current, TARGET_CPU[a]) for a in members)


def decide_conformal_envelope(current: float, prediction_set, **_) -> int:
    """Target-only hedging plus the warm pool the envelope would carry.

    ConformalScalingPolicy takes the max warm pool across the set; at decision
    level that is the one extra term that changes the replica count, so it is
    reproduced here rather than importing the full simulator strategy dict.
    """
    members = [a for a in prediction_set if a in TARGET_CPU] or [FALLBACK_ARCHETYPE]
    base = max(provision(current, TARGET_CPU[a]) for a in members)
    warm_pool = 3 if "SPIKE" in members else 1
    return base + warm_pool


CONTROLLERS = {
    "Reactive": decide_reactive,
    "AAPA_Beta": decide_aapa_beta,
    "Top1_NoMargin_control": decide_top1_no_margin,
    "Conformal_TargetOnly": decide_conformal_target_only,
    "Conformal_Envelope": decide_conformal_envelope,
}


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------

def build_windows(args):
    """Windowed features with ground-truth labels and load requirements."""
    loader = KSWDLoader(args.dataset_dir)
    traces = loader.load_invocation_data(max_workloads=args.max_workloads)
    active = {k: v for k, v in traces.items() if np.sum(v) >= args.min_invocations}
    logger.info(f"Loaded {len(active)} active workloads")

    processor = SlidingWindowProcessor(args.window_size, args.stride)
    df = processor.process_all_functions(
        active, min_activity_threshold=args.min_activity
    )

    labeler = WindowedWeakSupervisionLabeler()
    df = labeler.label_windows(df)

    if args.labels == "kswd":
        truth = loader.load_archetype_labels(sorted(df["function_hash"].unique()))
        df["label"] = df["function_hash"].map(truth)
        df = df[df["label"].notna()].copy()

    # Capacity actually demanded by this window and by the next one. Peak load
    # is the SLO-relevant quantity: a controller that provisions for the mean
    # still drops requests at the peak.
    cur, nxt = [], []
    for func_hash, group in df.groupby("function_hash", sort=False):
        trace = active[func_hash]
        starts = group["window_start_minute"].to_numpy()
        for start in starts:
            w = trace[start:start + args.window_size]
            n = trace[start + args.stride:start + args.stride + args.window_size]
            # Peak load is the SLO-relevant quantity, floored at one pod:
            # a deployment cannot run a fraction of a replica, and leaving
            # these as fractions makes the over-provisioning ratio explode
            # on low-traffic windows.
            scale = args.load_multiplier / POD_CAPACITY_RPM
            cur.append(max(MIN_REPLICAS, int(ceil(np.max(w) * scale))) if len(w) else MIN_REPLICAS)
            nxt.append(max(MIN_REPLICAS, int(ceil(np.max(n) * scale))) if len(n) else MIN_REPLICAS)

    df["current_req_replicas"] = cur
    df["next_req_replicas"] = nxt
    return df


def split_calibration(val_df, fraction, seed):
    """Hold out whole workloads, never rows - adjacent windows overlap."""
    ids = np.sort(val_df["function_hash"].unique())
    shuffled = np.random.default_rng(seed).permutation(ids)
    n = max(1, min(int(round(len(shuffled) * fraction)), len(shuffled) - 1))
    calib_ids = set(shuffled[:n].tolist())
    mask = val_df["function_hash"].isin(calib_ids)
    return val_df[~mask], val_df[mask]


# --------------------------------------------------------------------------

def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)

    df = build_windows(args)
    nontrivial = (df["next_req_replicas"] > 1).mean()
    logger.info(
        f"{len(df)} windows | load multiplier {args.load_multiplier}x | "
        f"{nontrivial:.1%} of windows need more than one pod "
        f"(median {df['next_req_replicas'].median():.0f}, "
        f"p99 {df['next_req_replicas'].quantile(0.99):.0f})"
    )
    if nontrivial < 0.10:
        logger.warning(
            "Fewer than 10%% of windows require scaling at all - controller "
            "differences will be dominated by trivially-satisfied decisions. "
            "Raise --load-multiplier."
        )

    if args.split == "workload":
        train_df, val_df, test_df = create_workload_split(df, seed=args.seed)
    else:
        train_df, val_df, test_df = create_train_test_split(df)
    val_df, calib_df = split_calibration(val_df, args.calib_fraction, args.seed)
    if min(len(train_df), len(val_df), len(calib_df), len(test_df)) == 0:
        raise SystemExit("A split came out empty; raise --max-workloads.")
    logger.info(
        f"train={len(train_df)} val={len(val_df)} calib={len(calib_df)} test={len(test_df)}"
    )

    trainer = WorkloadClassifierTrainer(output_dir=str(results_dir))
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_data(
        train_df, val_df, test_df
    )
    trainer.train_lightgbm(X_train, y_train, X_val, y_val, use_gpu=False)

    booster = trainer.models["lightgbm"]
    label_map = trainer.models["lightgbm_label_map"]
    classes = np.array(sorted(label_map, key=lambda k: label_map[k]))

    X_calib = trainer.scalers["standard"].transform(calib_df[trainer.feature_names].values)
    y_calib = calib_df["label"].values

    probs_calib = booster.predict(X_calib, num_iteration=booster.best_iteration)
    probs_test = booster.predict(X_test, num_iteration=booster.best_iteration)

    top_idx = np.argmax(probs_test, axis=1)
    accuracy = float((classes[top_idx] == y_test).mean())
    logger.info(f"Classifier test accuracy: {accuracy:.4f}")

    # --- Baseline mechanism: beta calibration -----------------------------
    class_to_idx = {c: i for i, c in enumerate(classes)}
    beta = BetaCalibrator(len(classes)).fit(
        probs_calib, np.array([class_to_idx[c] for c in y_calib])
    )
    conf_test, _ = beta.top_label_confidence(probs_test)
    raw_conf = probs_test.max(axis=1)
    correct = classes[top_idx] == y_test

    ece_raw = expected_calibration_error(raw_conf, correct)
    ece_beta = expected_calibration_error(conf_test, correct)
    logger.info(
        f"ECE  raw softmax={ece_raw:.4f}  beta-calibrated={ece_beta:.4f} "
        f"(mean conf {raw_conf.mean():.4f} -> {conf_test.mean():.4f}, "
        f"accuracy {accuracy:.4f})"
    )

    # --- Our mechanism: conformal prediction sets -------------------------
    conformal = ConformalClassifier(
        alpha=args.alpha, method=args.conformal_method,
        mondrian=args.mondrian, random_state=args.seed,
    ).calibrate(probs_calib, y_calib, classes=classes)

    cp_eval = conformal.evaluate(probs_test, y_test)
    logger.info(
        f"Conformal coverage {cp_eval['empirical_coverage']:.4f} "
        f"(target {cp_eval['target_coverage']:.2f}), "
        f"avg set size {cp_eval['avg_set_size']:.3f}, "
        f"singleton rate {cp_eval['singleton_rate']:.3f}"
    )

    pred_sets = conformal.predict_set(probs_test)

    # --- Run every controller over every test window ----------------------
    rows = []
    cur_all = test_df["current_req_replicas"].to_numpy()
    nxt_all = test_df["next_req_replicas"].to_numpy()

    for i in range(len(test_df)):
        pset = [str(a) for a in pred_sets[i]]
        ctx = dict(
            current=cur_all[i],
            archetype=str(classes[top_idx[i]]),
            confidence=float(conf_test[i]),
            prediction_set=pset,
        )
        required = nxt_all[i]

        for name, decide in CONTROLLERS.items():
            provisioned = decide(**ctx)
            violation = provisioned < required
            rows.append({
                "controller": name,
                "archetype_true": str(y_test[i]),
                "set_size": len(pset),
                "is_ambiguous": len(pset) > 1,
                "provisioned": provisioned,
                "required": required,
                "violation": bool(violation),
                # Shortfall: fraction of demanded capacity that was missing.
                "shortfall": float((required - provisioned) / required) if violation else 0.0,
                # Waste in absolute replicas, which stays interpretable on
                # low-traffic windows where a ratio would blow up.
                "excess_replicas": float(max(0, provisioned - required)),
            })

    res = pd.DataFrame(rows)
    res.to_csv(results_dir / "controller_decisions.csv", index=False)

    def summarise(frame, title):
        if frame.empty:
            print(f"\n{title}: no windows")
            return None
        out = frame.groupby("controller").agg(
            slo_violation_rate=("violation", "mean"),
            mean_shortfall=("shortfall", lambda s: s[s > 0].mean() if (s > 0).any() else 0.0),
            mean_provisioned=("provisioned", "mean"),
            mean_excess_replicas=("excess_replicas", "mean"),
        ).round(4)
        print("\n" + "=" * 78)
        print(f"{title}  (n={len(frame)//len(CONTROLLERS)} windows)")
        print("=" * 78)
        print(out.to_string())
        return out

    overall = summarise(res, "OVERALL")
    ambiguous = summarise(res[res.is_ambiguous], "AMBIGUOUS WINDOWS  (conformal set size > 1)")
    clean = summarise(res[~res.is_ambiguous], "CLEAN WINDOWS  (singleton set)")

    print("\n" + "=" * 78)
    print("PER-ARCHETYPE (SLO violation rate / mean replicas)")
    print("=" * 78)
    per_arch = res.pivot_table(
        index="archetype_true", columns="controller",
        values=["violation", "provisioned"], aggfunc="mean",
    ).round(4)
    print(per_arch.to_string())

    for name, frame in [("overall", overall), ("ambiguous", ambiguous), ("clean", clean)]:
        if frame is not None:
            frame.to_csv(results_dir / f"summary_{name}.csv")
    per_arch.to_csv(results_dir / "summary_per_archetype.csv")

    with open(results_dir / "run_report.json", "w", encoding="utf-8") as fh:
        json.dump({
            "accuracy": accuracy,
            "ece_raw_softmax": ece_raw,
            "ece_beta_calibrated": ece_beta,
            "mean_confidence_raw": float(raw_conf.mean()),
            "mean_confidence_beta": float(conf_test.mean()),
            "conformal": {k: v for k, v in cp_eval.items() if k != "per_class"},
            "conformal_per_class": cp_eval["per_class"],
            "n_test_windows": int(len(test_df)),
            "ambiguous_window_rate": float(res.is_ambiguous.mean()),
            "args": vars(args),
        }, fh, indent=2, default=str)

    print(f"\nWrote {results_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
