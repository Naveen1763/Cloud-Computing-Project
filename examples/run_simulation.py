#!/usr/bin/env python3
"""
Run the AAPA autoscaling simulation and compare it against baselines.

Simulates each test workload under four autoscalers - AAPA (archetype-aware),
AAPA_Conformal (archetype-aware with conformal prediction sets), Kubernetes
HPA, and a generic predictive autoscaler - then reports SLO violations,
response times, pod-minutes and scaling churn.

AAPA and AAPA_Conformal differ only in how they translate a classification
into a strategy: the former softens SPIKE's target when the softmax score
looks low, the latter provisions for the worst case across every archetype
the conformal predictor cannot rule out. They are identical whenever the
prediction set is a singleton, so any gap between them comes purely from the
windows the model is genuinely unsure about.

Run examples/run_classification.py first so the trained classifier exists.

Examples:
    python examples/run_simulation.py
    python examples/run_simulation.py --max-workloads 30 --sim-minutes 720
    python examples/run_simulation.py --model-dir models_gt --alpha 0.05
    python examples/run_simulation.py --no-conformal      # 3-arm comparison
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import simpy

from src.core.simulator import (
    AAPAAutoscaler,
    ConformalAAPAAutoscaler,
    ConformalTargetOnlyAutoscaler,
    Top1TargetOnlyAutoscaler,
    GenericPredictiveAutoscaler,
    HPAAutoscaler,
    ServerlessSimulator,
    calculate_rei,
)
from src.data.features import TimeSeriesFeatureExtractor
from src.data.kswd_loader import KSWDLoader
from src.utils.rei import compute_all as compute_all_rei

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="AAPA autoscaling simulation")
    parser.add_argument("--model-dir", default="models", help="Directory with the trained model")
    parser.add_argument("--dataset-dir", default="dataset", help="KSWD dataset directory")
    parser.add_argument("--results-dir", default="results", help="Where to write results")
    parser.add_argument("--max-workloads", type=int, default=20, help="Workloads to simulate")
    parser.add_argument(
        "--sim-minutes",
        type=int,
        default=360,
        help="Minutes of each trace to simulate (shorter = faster)",
    )
    parser.add_argument("--window-size", type=int, default=60, help="Feature window in minutes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override the conformal miscoverage level used at scaling time",
    )
    parser.add_argument(
        "--no-conformal",
        action="store_true",
        help="Skip the AAPA_Conformal arm",
    )
    return parser.parse_args()


def build_feature_list(trace, window_size, feature_names, scaler):
    """Build one scaled feature vector per simulated minute.

    Each minute uses a trailing window of window_size samples, matching the
    windowing used during training.
    """
    extractor = TimeSeriesFeatureExtractor()
    rows = []
    for minute in range(len(trace)):
        start = max(0, minute - window_size + 1)
        window = trace[start : minute + 1]
        if len(window) < window_size:  # left-pad the warm-up period
            window = np.concatenate([np.zeros(window_size - len(window)), window])
        rows.append(extractor.extract_features(window))

    features_df = pd.DataFrame(rows)
    # Align to the exact column order the classifier was trained on.
    features_df = features_df.reindex(columns=feature_names, fill_value=0.0)
    features_df = features_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    scaled = scaler.transform(features_df.values)
    return [row for row in scaled]


def summarise(metrics, autoscaler_name, workload_id, archetype, sim_minutes):
    """Turn simulator Metrics into a flat result row."""
    pod_minutes = metrics.pod_minutes
    # Mean utilisation proxy: completed work over provisioned pod-minutes.
    mean_cpu = (
        min(1.0, metrics.completed_requests / (pod_minutes * 60.0))
        if pod_minutes > 0
        else 0.0
    )
    return {
        "workload_id": workload_id,
        "workload_type": archetype,
        "autoscaler": autoscaler_name,
        "slo_violation_rate": metrics.get_slo_violation_rate(),
        "mean_response_time": metrics.get_mean_response_time(),
        "p99_response_time": metrics.get_p99_response_time(),
        "total_pod_minutes": pod_minutes,
        "mean_cpu_utilization": mean_cpu,
        "cold_starts": metrics.cold_starts,
        "scaling_actions": metrics.scaling_actions,
        "total_requests": metrics.total_requests,
        "completed_requests": metrics.completed_requests,
        "rei": calculate_rei(metrics),
        **{
            f"rei_{name}": value
            for name, value in compute_all_rei(
                metrics.get_slo_violation_rate(),
                metrics.pod_minutes,
                metrics.scaling_actions,
            ).items()
        },
        "sim_minutes": sim_minutes,
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)

    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    booster_path = model_dir / "lightgbm_model.txt"
    encoder_path = model_dir / "label_encoder.pkl"
    scaler_path = model_dir / "scaler.pkl"
    names_path = model_dir / "feature_names.txt"

    missing = [p for p in (booster_path, encoder_path, scaler_path, names_path) if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing model artifacts: "
            + ", ".join(str(p) for p in missing)
            + "\nRun: python examples/run_classification.py"
        )

    feature_names = names_path.read_text(encoding="utf-8").split()
    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)

    # Load test workloads
    logger.info("Loading test workloads...")
    loader = KSWDLoader(args.dataset_dir)
    traces = loader.load_invocation_data(max_workloads=args.max_workloads)
    archetypes = loader.load_archetype_labels(list(traces))

    # Trim traces so the simulation stays tractable.
    traces = {k: v[: args.sim_minutes] for k, v in traces.items() if np.sum(v) > 0}
    logger.info(f"Simulating {len(traces)} workloads for {args.sim_minutes} minutes each")

    # Each (workload, autoscaler) pair draws from its own stream, so adding or
    # reordering arms cannot shift the service times another arm sees. Without
    # this the between-arm differences are swamped by RNG-ordering artifacts.
    def make_execution_time_dist(workload_id: str, autoscaler_name: str):
        stream = np.random.default_rng(
            abs(hash((args.seed, workload_id))) % (2 ** 32)
        )
        return lambda: stream.exponential(0.1)

    autoscaler_factories = {
        "AAPA": lambda: AAPAAutoscaler(
            model_path=str(booster_path), encoder_path=str(encoder_path)
        ),
        "HPA": HPAAutoscaler,
        "Generic_Predictive": GenericPredictiveAutoscaler,
    }

    conformal_path = model_dir / "conformal.pkl"
    if not args.no_conformal:
        if conformal_path.exists():
            autoscaler_factories["AAPA_Conformal"] = lambda: ConformalAAPAAutoscaler(
                model_path=str(booster_path),
                encoder_path=str(encoder_path),
                conformal_path=str(conformal_path),
                alpha=args.alpha,
            )
            autoscaler_factories["Top1_TargetOnly_control"] = lambda: (
                Top1TargetOnlyAutoscaler(
                    model_path=str(booster_path),
                    encoder_path=str(encoder_path),
                    conformal_path=str(conformal_path),
                    alpha=args.alpha,
                )
            )
            autoscaler_factories["Conformal_TargetOnly"] = lambda: (
                ConformalTargetOnlyAutoscaler(
                    model_path=str(booster_path),
                    encoder_path=str(encoder_path),
                    conformal_path=str(conformal_path),
                    alpha=args.alpha,
                )
            )
        else:
            logger.warning(
                f"{conformal_path} not found; skipping the AAPA_Conformal arm. "
                "Re-run examples/run_classification.py to create it."
            )

    all_results = []
    for i, (workload_id, trace) in enumerate(traces.items(), start=1):
        archetype = archetypes.get(workload_id, "UNKNOWN")
        logger.info(f"[{i}/{len(traces)}] {workload_id} ({archetype})")

        features_list = build_feature_list(trace, args.window_size, feature_names, scaler)

        for name, factory in autoscaler_factories.items():
            env = simpy.Environment()
            sim = ServerlessSimulator(env, factory())
            metrics = sim.run(
                workload=list(trace),
                function_id=workload_id,
                execution_time_dist=make_execution_time_dist(workload_id, name),
                features_list=features_list if name.startswith("AAPA") else None,
            )
            row = summarise(metrics, name, workload_id, archetype, args.sim_minutes)
            sizes = getattr(sim.autoscaler, "set_sizes", None)
            row["avg_set_size"] = float(np.mean(sizes)) if sizes else float("nan")
            all_results.append(row)

    results_df = pd.DataFrame(all_results)
    results_path = results_dir / "simulation_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved detailed results to {results_path}")

    # Aggregate view
    agg = (
        results_df.groupby("autoscaler")[
            [
                "slo_violation_rate",
                "mean_response_time",
                "p99_response_time",
                "total_pod_minutes",
                "scaling_actions",
                "cold_starts",
                "rei",
                "avg_set_size",
            ]
        ]
        .mean()
        .round(4)
    )
    print("\n" + "=" * 78)
    print("Aggregate results by autoscaler")
    print("=" * 78)
    print(agg.to_string())

    # AAPA vs HPA, per workload
    aapa = results_df[results_df["autoscaler"] == "AAPA"].set_index("workload_id")
    hpa = results_df[results_df["autoscaler"] == "HPA"].set_index("workload_id")
    common = aapa.index.intersection(hpa.index)

    if len(common):
        print("\n" + "=" * 78)
        print("AAPA improvement over HPA (positive = AAPA better)")
        print("=" * 78)
        for metric in ["slo_violation_rate", "mean_response_time", "p99_response_time"]:
            a = aapa.loc[common, metric]
            h = hpa.loc[common, metric]
            denom = h.replace(0, np.nan)
            improvement = ((h - a) / denom * 100).mean()
            print(f"  {metric:<24} {improvement:6.1f}%")

        overhead = (
            aapa.loc[common, "total_pod_minutes"]
            / hpa.loc[common, "total_pod_minutes"].replace(0, np.nan)
        ).mean()
        print(f"  {'resource overhead':<24} {overhead:6.2f}x")

    # Per-archetype breakdown
    print("\n" + "=" * 78)
    print("SLO violation rate by archetype")
    print("=" * 78)
    pivot = results_df.pivot_table(
        index="workload_type", columns="autoscaler", values="slo_violation_rate", aggfunc="mean"
    ).round(4)
    print(pivot.to_string())

    logger.info("Simulation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
