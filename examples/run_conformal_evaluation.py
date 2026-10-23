#!/usr/bin/env python3
"""
Evaluate conformal prediction sets across alpha, score function, and
calibration mode.

Reads the cached score matrices written by run_classification.py, so it
re-calibrates in milliseconds instead of retraining. Reports empirical
coverage against the target, and the set-size distribution that says whether
the sets are actually informative.

Examples:
    python examples/run_conformal_evaluation.py
    python examples/run_conformal_evaluation.py --model-dir models_gt
    python examples/run_conformal_evaluation.py --plot
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.models.conformal import ConformalClassifier

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ALPHAS = [0.01, 0.05, 0.10, 0.20]


def parse_args():
    parser = argparse.ArgumentParser(description="Conformal prediction evaluation")
    parser.add_argument("--model-dir", default="models", help="Directory from run_classification.py")
    parser.add_argument("--results-dir", default="results", help="Where to write the report")
    parser.add_argument("--plot", action="store_true", help="Write a coverage/efficiency figure")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_scores(model_dir: Path):
    path = model_dir / "conformal_scores.npz"
    if not path.exists():
        raise SystemExit(
            f"{path} not found.\nRun: python examples/run_classification.py"
        )
    data = np.load(path, allow_pickle=True)
    return (
        data["calib_probs"],
        data["calib_labels"],
        data["test_probs"],
        data["test_labels"],
        data["classes"],
    )


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    calib_probs, calib_labels, test_probs, test_labels, classes = load_scores(model_dir)

    print(f"Calibration windows: {len(calib_labels)}   Test windows: {len(test_labels)}")
    print(f"Classes: {list(classes)}")

    top1 = classes[np.argmax(test_probs, axis=1)]
    accuracy = float((top1 == test_labels).mean())
    mean_conf = float(np.max(test_probs, axis=1).mean())
    print(f"Top-1 accuracy: {accuracy:.4f}   Mean softmax confidence: {mean_conf:.4f}")
    print(
        "Softmax gap (confidence - accuracy): "
        f"{mean_conf - accuracy:+.4f}  <- positive means overconfident\n"
    )

    header = (
        f"{'method':>7} {'calib':>10} {'alpha':>6} {'target':>7} "
        f"{'empirical':>10} {'avg size':>9} {'singleton':>10}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for method in ("lac", "aps"):
        for mondrian in (False, True):
            for alpha in ALPHAS:
                cp = ConformalClassifier(
                    alpha=alpha,
                    method=method,
                    mondrian=mondrian,
                    random_state=args.seed,
                ).calibrate(calib_probs, calib_labels, classes=classes)
                res = cp.evaluate(test_probs, test_labels)

                label = "per-class" if mondrian else "marginal"
                print(
                    f"{method:>7} {label:>10} {alpha:>6.2f} "
                    f"{res['target_coverage']:>7.2f} {res['empirical_coverage']:>10.4f} "
                    f"{res['avg_set_size']:>9.3f} {res['singleton_rate']:>10.3f}"
                )
                rows.append(
                    {
                        "method": method,
                        "mondrian": mondrian,
                        **{k: v for k, v in res.items() if k != "per_class"},
                        "per_class": res["per_class"],
                    }
                )

    # Class-conditional coverage is where marginal calibration can quietly fail.
    print("\nPer-class coverage at alpha=0.10:")
    for method in ("lac", "aps"):
        for mondrian in (False, True):
            match = next(
                r
                for r in rows
                if r["method"] == method
                and r["mondrian"] == mondrian
                and abs(r["alpha"] - 0.10) < 1e-9
            )
            label = "per-class" if mondrian else "marginal"
            parts = [
                f"{name}={stats['coverage']:.3f}"
                for name, stats in sorted(match["per_class"].items())
            ]
            print(f"  {method:>4} {label:>10}: " + "  ".join(parts))

    report_path = results_dir / "conformal_evaluation.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"accuracy": accuracy, "mean_confidence": mean_conf, "sweep": rows},
            fh,
            indent=2,
            default=str,
        )
    print(f"\nWrote {report_path}")

    if args.plot:
        write_plot(rows, results_dir / "figures" / "conformal_coverage.png")

    return 0


def write_plot(rows, output_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_cov, ax_size) = plt.subplots(1, 2, figsize=(12, 5))

    for method in ("lac", "aps"):
        for mondrian in (False, True):
            subset = sorted(
                (r for r in rows if r["method"] == method and r["mondrian"] == mondrian),
                key=lambda r: r["alpha"],
            )
            if not subset:
                continue
            label = f"{method.upper()} ({'per-class' if mondrian else 'marginal'})"
            targets = [r["target_coverage"] for r in subset]
            ax_cov.plot(
                targets, [r["empirical_coverage"] for r in subset], "o-", label=label
            )
            ax_size.plot(targets, [r["avg_set_size"] for r in subset], "o-", label=label)

    lo = min(r["target_coverage"] for r in rows)
    ax_cov.plot([lo, 1.0], [lo, 1.0], "k--", alpha=0.5, label="ideal")
    ax_cov.set_xlabel("Target coverage (1 - alpha)")
    ax_cov.set_ylabel("Empirical coverage")
    ax_cov.set_title("Coverage calibration")
    ax_cov.legend(fontsize=8)
    ax_cov.grid(alpha=0.3)

    ax_size.set_xlabel("Target coverage (1 - alpha)")
    ax_size.set_ylabel("Average prediction set size")
    ax_size.set_title("Efficiency (lower is more informative)")
    ax_size.legend(fontsize=8)
    ax_size.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    sys.exit(main())
