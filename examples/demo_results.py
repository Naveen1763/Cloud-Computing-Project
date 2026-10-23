#!/usr/bin/env python3
"""
Show the headline comparison from a completed run. Instant - reads saved
results, computes nothing.

Use this to present findings without waiting on the full pipeline. To
regenerate the underlying data see the command printed at the end.

Example:
    python examples/demo_results.py
    python examples/demo_results.py --results-dir results_v2 --figure
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

ORDER = [
    "Reactive",
    "Top1_NoMargin_control",
    "AAPA_Beta",
    "Conformal_TargetOnly",
    "Conformal_Envelope",
]

PRETTY = {
    "Reactive": "Reactive (HPA-style)",
    "Top1_NoMargin_control": "Top-1 only (ablation control)",
    "AAPA_Beta": "AAPA, beta-calibrated (paper)",
    "Conformal_TargetOnly": "Conformal sets (ours)",
    "Conformal_Envelope": "Conformal + warm pool (ours)",
}


def parse_args():
    p = argparse.ArgumentParser(description="Present saved controller results")
    p.add_argument("--results-dir", default="results_loadfix")
    p.add_argument("--figure", action="store_true", help="Also write a comparison PNG")
    return p.parse_args()


def rule(char="=", n=78):
    print(char * n)


def main():
    args = parse_args()
    d = Path(args.results_dir)
    csv = d / "controller_decisions.csv"
    if not csv.exists():
        raise SystemExit(
            f"{csv} not found.\n"
            "Run: python examples/run_conformal_controllers.py "
            "--max-workloads 200 --mondrian --split workload --load-multiplier 60"
        )

    df = pd.read_csv(csv)
    report = {}
    rp = d / "run_report.json"
    if rp.exists():
        report = json.load(open(rp, encoding="utf-8"))

    n_windows = len(df) // df.controller.nunique()
    present = [c for c in ORDER if c in set(df.controller)]

    stats = df.groupby("controller").agg(
        failures=("violation", "sum"),
        rate=("violation", "mean"),
        pods=("provisioned", "mean"),
    ).loc[present]

    base = stats.loc["AAPA_Beta"] if "AAPA_Beta" in stats.index else None

    print()
    rule()
    print("  AUTOSCALING CONTROLLER COMPARISON")
    rule()
    print(f"  {n_windows:,} scaling decisions, evaluated against actual next-window demand")
    if report:
        cfg = report.get("args", {})
        print(
            f"  split: {cfg.get('split', '?')}   "
            f"load: {cfg.get('load_multiplier', 1)}x   "
            f"classifier accuracy: {report.get('accuracy', float('nan')):.3f}"
        )
        cp = report.get("conformal", {})
        if cp:
            print(
                f"  conformal: avg set size {cp.get('avg_set_size', 0):.2f}, "
                f"coverage {cp.get('empirical_coverage', 0):.3f} "
                f"(target {cp.get('target_coverage', 0):.2f})"
            )
    rule()
    print()

    head = f"  {'Controller':<32}{'Failures':>10}{'Rate':>10}{'Pods':>10}"
    if base is not None:
        head += f"{'vs AAPA':>22}"
    print(head)
    print("  " + "-" * (len(head) - 2))

    for name in present:
        row = stats.loc[name]
        line = (
            f"  {PRETTY.get(name, name):<32}{int(row.failures):>10,}"
            f"{row.rate:>10.4f}{row.pods:>10.1f}"
        )
        if base is not None and name != "AAPA_Beta":
            dv = (row.rate - base.rate) / base.rate * 100
            dp = (row.pods - base.pods) / base.pods * 100
            line += f"{dv:>+11.1f}% {dp:>+8.1f}%"
        elif name == "AAPA_Beta":
            line += f"{'(baseline)':>22}"
        print(line)

    print()
    if base is not None and "Conformal_TargetOnly" in stats.index:
        ours = stats.loc["Conformal_TargetOnly"]
        dv = (ours.rate - base.rate) / base.rate * 100
        dp = (ours.pods - base.pods) / base.pods * 100
        rule("-")
        print(
            f"  HEADLINE: conformal prediction sets vs AAPA's beta-calibrated margin"
        )
        print(
            f"            {abs(dv):.0f}% {'fewer' if dv < 0 else 'more'} SLO violations, "
            f"{abs(dp):.1f}% {'less' if dp < 0 else 'more'} compute"
        )
        if "Top1_NoMargin_control" in stats.index:
            ctrl = stats.loc["Top1_NoMargin_control"]
            gap = (ctrl.rate - ours.rate) / ours.rate * 100
            print(
                f"  CONTROL:  the no-hedge ablation fails {gap:.0f}% more often "
                f"({int(ctrl.failures):,} vs {int(ours.failures):,}) -"
            )
            print( "            confirming the prediction set is doing the work")
        rule("-")

    # Per-archetype, if more than one is present
    arch = df.archetype_true.nunique()
    if arch > 1:
        print()
        print("  SLO violation rate by archetype")
        piv = df.pivot_table(
            index="archetype_true", columns="controller",
            values="violation", aggfunc="mean",
        )[present].round(4)
        piv.columns = [PRETTY.get(c, c) for c in piv.columns]
        print(piv.to_string().replace("\n", "\n  ").rjust(2))

    print()
    print("  Regenerate with:")
    cfg = report.get("args", {})
    print(
        f"    python examples/run_conformal_controllers.py "
        f"--max-workloads {cfg.get('max_workloads', 200)} --mondrian "
        f"--split {cfg.get('split', 'workload')} "
        f"--load-multiplier {cfg.get('load_multiplier', 60)} "
        f"--results-dir {args.results_dir}"
    )
    print()

    if args.figure:
        write_figure(stats, present, d / "figures" / "controller_comparison.png")

    return 0


def write_figure(stats, present, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {
        "Reactive": "#999999",
        "Top1_NoMargin_control": "#c8a2c8",
        "AAPA_Beta": "#d62728",
        "Conformal_TargetOnly": "#1f77b4",
        "Conformal_Envelope": "#2ca02c",
    }

    # Points can sit almost on top of each other, so stagger labels that would
    # otherwise overlap rather than letting them print over one another.
    placed = []
    for name in present:
        row = stats.loc[name]
        ax.scatter(
            row.pods, row.rate * 100, s=260, alpha=0.85,
            color=colors.get(name, "#333333"), edgecolor="black", zorder=3,
        )

        x, y = row.pods, row.rate * 100
        dy = 18
        for px, py in placed:
            close_x = abs(x - px) < 0.05 * max(stats.pods.max(), 1)
            close_y = abs(y - py) < 0.06 * max(stats.rate.max() * 100, 1e-6)
            if close_x and close_y:
                dy = -24 if dy > 0 else 18
        placed.append((x, y))

        ax.annotate(
            PRETTY.get(name, name), (x, y),
            textcoords="offset points", xytext=(0, dy),
            ha="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
        )

    ax.set_xlabel("Mean pods provisioned  (cost -->)", fontsize=11)
    ax.set_ylabel("SLO violation rate (%)  (<-- better)", fontsize=11)
    ax.set_title(
        "Cost vs reliability: conformal sets vs beta-calibrated confidence",
        fontsize=12,
    )
    ax.grid(alpha=0.3, zorder=0)
    ax.margins(0.18)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
