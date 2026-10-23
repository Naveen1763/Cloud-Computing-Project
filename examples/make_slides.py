#!/usr/bin/env python3
"""
Generate presentation-ready figures from a completed run.

Writes three PNGs sized and styled for slides (large type, no clutter):

  1_failures.png    bar chart of SLO violations per controller
  2_tradeoff.png    cost vs reliability scatter
  3_leak.png        effect of fixing the train/test leak

Example:
    python examples/make_slides.py --results-dir results_loadfix
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.facecolor": "white",
})

ORDER = ["Reactive", "Top1_NoMargin_control", "AAPA_Beta", "Conformal_TargetOnly"]
LABEL = {
    "Reactive": "HPA\n(reactive)",
    "Top1_NoMargin_control": "Ablation\ncontrol",
    "AAPA_Beta": "AAPA\n(paper)",
    "Conformal_TargetOnly": "Conformal\n(ours)",
}
COLOR = {
    "Reactive": "#9e9e9e",
    "Top1_NoMargin_control": "#b39ddb",
    "AAPA_Beta": "#e53935",
    "Conformal_TargetOnly": "#1e88e5",
}

# Grayscale palette for print reports.
MONO = {
    "Reactive": "#d9d9d9",
    "Top1_NoMargin_control": "#a6a6a6",
    "AAPA_Beta": "#6e6e6e",
    "Conformal_TargetOnly": "#1a1a1a",
}
ACCENT = "#1e88e5"


def parse_args():
    p = argparse.ArgumentParser(description="Slide figures")
    p.add_argument("--results-dir", default="results_loadfix")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--mono", action="store_true",
                   help="Grayscale palette for print reports")
    return p.parse_args()


def load(results_dir):
    csv = Path(results_dir) / "controller_decisions.csv"
    if not csv.exists():
        raise SystemExit(f"{csv} not found. Run the controller comparison first.")
    df = pd.read_csv(csv)
    present = [c for c in ORDER if c in set(df.controller)]
    stats = df.groupby("controller").agg(
        failures=("violation", "sum"),
        rate=("violation", "mean"),
        pods=("provisioned", "mean"),
    ).loc[present]
    return stats, present, len(df) // df.controller.nunique()


def fig_failures(stats, present, n, out):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    vals = [stats.loc[c].failures for c in present]
    bars = ax.bar(
        range(len(present)), vals,
        color=[COLOR[c] for c in present], edgecolor="black", linewidth=0.8, width=0.62,
    )
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.025,
                f"{int(v)}", ha="center", fontweight="bold", fontsize=15)

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([LABEL[c] for c in present])
    ax.set_ylabel("SLO violations")
    ax.set_title(f"Failures across {n:,} scaling decisions", pad=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # Call out the headline comparison.
    if "AAPA_Beta" in present and "Conformal_TargetOnly" in present:
        i_a, i_c = present.index("AAPA_Beta"), present.index("Conformal_TargetOnly")
        a, c = stats.loc["AAPA_Beta"].failures, stats.loc["Conformal_TargetOnly"].failures
        ax.annotate(
            f"{(a - c) / a * 100:.0f}% fewer",
            xy=(i_c, c), xytext=(i_a + 0.1, max(vals) * 0.62),
            arrowprops=dict(arrowstyle="->", lw=2, color=ACCENT),
            fontsize=15, fontweight="bold", color=ACCENT,
        )

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  {out}")


def fig_tradeoff(stats, present, out):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for c in present:
        r = stats.loc[c]
        ax.scatter(r.pods, r.rate * 100, s=520, color=COLOR[c],
                   edgecolor="black", linewidth=1.2, zorder=3)

    # Label placement: nudge apart when points nearly coincide.
    placed = []
    for c in present:
        r = stats.loc[c]
        dy = 34
        for px, py in placed:
            if abs(r.pods - px) < 40 and abs(r.rate * 100 - py) < 0.06:
                dy = -46 if dy > 0 else 34
        placed.append((r.pods, r.rate * 100))
        ax.annotate(LABEL[c].replace("\n", " "), (r.pods, r.rate * 100),
                    textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=14, fontweight="bold",
                    color=COLOR[c])

    ax.set_xlabel("Pods provisioned   (cost →)")
    ax.set_ylabel("SLO violation rate (%)")
    ax.set_title("Better on both axes: fewer failures, less compute", pad=15)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.margins(0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("better", xy=(0.04, 0.08), xycoords="axes fraction",
                fontsize=13, style="italic", color="#555")

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  {out}")


def fig_leak(out, MONO_ON=False):
    """Effect of removing the train/test leak. Values measured, see roadmap."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    for ax, (title, before, after, ylab) in zip(axes, [
        ("Classifier accuracy", 96.4, 90.7, "%"),
        ("Predictions with a single archetype", 95.2, 1.4, "%"),
    ]):
        bars = ax.bar([0, 1], [before, after], width=0.55,
                      color=["#d9d9d9", "#1a1a1a"] if MONO_ON else ["#bdbdbd", "#1e88e5"],
                      edgecolor="black", linewidth=0.8)
        for b, v in zip(bars, [before, after]):
            ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.1f}{ylab}",
                    ha="center", fontweight="bold", fontsize=15)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Split by day\n(original)", "Split by workload\n(fixed)"])
        ax.set_ylim(0, 112)
        ax.set_title(title, fontsize=15, pad=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    fig.suptitle("The original split let the classifier memorise workloads",
                 fontsize=17, y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  {out}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir or Path(args.results_dir) / "slides")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mono:
        COLOR.update(MONO)
        globals()["ACCENT"] = "#1a1a1a"

    stats, present, n = load(args.results_dir)
    print("Wrote:")
    fig_failures(stats, present, n, out_dir / "1_failures.png")
    fig_tradeoff(stats, present, out_dir / "2_tradeoff.png")
    fig_leak(out_dir / "3_leak.png", MONO_ON=args.mono)
    return 0


if __name__ == "__main__":
    sys.exit(main())
