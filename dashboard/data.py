"""Data access for the results dashboard.

Reads the CSV/JSON artifacts written by examples/run_conformal_controllers.py.
Nothing is recomputed here - the dashboard only presents saved runs, so it
loads instantly and cannot fail partway through a demo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Display order and labels, shared by every page.
CONTROLLER_ORDER = [
    "Reactive",
    "Top1_NoMargin_control",
    "AAPA_Beta",
    "Conformal_TargetOnly",
    "Conformal_Envelope",
]

CONTROLLER_LABEL = {
    "Reactive": "HPA (reactive)",
    "Top1_NoMargin_control": "Ablation control",
    "AAPA_Beta": "AAPA (paper)",
    "Conformal_TargetOnly": "Conformal (ours)",
    "Conformal_Envelope": "Conformal + warm pool",
}

CONTROLLER_COLOR = {
    "Reactive": "#9e9e9e",
    "Top1_NoMargin_control": "#b39ddb",
    "AAPA_Beta": "#e53935",
    "Conformal_TargetOnly": "#1e88e5",
    "Conformal_Envelope": "#43a047",
}

BASELINE = "AAPA_Beta"
OURS = "Conformal_TargetOnly"
CONTROL = "Top1_NoMargin_control"


@dataclass(frozen=True)
class Experiment:
    """One saved run of the controller comparison."""

    key: str
    name: str
    blurb: str
    directory: Path

    @property
    def available(self) -> bool:
        return (self.directory / "controller_decisions.csv").exists()


# Registered runs, in the order they should appear in the selector.
EXPERIMENTS: List[Experiment] = [
    Experiment(
        key="loadfix",
        name="Main result",
        blurb=(
            "Leak-free workload split at realistic load. This is the headline "
            "experiment: the archetype the classifier ranks first is usually "
            "not the most resource-intensive one, so hedging has room to act."
        ),
        directory=ROOT / "results_loadfix",
    ),
    Experiment(
        key="spike",
        name="With SPIKE restored",
        blurb=(
            "The activity filter that removes 98% of SPIKE workloads is "
            "disabled and load is raised further. Exposes the capacity ceiling: "
            "when SPIKE is predicted it is already the most demanding "
            "archetype, so there is nothing left to hedge toward."
        ),
        directory=ROOT / "results_v2",
    ),
    Experiment(
        key="workload_split",
        name="Leak-free split, low load",
        blurb=(
            "The workload-level split before the load regime was corrected. "
            "Most windows need a single pod here, so the controllers are "
            "barely separated."
        ),
        directory=ROOT / "results_workload_split",
    ),
    Experiment(
        key="day_split",
        name="Original day split",
        blurb=(
            "The paper's own protocol, which places the same workload in both "
            "training and test. The classifier memorises rather than "
            "generalises, so 95% of predictions name a single archetype."
        ),
        directory=ROOT / "results_ctrl_ablation",
    ),
]


def available_experiments() -> List[Experiment]:
    return [e for e in EXPERIMENTS if e.available]


def find_experiment(key: str) -> Optional[Experiment]:
    return next((e for e in EXPERIMENTS if e.key == key), None)


def load_decisions(exp: Experiment) -> pd.DataFrame:
    """Per-decision records: one row per (window, controller)."""
    return pd.read_csv(exp.directory / "controller_decisions.csv")


def load_report(exp: Experiment) -> Dict:
    """Classifier and conformal metadata for the run."""
    path = exp.directory / "run_report.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def summarise(decisions: pd.DataFrame) -> pd.DataFrame:
    """Per-controller totals, ordered for display.

    Returns a frame indexed by controller with failure counts, violation
    rate, mean pods provisioned, and the deltas against the AAPA baseline.
    """
    present = [c for c in CONTROLLER_ORDER if c in set(decisions.controller)]

    stats = decisions.groupby("controller").agg(
        failures=("violation", "sum"),
        rate=("violation", "mean"),
        pods=("provisioned", "mean"),
        excess=("excess_replicas", "mean"),
        shortfall=("shortfall", lambda s: s[s > 0].mean() if (s > 0).any() else 0.0),
    ).loc[present]

    if BASELINE in stats.index:
        base = stats.loc[BASELINE]
        stats["rate_vs_base"] = (stats.rate - base.rate) / base.rate * 100
        stats["pods_vs_base"] = (stats.pods - base.pods) / base.pods * 100
    else:
        stats["rate_vs_base"] = float("nan")
        stats["pods_vs_base"] = float("nan")

    stats["label"] = [CONTROLLER_LABEL.get(c, c) for c in stats.index]
    return stats


def headline(stats: pd.DataFrame) -> Dict[str, Optional[float]]:
    """The three numbers the overview page leads with."""
    out: Dict[str, Optional[float]] = {
        "violation_delta": None,
        "cost_delta": None,
        "control_gap": None,
        "ours_failures": None,
        "base_failures": None,
        "control_failures": None,
    }
    if BASELINE not in stats.index or OURS not in stats.index:
        return out

    base, ours = stats.loc[BASELINE], stats.loc[OURS]
    out["violation_delta"] = float(ours.rate_vs_base)
    out["cost_delta"] = float(ours.pods_vs_base)
    out["ours_failures"] = int(ours.failures)
    out["base_failures"] = int(base.failures)

    if CONTROL in stats.index and ours.rate > 0:
        control = stats.loc[CONTROL]
        out["control_gap"] = float((control.rate - ours.rate) / ours.rate * 100)
        out["control_failures"] = int(control.failures)
    return out


def per_archetype(decisions: pd.DataFrame, value: str) -> pd.DataFrame:
    """Pivot one metric by true archetype and controller."""
    present = [c for c in CONTROLLER_ORDER if c in set(decisions.controller)]
    piv = decisions.pivot_table(
        index="archetype_true", columns="controller", values=value, aggfunc="mean"
    )
    piv = piv[[c for c in present if c in piv.columns]]
    piv.columns = [CONTROLLER_LABEL.get(c, c) for c in piv.columns]
    return piv


def by_ambiguity(decisions: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split the run into clean (singleton) and ambiguous windows.

    Conformal hedging can only act where the prediction set holds more than
    one archetype, so this split is where the contribution shows up.
    """
    return {
        "ambiguous": summarise(decisions[decisions.is_ambiguous]),
        "clean": summarise(decisions[~decisions.is_ambiguous]),
    }


def window_counts(decisions: pd.DataFrame) -> Dict[str, int]:
    n_controllers = max(decisions.controller.nunique(), 1)
    one = decisions[decisions.controller == decisions.controller.iloc[0]]
    return {
        "total": len(decisions) // n_controllers,
        "ambiguous": int(one.is_ambiguous.sum()),
        "clean": int((~one.is_ambiguous).sum()),
    }


# Findings measured during the study. Sources are named so every number on
# the dashboard can be traced back to the run that produced it.
FINDINGS = {
    "leak": {
        "title": "The train/test split leaks workloads",
        "detail": (
            "The published pipeline splits by day, so days 1-9 of a workload "
            "train the model and days 12-13 of the same workload test it. A "
            "workload's archetype never changes, so the classifier can score "
            "well by recognising the workload and recalling its label. "
            "Verified: 100% of test workloads also appear in training."
        ),
        "metrics": [
            ("Classifier accuracy", 96.4, 90.7, "%"),
            ("Predictions naming one archetype", 95.2, 1.4, "%"),
        ],
        "labels": ("Split by day (original)", "Split by workload (fixed)"),
        "takeaway": (
            "Accuracy barely moves, but apparent certainty collapses from 95% "
            "to 1.4%. The confidence was an artefact of the leak."
        ),
    },
    "labels": {
        "title": "The weak-supervision labels are circular",
        "detail": (
            "Labels are computed from the same window statistics given to the "
            "classifier as features, so training against them measures how "
            "well the model reproduces the labelling functions rather than "
            "how well it recovers workload behaviour."
        ),
        "metrics": [("Agreement with ground-truth archetypes", 100.0, 18.6, "%")],
        "labels": ("If labels were sound", "Measured"),
        "takeaway": (
            "18.6% agreement explains the reported 99.8% accuracy: the task "
            "being scored is not archetype recovery."
        ),
    },
    "spike": {
        "title": "SPIKE is deleted before evaluation",
        "detail": (
            "The windowing stage drops any workload active in fewer than 10% "
            "of minutes. Bursty traffic is idle by definition, so the filter "
            "removes almost every SPIKE workload - the archetype on which the "
            "paper's strongest claims rest."
        ),
        "metrics": [
            ("SPIKE workloads removed", 0.0, 97.7, "%"),
            ("PERIODIC workloads removed", 0.0, 78.8, "%"),
            ("STATIONARY workloads removed", 0.0, 1.5, "%"),
        ],
        "labels": ("Intended", "Actually removed"),
        "takeaway": (
            "42 of 43 SPIKE workloads are discarded, leaving an evaluation "
            "that is overwhelmingly STATIONARY."
        ),
    },
}

# MAPIE cross-validation, from examples/validate_conformal_mapie.py on models_gt.
MAPIE_VALIDATION = pd.DataFrame(
    [
        (0.01, "lac", 0.9958, 0.9958, 1.053, 1.053, 1.000),
        (0.05, "lac", 0.9637, 0.9637, 0.999, 0.999, 1.000),
        (0.10, "lac", 0.9631, 0.9631, 0.994, 0.994, 1.000),
        (0.20, "lac", 0.8724, 0.8721, 0.872, 0.872, 1.000),
        (0.01, "aps", 0.9991, 0.9999, 1.070, 1.075, 0.995),
        (0.05, "aps", 0.9946, 0.9998, 1.055, 1.069, 0.986),
        (0.10, "aps", 0.9893, 0.9984, 1.044, 1.061, 0.983),
        (0.20, "aps", 0.9796, 0.9974, 1.027, 1.058, 0.970),
    ],
    columns=[
        "alpha", "method", "ours_coverage", "mapie_coverage",
        "ours_set_size", "mapie_set_size", "set_agreement",
    ],
)

# The archetype capacity ladder, from AAPA Table III.
CAPACITY_LADDER = pd.DataFrame(
    [
        ("PERIODIC", 0.75, 1 / 0.75),
        ("RAMP", 0.60, 1 / 0.60),
        ("STATIONARY", 0.55, 1 / 0.55),
        ("SPIKE", 0.30, 1 / 0.30),
    ],
    columns=["archetype", "target_cpu", "pods_per_unit_load"],
)
