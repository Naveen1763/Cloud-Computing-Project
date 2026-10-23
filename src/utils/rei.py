"""Resource Efficiency Index, with the weightings pinned explicitly.

The AAPA paper reports REI = a*S_slo + b*S_eff + c*S_stab but states the
weights inconsistently: Section III-C gives (0.5, 0.3, 0.2), while the
Table VI sensitivity baseline gives (0.4, 0.3, 0.3). The simulator in this
repository used a third setting, (0.4, 0.4, 0.2). Since REI is the metric on
which the paper concedes HPA outranks AAPA, any claim about closing that gap
has to say which weighting it used - so all three are named here and results
are reported under each.
"""

from typing import Dict

WEIGHTINGS: Dict[str, Dict[str, float]] = {
    # Section III-C of the paper.
    "paper_text": {"slo": 0.5, "eff": 0.3, "stab": 0.2},
    # Table VI sensitivity-analysis baseline.
    "paper_table6": {"slo": 0.4, "eff": 0.3, "stab": 0.3},
    # calculate_rei() as shipped in src/core/simulator.py.
    "repo_code": {"slo": 0.4, "eff": 0.4, "stab": 0.2},
}

# Normalisation constants, matching src/core/simulator.py:calculate_rei so
# scores stay comparable with previously published runs.
MAX_POD_MINUTES = 100.0
MAX_SCALING_ACTIONS = 50.0


def rei_components(slo_violation_rate: float,
                   pod_minutes: float,
                   scaling_actions: float,
                   max_pod_minutes: float = MAX_POD_MINUTES,
                   max_scaling_actions: float = MAX_SCALING_ACTIONS) -> Dict[str, float]:
    """Normalise the three REI sub-scores to [0, 1], higher being better."""
    return {
        "slo": 1.0 - slo_violation_rate,
        "eff": 1.0 - min(pod_minutes / max_pod_minutes, 1.0),
        "stab": 1.0 - min(scaling_actions / max_scaling_actions, 1.0),
    }


def compute_rei(slo_violation_rate: float,
                pod_minutes: float,
                scaling_actions: float,
                weighting: str = "repo_code",
                **kwargs) -> float:
    """REI under a named weighting.

    Args:
        weighting: One of ``WEIGHTINGS``.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(
            f"unknown weighting {weighting!r}; expected one of {sorted(WEIGHTINGS)}"
        )
    w = WEIGHTINGS[weighting]
    s = rei_components(slo_violation_rate, pod_minutes, scaling_actions, **kwargs)
    return sum(w[k] * s[k] for k in w)


def compute_all(slo_violation_rate: float,
                pod_minutes: float,
                scaling_actions: float,
                **kwargs) -> Dict[str, float]:
    """REI under every weighting, keyed by name."""
    return {
        name: compute_rei(
            slo_violation_rate, pod_minutes, scaling_actions, weighting=name, **kwargs
        )
        for name in WEIGHTINGS
    }
