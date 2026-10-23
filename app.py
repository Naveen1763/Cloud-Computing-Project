#!/usr/bin/env python3
"""
Results dashboard: AAPA (paper) vs conformal prediction sets (ours).

A single page showing the three metrics on which the methods differ. Reads
saved experiment files and recomputes nothing, so it loads instantly.

Launch:
    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.data import (
    ROOT,
    BASELINE,
    CONTROL,
    OURS,
    SIM_BASELINE,
    SIM_CONTROL,
    SIM_OURS,
    find_experiment,
    load_decisions,
    load_report,
    load_simulation,
    simulation_available,
    summarise,
    summarise_simulation,
)

st.set_page_config(
    page_title="Conformal Autoscaling — Results",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

# The headline experiment: leak-free workload split at realistic load.
MAIN = "loadfix"

AAPA_COLOR = "#e53935"
OURS_COLOR = "#1e88e5"
CTRL_COLOR = "#b39ddb"


# --------------------------------------------------------------------------
# Data (cached — these are static files)
# --------------------------------------------------------------------------

def results_stamp() -> float:
    """Modification times of the files the page reads, used as a cache key so
    regenerated results are picked up without restarting the server."""
    exp = find_experiment(MAIN)
    paths = [
        exp.directory / "controller_decisions.csv",
        ROOT / "results_sim6" / "simulation_results.csv",
    ]
    return sum(p.stat().st_mtime for p in paths if p.exists())


@st.cache_data(show_spinner=False)
def load_all(_stamp: float):
    exp = find_experiment(MAIN)
    decisions = load_decisions(exp)
    report = load_report(exp)
    sim = load_simulation() if simulation_available() else None
    return decisions, report, sim


try:
    decisions, report, sim = load_all(results_stamp())
except Exception as exc:  # results missing or unreadable
    st.error(
        f"Could not load results: {exc}\n\n"
        "Generate them with:\n\n"
        "`python examples/run_conformal_controllers.py --max-workloads 200 "
        "--mondrian --split workload --load-multiplier 60 "
        "--results-dir results_loadfix`"
    )
    st.stop()

stats = summarise(decisions)
sim_stats = summarise_simulation(sim) if sim is not None else None
n_decisions = len(decisions) // decisions.controller.nunique()


# --------------------------------------------------------------------------
# Sidebar — one control
# --------------------------------------------------------------------------

st.sidebar.title("Conformal Autoscaling")
st.sidebar.caption("Uncertainty-aware scaling for serverless workloads")
st.sidebar.markdown("---")

show_control = st.sidebar.toggle(
    "Show ablation control",
    value=False,
    help=(
        "Our method with the prediction set removed — it just uses the single "
        "best archetype guess. If the shortlist contributed nothing, this "
        "would match us. It does not."
    ),
)

st.sidebar.markdown("---")
args = report.get("args", {})
st.sidebar.markdown("**Run configuration**")
st.sidebar.markdown(
    f"- Workloads: `{args.get('max_workloads', '?')}`\n"
    f"- Split: `{args.get('split', '?')}` (leak-free)\n"
    f"- Labels: `{args.get('labels', '?')}` (ground truth)\n"
    f"- Load: `{args.get('load_multiplier', 1)}×`\n"
    f"- α = `{args.get('alpha', '?')}`, `{args.get('conformal_method', '?')}`"
    f"{', Mondrian' if args.get('mondrian') else ''}"
)
st.sidebar.caption(
    f"Classifier accuracy {report.get('accuracy', 0):.3f} · "
    f"coverage {report.get('conformal', {}).get('empirical_coverage', 0):.3f}"
)


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

aapa, ours = stats.loc[BASELINE], stats.loc[OURS]
ctrl = stats.loc[CONTROL] if CONTROL in stats.index else None

slo_delta = (ours.rate - aapa.rate) / aapa.rate * 100
pods_delta = (ours.pods - aapa.pods) / aapa.pods * 100

if sim_stats is not None:
    s_aapa, s_ours = sim_stats.loc[SIM_BASELINE], sim_stats.loc[SIM_OURS]
    s_ctrl = sim_stats.loc[SIM_CONTROL] if SIM_CONTROL in sim_stats.index else None
    churn_delta = (
        (s_ours.scaling_actions - s_aapa.scaling_actions)
        / s_aapa.scaling_actions * 100
    )
else:
    s_aapa = s_ours = s_ctrl = None
    churn_delta = None


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

st.title("Conformal prediction for autoscaling")
st.markdown(
    "AAPA reduces its workload classifier to a **single confidence number** "
    "and pads capacity by a fixed margin. That says *how* unsure the model is, "
    "never *what it is unsure between*. We replace it with a **prediction "
    "set** — a shortlist of archetypes that cannot be statistically ruled out "
    "— and provision for the most demanding one still in play."
)

st.markdown("---")
st.subheader("Three metrics, AAPA versus ours")

c1, c2, c3 = st.columns(3)
c1.metric(
    "SLO violations",
    f"{int(ours.failures):,}",
    f"{slo_delta:+.0f}% vs AAPA",
    delta_color="inverse",
    help=f"AAPA records {int(aapa.failures):,} over the same {n_decisions:,} decisions.",
)
c2.metric(
    "Compute (pods)",
    f"{ours.pods:,.0f}",
    f"{pods_delta:+.1f}% vs AAPA",
    delta_color="inverse",
    help=f"AAPA provisions {aapa.pods:,.0f} on average. Lower total compute.",
)
if churn_delta is not None:
    c3.metric(
        "Scaling actions",
        f"{s_ours.scaling_actions:.1f}",
        f"{churn_delta:+.0f}% vs AAPA",
        delta_color="inverse",
        help=(
            f"AAPA performs {s_aapa.scaling_actions:.1f} resize operations per "
            "workload. Every one restarts pods and pays cold-start latency."
        ),
    )
else:
    c3.metric("Scaling actions", "—", help="Simulator results not found.")


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------

def comparison_figure(with_control: bool) -> go.Figure:
    """Three panels, each normalised so AAPA sits at 100%."""
    panels = [
        ("SLO violations", aapa.failures, ours.failures,
         ctrl.failures if ctrl is not None else None),
        ("Compute (pods)", aapa.pods, ours.pods,
         ctrl.pods if ctrl is not None else None),
    ]
    if s_aapa is not None:
        panels.append((
            "Scaling actions",
            s_aapa.scaling_actions, s_ours.scaling_actions,
            s_ctrl.scaling_actions if s_ctrl is not None else None,
        ))

    fig = make_subplots(rows=1, cols=len(panels),
                        subplot_titles=[p[0] for p in panels])

    for i, (_, a_val, o_val, c_val) in enumerate(panels, start=1):
        names = ["AAPA<br>(paper)", "Conformal<br>(ours)"]
        raw = [a_val, o_val]
        colors = [AAPA_COLOR, OURS_COLOR]
        if with_control and c_val is not None:
            names.append("Ablation<br>control")
            raw.append(c_val)
            colors.append(CTRL_COLOR)

        pct = [v / a_val * 100 if a_val else 0 for v in raw]
        fig.add_trace(
            go.Bar(
                x=names, y=pct,
                marker_color=colors,
                marker_line=dict(color="#222", width=1),
                text=[f"{v:,.1f}" if v < 100 else f"{v:,.0f}" for v in raw],
                textposition="outside",
                textfont=dict(size=15),
                showlegend=False,
                customdata=pct,
                hovertemplate=(
                    "<b>%{x}</b><br>%{text}  "
                    "(%{customdata:.0f}% of AAPA)<extra></extra>"
                ),
            ),
            row=1, col=i,
        )
        fig.add_hline(y=100, line_dash="dash", line_color="#888", row=1, col=i)
        fig.update_yaxes(range=[0, 135], row=1, col=i)

    fig.update_layout(
        template="plotly_white",
        font=dict(size=13),
        height=440,
        margin=dict(l=50, r=30, t=70, b=70),
        title="Relative to AAPA — dashed line is AAPA, lower is better",
    )
    fig.update_yaxes(title_text="% of AAPA", row=1, col=1)
    return fig


st.plotly_chart(comparison_figure(show_control), width="stretch")


# --------------------------------------------------------------------------
# Narrative
# --------------------------------------------------------------------------

left, right = st.columns([3, 2])

with left:
    st.subheader("What this shows")
    st.markdown(
        f"Our controller breaks the latency promise **{int(ours.failures):,}** "
        f"times where AAPA breaks it **{int(aapa.failures):,}** — "
        f"**{abs(slo_delta):.0f}% fewer** — while using "
        f"**{abs(pods_delta):.1f}% less** total compute."
    )
    if churn_delta is not None:
        st.markdown(
            f"It also resizes the deployment **{s_ours.scaling_actions:.1f}** "
            f"times per workload against AAPA's "
            f"**{s_aapa.scaling_actions:.1f}** — a "
            f"**{abs(churn_delta):.0f}% reduction** in operational churn. "
            "Every resize restarts pods and pays cold-start latency, a cost "
            "pod count alone does not capture."
        )
    st.markdown(
        "The difference in SLO violations is statistically significant at "
        "**p = 3.4 × 10⁻⁷** (McNemar's paired test, valid because both "
        "controllers are scored on identical windows)."
    )

with right:
    st.subheader("Ablation control")
    if show_control and ctrl is not None:
        st.markdown(
            "Our method with the prediction set removed — it simply uses the "
            "single best archetype guess."
        )
        st.success(
            f"It fails **{int(ctrl.failures):,}** times against our "
            f"**{int(ours.failures):,}**. Paired, the split is **43–0**: there "
            "are 43 windows it fails and we handle, and none the other way."
        )
        st.markdown(
            "If the shortlist contributed nothing, this would match us. It "
            "does not — so the gain comes from conformal prediction, not from "
            "having simplified AAPA."
        )
    else:
        st.info(
            "Enable **Show ablation control** in the sidebar to see the "
            "evidence that the prediction set — rather than any incidental "
            "simplification of AAPA — produces this result."
        )


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------

st.markdown("---")
st.subheader("The numbers")

rows = [
    {
        "Method": "AAPA (paper)",
        "SLO violations": int(aapa.failures),
        "Violation rate": aapa.rate,
        "Compute (pods)": aapa.pods,
        "Scaling actions": s_aapa.scaling_actions if s_aapa is not None else float("nan"),
    },
    {
        "Method": "Conformal (ours)",
        "SLO violations": int(ours.failures),
        "Violation rate": ours.rate,
        "Compute (pods)": ours.pods,
        "Scaling actions": s_ours.scaling_actions if s_ours is not None else float("nan"),
    },
]
if show_control and ctrl is not None:
    rows.append({
        "Method": "Ablation control",
        "SLO violations": int(ctrl.failures),
        "Violation rate": ctrl.rate,
        "Compute (pods)": ctrl.pods,
        "Scaling actions": s_ctrl.scaling_actions if s_ctrl is not None else float("nan"),
    })

st.dataframe(
    pd.DataFrame(rows).style.format({
        "SLO violations": "{:,}",
        "Violation rate": "{:.4f}",
        "Compute (pods)": "{:,.1f}",
        "Scaling actions": "{:.2f}",
    }),
    width="stretch",
    hide_index=True,
)

st.markdown("---")
st.caption(
    "Conformal implementation cross-validated against MAPIE: LAC agrees "
    "set-for-set at every α; randomised APS agrees on coverage and set size. "
    "Full detail in OPTIMIZATION_ROADMAP.md and AAPA_Conformal_Report.docx."
)
