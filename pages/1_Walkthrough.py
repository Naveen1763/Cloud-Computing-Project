#!/usr/bin/env python3
"""
Replay one scaling decision at a time, revealed stage by stage.

Pick a window, explore what each archetype in the prediction set would cost,
then uncover AAPA's choice, ours, the demand that actually arrived, and the
outcome — one click at a time, so an audience follows the reasoning rather
than being shown a conclusion.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import BASELINE, OURS, find_experiment, load_decisions

st.set_page_config(page_title="Step-by-step", page_icon="footprints", layout="wide")

MAIN = "loadfix"
AAPA_COLOR = "#e53935"
OURS_COLOR = "#1e88e5"
DEMAND_COLOR = "#111111"

REQUIRED_COLUMNS = {"workload_id", "window_start_minute", "prediction_set"}

# Curated rather than a dropdown over every workload: both contain many
# windows where hedging changes the outcome.
FEATURED: list[tuple[str, str]] = [
    ("w_0056", "Stationary — 25 saves"),
    ("w_0295", "Periodic — 10 saves"),
]

# AAPA Table III. A lower target keeps each pod less loaded, so it needs more
# pods for the same traffic — which is why the choice within the set matters.
TARGET_CPU = {
    "SPIKE": 0.30,
    "RAMP": 0.60,
    "STATIONARY_NOISY": 0.55,
    "PERIODIC": 0.75,
}

STAGES = [
    "The prediction set",
    "What AAPA picks",
    "What we pick",
    "What demand actually was",
]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load(_stamp: float):
    """`_stamp` is the results file mtime, so regenerating invalidates this."""
    return load_decisions(find_experiment(MAIN))


def results_stamp() -> float:
    path = find_experiment(MAIN).directory / "controller_decisions.csv"
    return path.stat().st_mtime if path.exists() else 0.0


try:
    decisions = load(results_stamp())
except Exception as exc:
    st.error(f"Could not load results: {exc}")
    st.stop()

missing = REQUIRED_COLUMNS - set(decisions.columns)
if missing:
    st.warning(
        "This page needs per-window detail the saved results do not contain "
        f"(missing: {', '.join(sorted(missing))}).\n\n"
        "Regenerate with:\n\n"
        "`python examples/run_conformal_controllers.py --max-workloads 200 "
        "--mondrian --split workload --load-multiplier 60 --min-activity 0.1 "
        "--results-dir results_loadfix`"
    )
    st.stop()


st.title("How a violation gets prevented")
st.markdown(
    "One scaling decision, uncovered a step at a time. The classifier cannot "
    "always tell which archetype it is looking at — and because each archetype "
    "implies a different amount of capacity, *which one you commit to* decides "
    "whether the latency promise holds."
)


# --------------------------------------------------------------------------
# Workload and window
# --------------------------------------------------------------------------

available = set(decisions.workload_id.unique())
featured = [(w, note) for w, note in FEATURED if w in available]
if not featured:
    st.info("None of the featured workloads are present in this result set.")
    st.stop()

labels = {w: f"{w} · {note}" for w, note in featured}
workload = st.radio(
    "Workload", [w for w, _ in featured],
    format_func=lambda w: labels[w], horizontal=True,
)

wl = decisions[decisions.workload_id == workload]
wide = wl.pivot_table(
    index=["window_start_minute", "archetype_true", "archetype_predicted",
           "prediction_set", "set_size", "current_load", "required"],
    columns="controller", values=["provisioned", "violation"], aggfunc="first",
).reset_index()
wide.columns = [c[0] if not c[1] else f"{c[0]}_{c[1]}"
                for c in wide.columns.to_flat_index()]
wide = wide.sort_values("window_start_minute").reset_index(drop=True)

P_A, P_O = f"provisioned_{BASELINE}", f"provisioned_{OURS}"
V_A, V_O = f"violation_{BASELINE}", f"violation_{OURS}"

# Windows where AAPA fell short and we held — the ones worth walking through.
saves = wide.index[wide[V_A] & ~wide[V_O]].tolist()
if not saves:
    st.info("This workload has no window where hedging changed the outcome.")
    st.stop()

idx_key, stage_key = f"idx_{workload}", f"stage_{workload}"
st.session_state.setdefault(idx_key, 0)
st.session_state.setdefault(stage_key, 0)

pick = st.session_state[idx_key] % len(saves)
step = saves[pick]
row = wide.iloc[step]

nav1, nav2, nav3 = st.columns([1.3, 1.3, 4])
if nav1.button("◀ Previous case", width="stretch"):
    st.session_state[idx_key] = (pick - 1) % len(saves)
    st.session_state[stage_key] = 0
    st.rerun()
if nav2.button("Next case ▶", width="stretch"):
    st.session_state[idx_key] = (pick + 1) % len(saves)
    st.session_state[stage_key] = 0
    st.rerun()
nav3.caption(
    f"Case {pick + 1} of {len(saves)} on {workload} · window starting at "
    f"minute {int(row.window_start_minute):,} · true archetype "
    f"**{row.archetype_true.title()}**"
)


# --------------------------------------------------------------------------
# Decision facts
# --------------------------------------------------------------------------

pset = [p for p in str(row.prediction_set).split("|") if p]
load = int(row.current_load)
demand, prov_a, prov_o = int(row.required), int(row[P_A]), int(row[P_O])
top1 = str(row.archetype_predicted)
ours_pick = max(pset, key=lambda a: math.ceil(load / TARGET_CPU.get(a, 0.55)))


def pods_for(archetype: str) -> int:
    return math.ceil(load / TARGET_CPU.get(archetype, 0.55))


stage = st.session_state[stage_key]

st.markdown("---")
st.markdown(f"**Traffic this window peaks at {load:,} pods of raw load.**")

prog = st.columns([1, 1, 1, 1])
for i, name in enumerate(STAGES):
    mark = "●" if i <= stage else "○"
    prog[i].markdown(
        f"<span style='color:{'#1e88e5' if i <= stage else '#888'}'>"
        f"{mark} {name}</span>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Stage 0 — explore the set
# --------------------------------------------------------------------------

st.markdown("---")
st.markdown("#### The classifier cannot rule these out")

explore = st.radio(
    "Select an archetype to see the capacity it would require",
    pset,
    format_func=lambda a: a.title(),
    horizontal=True,
    key=f"explore_{workload}_{step}",
)

e1, e2 = st.columns([1, 2])
e1.metric(
    f"If it is {explore.title()}",
    f"{pods_for(explore):,} pods",
    help=f"{load:,} pods of load ÷ {TARGET_CPU.get(explore, 0.55):.0%} CPU target",
)
e2.markdown(
    f"At a **{TARGET_CPU.get(explore, 0.55):.0%}** CPU target, each pod is kept "
    f"that loaded — so serving {load:,} pods of traffic needs "
    f"**{pods_for(explore):,}** pods.\n\n"
    "A lower target means more headroom and therefore more pods. "
    + (
        f"Across this set the requirement ranges from "
        f"**{min(pods_for(a) for a in pset):,}** to "
        f"**{max(pods_for(a) for a in pset):,}** pods — which is the whole "
        "problem."
        if len(pset) > 1 else
        "Only one archetype is plausible here."
    )
)

if stage < len(STAGES) - 1:
    if st.button(f"Reveal: {STAGES[stage + 1].lower()} ▸", type="primary"):
        st.session_state[stage_key] = stage + 1
        st.rerun()
else:
    if st.button("Start over"):
        st.session_state[stage_key] = 0
        st.rerun()


# --------------------------------------------------------------------------
# Stages 1-3 — the reveal
# --------------------------------------------------------------------------

if stage >= 1:
    st.markdown("---")
    st.markdown("#### What each controller committed to")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f"<span style='color:{AAPA_COLOR};font-size:1.1rem'><b>AAPA</b>"
            "</span>", unsafe_allow_html=True,
        )
        st.markdown(
            f"Commits to its single best guess: **{top1.title()}**  \n"
            f"That archetype implies **{pods_for(top1):,}** pods, plus its "
            f"confidence margin."
        )
        st.metric("AAPA provisions", f"{prov_a:,} pods")

    with c2:
        if stage >= 2:
            st.markdown(
                f"<span style='color:{OURS_COLOR};font-size:1.1rem'><b>Ours"
                "</b></span>", unsafe_allow_html=True,
            )
            st.markdown(
                f"Commits to the most demanding archetype still plausible: "
                f"**{ours_pick.title()}**  \n"
                f"No margin — the set itself says how much to hedge."
            )
            st.metric("We provision", f"{prov_o:,} pods",
                      delta=f"{prov_o - prov_a:+,} vs AAPA")
        else:
            st.caption("Not revealed yet.")

if stage >= 3:
    st.markdown("---")
    st.markdown(f"#### Demand arrived: **{demand:,} pods** needed")

    r1, r2, r3 = st.columns(3)
    r1.metric("AAPA had", f"{prov_a:,}",
              delta=f"{prov_a - demand:+,} vs demand",
              delta_color="normal" if prov_a >= demand else "inverse")
    r2.metric("We had", f"{prov_o:,}",
              delta=f"{prov_o - demand:+,} vs demand",
              delta_color="normal" if prov_o >= demand else "inverse")
    r3.metric("True archetype", row.archetype_true.title())

    if prov_a < demand <= prov_o:
        st.success(
            f"**SLO violation in AAPA — prevented in ours.**\n\n"
            f"AAPA committed to {top1.title()} and had **{prov_a:,}** pods "
            f"against **{demand:,}** needed. It is short by "
            f"**{demand - prov_a:,}** — requests queue and the latency promise "
            f"breaks.\n\n"
            f"We could not rule out **{ours_pick.title()}**, which needs more "
            f"capacity, so we provisioned **{prov_o:,}** and absorbed it. "
            f"**That is what the prediction set buys.**"
        )
    elif prov_o < demand <= prov_a:
        st.warning(
            f"AAPA's margin carried it here and our hedging did not: "
            f"AAPA {prov_a:,}, ours {prov_o:,}, demand {demand:,}."
        )
    elif demand > max(prov_a, prov_o):
        st.error(
            f"Both fell short — demand {demand:,} against AAPA's {prov_a:,} "
            f"and our {prov_o:,}."
        )
    else:
        st.info(
            f"Both held. We carried {prov_o - prov_a:,} pods more than AAPA, "
            "which was not needed this window — the cost side of hedging."
        )


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------

st.markdown("---")
view = wide.iloc[: step + 1]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=view.window_start_minute, y=view.required, name="Demand",
    mode="lines", line=dict(color=DEMAND_COLOR, width=3)))
fig.add_trace(go.Scatter(
    x=view.window_start_minute, y=view[P_A], name="AAPA provisioned",
    mode="lines", line=dict(color=AAPA_COLOR, width=2, dash="dot")))
fig.add_trace(go.Scatter(
    x=view.window_start_minute, y=view[P_O], name="Ours provisioned",
    mode="lines", line=dict(color=OURS_COLOR, width=2, dash="dot")))
fig.add_vline(x=float(row.window_start_minute), line_width=2,
              line_dash="dash", line_color="#888")
fig.update_layout(
    template="plotly_white", height=330,
    margin=dict(l=60, r=30, t=30, b=50),
    xaxis_title="Minute of trace", yaxis_title="Pods",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    font=dict(size=13),
)
st.plotly_chart(fig, width="stretch")
st.caption(
    f"Capacity up to this window. Hedging prevents {len(saves)} violations on "
    f"{workload} alone."
)
