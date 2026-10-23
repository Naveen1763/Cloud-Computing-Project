#!/usr/bin/env python3
"""
Interactive sandbox: build a prediction set and watch the decision follow.

Everything on this page is driven by the controls — choose the traffic, choose
which archetypes the classifier cannot rule out, choose its top guess and what
the workload actually turns out to be. AAPA's choice, ours, and whether each
one holds are all recomputed from those inputs, so the mechanism can be
explored rather than merely shown.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Try it", page_icon="joystick", layout="wide")

AAPA_COLOR = "#e53935"
OURS_COLOR = "#1e88e5"
DEMAND_COLOR = "#111111"

# AAPA Table III. A lower CPU target keeps each pod less loaded, so the same
# traffic needs more pods — which is why the choice within the set matters.
TARGET_CPU = {
    "PERIODIC": 0.75,
    "RAMP": 0.60,
    "STATIONARY_NOISY": 0.55,
    "SPIKE": 0.30,
}
ARCHETYPES = list(TARGET_CPU)


def pretty(a: str) -> str:
    return a.replace("_", " ").title()


def pods_for(load: int, archetype: str) -> int:
    """Replicas needed to serve `load` at that archetype's CPU target."""
    return math.ceil(load / TARGET_CPU[archetype])


st.title("Build the set, see the decision")
st.markdown(
    "The classifier cannot always tell which archetype it is looking at. Each "
    "archetype implies a different amount of capacity, so **which one you "
    "commit to decides whether the latency promise holds**. Set the inputs "
    "below and the rest follows."
)


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------

st.markdown("---")
st.markdown("### 1 · Set up the situation")

c1, c2 = st.columns([1, 1.6])

with c1:
    load = st.slider(
        "Traffic this window (pods of raw load)",
        min_value=10, max_value=2000, value=400, step=10,
        help="The peak load the deployment has to serve.",
    )

with c2:
    chosen = st.multiselect(
        "Archetypes the classifier cannot rule out — the prediction set",
        ARCHETYPES,
        default=["STATIONARY_NOISY", "SPIKE"],
        format_func=pretty,
        help=(
            "Conformal prediction returns this set. Add or remove members and "
            "watch the decision change."
        ),
    )

if not chosen:
    st.warning("Pick at least one archetype to put in the set.")
    st.stop()

c3, c4, c5 = st.columns(3)
with c3:
    top1 = st.selectbox(
        "The classifier's top guess", chosen, format_func=pretty,
        help="AAPA commits to this one and ignores the rest of the set.",
    )
with c4:
    truth = st.selectbox(
        "What the workload actually is", ARCHETYPES,
        index=ARCHETYPES.index("SPIKE") if "SPIKE" in ARCHETYPES else 0,
        format_func=pretty,
        help=(
            "Determines the capacity actually needed. Conformal guarantees "
            "this lands inside the set ~90% of the time."
        ),
    )
with c5:
    confidence = st.slider(
        "Classifier confidence", 0.50, 1.00, 0.80, 0.01,
        help="Drives AAPA's safety margin: 1 + 0.5 × (1 − confidence).",
    )


# --------------------------------------------------------------------------
# The decision, computed from those inputs
# --------------------------------------------------------------------------

margin = 1 + 0.5 * (1 - confidence)
ours_pick = max(chosen, key=lambda a: pods_for(load, a))

prov_aapa = math.ceil(pods_for(load, top1) * margin)
prov_ours = pods_for(load, ours_pick)
demand = pods_for(load, truth)

st.markdown("---")
st.markdown("### 2 · What each archetype would require")

table = []
for a in sorted(chosen, key=lambda x: -TARGET_CPU[x]):
    who = []
    if a == top1:
        who.append("AAPA")
    if a == ours_pick:
        who.append("Ours")
    table.append({
        "Archetype in set": pretty(a),
        "CPU target": f"{TARGET_CPU[a]:.0%}",
        "Pods it implies": f"{pods_for(load, a):,}",
        "Committed to by": " + ".join(who) if who else "—",
    })

t1, t2 = st.columns([3, 2])
with t1:
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
with t2:
    spread = max(pods_for(load, a) for a in chosen) - min(pods_for(load, a) for a in chosen)
    if len(chosen) > 1:
        st.markdown(
            f"The set spans **{spread:,} pods** of disagreement. AAPA resolves "
            "it by picking one and padding; we resolve it by covering the "
            "most demanding member."
        )
    else:
        st.markdown(
            "Only one archetype is in the set, so there is nothing to hedge "
            "between — both methods differ only by AAPA's margin."
        )
    if truth not in chosen:
        st.warning(
            f"**{pretty(truth)}** is not in the set — this is the ~10% of "
            "cases the guarantee allows to fall outside. Hedging cannot help "
            "when the truth was ruled out."
        )


# --------------------------------------------------------------------------
# Outcome
# --------------------------------------------------------------------------

st.markdown("---")
st.markdown("### 3 · The outcome")

o1, o2, o3 = st.columns(3)
o1.metric(
    "AAPA provisions", f"{prov_aapa:,}",
    delta=f"{prov_aapa - demand:+,} vs demand",
    delta_color="normal" if prov_aapa >= demand else "inverse",
    help=f"{pretty(top1)} needs {pods_for(load, top1):,}, × {margin:.3f} margin.",
)
o2.metric(
    "We provision", f"{prov_ours:,}",
    delta=f"{prov_ours - demand:+,} vs demand",
    delta_color="normal" if prov_ours >= demand else "inverse",
    help=f"Most demanding in the set: {pretty(ours_pick)}.",
)
o3.metric(
    "Demand", f"{demand:,}",
    help=f"The workload is really {pretty(truth)}, which needs this much.",
)

aapa_fails = prov_aapa < demand
ours_fails = prov_ours < demand

if aapa_fails and not ours_fails:
    st.success(
        f"**SLO violation in AAPA — prevented in ours.**\n\n"
        f"AAPA committed to {pretty(top1)} and provisioned **{prov_aapa:,}** "
        f"pods against **{demand:,}** needed — short by "
        f"**{demand - prov_aapa:,}**, so requests queue and the latency "
        f"promise breaks.\n\n"
        f"We could not rule out **{pretty(ours_pick)}**, provisioned "
        f"**{prov_ours:,}**, and absorbed it. **That is what the prediction "
        f"set buys.**"
    )
elif aapa_fails and ours_fails:
    st.error(
        f"**Both fall short.** Demand is {demand:,}; AAPA has {prov_aapa:,} "
        f"and we have {prov_ours:,}. "
        + (
            f"{pretty(truth)} was not in the set, so there was nothing to "
            "hedge toward — the guarantee allows this in ~10% of windows."
            if truth not in chosen else
            "Even the most demanding archetype in the set does not cover this "
            "demand."
        )
    )
elif not aapa_fails and ours_fails:
    st.warning(
        f"**AAPA holds and we do not.** Its margin carried it to "
        f"{prov_aapa:,} where our hedging reached only {prov_ours:,}, against "
        f"{demand:,} needed. This happens when the top guess is already the "
        "most demanding archetype in the set — hedging has nothing above it "
        "to reach for, while AAPA's margin keeps multiplying."
    )
else:
    waste = prov_ours - demand
    st.info(
        f"**Both hold.** Demand {demand:,}; AAPA {prov_aapa:,}, ours "
        f"{prov_ours:,}. "
        + (
            f"We carried {waste:,} pods more than needed — the cost side of "
            "hedging, paid on windows where the top guess happened to be right."
            if waste > 0 else "Neither was stressed."
        )
    )


# --------------------------------------------------------------------------
# Picture
# --------------------------------------------------------------------------

fig = go.Figure()
fig.add_trace(go.Bar(
    x=["AAPA", "Ours"], y=[prov_aapa, prov_ours],
    marker_color=[AAPA_COLOR, OURS_COLOR],
    marker_line=dict(color="#222", width=1),
    text=[f"{prov_aapa:,}", f"{prov_ours:,}"],
    textposition="outside", textfont=dict(size=16),
    width=0.45, showlegend=False,
    hovertemplate="%{x}: %{y:,} pods<extra></extra>",
))
fig.add_hline(
    y=demand, line_width=3, line_color=DEMAND_COLOR,
    annotation_text=f"demand {demand:,}", annotation_position="right",
)
fig.update_layout(
    template="plotly_white", height=360,
    margin=dict(l=60, r=120, t=50, b=40),
    yaxis_title="Pods provisioned",
    title="Anything below the demand line is an SLO violation",
    font=dict(size=13),
)
fig.update_yaxes(range=[0, max(prov_aapa, prov_ours, demand) * 1.25])
st.plotly_chart(fig, width="stretch")

st.caption(
    "Pods = load ÷ CPU target, rounded up. AAPA then multiplies by "
    f"1 + 0.5 × (1 − confidence) = {margin:.3f}. We take the maximum over the "
    "prediction set and add no margin — the set itself says how much to hedge. "
    "Targets are AAPA's Table III: Periodic 75%, Ramp 60%, Stationary 55%, "
    "Spike 30%."
)
