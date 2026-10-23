#!/usr/bin/env python3
"""
Replay one workload decision by decision, AAPA against ours.

The headline numbers say the method works. This page shows the mechanism:
traffic arrives, the classifier guesses, the conformal set names what cannot
be ruled out, both controllers commit to a pod count, and then real demand
arrives and settles it. A running scoreboard accumulates violations so the
moment hedging prevents one is visible as it happens.
"""

import sys
import time
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

# Curated examples rather than a dropdown over every workload: both contain
# many windows where hedging changed the outcome, so the replay lands on
# something worth watching.
FEATURED: list[tuple[str, str]] = [
    ("w_0056", "Stationary — 25 saves"),
    ("w_0295", "Periodic — 10 saves"),
]

AUTOPLAY_DELAY = 0.9  # seconds between steps

# AAPA Table III. A lower target keeps each pod less loaded, so it needs more
# pods for the same traffic - which is why the choice within the set matters.
TARGET_CPU = {
    "SPIKE": 0.30,
    "RAMP": 0.60,
    "STATIONARY_NOISY": 0.55,
    "PERIODIC": 0.75,
}


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
    "One workload, replayed decision by decision. Each step is an hour-long "
    "window: traffic arrives, the classifier guesses an archetype, the "
    "conformal set names what it cannot rule out, both controllers commit to "
    "a pod count — then real demand arrives and settles it."
)


# --------------------------------------------------------------------------
# Pick the workload
# --------------------------------------------------------------------------

available = set(decisions.workload_id.unique())
featured = [(w, note) for w, note in FEATURED if w in available]
if not featured:
    st.info("None of the featured workloads are present in this result set.")
    st.stop()

if len(featured) == 1:
    workload = featured[0][0]
    st.caption(f"Workload **{workload}** — {featured[0][1]}")
else:
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
n_steps = len(wide)

# Windows where AAPA fell short and we held — the point of the exercise.
saves = wide.index[wide[V_A] & ~wide[V_O]].tolist()


# --------------------------------------------------------------------------
# Replay state
# --------------------------------------------------------------------------

key = f"step_{workload}"
if key not in st.session_state:
    st.session_state[key] = 0
step = min(st.session_state[key], n_steps - 1)

c1, c2, c3, c4, c5 = st.columns([1, 1, 1.4, 1.4, 3])
if c1.button("Restart", width="stretch"):
    st.session_state[key] = 0
    st.rerun()
if c2.button("Step", width="stretch", type="primary"):
    st.session_state[key] = min(step + 1, n_steps - 1)
    st.rerun()
if c3.button("Next save", width="stretch",
             help="Jump to the next window where hedging prevents a failure"):
    nxt = next((s for s in saves if s > step), saves[0] if saves else step)
    st.session_state[key] = nxt
    st.rerun()
if c4.button("Run to end", width="stretch"):
    st.session_state[key] = n_steps - 1
    st.rerun()
autoplay = c5.toggle("Auto-play", value=False,
                     help=f"Advance a step every {AUTOPLAY_DELAY:g}s")

row = wide.iloc[step]
elapsed = wide.iloc[: step + 1]
aapa_fails = int(elapsed[V_A].sum())
ours_fails = int(elapsed[V_O].sum())
saves_so_far = int((elapsed[V_A] & ~elapsed[V_O]).sum())

st.progress((step + 1) / n_steps,
            text=f"Window {step + 1} of {n_steps} · minute "
                 f"{int(row.window_start_minute):,}")


# --------------------------------------------------------------------------
# Scoreboard
# --------------------------------------------------------------------------

s1, s2, s3 = st.columns(3)
s1.metric("AAPA — violations so far", aapa_fails,
          help="Windows where AAPA provisioned less than demand required.")
s2.metric("Ours — violations so far", ours_fails,
          delta=f"{ours_fails - aapa_fails:+d} vs AAPA" if step else None,
          delta_color="inverse",
          help="Same windows, our controller.")
s3.metric("Violations prevented", saves_so_far,
          help="Windows where AAPA fell short and we held.")


# --------------------------------------------------------------------------
# The current decision, in order
# --------------------------------------------------------------------------

st.markdown("---")
pset = [p for p in str(row.prediction_set).split("|") if p]
hedged = len(pset) > 1
demand, prov_a, prov_o = int(row.required), int(row[P_A]), int(row[P_O])

d1, d2 = st.columns([1, 1])

with d1:
    st.markdown("##### 1 · Traffic arrives")
    st.markdown(
        f"This window peaks at **{int(row.current_load)} pods** of raw load.\n\n"
        f"True archetype: **{row.archetype_true.title()}**"
    )

with d2:
    st.markdown("##### 2 · The classifier cannot decide")
    st.markdown(f"Top guess: **{row.archetype_predicted.title()}**")
    st.markdown("Cannot rule out: " + " ".join(f"`{p.title()}`" for p in pset))

st.markdown("##### 3 · Each controller picks from that set")

import math
load = int(row.current_load)
top1 = str(row.archetype_predicted)
# Ours sizes for whichever member needs the most pods.
ours_pick = max(pset, key=lambda a: math.ceil(load / TARGET_CPU.get(a, 0.55)))

choice_rows = []
for a in sorted(pset, key=lambda x: -TARGET_CPU.get(x, 0.55)):
    target = TARGET_CPU.get(a, 0.55)
    implied = math.ceil(load / target)
    who = []
    if a == top1:
        who.append("AAPA")
    if a == ours_pick:
        who.append("Ours")
    choice_rows.append({
        "Archetype in set": a.title(),
        "CPU target": f"{target:.0%}",
        "Pods it implies": f"{implied:,}",
        "Chosen by": " + ".join(who) if who else "—",
    })

cc1, cc2 = st.columns([3, 2])
with cc1:
    st.dataframe(pd.DataFrame(choice_rows), width="stretch", hide_index=True)
    st.caption(
        "A lower CPU target keeps each pod less loaded, so it needs more pods "
        "for the same traffic. AAPA commits to its single top guess (then adds "
        "its confidence margin); we commit to whichever member of the set "
        "needs the most."
    )
with cc2:
    st.markdown(
        f"<div style='line-height:1.9'>"
        f"<span style='color:{AAPA_COLOR}'><b>AAPA</b></span> picks "
        f"<b>{top1.title()}</b> → provisions <b>{prov_a:,}</b> pods<br>"
        f"<span style='color:{OURS_COLOR}'><b>Ours</b></span> picks "
        f"<b>{ours_pick.title()}</b> → provisions <b>{prov_o:,}</b> pods"
        f"</div>",
        unsafe_allow_html=True,
    )
    if not hedged:
        st.caption(
            "Only one archetype is plausible here, so both pick the same one "
            "— they differ only by AAPA's margin."
        )

st.markdown(f"##### 4 · Demand arrives: **{demand} pods** needed")

if prov_a < demand <= prov_o:
    st.success(
        f"**Violation prevented.** AAPA had {prov_a} pods against {demand} "
        f"needed — requests queue, the latency promise breaks. We had "
        f"{prov_o}, because {', '.join(p.title() for p in pset if p != row.archetype_predicted)} "
        f"could not be ruled out and needed more capacity than the top guess. "
        f"**This is what the prediction set buys.**"
    )
elif prov_o < demand <= prov_a:
    st.warning(
        f"AAPA's margin carried it here and our hedging did not: AAPA "
        f"{prov_a}, ours {prov_o}, demand {demand}."
    )
elif demand > max(prov_a, prov_o):
    st.error(f"Both fell short — demand {demand}, AAPA {prov_a}, ours {prov_o}.")
else:
    extra = prov_o - prov_a
    st.info(
        f"Both held. "
        + (f"We carried {extra} pod{'s' if extra != 1 else ''} more than "
           "needed this window — that is the cost side of hedging."
           if extra > 0 else "Neither was stressed.")
    )


# --------------------------------------------------------------------------
# Timeline up to the current step
# --------------------------------------------------------------------------

st.markdown("---")
view = wide.iloc[: step + 1]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=view.window_start_minute, y=view.required, name="Demand",
    mode="lines", line=dict(color=DEMAND_COLOR, width=3),
    hovertemplate="minute %{x}<br>demand %{y} pods<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=view.window_start_minute, y=view[P_A], name="AAPA provisioned",
    mode="lines", line=dict(color=AAPA_COLOR, width=2, dash="dot"),
    hovertemplate="minute %{x}<br>AAPA %{y} pods<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=view.window_start_minute, y=view[P_O], name="Ours provisioned",
    mode="lines", line=dict(color=OURS_COLOR, width=2, dash="dot"),
    hovertemplate="minute %{x}<br>ours %{y} pods<extra></extra>",
))

saved_rows = view[view[V_A] & ~view[V_O]]
if len(saved_rows):
    fig.add_trace(go.Scatter(
        x=saved_rows.window_start_minute, y=saved_rows.required,
        name="Violation prevented", mode="markers",
        marker=dict(symbol="star", size=15, color="#2e7d32",
                    line=dict(width=1, color="#111")),
        hovertemplate="minute %{x}<br>AAPA fell short, we held<extra></extra>",
    ))

fig.add_vline(x=float(row.window_start_minute), line_width=2,
              line_dash="dash", line_color="#888")
fig.update_layout(
    template="plotly_white", height=380,
    margin=dict(l=60, r=30, t=30, b=50),
    xaxis_title="Minute of trace", yaxis_title="Pods",
    xaxis=dict(range=[float(wide.window_start_minute.min()),
                      float(wide.window_start_minute.max())]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    font=dict(size=13),
)
st.plotly_chart(fig, width="stretch")

st.caption(
    f"Replaying {workload}. Across the full run, hedging prevents "
    f"{len(saves)} violation{'s' if len(saves) != 1 else ''} on this workload "
    "alone. Pod counts follow each archetype's CPU target: Periodic 1.33×, "
    "Ramp 1.67×, Stationary 1.82×, Spike 3.33× the raw load."
)

# Auto-advance last, so the page above is already drawn.
if autoplay and step < n_steps - 1:
    time.sleep(AUTOPLAY_DELAY)
    st.session_state[key] = step + 1
    st.rerun()
