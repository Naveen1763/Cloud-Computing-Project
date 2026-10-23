#!/usr/bin/env python3
"""
Results dashboard for the conformal autoscaling project.

Presents saved experiment runs - nothing is recomputed, so the app loads
instantly and cannot fail partway through a demo.

Launch:
    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.data import (
    SIM_COLOR,
    SIM_FLAT_METRICS,
    SIM_LABEL,
    SIM_OURS,
    BASELINE,
    CAPACITY_LADDER,
    CONTROL,
    CONTROLLER_LABEL,
    FINDINGS,
    MAPIE_VALIDATION,
    OURS,
    available_experiments,
    by_ambiguity,
    headline,
    load_decisions,
    load_report,
    per_archetype,
    load_simulation,
    metric_separation,
    simulation_available,
    summarise,
    summarise_simulation,
    window_counts,
)

st.set_page_config(
    page_title="Conformal Autoscaling — Results",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


# --------------------------------------------------------------------------
# Caching: results are static files, so read them once per session.
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _decisions(directory: str) -> pd.DataFrame:
    return pd.read_csv(Path(directory) / "controller_decisions.csv")


@st.cache_data(show_spinner=False)
def _simulation():
    return load_simulation()


def get_run(exp):
    return _decisions(str(exp.directory)), load_report(exp)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

experiments = available_experiments()
if not experiments:
    st.error(
        "No result sets found. Run the controller comparison first:\n\n"
        "`python examples/run_conformal_controllers.py --max-workloads 200 "
        "--mondrian --split workload --load-multiplier 60 "
        "--results-dir results_loadfix`"
    )
    st.stop()

st.sidebar.title("Conformal Autoscaling")
st.sidebar.caption("Uncertainty-aware scaling for serverless workloads")

page = st.sidebar.radio(
    "Page",
    [
        "Overview",
        "Controller comparison",
        "Evaluation findings",
        "Conformal analysis",
        "Stability & churn",
        "Limitation",
    ],
)

st.sidebar.markdown("---")
exp = st.sidebar.selectbox(
    "Experiment",
    experiments,
    format_func=lambda e: e.name,
    help="Every page reflects the run selected here.",
)
st.sidebar.caption(exp.blurb)

decisions, report = get_run(exp)
stats = summarise(decisions)
head = headline(stats)
counts = window_counts(decisions)
args = report.get("args", {})

st.sidebar.markdown("---")
st.sidebar.markdown("**Run configuration**")
st.sidebar.markdown(
    f"- Split: `{args.get('split', '?')}`\n"
    f"- Load multiplier: `{args.get('load_multiplier', 1)}×`\n"
    f"- Labels: `{args.get('labels', '?')}`\n"
    f"- α = `{args.get('alpha', '?')}`, "
    f"method `{args.get('conformal_method', '?')}`"
    f"{', Mondrian' if args.get('mondrian') else ''}\n"
    f"- Workloads: `{args.get('max_workloads', '?')}`"
)


def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, delta, help_text) in zip(cols, items):
        col.metric(label, value, delta, help=help_text)


# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------

if page == "Overview":
    st.title("Conformal prediction for autoscaling")
    st.markdown(
        "Replacing AAPA's single confidence score with a **prediction set** — "
        "a shortlist of archetypes that cannot be statistically ruled out — so "
        "the controller hedges against the alternatives that actually remain "
        "rather than padding blindly."
    )

    if head["violation_delta"] is not None:
        vd, cd = head["violation_delta"], head["cost_delta"]
        metric_row([
            (
                "SLO violations vs AAPA",
                f"{head['ours_failures']:,}",
                f"{vd:+.1f}%",
                f"AAPA records {head['base_failures']:,} failures over the same decisions.",
            ),
            (
                "Compute vs AAPA",
                f"{stats.loc[OURS].pods:.1f} pods",
                f"{cd:+.1f}%",
                "Mean pods provisioned per decision.",
            ),
            (
                "Ablation control",
                f"{head['control_failures']:,}" if head["control_failures"] else "—",
                f"{head['control_gap']:+.0f}%" if head["control_gap"] else None,
                "Our method with the prediction set removed. The gap is what "
                "the set contributes.",
            ),
            (
                "Decisions evaluated",
                f"{counts['total']:,}",
                None,
                "Each scored against the capacity the next window actually demanded.",
            ),
        ])

    st.markdown("---")
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(
            charts.failures_bar(stats, counts["total"]), width="stretch"
        )
    with right:
        st.subheader("Reading this")
        if head["control_gap"] and head["violation_delta"] and head["violation_delta"] < 0:
            st.markdown(
                f"Our controller breaks the latency promise **"
                f"{head['ours_failures']:,}** times where AAPA breaks it "
                f"**{head['base_failures']:,}** — "
                f"**{abs(head['violation_delta']):.0f}% fewer**, at "
                f"**{abs(head['cost_delta']):.1f}% "
                f"{'less' if head['cost_delta'] < 0 else 'more'}** compute."
            )
            st.markdown(
                f"The **ablation control** is the same method with the "
                f"shortlist removed. It fails "
                f"**{head['control_failures']:,}** times — "
                f"**{head['control_gap']:.0f}% more** than ours. That gap is "
                f"the evidence the prediction set is doing the work, rather "
                f"than some incidental simplification."
            )
        else:
            st.markdown(
                "On this run the conformal controller does **not** improve on "
                "AAPA. See the **Limitation** page for why — when the "
                "top-ranked archetype is already the most demanding one, "
                "there is nothing left to hedge toward."
            )

        st.info(exp.blurb)

    st.markdown("---")
    st.subheader("Results by controller")
    table = stats[["label", "failures", "rate", "pods", "rate_vs_base", "pods_vs_base"]].copy()
    table.columns = [
        "Controller", "Failures", "Violation rate", "Mean pods",
        "Violations vs AAPA (%)", "Compute vs AAPA (%)",
    ]
    st.dataframe(
        table.style.format({
            "Failures": "{:,.0f}",
            "Violation rate": "{:.4f}",
            "Mean pods": "{:.1f}",
            "Violations vs AAPA (%)": "{:+.1f}",
            "Compute vs AAPA (%)": "{:+.1f}",
        }),
        width="stretch",
        hide_index=True,
    )


# --------------------------------------------------------------------------
# Controller comparison
# --------------------------------------------------------------------------

elif page == "Controller comparison":
    st.title("Controller comparison")
    st.caption(
        "Every controller sees the same window and is judged against the same "
        "next-window demand, so differences come from decision logic alone."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            charts.failures_bar(stats, counts["total"]), width="stretch"
        )
    with c2:
        st.plotly_chart(charts.tradeoff_scatter(stats), width="stretch")

    st.markdown("---")
    st.subheader("Where the prediction set can act")
    st.markdown(
        f"Conformal hedging only changes a decision when the set holds more "
        f"than one archetype. This run has **{counts['ambiguous']:,} ambiguous** "
        f"and **{counts['clean']:,} single-archetype** windows."
    )

    split = by_ambiguity(decisions)
    a, b = st.columns(2)
    for col, key, title in [
        (a, "ambiguous", f"Ambiguous windows ({counts['ambiguous']:,})"),
        (b, "clean", f"Single-archetype windows ({counts['clean']:,})"),
    ]:
        with col:
            st.markdown(f"**{title}**")
            frame = split[key]
            if frame.empty:
                st.caption("No windows in this group.")
                continue
            show = frame[["label", "failures", "rate", "pods"]].copy()
            show.columns = ["Controller", "Failures", "Rate", "Mean pods"]
            st.dataframe(
                show.style.format({
                    "Failures": "{:,.0f}", "Rate": "{:.4f}", "Mean pods": "{:.1f}",
                }),
                width="stretch", hide_index=True,
            )

    st.caption(
        "On single-archetype windows the conformal controller reduces "
        "algebraically to the ablation control, so identical numbers there "
        "are expected, not a bug."
    )

    st.markdown("---")
    st.subheader("By archetype")
    metric = st.radio(
        "Metric", ["Violation rate", "Mean pods provisioned"],
        horizontal=True, label_visibility="collapsed",
    )
    if metric == "Violation rate":
        st.plotly_chart(
            charts.grouped_bar(
                per_archetype(decisions, "violation"),
                "SLO violation rate by archetype", "Violation rate",
            ),
            width="stretch",
        )
    else:
        st.plotly_chart(
            charts.grouped_bar(
                per_archetype(decisions, "provisioned"),
                "Pods provisioned by archetype", "Mean pods",
            ),
            width="stretch",
        )


# --------------------------------------------------------------------------
# Evaluation findings
# --------------------------------------------------------------------------

elif page == "Evaluation findings":
    st.title("Defects in the original evaluation")
    st.markdown(
        "Three problems were found while reproducing AAPA. Each inflates the "
        "apparent reliability of the published classifier, and together they "
        "explain why its uncertainty mechanism looked unnecessary."
    )

    tabs = st.tabs([
        "1. Train/test leak",
        "2. Circular labels",
        "3. SPIKE removed",
    ])

    for tab, key in zip(tabs, ["leak", "labels", "spike"]):
        finding = FINDINGS[key]
        with tab:
            st.subheader(finding["title"])
            st.markdown(finding["detail"])
            st.plotly_chart(
                charts.before_after(
                    finding["metrics"], finding["labels"], finding["title"]
                ),
                width="stretch",
            )
            st.success(finding["takeaway"])

    st.markdown("---")
    st.subheader("Why this matters for the contribution")
    st.markdown(
        "Under the original split, 95% of predictions name a single "
        "archetype. A prediction set with one member is just a point "
        "prediction, so any set-based method has nothing to act on. The leak "
        "made uncertainty handling look pointless by manufacturing certainty "
        "that was not there."
    )


# --------------------------------------------------------------------------
# Conformal analysis
# --------------------------------------------------------------------------

elif page == "Conformal analysis":
    st.title("Conformal prediction internals")

    conformal = report.get("conformal", {})
    per_class = report.get("conformal_per_class", {})

    if not conformal:
        st.warning("No conformal metadata saved for this run.")
    else:
        metric_row([
            ("Classifier accuracy", f"{report.get('accuracy', 0):.3f}", None,
             "Top-1 accuracy on held-out workloads."),
            ("Empirical coverage",
             f"{conformal.get('empirical_coverage', 0):.3f}",
             f"target {conformal.get('target_coverage', 0):.2f}",
             "How often the true archetype falls inside the prediction set."),
            ("Mean set size", f"{conformal.get('avg_set_size', 0):.2f}", None,
             "Smaller is more informative, as long as coverage holds."),
            ("Single-archetype rate",
             f"{conformal.get('singleton_rate', 0):.1%}", None,
             "Windows where the model is confident enough to name one archetype."),
        ])

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            dist = conformal.get("set_size_distribution", {})
            if dist:
                st.plotly_chart(
                    charts.set_size_distribution(dist), width="stretch"
                )
        with c2:
            if per_class:
                st.plotly_chart(
                    charts.per_class_coverage(
                        per_class, conformal.get("target_coverage", 0.9)
                    ),
                    width="stretch",
                )

        if per_class:
            target = conformal.get("target_coverage", 0.9)
            short = [k for k, v in per_class.items() if v["coverage"] < target - 0.005]
            if short:
                st.warning(
                    f"**{', '.join(short)}** fall short of the {target:.0%} "
                    "guarantee. Marginal coverage can hold overall while "
                    "failing on individual classes — which is why "
                    "class-conditional (Mondrian) calibration matters, and why "
                    "residual shortfall points at distribution drift between "
                    "calibration and test."
                )

    st.markdown("---")
    st.subheader("Validation against MAPIE")
    st.markdown(
        "Our implementation and MAPIE, the reference conformal library, were "
        "run on byte-identical probability matrices."
    )
    st.dataframe(
        MAPIE_VALIDATION.style.format({
            "alpha": "{:.2f}",
            "ours_coverage": "{:.4f}", "mapie_coverage": "{:.4f}",
            "ours_set_size": "{:.3f}", "mapie_set_size": "{:.3f}",
            "set_agreement": "{:.1%}",
        }),
        width="stretch", hide_index=True,
    )
    st.caption(
        "LAC agrees set-for-set at every α. Randomised APS agrees on coverage "
        "and set size; exact set identity is not expected because each library "
        "draws its own uniform variates."
    )


# --------------------------------------------------------------------------
# Limitation
# --------------------------------------------------------------------------

elif page == "Stability & churn":
    st.title("Stability and operational churn")
    st.markdown(
        "A controller that constantly resizes the deployment is worse to "
        "operate even at identical cost and reliability — every scaling action "
        "terminates and starts pods, drains connections, and pays cold-start "
        "latency on the replacements. This is measured in the discrete-event "
        "simulator, which models cooldowns and pod startup; the decision-level "
        "harness scores windows independently and so cannot express churn."
    )

    if not simulation_available():
        st.warning(
            "Simulator results not found. Generate them with:\n\n"
            "`python examples/run_simulation.py --model-dir models_gt "
            "--max-workloads 24 --sim-minutes 360 --results-dir results_sim6`"
        )
    else:
        sim = _simulation()
        sim_stats = summarise_simulation(sim)

        if "AAPA" in sim_stats.index and SIM_OURS in sim_stats.index:
            base, ours = sim_stats.loc["AAPA"], sim_stats.loc[SIM_OURS]
            metric_row([
                ("Scaling actions (ours)", f"{ours.scaling_actions:.1f}",
                 f"{ours.churn_vs_base:+.0f}% vs AAPA",
                 "Mean resize operations per simulated workload."),
                ("Scaling actions (AAPA)", f"{base.scaling_actions:.1f}", None,
                 "The paper's controller chases per-archetype policies."),
                ("Compute (ours)", f"{ours.total_pod_minutes:.0f} pod-min",
                 f"{ours.pods_vs_base:+.0f}% vs AAPA", None),
                ("Workloads simulated", f"{sim.workload_id.nunique()}", None,
                 "Each replayed through the full SimPy model."),
            ])

        st.markdown("---")
        st.plotly_chart(
            charts.stability_bar(sim_stats, SIM_LABEL, SIM_COLOR), width="stretch"
        )
        st.success(
            "Our controller settles on a stable target instead of chasing "
            "archetype-specific policies, cutting resize operations by roughly "
            "97% against AAPA while using less compute."
        )

        st.markdown("---")
        st.subheader("All three axes, relative to AAPA")
        st.plotly_chart(
            charts.three_axis_compare(sim_stats, SIM_COLOR), width="stretch"
        )

        st.markdown("---")
        st.subheader("Which metrics actually distinguish the controllers")
        st.markdown(
            "A metric whose spread across controllers is near 1× tells you "
            "nothing. Reporting it would imply a difference the data does not "
            "show, so these are excluded from the headline comparison."
        )
        sep = metric_separation(sim_stats)
        sep_show = sep.copy()
        sep_show["verdict"] = sep_show.discriminates.map(
            {True: "discriminates", False: "flat — not reported"}
        )
        sep_show = sep_show[["metric", "min", "max", "ratio", "verdict"]]
        sep_show.columns = ["Metric", "Min", "Max", "Spread (×)", "Verdict"]
        st.dataframe(
            sep_show.style.format({
                "Min": "{:.4g}", "Max": "{:.4g}", "Spread (×)": "{:.2f}",
            }),
            width="stretch", hide_index=True,
        )
        st.caption(
            "Latency, CPU utilisation and cold starts land within ~1% across "
            "every controller in this simulation, so they are reported here "
            "for transparency rather than used as evidence."
        )

        st.markdown("---")
        st.subheader("Full simulator results")
        full = sim_stats[[
            "label", "slo_violation_rate", "total_pod_minutes",
            "scaling_actions", "rei_paper_text",
        ]].copy()
        full.columns = [
            "Controller", "SLO violation rate", "Pod-minutes",
            "Scaling actions", "REI (paper weights)",
        ]
        st.dataframe(
            full.style.format({
                "SLO violation rate": "{:.4f}", "Pod-minutes": "{:.1f}",
                "Scaling actions": "{:.2f}", "REI (paper weights)": "{:.4f}",
            }),
            width="stretch", hide_index=True,
        )
        st.caption(
            "Note: these come from the SimPy simulator over 24 workloads, a "
            "different harness from the decision-level results on the other "
            "pages. Cost units differ, so the two should not be pooled."
        )


elif page == "Limitation":
    st.title("The capacity ceiling")
    st.markdown(
        "The conformal advantage holds on workloads where the top-ranked "
        "archetype is **not** the most resource-intensive one. When it is, the "
        "mechanism has nothing to offer. This is structural, not a tuning "
        "problem."
    )

    c1, c2 = st.columns([2, 3])
    with c1:
        st.plotly_chart(charts.capacity_ladder(CAPACITY_LADDER), width="stretch")
    with c2:
        st.subheader("Why hedging stalls")
        st.markdown(
            "Our rule sizes for the **most demanding archetype in the set**. "
            "That adds capacity only when the set contains something more "
            "demanding than the top-ranked prediction."
        )
        st.markdown(
            "- Top-1 is PERIODIC (1.33×), set also holds STATIONARY (1.82×) → "
            "hedging adds **37%** capacity. It helps.\n"
            "- Top-1 is SPIKE (3.33×), the highest rung → `max` returns the "
            "same answer. Hedging adds **nothing**."
        )
        st.markdown(
            "AAPA's margin has no ceiling: it multiplies *on top of* the "
            "target, so it keeps adding capacity exactly where it is needed. "
            "That is why it wins once SPIKE is present."
        )

    st.markdown("---")
    st.subheader("Measured with SPIKE restored")
    spike_exp = next((e for e in experiments if e.key == "spike"), None)
    if spike_exp:
        spike_dec, _ = get_run(spike_exp)
        spike_stats = summarise(spike_dec)
        show = spike_stats[["label", "failures", "rate", "pods"]].copy()
        show.columns = ["Controller", "Failures", "Violation rate", "Mean pods"]
        st.dataframe(
            show.style.format({
                "Failures": "{:,.0f}", "Violation rate": "{:.4f}", "Mean pods": "{:.1f}",
            }),
            width="stretch", hide_index=True,
        )
        if OURS in spike_stats.index and CONTROL in spike_stats.index:
            st.error(
                f"Our controller records "
                f"**{int(spike_stats.loc[OURS].failures):,}** failures against "
                f"the ablation control's "
                f"**{int(spike_stats.loc[CONTROL].failures):,}** — effectively "
                f"identical. With SPIKE present the prediction set contributes "
                f"nothing, exactly as the ladder predicts."
            )
    else:
        st.info("The SPIKE-inclusive run is not available in this workspace.")

    st.markdown("---")
    st.subheader("Where this leads")
    st.markdown(
        "The method answers *which archetype is this?* when the question that "
        "decides success is *how much capacity do I need?* Those coincide only "
        "while the archetype-to-capacity mapping has headroom left."
    )
    st.markdown(
        "**Next step — conformal prediction on demand.** Predict next-window "
        "peak demand directly with a calibrated upper bound, rather than "
        "classifying archetype and looking up a CPU target. It has no ceiling, "
        "the guarantee `P(demand ≤ provisioned) ≥ 1−α` *is* the SLO promise "
        "rather than a proxy, and it bypasses the labelling and filtering "
        "defects upstream entirely."
    )
