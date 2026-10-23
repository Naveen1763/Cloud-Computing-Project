"""Plotly figures for the results dashboard."""

from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.graph_objects as go

from dashboard.data import CONTROLLER_COLOR, CONTROLLER_LABEL

LAYOUT = dict(
    template="plotly_white",
    font=dict(size=13),
    margin=dict(l=60, r=30, t=50, b=60),
    hoverlabel=dict(font_size=13),
)


def failures_bar(stats: pd.DataFrame, n_windows: int) -> go.Figure:
    """Absolute failure counts per controller."""
    fig = go.Figure(
        go.Bar(
            x=stats.label,
            y=stats.failures,
            marker_color=[CONTROLLER_COLOR.get(c, "#666") for c in stats.index],
            marker_line=dict(color="#222", width=1),
            text=[f"{int(v):,}" for v in stats.failures],
            textposition="outside",
            textfont=dict(size=14),
            hovertemplate="<b>%{x}</b><br>%{y:,} failures<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"SLO violations across {n_windows:,} scaling decisions",
        yaxis_title="SLO violations",
        showlegend=False,
        **LAYOUT,
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def tradeoff_scatter(stats: pd.DataFrame) -> go.Figure:
    """Cost against reliability. Bottom-left is best."""
    fig = go.Figure()
    for controller, row in stats.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row.pods],
                y=[row.rate * 100],
                mode="markers+text",
                marker=dict(
                    size=26,
                    color=CONTROLLER_COLOR.get(controller, "#666"),
                    line=dict(color="#222", width=1.5),
                ),
                text=[row.label],
                textposition="top center",
                textfont=dict(size=12),
                name=row.label,
                hovertemplate=(
                    f"<b>{row.label}</b><br>"
                    f"{row.pods:.1f} pods<br>"
                    f"{row.rate * 100:.3f}% violations<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Cost vs reliability — bottom-left is better",
        xaxis_title="Mean pods provisioned  (cost →)",
        yaxis_title="SLO violation rate (%)",
        showlegend=False,
        **LAYOUT,
    )
    return fig


def grouped_bar(pivot: pd.DataFrame, title: str, ytitle: str,
                percent: bool = False) -> go.Figure:
    """Metric by archetype, one bar group per controller."""
    inverse = {v: k for k, v in CONTROLLER_LABEL.items()}
    fig = go.Figure()
    for column in pivot.columns:
        vals = pivot[column] * (100 if percent else 1)
        fig.add_trace(
            go.Bar(
                name=column,
                x=pivot.index,
                y=vals,
                marker_color=CONTROLLER_COLOR.get(inverse.get(column, ""), "#666"),
                hovertemplate="<b>%{x}</b><br>" + column + ": %{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title, yaxis_title=ytitle, barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), **LAYOUT,
    )
    return fig


def before_after(metrics, labels, title: str) -> go.Figure:
    """Paired bars for a measured before/after finding."""
    names = [m[0] for m in metrics]
    before = [m[1] for m in metrics]
    after = [m[2] for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=labels[0], x=names, y=before, marker_color="#bdbdbd",
        marker_line=dict(color="#222", width=1),
        text=[f"{v:.1f}%" for v in before], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name=labels[1], x=names, y=after, marker_color="#1e88e5",
        marker_line=dict(color="#222", width=1),
        text=[f"{v:.1f}%" for v in after], textposition="outside",
    ))
    fig.update_layout(
        title=title, yaxis_title="%", barmode="group",
        yaxis=dict(range=[0, 115]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **LAYOUT,
    )
    return fig


def set_size_distribution(dist: Dict[str, int]) -> go.Figure:
    """How many archetypes the prediction set holds, per window."""
    sizes = sorted(int(k) for k in dist)
    counts = [dist[str(s)] if str(s) in dist else dist[s] for s in sizes]
    total = sum(counts) or 1
    fig = go.Figure(go.Bar(
        x=[str(s) for s in sizes],
        y=counts,
        marker_color="#1e88e5",
        marker_line=dict(color="#222", width=1),
        text=[f"{c:,}<br>({c / total:.1%})" for c in counts],
        textposition="outside",
        hovertemplate="Set size %{x}<br>%{y:,} windows<extra></extra>",
    ))
    fig.update_layout(
        title="How many archetypes remain plausible",
        xaxis_title="Archetypes in the prediction set",
        yaxis_title="Windows",
        showlegend=False,
        **LAYOUT,
    )
    return fig


def per_class_coverage(per_class: Dict[str, Dict], target: float) -> go.Figure:
    """Coverage achieved for each archetype against the target."""
    names = sorted(per_class)
    cov = [per_class[n]["coverage"] for n in names]
    colors = ["#e53935" if c < target - 0.005 else "#1e88e5" for c in cov]

    fig = go.Figure(go.Bar(
        x=names, y=cov, marker_color=colors,
        marker_line=dict(color="#222", width=1),
        text=[f"{c:.3f}" for c in cov], textposition="outside",
        hovertemplate="<b>%{x}</b><br>coverage %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(
        y=target, line_dash="dash", line_color="#222",
        annotation_text=f"target {target:.2f}", annotation_position="right",
    )
    fig.update_layout(
        title="Coverage by archetype — red falls short of the guarantee",
        yaxis_title="Empirical coverage",
        yaxis=dict(range=[0, 1.08]),
        showlegend=False,
        **LAYOUT,
    )
    return fig


def capacity_ladder(ladder: pd.DataFrame) -> go.Figure:
    """Pods required per unit of load, by archetype."""
    colors = ["#1e88e5" if a != "SPIKE" else "#e53935" for a in ladder.archetype]
    fig = go.Figure(go.Bar(
        x=ladder.archetype, y=ladder.pods_per_unit_load,
        marker_color=colors, marker_line=dict(color="#222", width=1),
        text=[f"{v:.2f}×" for v in ladder.pods_per_unit_load],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>CPU target %{customdata:.0%}"
            "<br>%{y:.2f}× pods<extra></extra>"
        ),
        customdata=ladder.target_cpu,
    ))
    fig.update_layout(
        title="The capacity ladder — SPIKE is the top rung",
        yaxis_title="Pods per unit of load",
        showlegend=False,
        **LAYOUT,
    )
    return fig


def stability_bar(stats, label_map, color_map) -> go.Figure:
    """Scaling actions per controller - the operational churn metric.

    Every scaling action terminates and starts pods, so a controller that
    thrashes is worse to operate even at equal cost and SLO.
    """
    fig = go.Figure(go.Bar(
        x=stats.label,
        y=stats.scaling_actions,
        marker_color=[color_map.get(c, "#666") for c in stats.index],
        marker_line=dict(color="#222", width=1),
        text=[f"{v:.1f}" for v in stats.scaling_actions],
        textposition="outside",
        textfont=dict(size=14),
        hovertemplate="<b>%{x}</b><br>%{y:.1f} scaling actions<extra></extra>",
    ))
    fig.update_layout(
        title="Scaling actions per workload — lower is more stable",
        yaxis_title="Scaling actions",
        showlegend=False,
        **LAYOUT,
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def three_axis_compare(stats, color_map) -> go.Figure:
    """Reliability, cost and churn side by side, each normalised to AAPA.

    Normalising lets three different units share one axis; 100% is AAPA on
    every measure, and lower is better throughout.
    """
    from plotly.subplots import make_subplots

    specs = [
        ("slo_violation_rate", "SLO violations"),
        ("total_pod_minutes", "Compute (pod-minutes)"),
        ("scaling_actions", "Scaling actions"),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[s[1] for s in specs])

    for i, (col, _) in enumerate(specs, start=1):
        base = stats.loc["AAPA", col] if "AAPA" in stats.index else 1.0
        vals = stats[col] / base * 100 if base else stats[col]
        fig.add_trace(
            go.Bar(
                x=stats.label, y=vals,
                marker_color=[color_map.get(c, "#666") for c in stats.index],
                marker_line=dict(color="#222", width=1),
                text=[f"{v:.0f}%" for v in vals],
                textposition="outside",
                showlegend=False,
                hovertemplate="<b>%{x}</b><br>%{y:.0f}% of AAPA<extra></extra>",
            ),
            row=1, col=i,
        )
        fig.add_hline(y=100, line_dash="dash", line_color="#888", row=1, col=i)

    fig.update_layout(
        title="Relative to AAPA (100% = AAPA, lower is better)",
        template="plotly_white", font=dict(size=12),
        margin=dict(l=50, r=30, t=80, b=110), height=460,
    )
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(title_text="% of AAPA", row=1, col=1)
    return fig
