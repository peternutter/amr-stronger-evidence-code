"""Probe performance visualization component for Streamlit dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.components.figures import create_line_chart_figure
from src.plots.common import format_pooling_label

if TYPE_CHECKING:
    from src.dashboard.data_loader import DashboardDataLoader

# Default pooling to show
DEFAULT_POOLING = "flat"


def render_probe_performance(
    loader: DashboardDataLoader,
    train_dataset: str,
    probe: str,
    model: str,
    eval_dataset: str,
    metric: str,
    use_calibrated_threshold: bool = False,
):
    """Render the probe performance line chart.

    Shows metrics across layers for different pooling/aggregation configurations.
    """
    st.subheader("📈 Probe Performance Across Layers")

    df = loader.get_probe_metrics(train_dataset, probe, model, eval_dataset)

    if df.empty:
        st.warning("No metrics available for this configuration.")
        return

    # Validate that the metric column exists
    if metric not in df.columns:
        available = [
            c
            for c in df.columns
            if c
            not in [
                "train_dataset",
                "probe",
                "model",
                "eval_dataset",
                "layer",
                "pooling",
                "aggregation",
            ]
        ]
        st.warning(f"Metric '{metric}' is not available in this data. Available metrics: {', '.join(available)}")
        return

    # When using calibrated threshold, recompute metrics for applicable metrics
    if use_calibrated_threshold and metric in ("recall", "precision", "accuracy", "f1"):
        df = loader.get_probe_metrics_calibrated(train_dataset, probe, model, eval_dataset)
        if df.empty:
            st.warning("Could not compute calibrated metrics.")
            return
        st.caption(f"📏 Showing **{metric}** computed at calibrated threshold for each layer")

    # Find best config for current metric
    best_config = loader.get_best_config(train_dataset, probe, model, eval_dataset, metric)
    # If using calibrated threshold, find best from calibrated data
    if use_calibrated_threshold and metric in ("recall", "precision", "accuracy", "f1"):
        if not df.empty and metric in df.columns:
            best_row = df.loc[df[metric].idxmax()]
            best_config = {
                "layer": int(best_row["layer"]),
                "pooling": best_row["pooling"],
                "aggregation": best_row.get("aggregation", "legacy"),
                "value": float(best_row[metric]),
            }

    # Show best config info
    if best_config:
        best_label = format_pooling_label(best_config["pooling"], best_config.get("aggregation"))
        st.success(
            f"🏆 **Best {metric.replace('_', ' ').title()}:** {best_config['value']:.4f} "
            f"at Layer {best_config['layer']} ({best_label})"
        )

    # Get all available configurations
    all_combinations = df[["pooling", "aggregation"]].drop_duplicates().sort_values(["pooling", "aggregation"])
    all_config_tuples = [tuple(row) for row in all_combinations.values]
    all_config_labels = [format_pooling_label(p, a) for p, a in all_config_tuples]

    # Default: show one pooling with all aggregations (prefer DEFAULT_POOLING if available)
    unique_poolings = df["pooling"].unique()
    if DEFAULT_POOLING in unique_poolings:
        default_pooling = DEFAULT_POOLING
    elif "last" in unique_poolings:
        default_pooling = "last"
    else:
        default_pooling = df["pooling"].iloc[0]
    default_configs = [(p, a) for p, a in all_config_tuples if p == default_pooling]
    # If no matches, fall back to first 3 configs
    if not default_configs:
        default_configs = all_config_tuples[:3]

    # Multiselect for configurations to display
    selected_indices = st.multiselect(
        "📊 Configs to Display",
        options=range(len(all_config_tuples)),
        default=[all_config_tuples.index(c) for c in default_configs if c in all_config_tuples],
        format_func=lambda i: all_config_labels[i],
        help="Select which pooling/aggregation combinations to show on the chart",
    )
    selected_configs = {all_config_tuples[i] for i in selected_indices}

    # Create line chart figure
    fig = create_line_chart_figure(
        title="",
        xaxis_title="Layer",
        yaxis_title=metric.replace("_", " ").title(),
        height=450,
        show_legend=True,
    )

    marker_cycle = [
        "circle",
        "square",
        "diamond",
        "triangle-up",
        "triangle-down",
        "pentagon",
        "hexagon",
        "star",
        "cross",
        "x",
    ]

    for idx, row in enumerate(all_combinations.itertuples(index=False)):
        pooling_val = str(row.pooling)
        agg_val = str(row.aggregation)

        # Skip if not in selected configs
        if (pooling_val, agg_val) not in selected_configs:
            continue

        subset = df[(df["pooling"] == pooling_val) & (df["aggregation"] == agg_val)]

        # Filter out layers with 0 values (untrained layers)
        subset = subset[subset[metric] > 0]

        if subset.empty:
            continue

        # Sort by layer for proper line connection
        subset = subset.sort_values("layer")

        label = format_pooling_label(pooling_val, agg_val)
        marker = marker_cycle[idx % len(marker_cycle)]

        # Highlight best config line
        is_best = best_config and best_config["pooling"] == pooling_val

        fig.add_trace(
            go.Scatter(
                x=subset["layer"],
                y=subset[metric],
                mode="lines+markers",
                name=label + (" ⭐" if is_best else ""),
                marker=dict(symbol=marker, size=10 if is_best else 8),
                line=dict(width=3 if is_best else 2),
            )
        )

    # Add marker for best point
    if best_config:
        fig.add_trace(
            go.Scatter(
                x=[best_config["layer"]],
                y=[best_config["value"]],
                mode="markers",
                name="Best",
                marker=dict(symbol="star", size=20, color="#FFD700", line=dict(color="black", width=1)),
                showlegend=False,
            )
        )

    # Adjust y-axis range based on metric type
    if metric in ["fpr", "fnr"]:
        y_range = [0, max(0.5, df[metric].max() * 1.1)]
    elif metric == "mcc":
        y_range = [-1, 1]
    else:
        y_range = [0, 1]

    fig.update_layout(
        yaxis=dict(range=y_range),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    st.plotly_chart(fig, width="stretch")

    # Best metrics summary
    with st.expander("📊 Best Metrics Summary (All Metrics)"):
        best_data = []
        all_metrics = [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "specificity",
            "mcc",
        ]
        for m in all_metrics:
            if m in df.columns and not df[m].isna().all():
                idx_max = df[m].idxmax()
                best_val = df.loc[idx_max, m]
                best_layer = df.loc[idx_max, "layer"]
                best_cfg = format_pooling_label(
                    df.loc[idx_max, "pooling"],
                    df.loc[idx_max, "aggregation"],
                )
                best_data.append(
                    {
                        "Metric": m.replace("_", " ").title(),
                        "Best Value": f"{best_val:.4f}",
                        "Layer": int(best_layer),
                        "Config": best_cfg,
                    }
                )
        if best_data:
            st.dataframe(pd.DataFrame(best_data), hide_index=True, width="stretch")
