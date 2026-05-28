"""Judge comparison component for Streamlit dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

if TYPE_CHECKING:
    from src.dashboard.data_loader import DashboardDataLoader


def _extract_model_name(judge_name: str) -> str:
    """Extract model name from judge column name.

    Examples:
        'gpt_4o' -> 'gpt-4o'
        'gpt_4o_mini' -> 'gpt-4o-mini'
        'claude_opus_4_5_20251101_run_2' -> 'claude-opus-4.5'
        'gpt_5_1' -> 'gpt-5.1'
        'claude_haiku_4_5_20251001' -> 'claude-haiku-4.5'
    """
    import re

    # Remove run suffix if present
    if "_run_" in judge_name:
        judge_name = judge_name.split("_run_")[0]

    # Remove date suffix (8-digit numbers at the end like 20251101)
    judge_name = re.sub(r"_\d{8}$", "", judge_name)

    # Replace underscores with hyphens
    model_name = judge_name.replace("_", "-")

    # Fix version numbers (e.g., "4-5" -> "4.5", "5-1" -> "5.1")
    # This handles cases like "claude-opus-4-5" -> "claude-opus-4.5"
    model_name = re.sub(r"-(\d)-(\d)(?![\d])", r"-\1.\2", model_name)

    return model_name


def _infer_reasoning(model_name: str, reasoning_config: dict | None) -> str:
    """Infer if reasoning is used from model name and config.

    Returns:
        - "thinking" if Anthropic thinking/budget is detected
        - "reasoning" if OpenAI o1 models detected
        - "" if no reasoning detected
    """
    # If we have config with thinking_budget, it's using thinking
    if reasoning_config and "thinking_budget" in reasoning_config:
        return "thinking"

    # Check model name for indicators
    model_lower = model_name.lower()

    # Anthropic Claude with thinking capability
    if "claude" in model_lower and ("opus" in model_lower or "sonnet" in model_lower):
        # These models support thinking in newer versions, but we'll only mark if config says so
        if reasoning_config and "thinking_budget" in reasoning_config:
            return "thinking"

    # OpenAI o1 models have reasoning
    if "o1" in model_lower or "o-1" in model_lower:
        return "reasoning"

    return ""


def _format_judge_label(judge_name: str, model_name: str, reasoning: str) -> str:
    """Format judge display label with model name and reasoning info.

    Examples:
        'gpt-4o' + 'reasoning' -> 'gpt-4o (reasoning)'
        'claude-opus-4.5' + 'thinking' -> 'claude-opus-4.5 (thinking)'
        'gpt-4o' + '' -> 'gpt-4o'
    """
    label = model_name

    if reasoning:
        label += f" ({reasoning})"

    return label


def render_judge_comparison(loader: DashboardDataLoader, sidebar_mode: bool = False):
    """Render the Judge Comparison section for multi-judge analysis.

    Args:
        loader: Dashboard data loader
        sidebar_mode: If True, render pickers in sidebar instead of main body
    """
    # Check if any datasets have multi-judge results
    multi_judge_datasets = loader.get_datasets_with_multi_judge()

    if not multi_judge_datasets:
        st.info("No datasets with multiple judge results found.")
        return

    st.subheader("⚖️ Judge Comparison")
    st.markdown("Compare deception labels from multiple judge models.")

    # Dataset selection
    dataset_options = [f"{ds}/{model}" for ds, model, _ in multi_judge_datasets]

    if sidebar_mode:
        st.sidebar.header("🎛️ Filters")
        selected_idx = st.sidebar.selectbox(
            "Dataset with Multi-Judge Results",
            options=range(len(dataset_options)),
            format_func=lambda i: dataset_options[i],
            key="judge_comparison_dataset",
        )
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_idx = st.selectbox(
                "Dataset with Multi-Judge Results",
                options=range(len(dataset_options)),
                format_func=lambda i: dataset_options[i],
                key="judge_comparison_dataset",
            )

    dataset_name, model_name, _ = multi_judge_datasets[selected_idx]

    # Get summary stats
    summary = loader.get_multi_judge_summary(dataset_name, model_name)
    if not summary:
        st.warning("Could not load multi-judge summary.")
        return

    judges = summary.get("judges", [])
    n_samples = summary.get("n_samples", 0)
    completeness = summary.get("completeness", {})

    # Filter out incomplete judges (exclude from comparison)
    incomplete_judges = {j: pct for j, pct in completeness.items() if pct > 0}
    complete_judges = [j for j in judges if j not in incomplete_judges]

    # Warn about excluded incomplete judges
    if incomplete_judges:
        warning_msg = "⚠️ **Excluding incomplete judge data from comparison:**\n\n"
        for judge, pct in incomplete_judges.items():
            warning_msg += f"- **{judge}**: {pct:.1%} missing\n"
        st.warning(warning_msg)

    # Check if we have at least one judge to display
    if not complete_judges or len(complete_judges) == 1:
        st.error("Not enough complete judges for comparison (need at least 2).")
        return

    # Build judge display labels with model name and reasoning info (only for complete judges)
    judge_display_labels = {}
    for judge in judges:
        if judge == "primary":
            judge_display_labels[judge] = "Primary"
        else:
            # Extract model name
            model_short = _extract_model_name(judge)

            # Get reasoning config
            reasoning_config = loader.get_judge_reasoning_config(dataset_name, model_name, judge)
            reasoning = _infer_reasoning(model_short, reasoning_config)

            # Format label
            judge_display_labels[judge] = _format_judge_label(judge, model_short, reasoning)

    # Recompute metrics using only complete judges
    from src.utils.judge_metrics import (
        compute_agreement_rate,
        compute_fleiss_kappa,
        compute_pairwise_agreement_rate,
        compute_pairwise_fleiss_kappa,
        compute_pairwise_kappa,
        get_disagreement_indices,
    )

    labels_df = loader.get_judge_labels_df(dataset_name, model_name)
    if labels_df.empty:
        st.error("Could not load judge labels.")
        return

    # Filter to only complete judges
    judge_cols_to_use = []
    labels_by_judge_filtered = {}
    for col in labels_df.columns:
        if col.startswith("judge_"):
            judge_name = col.replace("judge_", "")
            if judge_name in complete_judges:
                judge_cols_to_use.append(col)
                labels_by_judge_filtered[judge_name] = labels_df[col].tolist()

    if "primary_label" in labels_df.columns:
        labels_by_judge_filtered["primary"] = labels_df["primary_label"].tolist()

    # Recompute pairwise metrics with filtered judges
    pairwise_kappa_filtered = compute_pairwise_kappa(labels_by_judge_filtered)
    pairwise_agreement_filtered = compute_pairwise_agreement_rate(labels_by_judge_filtered)
    pairwise_fleiss_filtered = compute_pairwise_fleiss_kappa(labels_by_judge_filtered)

    # Compute overall agreement metrics
    overall_fleiss_kappa = compute_fleiss_kappa(labels_by_judge_filtered)
    overall_agreement = compute_agreement_rate(labels_by_judge_filtered)
    disagreement_indices_filtered = get_disagreement_indices(labels_by_judge_filtered)

    # Summary metrics
    if sidebar_mode:
        st.sidebar.metric("Judges", len(complete_judges))
    else:
        with col2:
            st.metric("Judges", len(complete_judges))

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Samples", n_samples)
    with col_b:
        st.metric("Overall Agreement", f"{overall_agreement:.1%}")
    with col_c:
        st.metric("Fleiss' κ (All)", f"{overall_fleiss_kappa:.2f}")
    with col_d:
        st.metric("Disagreements", len(disagreement_indices_filtered))

    # Metric selection for pairwise matrices
    metric_options = ["Pairwise Kappa", "Pairwise Agreement Rate", "Pairwise Fleiss' Kappa"]
    selected_metric = st.selectbox(
        "Comparison Metric (Matrix)",
        options=metric_options,
        index=0,
        key="judge_comparison_metric",
    )

    # Two columns: Metric matrix and Label distribution
    col_left, col_right = st.columns(2)

    with col_left:
        if selected_metric == "Pairwise Kappa":
            st.markdown("**Pairwise Cohen's Kappa**")
            kappa_df = pairwise_kappa_filtered
            metric_data = kappa_df
        elif selected_metric == "Pairwise Agreement Rate":
            st.markdown("**Pairwise Agreement Rate**")
            metric_data = pairwise_agreement_filtered
        else:  # Pairwise Fleiss' Kappa
            st.markdown("**Pairwise Fleiss' Kappa**")
            metric_data = pairwise_fleiss_filtered

        if metric_data is not None and not metric_data.empty:
            # Rename columns and index to use display labels
            display_df = metric_data.copy()
            display_df.columns = [judge_display_labels.get(c, c) for c in display_df.columns]
            display_df.index = [judge_display_labels.get(i, i) for i in display_df.index]

            # Mask upper triangle (keep only lower triangle)
            mask = np.triu(np.ones_like(display_df.values, dtype=bool), k=0)
            masked_values = np.where(mask, np.nan, display_df.values)

            # Determine colorscale and range based on metric
            if "Kappa" in selected_metric:
                # Custom Kappa colorscale with 0.9 as gold standard threshold
                # Range is -1 to 1, so we normalize: (kappa + 1) / 2
                # 0.9 kappa = 0.95 normalized position
                colorscale = [
                    [0.0, "#d73027"],  # Low agreement - red
                    [0.5, "#d73027"],  # kappa=0 - red
                    [0.5, "#fc8d59"],  # boundary
                    [0.65, "#fc8d59"],  # kappa=0.30
                    [0.65, "#fee090"],  # boundary
                    [0.75, "#fee090"],  # kappa=0.50
                    [0.75, "#d9ef8b"],  # boundary
                    [0.85, "#d9ef8b"],  # kappa=0.70
                    [0.85, "#91cf60"],  # boundary
                    [0.95, "#91cf60"],  # kappa=0.90 - gold standard threshold
                    [0.95, "#1a9850"],  # excellent
                    [1.0, "#1a9850"],  # kappa=1.0 - perfect
                ]
                zmin, zmax = -1, 1
            else:  # Agreement Rate
                colorscale = "Blues"
                zmin, zmax = 0, 1

            # Create text matrix with lower triangle only
            text_matrix = [[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in masked_values]

            fig = go.Figure(
                data=go.Heatmap(
                    z=masked_values,
                    x=display_df.columns.tolist(),
                    y=display_df.index.tolist(),
                    colorscale=colorscale,
                    zmin=zmin,
                    zmax=zmax,
                    text=text_matrix,
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    hovertemplate="%{y} vs %{x}<br>Score: %{z:.3f}<extra></extra>",
                    showscale=True,
                )
            )
            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="",
                yaxis_title="",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Not enough judges for pairwise comparison.")

    with col_right:
        st.markdown("**Deceptive Labels by Judge**")
        if not labels_df.empty:
            # Only show complete judges in distribution - deceptive counts only
            judge_cols = [c for c in labels_df.columns if c.startswith("judge_") or c == "primary_label"]
            dist_data = []
            for col in judge_cols:
                if col == "primary_label":
                    judge_name = "primary"
                else:
                    judge_name = col.replace("judge_", "")

                # Skip incomplete judges
                if judge_name != "primary" and judge_name not in complete_judges:
                    continue

                display_label = judge_display_labels.get(judge_name, judge_name)
                valid = labels_df[col].dropna()
                if len(valid) > 0:
                    n_deceptive = (valid == 0).sum()
                    dist_data.append({"Judge": display_label, "Count": n_deceptive})

            if dist_data:
                dist_df = pd.DataFrame(dist_data)
                # Sort by deceptive count for easier comparison
                dist_df = dist_df.sort_values("Count", ascending=True)

                fig = go.Figure(
                    go.Bar(
                        x=dist_df["Count"],
                        y=dist_df["Judge"],
                        orientation="h",
                        marker_color="#FF6B6B",
                        text=dist_df["Count"],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="Deceptive Count",
                    yaxis_title="",
                )
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("No label data available.")

    # Disagreement Analysis
    st.markdown("---")
    st.markdown("### 🔬 Disagreement Analysis")

    # Analyze if disagreements are on "hard samples" or random noise
    if disagreement_indices_filtered:
        # Count how many times each sample is disagreed upon across all judge pairs
        disagreement_counts = {}
        for idx in disagreement_indices_filtered:
            row = labels_df.iloc[idx]
            judge_cols = [
                c for c in labels_df.columns if c.startswith("judge_") and c.replace("judge_", "") in complete_judges
            ]
            labels_for_sample = [row[c] for c in judge_cols if pd.notna(row[c])]
            # Store the split (how many judges voted each way)
            if len(labels_for_sample) > 0:
                n_deceptive = labels_for_sample.count(0)
                n_honest = labels_for_sample.count(1)
                disagreement_counts[idx] = {
                    "deceptive": n_deceptive,
                    "honest": n_honest,
                    "total": len(labels_for_sample),
                }

        # Calculate disagreement patterns
        if disagreement_counts:
            # Categorize disagreements
            close_splits = []  # Near 50-50 splits (hard samples)
            one_outlier = []  # One judge disagrees (potential noise)

            for idx, counts in disagreement_counts.items():
                total = counts["total"]
                minority = min(counts["deceptive"], counts["honest"])

                # If minority is 1 and total > 2, it's likely one outlier
                if minority == 1 and total > 2:
                    one_outlier.append(idx)
                # If split is close (minority >= 40% of total), it's a hard sample
                elif minority / total >= 0.4:
                    close_splits.append(idx)
                else:
                    # Everything else is somewhere in between
                    close_splits.append(idx)

            col_analysis_a, col_analysis_b = st.columns(2)
            with col_analysis_a:
                pct_close = len(close_splits) / len(disagreement_counts) * 100 if disagreement_counts else 0
                st.metric(
                    "Hard Samples (close splits)",
                    len(close_splits),
                    delta=f"{pct_close:.0f}% of disagreements",
                    delta_color="off",
                )
            with col_analysis_b:
                pct_outlier = len(one_outlier) / len(disagreement_counts) * 100 if disagreement_counts else 0
                st.metric(
                    "Single Outlier Votes",
                    len(one_outlier),
                    delta=f"{pct_outlier:.0f}% of disagreements",
                    delta_color="off",
                )

            # How to read this metric
            st.caption(
                "ℹ️ **How to read**: High 'Single Outlier' count suggests noise (one judge disagrees). "
                "High 'Hard Samples' count suggests genuinely ambiguous samples (judges split evenly)."
            )

    # Disagreement Explorer
    with st.expander(f"🔍 Disagreement Explorer ({len(disagreement_indices_filtered)} samples)", expanded=False):
        if disagreement_indices_filtered:
            max_show = 20
            show_indices = disagreement_indices_filtered[:max_show]

            for idx in show_indices:
                row = labels_df.iloc[idx]
                # Only show labels from complete judges
                judge_cols = [c for c in labels_df.columns if c.startswith("judge_")]
                labels_str = ", ".join(
                    f"{judge_display_labels.get(c.replace('judge_', ''), c.replace('judge_', ''))}: "
                    f"{'D' if row[c] == 0 else 'H' if row[c] == 1 else '?'}"
                    for c in judge_cols
                    if pd.notna(row[c]) and c.replace("judge_", "") in complete_judges
                )
                primary = row.get("primary_label")
                primary_str = f"Primary: {'D' if primary == 0 else 'H'}" if pd.notna(primary) else ""
                st.markdown(f"**Sample {idx}**: {primary_str} | {labels_str}")

            if len(disagreement_indices_filtered) > max_show:
                st.info(f"Showing first {max_show} of {len(disagreement_indices_filtered)} disagreements.")
        else:
            st.success("All judges agree on all samples!")
