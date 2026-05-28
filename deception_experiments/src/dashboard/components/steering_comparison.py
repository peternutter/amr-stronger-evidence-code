"""Steering comparison component for visualizing effects of activation steering."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

if TYPE_CHECKING:
    from src.dashboard.data_loader import DashboardDataLoader


def _extract_strength(model_name: str) -> str:
    """Extract steering strength from model name."""
    match = re.search(r"_steered-.+-(-?\w+)$", model_name)
    return match.group(1) if match else model_name


def _format_strength_label(strength: str) -> str:
    """Format steering strength for display."""
    if strength == "neg2":
        return "-2 (less deceptive)"
    elif strength == "2":
        return "+2 (more deceptive)"
    elif strength == "0":
        return "0 (projected out)"
    elif strength == "base":
        return "Base (no steering)"
    return strength


def render_steering_comparison(loader: DashboardDataLoader, sidebar_mode: bool = False):
    """Render the Steering Comparison section.

    Args:
        loader: Dashboard data loader
        sidebar_mode: If True, render pickers in sidebar instead of main body
    """
    st.subheader("🔀 Steering Comparison")
    st.caption("Compare how activation steering affects judge deception labels across model variants.")

    # Get all steered datasets to discover unique vector names
    all_steered_datasets = loader.get_steered_datasets()

    if not all_steered_datasets:
        st.warning(
            "No datasets with multiple steered model variants found. "
            "Run calculate_activations with steered models first."
        )
        return

    # Extract unique vector names for the dropdown
    unique_vectors = sorted({vec for _, _, vec, _ in all_steered_datasets})

    # Picker location depends on mode
    picker_container = st.sidebar if sidebar_mode else st

    # Vector selector dropdown
    selected_vector = picker_container.selectbox(
        "Steering Vector",
        unique_vectors,
        key="steering_vector_picker",
        help="Select which steering vector's results to view",
    )

    if selected_vector is None:
        return

    # Filter datasets by selected vector
    steered_datasets = [(ds, base, vec, vars) for ds, base, vec, vars in all_steered_datasets if vec == selected_vector]

    # Build picker options for filtered datasets
    picker_options = []
    for dataset_name, base_model, vector_name, variants in steered_datasets:
        label = f"{dataset_name} ({len(variants)} variants)"
        picker_options.append((label, dataset_name, base_model, vector_name, variants))

    selected_idx = picker_container.selectbox(
        "Dataset with Steered Variants",
        range(len(picker_options)),
        format_func=lambda i: picker_options[i][0],
        key="steering_dataset_picker",
    )

    if selected_idx is None:
        return

    _, dataset_name, base_model, vector_name, variants = picker_options[selected_idx]

    # Show variant info
    st.info(
        f"**Dataset:** {dataset_name}  \n"
        f"**Base Model:** {base_model}  \n"
        f"**Steering Vector:** {vector_name}  \n"
        f"**Variants:** {', '.join(_extract_strength(v) for v in variants)}"
    )

    # Load comparison data
    with st.spinner("Loading comparison data..."):
        df = loader.get_steering_comparison_df(dataset_name, base_model, vector_name)

    if df.empty:
        st.warning("No comparison data available.")
        return

    # Find judge label columns
    judge_label_cols = [c for c in df.columns if c.startswith("judge_label_")]

    if not judge_label_cols:
        st.warning("No judge labels found in the data.")
        return

    # ==========================================================================
    # Summary Statistics Bar Chart
    # ==========================================================================
    st.markdown("### 📊 Deception Rate by Steering Strength")

    # Calculate % flagged as deceptive for each variant
    summary_data = []
    for col in judge_label_cols:
        strength = col.replace("judge_label_", "")
        labels = df[col].dropna()
        if len(labels) > 0:
            pct_deceptive = (labels.sum() / len(labels)) * 100
            summary_data.append(
                {
                    "strength": strength,
                    "strength_label": _format_strength_label(strength),
                    "pct_deceptive": pct_deceptive,
                    "n_samples": len(labels),
                    "n_deceptive": int(labels.sum()),
                }
            )

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        # Sort by numeric strength (base first, then by number)
        strength_order = {"base": -999, "neg2": -2, "0": 0, "2": 2}
        summary_df["sort_key"] = summary_df["strength"].map(
            lambda s: strength_order.get(s, int(s) if s.lstrip("-").isdigit() else 0)
        )
        summary_df = summary_df.sort_values("sort_key")

        # Create bar chart
        fig = go.Figure(
            data=[
                go.Bar(
                    x=summary_df["strength_label"],
                    y=summary_df["pct_deceptive"],
                    text=[f"{v:.1f}%" for v in summary_df["pct_deceptive"]],
                    textposition="auto",
                    marker_color=[
                        "#ff6b6b" if s == "2" else "#51cf66" if s == "neg2" else "#868e96" if s == "base" else "#748ffc"
                        for s in summary_df["strength"]
                    ],
                    hovertemplate=("<b>%{x}</b><br>" "Deceptive: %{y:.1f}%<br>" "<extra></extra>"),
                )
            ]
        )

        fig.update_layout(
            xaxis_title="Steering Strength",
            yaxis_title="% Samples Flagged Deceptive",
            yaxis_range=[0, 100],
            template="plotly_dark",
            height=350,
        )

        st.plotly_chart(fig, width="stretch")

        # Show summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df))
        if len(summary_df) >= 2:
            min_row = summary_df.loc[summary_df["pct_deceptive"].idxmin()]
            max_row = summary_df.loc[summary_df["pct_deceptive"].idxmax()]
            with col2:
                st.metric(
                    f"Lowest ({min_row['strength']})",
                    f"{min_row['pct_deceptive']:.1f}%",
                )
            with col3:
                st.metric(
                    f"Highest ({max_row['strength']})",
                    f"{max_row['pct_deceptive']:.1f}%",
                )

    # ==========================================================================
    # Sample-Level Comparison Table
    # ==========================================================================
    st.markdown("### 📋 Sample-Level Comparison")

    # Add a column indicating if any label changed across variants
    def check_label_changed(row):
        labels = [row[c] for c in judge_label_cols if pd.notna(row.get(c))]
        if len(labels) < 2:
            return False
        return len(set(labels)) > 1

    df["label_changed"] = df.apply(check_label_changed, axis=1)
    n_changed = df["label_changed"].sum()

    st.caption(
        f"**{n_changed}** samples ({n_changed/len(df)*100:.1f}%) have different labels across steering strengths"
    )

    # Filter toggle
    show_changed_only = st.checkbox(
        "Show only samples with label changes",
        value=False,
        key="steering_changed_only",
    )

    display_df = df[df["label_changed"]] if show_changed_only else df

    # Prepare display columns (no prompt - too long)
    display_cols = ["sample_index"]
    if "ground_truth" in df.columns:
        display_cols.append("ground_truth")
    display_cols.extend(judge_label_cols)

    # Format for display - handle numpy bool properly
    styled_df = display_df[display_cols].copy()
    for col in judge_label_cols:
        styled_df[col] = styled_df[col].apply(
            lambda x: "🔴 Deceptive" if x is True else "🟢 Truthful" if x is False else "❓"
        )

    if "ground_truth" in styled_df.columns:
        styled_df["ground_truth"] = styled_df["ground_truth"].apply(
            lambda x: "🔴" if x is True else "🟢" if x is False else "❓"
        )

    # Rename columns for readability
    rename_map = {col: col.replace("judge_label_", "Steer ") for col in judge_label_cols}
    rename_map["ground_truth"] = "GT"
    rename_map["sample_index"] = "#"
    styled_df = styled_df.rename(columns=rename_map)

    st.dataframe(
        styled_df.head(100),
        width="stretch",
        height=400,
    )

    if len(display_df) > 100:
        st.caption(f"Showing first 100 of {len(display_df)} samples")

    # ==========================================================================
    # Detailed Sample Viewer
    # ==========================================================================
    st.markdown("### 🔍 Detailed Sample Viewer")

    sample_idx = st.slider(
        "Sample Index",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        key=f"steering_sample_idx_{dataset_name}_{vector_name}",
    )

    sample = df.iloc[sample_idx]

    # Show prompt
    if "prompt" in sample:
        with st.expander("📝 Prompt", expanded=False):
            # Get full prompt from first variant
            first_variant = variants[0]
            full_ds = loader.load_dataset_metadata(dataset_name, first_variant)
            if full_ds and "prompt_text" in full_ds.column_names:
                st.text(full_ds[sample_idx]["prompt_text"])

    # Show responses side by side
    response_cols = [c for c in df.columns if c.startswith("response_")]
    if response_cols:
        st.markdown("**Responses by Steering Strength:**")
        cols = st.columns(len(response_cols))
        for i, col in enumerate(response_cols):
            strength = col.replace("response_", "")
            with cols[i]:
                st.markdown(f"**Steer {strength}**")
                # Get full response - handle base model separately
                if strength == "base":
                    variant_name = base_model
                else:
                    variant_name = next((v for v in variants if v.endswith(f"-{strength}")), None)
                if variant_name:
                    full_ds = loader.load_dataset_metadata(dataset_name, variant_name)
                    if full_ds and "response_text" in full_ds.column_names:
                        st.text_area(
                            f"Response {strength}",
                            value=full_ds[sample_idx]["response_text"] or "(empty)",
                            height=200,
                            key=f"resp_{dataset_name}_{strength}_{sample_idx}",
                            label_visibility="collapsed",
                        )

    # Show judge labels
    st.markdown("**Judge Labels:**")
    label_cols = st.columns(len(judge_label_cols))
    for i, col in enumerate(judge_label_cols):
        strength = col.replace("judge_label_", "")
        label = sample.get(col)
        with label_cols[i]:
            # Handle numpy bools and None properly
            if pd.notna(label):
                if bool(label):
                    st.error(f"Steer {strength}: 🔴 Deceptive")
                else:
                    st.success(f"Steer {strength}: 🟢 Truthful")
            else:
                st.warning(f"Steer {strength}: ❓ Unknown")
