"""Cross-dataset generalization visualization component for Streamlit dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import streamlit as st

if TYPE_CHECKING:
    from src.dashboard.data_loader import DashboardDataLoader


def render_cross_dataset(
    loader: DashboardDataLoader,
    model: str,
    layer: int,
    pooling: str,
    metric: str,
    aggregation: str,
    use_calibrated_threshold: bool = False,
):
    """Render the cross-dataset generalization heatmap.

    The matrix is organized as:
    - Left side: Square matrix where train datasets match eval datasets (diagonal = in-domain)
    - Right side: Columns for datasets only used for evaluation (never trained on)
    """
    st.subheader("🔄 Cross-Dataset Generalization")

    # When using calibrated threshold, recompute metrics for each cell
    if use_calibrated_threshold and metric in ("recall", "precision", "accuracy", "f1"):
        st.caption(f"Showing **{metric}** at calibrated threshold (each row uses its own threshold)")
        train_datasets, eval_datasets, matrix = loader.get_cross_dataset_matrix_calibrated(
            model, layer, pooling, metric, aggregation
        )
    else:
        st.caption(f"Showing **{metric}** for all training datasets (rows) × evaluation datasets (columns)")
        train_datasets, eval_datasets, matrix = loader.get_cross_dataset_matrix_all_probes(
            model, layer, pooling, metric, aggregation
        )

    if not train_datasets or not eval_datasets:
        st.warning("No cross-dataset data available for this configuration.")
        return

    # Reorder columns: first datasets that are both train and eval (forming square),
    # then datasets that are eval-only (never trained on)
    train_set = set(train_datasets)
    eval_set = set(eval_datasets)

    # Datasets that appear in both train and eval (will form the square part)
    shared_datasets = sorted(train_set & eval_set)
    # Datasets only used for evaluation (will be on the right)
    eval_only_datasets = sorted(eval_set - train_set)

    # New column order: shared first (matching row order), then eval-only
    reordered_eval_datasets = shared_datasets + eval_only_datasets

    # Build a mapping from original eval index to new position
    original_eval_to_idx = {ds: i for i, ds in enumerate(eval_datasets)}

    # Reorder rows to match shared_datasets order
    train_in_shared = [ds for ds in shared_datasets if ds in train_set]
    train_not_in_shared = sorted(train_set - set(shared_datasets))
    reordered_train_datasets = train_in_shared + train_not_in_shared

    original_train_to_idx = {ds: i for i, ds in enumerate(train_datasets)}

    # Rebuild matrix with reordered rows and columns
    reordered_matrix: list[list[float | None]] = []
    for train_ds in reordered_train_datasets:
        train_idx = original_train_to_idx[train_ds]
        row: list[float | None] = []
        for eval_ds in reordered_eval_datasets:
            eval_idx = original_eval_to_idx.get(eval_ds)
            if eval_idx is not None:
                row.append(matrix[train_idx][eval_idx])
            else:
                row.append(None)
        reordered_matrix.append(row)

    # Use reordered data for display
    display_train = reordered_train_datasets
    display_eval = reordered_eval_datasets
    display_values = reordered_matrix

    # Convert matrix for display
    display_matrix = [[("N/A" if v is None else f"{v:.3f}") for v in row] for row in display_values]

    hover_text = [
        [
            f"Train: {display_train[i]}<br>Eval: {display_eval[j]}<br>{metric}: {display_values[i][j]:.3f}"
            if display_values[i][j] is not None
            else "No data"
            for j in range(len(display_eval))
        ]
        for i in range(len(display_train))
    ]

    # Replace None with np.nan for plotting
    z_matrix = [[v if v is not None else np.nan for v in row] for row in display_values]

    # Add visual separator between shared and eval-only columns
    n_shared = len(shared_datasets)
    n_eval_only = len(eval_only_datasets)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=display_eval,
            y=display_train,
            text=display_matrix,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorscale="RdYlGn",
            zmid=0.5,
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar=dict(title=metric.replace("_", " ").title()),
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )

    # Add a vertical line to separate the square part from eval-only columns
    if n_shared > 0 and n_eval_only > 0:
        fig.add_vline(
            x=n_shared - 0.5,
            line=dict(color="white", width=3, dash="solid"),
            annotation_text="← In-Domain | OOD →",
            annotation_position="top",
            annotation_font_size=10,
        )

    fig.update_layout(
        xaxis=dict(title="Evaluation Dataset", tickangle=-45),
        yaxis=dict(title="Training Dataset", autorange="reversed"),
        template="plotly_dark",
        height=max(400, 50 * len(display_train)),
    )

    st.plotly_chart(fig, width="stretch")

    # Summary stats
    flat_values = [v for row in matrix for v in row if v is not None]
    if flat_values:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Mean {metric.title()}", f"{np.mean(flat_values):.4f}")
        with col2:
            st.metric(f"Min {metric.title()}", f"{np.min(flat_values):.4f}")
        with col3:
            st.metric(f"Max {metric.title()}", f"{np.max(flat_values):.4f}")
