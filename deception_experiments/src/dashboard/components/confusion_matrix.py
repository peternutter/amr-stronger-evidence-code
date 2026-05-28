"""Confusion matrix visualization component for Streamlit dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from src.dashboard.components.components import display_error_rates
from src.dashboard.components.figures import create_confusion_matrix_figure, create_roc_curve_figure
from src.plots.metrics_shared import compute_rate_metrics
from src.utils.types import Label

if TYPE_CHECKING:
    from src.dashboard.data_loader import DashboardDataLoader


def render_confusion_matrix(
    loader: DashboardDataLoader,
    train_dataset: str,
    probe: str,
    model: str,
    eval_dataset: str,
    layer: int,
    pooling: str,
    aggregation: str,
    calibrated_threshold: float | None = None,
):
    """Render the confusion matrix heatmap and detailed metrics."""
    st.subheader("🎯 Confusion Matrix & Detailed Metrics")

    result_data = loader.get_confusion_data(train_dataset, probe, model, eval_dataset, layer, pooling, aggregation)

    if result_data is None:
        st.warning("No data available for this configuration.")
        return

    y_true = np.array(result_data.get("ground_truth", []))

    # If calibrated threshold is set, recompute predictions from raw scores
    raw_scores_key = "raw_scores" if "raw_scores" in result_data else "logits_honest"
    if calibrated_threshold is not None and raw_scores_key in result_data:
        raw_scores = np.array(result_data.get(raw_scores_key, []))
        y_pred = (raw_scores > calibrated_threshold).astype(int)
    else:
        y_pred = np.array(result_data.get("predictions", []))

    if len(y_true) == 0:
        st.warning("No predictions available.")
        return

    cm = confusion_matrix(y_true, y_pred, labels=[Label.HONEST, Label.DECEPTIVE])
    total = len(y_true)
    labels = [f"Honest ({Label.HONEST})", f"Deceptive ({Label.DECEPTIVE})"]

    # Extract confusion matrix values
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    # Create confusion matrix figure
    fig = create_confusion_matrix_figure(cm.tolist(), labels)

    # Show threshold info if using calibrated threshold
    if calibrated_threshold is not None:
        st.caption(f"📏 Using calibrated threshold: {calibrated_threshold:.4f}")

    col1, col2 = st.columns([1, 1])
    with col1:
        chart_key = f"cm_{train_dataset}_{probe}_{layer}_{calibrated_threshold}"
        st.plotly_chart(fig, key=chart_key, width="stretch")

    with col2:
        # Calculate all metrics using shared function
        rate_metrics = compute_rate_metrics(tp, tn, fp, fn)
        fpr = rate_metrics["fpr"]
        fnr = rate_metrics["fnr"]
        tpr = rate_metrics["tpr"]
        tnr = rate_metrics["tnr"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        balanced_acc = (tpr + tnr) / 2

        # Per-class metrics
        st.markdown("**Per-Class Metrics:**")
        class_metrics = pd.DataFrame(
            {
                "Class": [f"Deceptive ({Label.DECEPTIVE})", f"Honest ({Label.HONEST})"],
                "Precision": [f"{tnr:.4f}", f"{precision:.4f}"],
                "Recall": [f"{tnr:.4f}", f"{tpr:.4f}"],
                "Support": [tn + fp, tp + fn],
            }
        )
        st.dataframe(class_metrics, hide_index=True, width="stretch")

        # Error rates
        display_error_rates(fpr, fnr)

    # ROC Curve
    st.markdown("---")
    roc_data = loader.get_roc_data(train_dataset, probe, model, eval_dataset, layer, pooling, aggregation)

    col3, col4 = st.columns([1, 1])
    with col3:
        if roc_data is not None:
            fpr_curve, tpr_curve, auc_val = roc_data
            fig_roc = create_roc_curve_figure(fpr_curve, tpr_curve, auc_val)
            st.plotly_chart(fig_roc, width="stretch")
        else:
            st.info("ROC curve requires probability scores (not available for this config).")

    with col4:
        # Summary metrics table
        st.markdown("**All Metrics:**")
        metrics = result_data.get("metrics", {})

        # Compute F1 from current confusion matrix to match threshold
        f1_computed = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        all_metrics = {
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_acc,
            "Precision": precision,
            "Recall (Sensitivity)": tpr,
            "Specificity": tnr,
            "F1 Score": f1_computed,
            "ROC AUC": metrics.get("roc_auc", 0.0),
            "NPV": npv,
            "FPR": fpr,
            "FNR": fnr,
        }
        metric_df = pd.DataFrame([{"Metric": k, "Value": f"{v:.4f}"} for k, v in all_metrics.items()])
        st.dataframe(metric_df, hide_index=True, width="stretch")
