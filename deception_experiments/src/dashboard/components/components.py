"""Reusable UI components for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st


def display_dataset_balance(balance: tuple[int, int] | None, title: str, show_label: bool = True) -> None:
    """Display dataset balance metrics in Streamlit columns.

    Args:
        balance: Tuple of (n_deceptive, n_honest) or None
        title: Title to display above the balance info
        show_label: Whether to show the title label
    """
    if show_label:
        st.markdown(f"*{title}:*")

    if balance:
        n_dec, n_hon = balance
        total = n_dec + n_hon
        subcol1, subcol2, subcol3 = st.columns(3)
        with subcol1:
            st.metric("Total", total)
        with subcol2:
            st.metric("Deceptive", f"{n_dec} ({100 * n_dec / total:.1f}%)")
        with subcol3:
            st.metric("Honest", f"{n_hon} ({100 * n_hon / total:.1f}%)")
    else:
        st.info(f"No {title.lower()} balance available")


def display_error_rates(fpr: float, fnr: float) -> None:
    """Display error rates (FPR and FNR) in two columns.

    Args:
        fpr: False Positive Rate
        fnr: False Negative Rate
    """
    st.markdown("**Error Rates:**")
    error_cols = st.columns(2)
    with error_cols[0]:
        st.metric("FPR", f"{fpr:.4f}", help="False Positive Rate = FP/(FP+TN)")
    with error_cols[1]:
        st.metric("FNR", f"{fnr:.4f}", help="False Negative Rate = FN/(FN+TP)")


def display_metric_columns(metrics_dict: dict[str, float], num_columns: int = 3) -> None:
    """Display metrics in evenly-spaced columns.

    Args:
        metrics_dict: Dictionary mapping metric names to values
        num_columns: Number of columns to display
    """
    cols = st.columns(num_columns)
    for idx, (metric_name, value) in enumerate(metrics_dict.items()):
        with cols[idx % num_columns]:
            st.metric(metric_name.upper(), f"{value:.4f}")


def display_per_class_metrics(
    precision_dec: float,
    recall_dec: float,
    precision_hon: float,
    recall_hon: float,
    support_dec: int,
    support_hon: int,
    label_deceptive: int,
    label_honest: int,
) -> None:
    """Display per-class classification metrics in a table.

    Args:
        precision_dec: Precision for deceptive class
        recall_dec: Recall for deceptive class
        precision_hon: Precision for honest class
        recall_hon: Recall for honest class
        support_dec: Number of deceptive samples
        support_hon: Number of honest samples
        label_deceptive: Numeric label for deceptive class
        label_honest: Numeric label for honest class
    """
    import pandas as pd

    st.markdown("**Per-Class Metrics:**")
    class_metrics = pd.DataFrame(
        {
            "Class": [f"Deceptive ({label_deceptive})", f"Honest ({label_honest})"],
            "Precision": [f"{precision_dec:.4f}", f"{precision_hon:.4f}"],
            "Recall": [f"{recall_dec:.4f}", f"{recall_hon:.4f}"],
            "Support": [support_dec, support_hon],
        }
    )
    st.dataframe(class_metrics, hide_index=True, width="stretch")
