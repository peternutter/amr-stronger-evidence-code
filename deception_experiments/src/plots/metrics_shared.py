"""Shared metric calculation utilities for probe evaluation.

This module provides consolidated metric calculation functions used across
both Streamlit dashboards and other evaluation tools.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import matthews_corrcoef

from src.utils.types import Label


def compute_all_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute comprehensive classification metrics from ground truth and predictions.

    Returns accuracy, precision, recall, F1, and extended metrics (FPR, FNR,
    specificity, balanced accuracy, MCC). Assumes binary classification.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary containing all metrics and confusion matrix values
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return {}

    # Confusion matrix values
    tn = int(np.sum((y_true == Label.HONEST) & (y_pred == Label.HONEST)))
    fp = int(np.sum((y_true == Label.HONEST) & (y_pred == Label.DECEPTIVE)))
    fn = int(np.sum((y_true == Label.DECEPTIVE) & (y_pred == Label.HONEST)))
    tp = int(np.sum((y_true == Label.DECEPTIVE) & (y_pred == Label.DECEPTIVE)))

    # Calculate rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Sensitivity/Recall)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate (Specificity)

    # Basic metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr  # Same as TPR
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Balanced accuracy
    balanced_acc = (tpr + tnr) / 2

    # Matthews Correlation Coefficient
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "tpr": tpr,
        "tnr": tnr,
        "specificity": tnr,
        "sensitivity": tpr,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_rate_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    """Compute rate metrics from confusion matrix values.

    Args:
        tp: True positives
        tn: True negatives
        fp: False positives
        fn: False negatives

    Returns:
        Dictionary containing FPR, FNR, TPR, TNR
    """
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "fpr": fpr,
        "fnr": fnr,
        "tpr": tpr,
        "tnr": tnr,
    }
