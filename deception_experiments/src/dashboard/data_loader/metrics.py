"""Metrics mixin providing probe metrics methods for DashboardDataLoader."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np

from src.utils.types import Label

if TYPE_CHECKING:
    import pandas as pd


class MetricsMixin:
    """Mixin class providing metrics retrieval methods.

    Requires: all_metrics (property), get_confusion_data, _get_experiment_path
    """

    all_metrics: pd.DataFrame

    def get_probe_metrics(
        self,
        train_dataset: str,
        probe: str,
        model: str,
        eval_dataset: str,
    ) -> pd.DataFrame:
        """Get metrics DataFrame for probe performance chart."""
        df = self.all_metrics
        mask = (
            (df["train_dataset"] == train_dataset)
            & (df["probe"] == probe)
            & (df["model"] == model)
            & (df["eval_dataset"] == eval_dataset)
        )
        return df[mask].copy()

    def get_probe_metrics_calibrated(
        self,
        train_dataset: str,
        probe: str,
        model: str,
        eval_dataset: str,
    ) -> pd.DataFrame:
        """Get metrics DataFrame with values computed at calibrated threshold.

        For each layer/pooling/aggregation, loads the result file and recomputes
        recall, precision, accuracy, f1 using that layer's calibrated threshold.
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        df = self.get_probe_metrics(train_dataset, probe, model, eval_dataset)
        if df.empty:
            return df

        df = df.copy()

        for idx in df.index:
            row = df.loc[idx]
            layer = int(row["layer"])
            pooling = row["pooling"]
            aggregation = row.get("aggregation", "legacy")

            result_data = self.get_confusion_data(
                train_dataset, probe, model, eval_dataset, layer, pooling, aggregation
            )
            if result_data is None:
                continue

            y_true = np.array(result_data.get("ground_truth", []))
            raw_scores = result_data.get("raw_scores") or result_data.get("logits_honest")
            calibrated_threshold = result_data.get("calibrated_threshold")

            if raw_scores is None or calibrated_threshold is None:
                continue

            raw_scores = np.array(raw_scores)
            y_pred = (raw_scores > calibrated_threshold).astype(int)

            try:
                df.at[idx, "recall"] = recall_score(y_true, y_pred)
                df.at[idx, "precision"] = precision_score(y_true, y_pred, zero_division=0)
                df.at[idx, "accuracy"] = accuracy_score(y_true, y_pred)
                df.at[idx, "f1"] = f1_score(y_true, y_pred, zero_division=0)
            except Exception:
                pass

        return df

    def get_confusion_data(
        self,
        train_dataset: str,
        probe: str,
        model: str,
        eval_dataset: str,
        layer: int,
        pooling: str,
        aggregation: str = "legacy",
    ) -> dict[str, Any] | None:
        """Get confusion matrix data for specific configuration."""
        result_path = self._get_experiment_path(train_dataset, probe, model, eval_dataset)
        if result_path is None:
            return None
        return self._load_result_file(result_path, layer, pooling, aggregation)

    def get_class_balance(
        self,
        train_dataset: str,
        probe: str,
        model: str,
        eval_dataset: str,
    ) -> tuple[int, int] | None:
        """Return (n_deceptive, n_honest) for the given experiment."""
        result_path = self._get_experiment_path(train_dataset, probe, model, eval_dataset)
        if result_path is None:
            return None

        result_files = list(Path(result_path).glob("results_layer_*.pkl"))
        if not result_files:
            return None

        data = joblib.load(result_files[0])
        y_true = np.array(data.get("ground_truth", []))
        if y_true.size == 0:
            return None

        n_honest = int(np.sum(y_true == Label.HONEST))
        n_deceptive = int(np.sum(y_true == Label.DECEPTIVE))
        return (n_deceptive, n_honest)

    def get_roc_data(
        self,
        train_dataset: str,
        probe: str,
        model: str,
        eval_dataset: str,
        layer: int,
        pooling: str,
        aggregation: str = "legacy",
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Get ROC curve data (fpr, tpr, auc) for a specific configuration.

        Returns (fpr_array, tpr_array, roc_auc) or None if not available.
        """
        from sklearn.metrics import roc_auc_score, roc_curve

        result_data = self.get_confusion_data(train_dataset, probe, model, eval_dataset, layer, pooling, aggregation)
        if result_data is None:
            return None

        y_true = np.array(result_data.get("ground_truth", []))
        y_scores = result_data.get("probabilities")

        if y_scores is None:
            y_scores = result_data.get("predictions")

        if y_scores is None or len(y_true) == 0:
            return None

        y_scores = np.array(y_scores)

        if y_scores.ndim == 2:
            y_scores = y_scores[:, 1]

        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc = roc_auc_score(y_true, y_scores)
            return fpr, tpr, auc
        except Exception:
            return None

    def get_best_config(
        self,
        train_dataset: str,
        probe: str,
        model: str,
        eval_dataset: str,
        metric: str = "f1",
    ) -> dict[str, Any] | None:
        """Find the best layer/config for a given metric.

        Returns dict with 'layer', 'pooling', and metric value.
        """
        df = self.get_probe_metrics(train_dataset, probe, model, eval_dataset)
        if df.empty or metric not in df.columns:
            return None

        valid_df = df[df[metric].notna()]
        if valid_df.empty:
            return None

        idx = valid_df[metric].idxmax()
        row = df.loc[idx]
        return {
            "layer": int(row["layer"]),
            "pooling": row["pooling"],
            "aggregation": row["aggregation"],
            "value": float(row[metric]),
        }
