"""Base classes and configuration for export functionality."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.dashboard.data_loader import DashboardDataLoader
from src.plots.metrics_shared import compute_all_classification_metrics


@dataclass
class ExportConfig:
    """Configuration for exports.

    Attributes:
        data_dir: Path to the data directory containing evaluation results.
        output_dir: Directory to save exported files.
        model: Model safe name (e.g., "Llama_Llama-3.3-70B-Instruct").
        train_dataset: Training dataset name (e.g., "instructed_pairs").
        layer: Layer number to use.
        pooling: Pooling strategy (e.g., "flat").
        aggregation: Aggregation strategy (e.g., "mean").
        probe: Probe type (e.g., "logistic_regression").
    """

    data_dir: Path | str
    output_dir: Path | str
    model: str = "Llama_Llama-3.3-70B-Instruct"
    train_dataset: str = "instructed_pairs"
    layer: int = 22
    pooling: str = "flat"
    aggregation: str = "mean"
    probe: str = "logistic_regression"

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class BaseExporter:
    """Base class for exporters with common utilities."""

    def __init__(self, config: ExportConfig):
        self.config = config
        self._loader: DashboardDataLoader | None = None

    @property
    def loader(self) -> DashboardDataLoader:
        """Lazy-load the data loader."""
        if self._loader is None:
            self._loader = DashboardDataLoader(self.config.data_dir)
        return self._loader

    def get_metrics_at_calibrated_threshold(
        self,
        train_dataset: str,
        eval_dataset: str,
    ) -> dict[str, Any] | None:
        """Compute classification metrics using calibrated threshold.

        Args:
            train_dataset: Training dataset name.
            eval_dataset: Evaluation dataset name.

        Returns:
            Dictionary with all classification metrics, or None if data unavailable.
        """
        result_data = self.loader.get_confusion_data(
            train_dataset=train_dataset,
            probe=self.config.probe,
            model=self.config.model,
            eval_dataset=eval_dataset,
            layer=self.config.layer,
            pooling=self.config.pooling,
            aggregation=self.config.aggregation,
        )

        if result_data is None:
            return None

        y_true = np.array(result_data.get("ground_truth", []))
        raw_scores = result_data.get("raw_scores") or result_data.get("logits_honest")
        calibrated_threshold = result_data.get("calibrated_threshold")

        if raw_scores is None or len(y_true) == 0:
            return None

        raw_scores = np.array(raw_scores)

        # Use calibrated threshold if available, otherwise use 0 (default decision boundary)
        threshold = calibrated_threshold if calibrated_threshold is not None else 0.0
        y_pred = (raw_scores > threshold).astype(int)

        metrics = compute_all_classification_metrics(y_true, y_pred)
        metrics["calibrated_threshold"] = threshold
        metrics["n_samples"] = len(y_true)

        return metrics

    def get_metrics_at_default_threshold(
        self,
        train_dataset: str,
        eval_dataset: str,
    ) -> dict[str, Any] | None:
        """Compute classification metrics using default threshold (0).

        Args:
            train_dataset: Training dataset name.
            eval_dataset: Evaluation dataset name.

        Returns:
            Dictionary with all classification metrics, or None if data unavailable.
        """
        result_data = self.loader.get_confusion_data(
            train_dataset=train_dataset,
            probe=self.config.probe,
            model=self.config.model,
            eval_dataset=eval_dataset,
            layer=self.config.layer,
            pooling=self.config.pooling,
            aggregation=self.config.aggregation,
        )

        if result_data is None:
            return None

        y_true = np.array(result_data.get("ground_truth", []))
        raw_scores = result_data.get("raw_scores") or result_data.get("logits_honest")

        if raw_scores is None or len(y_true) == 0:
            return None

        raw_scores = np.array(raw_scores)

        # Use default threshold (0 = decision boundary for logistic regression)
        y_pred = (raw_scores > 0).astype(int)

        metrics = compute_all_classification_metrics(y_true, y_pred)
        metrics["threshold"] = 0.0
        metrics["n_samples"] = len(y_true)

        return metrics

    def get_available_eval_datasets(self) -> list[str]:
        """Get list of evaluation datasets available for the configured probe."""
        df = self.loader.all_metrics
        mask = (
            (df["train_dataset"] == self.config.train_dataset)
            & (df["probe"] == self.config.probe)
            & (df["model"] == self.config.model)
            & (df["layer"] == self.config.layer)
            & (df["pooling"] == self.config.pooling)
            & (df["aggregation"] == self.config.aggregation)
        )
        return sorted(df[mask]["eval_dataset"].unique().tolist())
