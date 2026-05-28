"""Cross-dataset mixin providing generalization matrix methods for DashboardDataLoader."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


class CrossDatasetMixin:
    """Mixin class providing cross-dataset matrix methods.

    Requires: all_metrics (property), get_confusion_data
    """

    all_metrics: pd.DataFrame

    def get_cross_dataset_matrix(
        self,
        probe: str,
        model: str,
        layer: int,
        pooling: str,
        metric: str,
        aggregation: str = "legacy",
    ) -> tuple[list[str], list[str], list[list[float | None]]]:
        """Build cross-dataset generalization matrix.

        Returns:
            (train_datasets, eval_datasets, matrix) where matrix[i][j] is the
            metric value for train_datasets[i] evaluated on eval_datasets[j].
        """
        df = self.all_metrics
        mask = (
            (df["probe"] == probe)
            & (df["model"] == model)
            & (df["layer"] == layer)
            & (df["pooling"] == pooling)
            & (df["aggregation"] == aggregation)
        )
        filtered = df[mask]

        train_datasets = sorted(filtered["train_dataset"].unique())
        eval_datasets = sorted(filtered["eval_dataset"].unique())

        matrix: list[list[float | None]] = []
        for train_ds in train_datasets:
            row: list[float | None] = []
            for eval_ds in eval_datasets:
                cell = filtered[(filtered["train_dataset"] == train_ds) & (filtered["eval_dataset"] == eval_ds)]
                if cell.empty or metric not in cell.columns:
                    row.append(None)
                else:
                    row.append(float(cell[metric].iloc[0]))
            matrix.append(row)

        return train_datasets, eval_datasets, matrix

    def get_cross_dataset_matrix_all_probes(
        self,
        model: str,
        layer: int,
        pooling: str,
        metric: str,
        aggregation: str = "legacy",
    ) -> tuple[list[str], list[str], list[list[float | None]]]:
        """Build cross-dataset generalization matrix including all probe types.

        Unlike get_cross_dataset_matrix, this method does NOT filter by probe,
        so it shows all training datasets (including Apollo pretrained probes)
        in the same view.

        Returns:
            (train_datasets, eval_datasets, matrix) where matrix[i][j] is the
            metric value for train_datasets[i] evaluated on eval_datasets[j].
        """
        df = self.all_metrics
        mask = (
            (df["model"] == model)
            & (df["layer"] == layer)
            & (df["pooling"] == pooling)
            & (df["aggregation"] == aggregation)
        )
        filtered = df[mask]

        train_datasets = sorted(filtered["train_dataset"].unique())
        eval_datasets = sorted(filtered["eval_dataset"].unique())

        matrix: list[list[float | None]] = []
        for train_ds in train_datasets:
            row: list[float | None] = []
            for eval_ds in eval_datasets:
                cell = filtered[(filtered["train_dataset"] == train_ds) & (filtered["eval_dataset"] == eval_ds)]
                if cell.empty or metric not in cell.columns:
                    row.append(None)
                else:
                    row.append(float(cell[metric].iloc[0]))
            matrix.append(row)

        return train_datasets, eval_datasets, matrix

    def get_cross_dataset_matrix_calibrated(
        self,
        model: str,
        layer: int,
        pooling: str,
        metric: str,
        aggregation: str = "legacy",
    ) -> tuple[list[str], list[str], list[list[float | None]]]:
        """Build cross-dataset matrix with metrics computed at calibrated threshold.

        Each cell uses the calibrated threshold specific to that (train_dataset, eval_dataset) pair.
        This properly respects each probe's calibration instead of using stored metrics.

        Returns:
            (train_datasets, eval_datasets, matrix) where matrix[i][j] is the
            metric value computed at the calibrated threshold for train_datasets[i]
            evaluated on eval_datasets[j].
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        df = self.all_metrics
        mask = (
            (df["model"] == model)
            & (df["layer"] == layer)
            & (df["pooling"] == pooling)
            & (df["aggregation"] == aggregation)
        )
        filtered = df[mask]

        train_datasets = sorted(filtered["train_dataset"].unique())
        eval_datasets = sorted(filtered["eval_dataset"].unique())

        metric_funcs = {
            "recall": recall_score,
            "precision": precision_score,
            "accuracy": accuracy_score,
            "f1": f1_score,
        }
        metric_func = metric_funcs.get(metric)

        if metric_func is None:
            return self.get_cross_dataset_matrix_all_probes(model, layer, pooling, metric, aggregation)

        matrix: list[list[float | None]] = []
        for train_ds in train_datasets:
            row: list[float | None] = []
            for eval_ds in eval_datasets:
                cell = filtered[(filtered["train_dataset"] == train_ds) & (filtered["eval_dataset"] == eval_ds)]
                if cell.empty:
                    row.append(None)
                    continue

                probe = cell["probe"].iloc[0]

                result_data = self.get_confusion_data(train_ds, probe, model, eval_ds, layer, pooling, aggregation)
                if result_data is None:
                    row.append(None)
                    continue

                y_true = np.array(result_data.get("ground_truth", []))
                raw_scores = result_data.get("raw_scores") or result_data.get("logits_honest")
                calibrated_threshold = result_data.get("calibrated_threshold")

                if raw_scores is None or calibrated_threshold is None:
                    if metric in cell.columns:
                        row.append(float(cell[metric].iloc[0]))
                    else:
                        row.append(None)
                    continue

                raw_scores = np.array(raw_scores)
                y_pred = (raw_scores > calibrated_threshold).astype(int)

                try:
                    value = metric_func(y_true, y_pred)
                    row.append(float(value))
                except Exception:
                    row.append(None)

            matrix.append(row)

        return train_datasets, eval_datasets, matrix
