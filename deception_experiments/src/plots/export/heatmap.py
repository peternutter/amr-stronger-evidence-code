"""Heatmap exporter for cross-dataset generalization visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from src.plots.export.base import BaseExporter, ExportConfig


class HeatmapExporter(BaseExporter):
    """Generates publication-ready heatmaps for cross-dataset evaluation.

    Creates matplotlib heatmaps showing metric values across training and
    evaluation dataset combinations, suitable for LaTeX documents.
    """

    # Color schemes for different metrics
    COLORMAPS = {
        "accuracy": "RdYlGn",
        "f1": "RdYlGn",
        "recall": "RdYlGn",
        "precision": "RdYlGn",
    }

    def __init__(self, config: ExportConfig):
        super().__init__(config)
        # Styling configuration
        self.figure_dpi = 300
        self.font_family = "serif"
        self.font_size = 10
        self.annotation_size = 9

    def export_heatmap(
        self,
        metric: Literal["accuracy", "f1", "recall", "precision"] = "accuracy",
        use_calibrated: bool = True,
        filename: str | None = None,
        title: str | None = None,
        eval_datasets: list[str] | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Path:
        """Export a single heatmap for the specified metric.

        Args:
            metric: Metric to visualize.
            use_calibrated: If True, use calibrated threshold; otherwise use default.
            filename: Output filename (without extension). Defaults to metric name.
            title: Custom title for the heatmap.
            eval_datasets: Specific eval datasets to include. If None, uses all available.
            figsize: Figure size as (width, height) in inches.

        Returns:
            Path to the exported PDF file.
        """
        if filename is None:
            threshold_suffix = "calibrated" if use_calibrated else "default"
            filename = f"heatmap_{metric}_{threshold_suffix}"

        if title is None:
            metric_label = metric.replace("_", " ").title()
            threshold_label = "Calibrated" if use_calibrated else "Default"
            title = f"{metric_label} ({threshold_label} Threshold)"

        # Get evaluation datasets
        if eval_datasets is None:
            eval_datasets = self.get_available_eval_datasets()

        if not eval_datasets:
            raise ValueError("No evaluation datasets available for the configured probe.")

        # Build matrix: single row (one train dataset) × multiple eval datasets
        train_datasets = [self.config.train_dataset]
        matrix, row_labels, col_labels = self._build_matrix(
            train_datasets=train_datasets,
            eval_datasets=eval_datasets,
            metric=metric,
            use_calibrated=use_calibrated,
        )

        # Create and save figure
        output_path = self.config.output_dir / f"{filename}.pdf"
        self._create_heatmap_figure(
            matrix=matrix,
            row_labels=row_labels,
            col_labels=col_labels,
            metric=metric,
            title=title,
            output_path=output_path,
            figsize=figsize,
        )

        return output_path

    def export_cross_dataset_heatmap(
        self,
        metric: Literal["accuracy", "f1", "recall", "precision"] = "accuracy",
        use_calibrated: bool = True,
        filename: str | None = None,
        title: str | None = None,
        train_datasets: list[str] | None = None,
        eval_datasets: list[str] | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Path:
        """Export a cross-dataset heatmap with multiple training datasets.

        Args:
            metric: Metric to visualize.
            use_calibrated: If True, use calibrated threshold; otherwise use default.
            filename: Output filename (without extension).
            title: Custom title for the heatmap.
            train_datasets: Training datasets for rows. If None, gets all available.
            eval_datasets: Evaluation datasets for columns. If None, gets all available.
            figsize: Figure size as (width, height) in inches.

        Returns:
            Path to the exported PDF file.
        """
        if filename is None:
            threshold_suffix = "calibrated" if use_calibrated else "default"
            filename = f"heatmap_cross_{metric}_{threshold_suffix}"

        if title is None:
            metric_label = metric.replace("_", " ").title()
            threshold_label = "Calibrated" if use_calibrated else "Default"
            title = f"Cross-Dataset {metric_label} ({threshold_label} Threshold)"

        # Get datasets
        if train_datasets is None:
            train_datasets = self._get_available_train_datasets()
        if eval_datasets is None:
            eval_datasets = self._get_all_eval_datasets(train_datasets)

        matrix, row_labels, col_labels = self._build_matrix(
            train_datasets=train_datasets,
            eval_datasets=eval_datasets,
            metric=metric,
            use_calibrated=use_calibrated,
        )

        output_path = self.config.output_dir / f"{filename}.pdf"
        self._create_heatmap_figure(
            matrix=matrix,
            row_labels=row_labels,
            col_labels=col_labels,
            metric=metric,
            title=title,
            output_path=output_path,
            figsize=figsize,
        )

        return output_path

    def _build_matrix(
        self,
        train_datasets: list[str],
        eval_datasets: list[str],
        metric: str,
        use_calibrated: bool,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Build the metric matrix for the heatmap.

        Returns:
            Tuple of (matrix, row_labels, col_labels).
        """
        matrix = np.full((len(train_datasets), len(eval_datasets)), np.nan)

        for i, train_ds in enumerate(train_datasets):
            for j, eval_ds in enumerate(eval_datasets):
                # Temporarily update config for different train datasets
                original_train = self.config.train_dataset
                self.config.train_dataset = train_ds

                if use_calibrated:
                    metrics = self.get_metrics_at_calibrated_threshold(train_ds, eval_ds)
                else:
                    metrics = self.get_metrics_at_default_threshold(train_ds, eval_ds)

                self.config.train_dataset = original_train

                if metrics is not None and metric in metrics:
                    matrix[i, j] = metrics[metric]

        # Format labels for display
        row_labels = [self._format_dataset_label(ds) for ds in train_datasets]
        col_labels = [self._format_dataset_label(ds) for ds in eval_datasets]

        return matrix, row_labels, col_labels

    def _create_heatmap_figure(
        self,
        matrix: np.ndarray,
        row_labels: list[str],
        col_labels: list[str],
        metric: str,
        title: str,
        output_path: Path,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Create and save the heatmap figure."""
        # Configure matplotlib for publication quality
        plt.rcParams.update(
            {
                "font.family": self.font_family,
                "font.size": self.font_size,
                "axes.titlesize": self.font_size + 2,
                "axes.labelsize": self.font_size,
                "xtick.labelsize": self.font_size - 1,
                "ytick.labelsize": self.font_size - 1,
            }
        )

        # Auto-calculate figure size if not provided
        if figsize is None:
            width = max(6, len(col_labels) * 0.8 + 2)
            height = max(4, len(row_labels) * 0.6 + 2)
            figsize = (width, height)

        fig, ax = plt.subplots(figsize=figsize)

        # Create colormap
        cmap = plt.get_cmap(self.COLORMAPS.get(metric, "RdYlGn"))
        cmap.set_bad(color="lightgray")  # Color for NaN values

        # Create heatmap
        im = ax.imshow(
            matrix,
            cmap=cmap,
            aspect="auto",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.set_ylabel(metric.replace("_", " ").title(), rotation=270, labelpad=15)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(row_labels)

        # Add cell annotations
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                value = matrix[i, j]
                if np.isnan(value):
                    text = "—"
                    color = "gray"
                else:
                    text = f"{value:.2f}"
                    # Choose text color based on background brightness
                    color = "white" if value < 0.5 else "black"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=self.annotation_size,
                    fontweight="medium",
                )

        # Labels and title
        ax.set_xlabel("Evaluation Dataset")
        ax.set_ylabel("Training Dataset")
        ax.set_title(title, pad=15)

        # Add grid lines
        ax.set_xticks(np.arange(len(col_labels) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Tight layout and save
        plt.tight_layout()
        fig.savefig(output_path, dpi=self.figure_dpi, bbox_inches="tight", format="pdf")
        plt.close(fig)

        print(f"✓ Exported heatmap: {output_path}")

    def _format_dataset_label(self, dataset: str) -> str:
        """Format dataset name for display in plots."""
        # Handle nested dataset names like "roleplaying-plain"
        parts = dataset.replace("_", "-").split("-")

        # Capitalize and join
        formatted = " ".join(p.capitalize() for p in parts)

        # Abbreviate long names
        abbreviations = {
            "Instructed Pairs": "Instr. Pairs",
            "Offpolicy Train": "RP Off-Policy",
            "Roleplaying Plain": "RP Plain",
            "Roleplaying Actor": "RP Actor",
            "Roleplaying Recital": "RP Recital",
            "Instructed Alien": "Instr. Alien",
            "Instructed Peasant": "Instr. Peasant",
            "Instructed Sarcasm": "Instr. Sarcasm",
            "Instructed Counterfactual": "Instr. Counter.",
            "Instructed Wrong Answers": "Instr. Wrong Ans.",
        }

        return abbreviations.get(formatted, formatted)

    def _get_available_train_datasets(self) -> list[str]:
        """Get list of training datasets that have probes."""
        df = self.loader.all_metrics
        mask = (
            (df["probe"] == self.config.probe)
            & (df["model"] == self.config.model)
            & (df["layer"] == self.config.layer)
            & (df["pooling"] == self.config.pooling)
            & (df["aggregation"] == self.config.aggregation)
        )
        return sorted(df[mask]["train_dataset"].unique().tolist())

    def _get_all_eval_datasets(self, train_datasets: list[str]) -> list[str]:
        """Get union of all eval datasets across train datasets."""
        all_eval = set()
        df = self.loader.all_metrics
        for train_ds in train_datasets:
            mask = (
                (df["train_dataset"] == train_ds)
                & (df["probe"] == self.config.probe)
                & (df["model"] == self.config.model)
                & (df["layer"] == self.config.layer)
                & (df["pooling"] == self.config.pooling)
                & (df["aggregation"] == self.config.aggregation)
            )
            all_eval.update(df[mask]["eval_dataset"].unique().tolist())
        return sorted(all_eval)
