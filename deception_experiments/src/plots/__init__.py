"""Plotting utilities for deception detection experiments.

This package contains visualization utilities (not the dashboard).
The dashboard has been moved to src.dashboard.
"""

from src.plots.common import format_pooling_label
from src.plots.export import ExportConfig, HeatmapExporter, MetricsTableExporter
from src.plots.metrics_shared import compute_all_classification_metrics, compute_rate_metrics

__all__ = [
    "format_pooling_label",
    "compute_all_classification_metrics",
    "compute_rate_metrics",
    # Export module
    "ExportConfig",
    "HeatmapExporter",
    "MetricsTableExporter",
]
