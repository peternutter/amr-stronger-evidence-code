"""Streamlit app package for modular dashboard components."""

from src.dashboard.components.components import display_dataset_balance, display_error_rates, display_metric_columns
from src.dashboard.components.figures import (
    create_confusion_matrix_figure,
    create_heatmap_figure,
    create_line_chart_figure,
    create_roc_curve_figure,
)
from src.dashboard.components.filters import (
    create_dataset_selector,
    create_layer_selector,
    create_pooling_selector,
    setup_sidebar_filters,
)
from src.dashboard.components.styles import CUSTOM_CSS

__all__ = [
    # Components
    "display_dataset_balance",
    "display_error_rates",
    "display_metric_columns",
    # Figures
    "create_confusion_matrix_figure",
    "create_heatmap_figure",
    "create_line_chart_figure",
    "create_roc_curve_figure",
    # Filters
    "create_dataset_selector",
    "create_layer_selector",
    "create_pooling_selector",
    "setup_sidebar_filters",
    # Styles
    "CUSTOM_CSS",
]
