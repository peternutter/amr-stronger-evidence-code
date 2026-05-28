"""Sidebar filter components for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from src.dashboard.data_loader import DashboardDataLoader


def _get_index_from_session_state(key: str, options: list, default_value) -> int | None:
    """Get the index for a selectbox, preferring session state value if valid.

    Args:
        key: The session state key for this widget
        options: List of available options
        default_value: Default value to use if session state is not set or invalid

    Returns:
        Index into options list, or None if options is empty
    """
    if not options:
        return None

    # Check if we have a previously selected value in session state
    if key in st.session_state:
        prev_value = st.session_state[key]
        if prev_value in options:
            return options.index(prev_value)

    # Fall back to default
    if default_value in options:
        return options.index(default_value)

    # Fall back to first option
    return 0


def setup_sidebar_filters(
    loader: DashboardDataLoader,
    default_train_dataset: str,
    default_eval_dataset: str,
    default_layer: int,
    default_pooling: str,
    default_aggregation: str,
) -> dict[str, any]:
    """Set up sidebar filters and return selected values.

    Args:
        loader: Data loader instance
        default_train_dataset: Default training dataset name
        default_eval_dataset: Default evaluation dataset name
        default_layer: Default layer number
        default_pooling: Default pooling strategy
        default_aggregation: Default aggregation strategy

    Returns:
        Dictionary containing all filter selections
    """
    st.sidebar.header("🎛️ Filters")

    # Training dataset
    train_datasets = loader.get_train_datasets()
    train_idx = _get_index_from_session_state("filter_train_dataset", train_datasets, default_train_dataset)

    train_dataset = st.sidebar.selectbox(
        "Training Dataset",
        options=train_datasets,
        index=train_idx,
        key="filter_train_dataset",
    )

    # Probe type
    probes = loader.get_probes()
    probe_idx = _get_index_from_session_state("filter_probe", probes, probes[0] if probes else None)

    probe = st.sidebar.selectbox(
        "Probe Type",
        options=probes,
        index=probe_idx,
        key="filter_probe",
    )

    # Model
    models = loader.get_models()
    model_idx = _get_index_from_session_state("filter_model", models, models[0] if models else None)

    model = st.sidebar.selectbox(
        "Model",
        options=models,
        index=model_idx,
        key="filter_model",
    )

    # Evaluation dataset
    eval_datasets = loader.get_eval_datasets(train_dataset, probe, model)
    eval_idx = _get_index_from_session_state("filter_eval_dataset", eval_datasets, default_eval_dataset)

    eval_dataset = st.sidebar.selectbox(
        "Evaluation Dataset",
        options=eval_datasets,
        index=eval_idx,
        key="filter_eval_dataset",
    )

    # Layer
    layers = loader.get_layers(train_dataset, probe, model, eval_dataset)
    # For layer, use middle layer as fallback if default not available
    layer_default = default_layer if default_layer in layers else (layers[len(layers) // 2] if layers else None)
    layer_idx = _get_index_from_session_state("filter_layer", layers, layer_default)

    layer = st.sidebar.selectbox(
        "Layer",
        options=layers,
        index=layer_idx,
        key="filter_layer",
        help="Layer for confusion matrix and cross-dataset views",
    )

    # Pooling strategy
    pooling_strategies = loader.get_pooling_strategies(train_dataset, probe, model, eval_dataset)
    pooling_idx = _get_index_from_session_state("filter_pooling", pooling_strategies, default_pooling)

    pooling = st.sidebar.selectbox(
        "Pooling Strategy",
        options=pooling_strategies,
        index=pooling_idx,
        key="filter_pooling",
        help="Token pooling strategy used during probe training",
    )

    # Aggregation strategy
    aggregations = loader.get_aggregations(train_dataset, probe, model, eval_dataset)
    agg_idx = _get_index_from_session_state("filter_aggregation", aggregations, default_aggregation)

    aggregation = st.sidebar.selectbox(
        "Aggregation Strategy",
        options=aggregations,
        index=agg_idx,
        key="filter_aggregation",
        help="How to aggregate per-token predictions during evaluation",
    )

    # Metric
    metric_options = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "specificity",
        "fpr",
        "fnr",
        "mcc",
        "recall_at_1pct_fpr",
    ]
    metric_idx = _get_index_from_session_state("filter_metric", metric_options, "accuracy")

    metric = st.sidebar.selectbox(
        "Metric",
        options=metric_options,
        index=metric_idx,
        key="filter_metric",
        help="FPR=False Positive Rate, FNR=False Negative Rate, MCC=Matthews Correlation Coefficient",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Loaded:** {len(loader.experiments)} experiments, {len(loader.all_metrics)} metric records")

    return {
        "train_dataset": train_dataset,
        "probe": probe,
        "model": model,
        "eval_dataset": eval_dataset,
        "layer": layer,
        "pooling": pooling,
        "aggregation": aggregation,
        "metric": metric,
    }


def create_dataset_selector(label: str, datasets: list[str], default: str | None = None) -> str | None:
    """Create a dataset selector widget.

    Args:
        label: Label for the selector
        datasets: List of available datasets
        default: Default dataset to select

    Returns:
        Selected dataset name or None
    """
    try:
        idx = datasets.index(default) if default else 0
    except ValueError:
        idx = 0 if datasets else None

    return st.sidebar.selectbox(label, options=datasets, index=idx)


def create_layer_selector(layers: list[int], default: int | None = None, help_text: str | None = None) -> int | None:
    """Create a layer selector widget.

    Args:
        layers: List of available layers
        default: Default layer to select
        help_text: Optional help text

    Returns:
        Selected layer number or None
    """
    try:
        idx = layers.index(default) if default else len(layers) // 2
    except ValueError:
        idx = len(layers) // 2 if layers else None

    return st.sidebar.selectbox("Layer", options=layers, index=idx, help=help_text)


def create_pooling_selector(
    pooling_strategies: list[str], default: str | None = None, help_text: str | None = None
) -> str | None:
    """Create a pooling strategy selector widget.

    Args:
        pooling_strategies: List of available pooling strategies
        default: Default pooling strategy to select
        help_text: Optional help text

    Returns:
        Selected pooling strategy or None
    """
    try:
        idx = pooling_strategies.index(default) if default else 0
    except ValueError:
        idx = 0 if pooling_strategies else None

    return st.sidebar.selectbox("Pooling Strategy", options=pooling_strategies, index=idx, help=help_text)


def display_cache_status(loader: DashboardDataLoader, load_data_func) -> None:
    """Display cache status and refresh button in sidebar.

    Args:
        loader: Data loader instance
        load_data_func: The cached load_data function for clearing
    """
    cache_status = loader.get_cache_status()

    if cache_status["cache_stale"]:
        with st.sidebar:
            st.warning(f"⚠️ **Cache is out of date**: {cache_status['stale_reason']}")
            if st.button("🔄 Refresh Cache Now", help="Re-scans all result files and updates the cache"):
                with st.spinner("Recomputing dashboard cache... this may take a few minutes"):
                    # Clear the streamlit cache for load_data
                    load_data_func.clear()
                    # Recompute the data loader's cache
                    loader.refresh_cache()
                    # Force a rerun to pick up new data
                    st.rerun()


def display_no_experiments_warning(data_dir: Path) -> None:
    """Display warning and debug info when no experiments are found.

    Args:
        data_dir: Path to the data directory
    """
    st.warning("⚠️ No experiment results found in the data directory.")

    # Debug information
    with st.expander("🔍 Debug: Directory Contents", expanded=True):
        st.code(f"Searching in: {data_dir}")
        st.markdown("**Expected structure:**")
        st.code(
            """data/
  <train_dataset>/
    probe-<probe_name>-eval/
      <model_name>/
        <eval_dataset>/
          results_layer_<layer>_<pooling>_<source>.pkl""",
            language="text",
        )

        st.markdown("**Looking for:**")
        st.code(
            """- Directories matching: probe-*-eval
- Files matching: results_layer_*.pkl
- File pattern: results_layer_<number>_<pooling>_<aggregation>.pkl""",
            language="text",
        )
