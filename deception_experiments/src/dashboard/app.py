"""Streamlit dashboard for probe evaluation visualization."""

import sys
from pathlib import Path

# Add project root to path for imports (must be before other imports)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os  # noqa: E402

import streamlit as st  # noqa: E402

from src.dashboard.components.components import display_dataset_balance  # noqa: E402
from src.dashboard.components.confusion_matrix import render_confusion_matrix  # noqa: E402
from src.dashboard.components.cross_dataset import render_cross_dataset  # noqa: E402
from src.dashboard.components.data_inspector import render_data_inspector  # noqa: E402
from src.dashboard.components.filters import display_cache_status, display_no_experiments_warning  # noqa: E402
from src.dashboard.components.judge_comparison import render_judge_comparison  # noqa: E402

# Import extracted render functions
from src.dashboard.components.probe_performance import render_probe_performance  # noqa: E402
from src.dashboard.components.steering_comparison import render_steering_comparison  # noqa: E402
from src.dashboard.components.styles import CUSTOM_CSS  # noqa: E402
from src.dashboard.components.token_viewer_section import render_token_viewer  # noqa: E402
from src.dashboard.data_loader import DashboardDataLoader  # noqa: E402

# Default Configuration
DEFAULT_TRAIN_DATASET = "instructed_pairs"
DEFAULT_EVAL_DATASET = "roleplaying-offpolicy_train"
DEFAULT_LAYER = 22
DEFAULT_POOLING = "flat"
DEFAULT_AGGREGATION = "mean"
DEFAULT_MODEL_PATTERN = "Llama-3.3-70B"
DEFAULT_PROBE_PATTERN = "logistic_regression"


def find_default_model_index(models: list[str]) -> int:
    """Find the index of the default model (Llama 3.3 70B) in the list."""
    for i, model in enumerate(models):
        if DEFAULT_MODEL_PATTERN in model:
            return i
    for i, model in enumerate(models):
        if "70B" in model or "70b" in model:
            return i
    return 0


def find_default_index(options: list[str], key: str, default_value: str) -> int:
    """Find the index of a default value, using session_state if available."""
    if key in st.session_state and st.session_state[key] in options:
        return options.index(st.session_state[key])
    try:
        return options.index(default_value)
    except ValueError:
        return 0 if options else None


def get_default_data_dir() -> Path:
    """Get the default data directory based on environment variable."""
    cluster_work_dir = os.environ.get("CLUSTER_WORK_DIR")
    if cluster_work_dir:
        return Path(cluster_work_dir) / "data"
    else:
        return Path(__file__).parent.parent / "data"


# Page config
st.set_page_config(
    page_title="Probe Evaluation Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_data(data_dir: str, allow_stale: bool = True) -> DashboardDataLoader:
    """Load and cache the data loader with eager initialization."""
    loader = DashboardDataLoader(data_dir, allow_stale=allow_stale)
    _ = loader.experiments
    _ = loader.all_metrics
    return loader


def main():
    st.title("🔬 Probe Evaluation Dashboard")

    # Data directory selection (always in sidebar)
    default_data_dir = get_default_data_dir()
    data_dir = st.sidebar.text_input(
        "Data Directory",
        value=str(default_data_dir),
        help="Path to the data directory containing experiment results",
    )

    if not Path(data_dir).exists():
        st.error(f"Data directory not found: {data_dir}")
        return

    # Load data
    with st.spinner("Loading experiment data..."):
        loader = load_data(data_dir)

    display_cache_status(loader, load_data)

    has_experiments = not loader.experiments.empty
    dataset_names = loader.get_dataset_names()
    has_datasets = len(dataset_names) > 0

    if not has_experiments and not has_datasets:
        display_no_experiments_warning(Path(data_dir))
        st.info("💡 No datasets found either. Check the directory path and structure.")
        return

    # Build tab names dynamically based on available data
    tab_names = []
    if has_experiments:
        tab_names.append("📊 Probe Performance")
    if has_datasets:
        tab_names.append("🔍 Data Inspector")
        tab_names.append("⚖️ Judge Comparison")
        tab_names.append("🔀 Steering Comparison")

    if not tab_names:
        st.warning("No data available to display.")
        return

    # Create tabs for view switching (faster than radio - just hides/shows content)
    tabs = st.tabs(tab_names)

    tab_idx = 0

    # =========================================================================
    # View 1: Probe Performance
    # =========================================================================
    if has_experiments:
        with tabs[tab_idx]:
            # Sidebar filters for probe performance
            st.sidebar.header("🎛️ Filters")
            experiments_df = loader.experiments

            # Get unique datasets
            dataset_options = sorted(experiments_df["train_dataset"].unique().tolist())

            train_dataset = st.sidebar.selectbox(
                "Training Dataset",
                options=dataset_options,
                index=find_default_index(dataset_options, "sidebar_dataset", DEFAULT_TRAIN_DATASET),
                key="sidebar_dataset",
                help="Dataset used to train the probe.",
            )

            # Get probe types for selected dataset
            probe_options = sorted(
                experiments_df[experiments_df["train_dataset"] == train_dataset]["probe"].unique().tolist()
            )

            def get_default_probe_idx(probes: list[str]) -> int:
                if "sidebar_probe" in st.session_state and st.session_state["sidebar_probe"] in probes:
                    return probes.index(st.session_state["sidebar_probe"])
                for i, p in enumerate(probes):
                    if DEFAULT_PROBE_PATTERN in p:
                        return i
                return 0 if probes else None

            probe = st.sidebar.selectbox(
                "Probe Type",
                options=probe_options,
                index=get_default_probe_idx(probe_options),
                key="sidebar_probe",
                help="Probe architecture. Apollo probes are pretrained detectors from ApolloResearch.",
            )

            models = loader.get_models(train_dataset, probe)
            if "sidebar_model" in st.session_state and st.session_state["sidebar_model"] in models:
                model_idx = models.index(st.session_state["sidebar_model"])
            else:
                model_idx = find_default_model_index(models) if models else None

            model = st.sidebar.selectbox(
                "Model",
                options=models,
                index=model_idx,
                key="sidebar_model",
            )

            eval_datasets = loader.get_eval_datasets(train_dataset, probe, model)
            eval_dataset = st.sidebar.selectbox(
                "Evaluation Dataset",
                options=eval_datasets,
                index=find_default_index(eval_datasets, "sidebar_eval_dataset", DEFAULT_EVAL_DATASET),
                key="sidebar_eval_dataset",
            )

            # Layer and config filters
            layers = loader.get_layers(train_dataset, probe, model, eval_dataset)
            if "sidebar_layer" in st.session_state and st.session_state["sidebar_layer"] in layers:
                layer_idx = layers.index(st.session_state["sidebar_layer"])
            else:
                try:
                    layer_idx = layers.index(DEFAULT_LAYER)
                except ValueError:
                    layer_idx = len(layers) // 2 if layers else None

            layer = st.sidebar.selectbox(
                "Layer",
                options=layers,
                index=layer_idx,
                key="sidebar_layer",
                help="Layer for confusion matrix and cross-dataset views",
            )

            pooling_strategies = loader.get_pooling_strategies(train_dataset, probe, model, eval_dataset)
            pooling = st.sidebar.selectbox(
                "Pooling Strategy",
                options=pooling_strategies,
                index=find_default_index(pooling_strategies, "sidebar_pooling", DEFAULT_POOLING),
                key="sidebar_pooling",
                help="Token pooling strategy used during probe training",
            )

            aggregations = loader.get_aggregations(train_dataset, probe, model, eval_dataset)
            aggregation = st.sidebar.selectbox(
                "Aggregation Strategy",
                options=aggregations,
                index=find_default_index(aggregations, "sidebar_aggregation", DEFAULT_AGGREGATION),
                key="sidebar_aggregation",
                help="How to aggregate per-token predictions during evaluation",
            )

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
            metric = st.sidebar.selectbox(
                "Metric",
                options=metric_options,
                index=find_default_index(metric_options, "sidebar_metric", "accuracy"),
                key="sidebar_metric",
                help="FPR=False Positive Rate, FNR=False Negative Rate, MCC=Matthews Correlation Coefficient",
            )

            st.sidebar.markdown("---")

            # Calibrated threshold toggle
            use_calibrated_threshold = st.sidebar.toggle(
                "📏 Use Calibrated Threshold",
                value=False,
                key="sidebar_calibrated",
                help="Use 99th percentile threshold from control data (Alpaca) instead of 0.",
            )

            calibrated_threshold = None
            if use_calibrated_threshold and layer is not None and pooling and aggregation:
                from src.data.alpaca import load_or_compute_threshold

                calibrated_threshold = load_or_compute_threshold(
                    data_dir=Path(data_dir),
                    train_dataset=train_dataset,
                    probe_type=probe,
                    model_safe_name=model,
                    layer=layer,
                    pooling=pooling,
                    aggregation=aggregation,
                    eval_dataset=eval_dataset,
                )
                if calibrated_threshold is not None:
                    st.sidebar.caption(f"Threshold: {calibrated_threshold:.3f}")
                else:
                    st.sidebar.warning("Could not load threshold")
                    use_calibrated_threshold = False

            st.sidebar.markdown("---")
            st.sidebar.markdown(
                f"**Loaded:** {len(loader.experiments)} experiments, {len(loader.all_metrics)} metric records"
            )

            # Main content area
            if not all([train_dataset, probe, model, eval_dataset]):
                st.info("Please select all filter options to view probe evaluation results.")
            else:
                # Class balance info
                train_balance = loader.get_class_balance(train_dataset, probe, model, train_dataset)
                eval_balance = loader.get_class_balance(train_dataset, probe, model, eval_dataset)

                st.markdown("**Dataset Balance:**")
                col1, col2 = st.columns(2)
                with col1:
                    display_dataset_balance(train_balance, "Training Set")
                with col2:
                    display_dataset_balance(eval_balance, "Evaluation Set")

                # 1. Probe Performance Chart
                st.markdown("---")
                render_probe_performance(
                    loader,
                    train_dataset,
                    probe,
                    model,
                    eval_dataset,
                    metric,
                    use_calibrated_threshold=use_calibrated_threshold,
                )

                # 2. Confusion Matrix
                st.markdown("---")
                if layer is not None and pooling and aggregation:
                    render_confusion_matrix(
                        loader,
                        train_dataset,
                        probe,
                        model,
                        eval_dataset,
                        layer,
                        pooling,
                        aggregation,
                        calibrated_threshold=calibrated_threshold,
                    )
                else:
                    st.warning("Select layer and config to view confusion matrix.")

                # 3. Cross-Dataset Generalization
                st.markdown("---")
                if layer is not None and pooling and aggregation:
                    render_cross_dataset(
                        loader,
                        model,
                        layer,
                        pooling,
                        metric,
                        aggregation,
                        use_calibrated_threshold=use_calibrated_threshold,
                    )
                else:
                    st.warning("Select layer and config to view cross-dataset generalization.")

                # 4. Token Score Viewer
                st.markdown("---")
                if layer is not None and pooling and aggregation:
                    render_token_viewer(
                        loader,
                        train_dataset,
                        probe,
                        model,
                        eval_dataset,
                        layer,
                        pooling,
                        aggregation,
                        calibrated_threshold=calibrated_threshold,
                    )

        tab_idx += 1

    # =========================================================================
    # View 2: Data Inspector
    # =========================================================================
    if has_datasets:
        with tabs[tab_idx]:
            render_data_inspector(loader, sidebar_mode=False)
        tab_idx += 1

    # =========================================================================
    # View 3: Judge Comparison
    # =========================================================================
    if has_datasets:
        with tabs[tab_idx]:
            render_judge_comparison(loader, sidebar_mode=False)
        tab_idx += 1

    # =========================================================================
    # View 4: Steering Comparison
    # =========================================================================
    if has_datasets:
        with tabs[tab_idx]:
            render_steering_comparison(loader, sidebar_mode=False)


if __name__ == "__main__":
    main()
