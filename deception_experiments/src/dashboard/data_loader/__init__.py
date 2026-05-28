"""Data loader package providing modular DashboardDataLoader.

This package contains mixin classes that compose to form the full DashboardDataLoader.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.dashboard.data_loader.cache import CacheMixin
from src.dashboard.data_loader.cross_dataset import CrossDatasetMixin
from src.dashboard.data_loader.filters import FiltersMixin
from src.dashboard.data_loader.inspector import InspectorMixin
from src.dashboard.data_loader.judge import JudgeMixin
from src.dashboard.data_loader.metrics import MetricsMixin


class DashboardDataLoader(
    FiltersMixin,
    MetricsMixin,
    CrossDatasetMixin,
    InspectorMixin,
    JudgeMixin,
    CacheMixin,
):
    """Loads and caches all experiment data for fast dashboard rendering.

    This class scans the data directory once and builds aggregated DataFrames
    that can be quickly filtered for different visualizations.

    Composed from mixins:
    - FiltersMixin: Filter option methods (get_train_datasets, get_probes, etc.)
    - MetricsMixin: Metrics retrieval (get_probe_metrics, get_confusion_data, etc.)
    - CrossDatasetMixin: Cross-dataset matrix methods
    - InspectorMixin: Data inspection utilities
    - JudgeMixin: Multi-judge analysis methods
    - CacheMixin: Caching and loading infrastructure
    """

    def __init__(self, data_dir: Path | str, allow_stale: bool = True):
        """Initialize the data loader.

        Args:
            data_dir: Path to the data directory containing experiment results.
            allow_stale: If True, use cached data even if it may be stale.
        """
        self.data_dir = Path(data_dir)
        self.allow_stale = allow_stale

        # Internal state
        self._experiments_df: pd.DataFrame | None = None
        self._all_metrics_df: pd.DataFrame | None = None
        self._result_cache: dict[str, Any] = {}
        self._cache_was_stale = False
        self._stale_reason: str | None = None
        self._cached_fingerprint: str | None = None

    @property
    def experiments(self) -> pd.DataFrame:
        """DataFrame of all discovered experiments."""
        if self._experiments_df is None:
            self._experiments_df = self._load_experiments()
        return self._experiments_df

    @property
    def all_metrics(self) -> pd.DataFrame:
        """Aggregated DataFrame of all metrics across all experiments/layers."""
        if self._all_metrics_df is None:
            self._all_metrics_df = self._load_all_metrics()
        return self._all_metrics_df

    @property
    def cache_is_stale(self) -> bool:
        """Check if cached data was stale when loaded."""
        return self._cache_was_stale

    @property
    def stale_reason(self) -> str | None:
        """Reason why cache was stale, if applicable."""
        return self._stale_reason


@lru_cache(maxsize=1)
def get_cached_loader(data_dir: str) -> DashboardDataLoader:
    """Return a cached DashboardDataLoader instance.

    This function is designed to be used with Streamlit's caching.
    """
    return DashboardDataLoader(data_dir)


__all__ = [
    "DashboardDataLoader",
    "get_cached_loader",
    "FiltersMixin",
    "MetricsMixin",
    "CrossDatasetMixin",
    "InspectorMixin",
    "JudgeMixin",
    "CacheMixin",
]
