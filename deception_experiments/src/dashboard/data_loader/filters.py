"""Filters mixin providing filter option methods for DashboardDataLoader."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class FiltersMixin:
    """Mixin class providing filter option methods.

    These methods extract unique values for dropdown selectors.
    Requires: experiments (property), all_metrics (property)
    """

    experiments: pd.DataFrame
    all_metrics: pd.DataFrame

    def get_train_datasets(self) -> list[str]:
        """Return sorted list of unique training datasets."""
        return sorted(self.experiments["train_dataset"].unique())

    def get_probes(self) -> list[str]:
        """Return sorted list of unique probe types."""
        return sorted(self.experiments["probe"].unique())

    def get_models(
        self,
        train_dataset: str | None = None,
        probe: str | None = None,
    ) -> list[str]:
        """Return sorted list of unique models, optionally filtered by train_dataset and probe."""
        df = self.experiments
        if train_dataset:
            df = df[df["train_dataset"] == train_dataset]
        if probe:
            df = df[df["probe"] == probe]
        return sorted(df["model"].unique())

    def get_eval_datasets(
        self,
        train_dataset: str | None = None,
        probe: str | None = None,
        model: str | None = None,
    ) -> list[str]:
        """Return eval datasets available for the given filters."""
        df = self.experiments
        if train_dataset:
            df = df[df["train_dataset"] == train_dataset]
        if probe:
            df = df[df["probe"] == probe]
        if model:
            df = df[df["model"] == model]
        return sorted(df["eval_dataset"].unique())

    def get_layers(
        self,
        train_dataset: str | None = None,
        probe: str | None = None,
        model: str | None = None,
        eval_dataset: str | None = None,
    ) -> list[int]:
        """Return layers available for the given filters."""
        df = self.all_metrics
        if train_dataset:
            df = df[df["train_dataset"] == train_dataset]
        if probe:
            df = df[df["probe"] == probe]
        if model:
            df = df[df["model"] == model]
        if eval_dataset:
            df = df[df["eval_dataset"] == eval_dataset]
        if df.empty:
            return []
        return sorted(df["layer"].unique())

    def get_aggregations(
        self,
        train_dataset: str | None = None,
        probe: str | None = None,
        model: str | None = None,
        eval_dataset: str | None = None,
    ) -> list[str]:
        """Return aggregation strategies available for the given filters."""
        df = self.all_metrics
        if train_dataset:
            df = df[df["train_dataset"] == train_dataset]
        if probe:
            df = df[df["probe"] == probe]
        if model:
            df = df[df["model"] == model]
        if eval_dataset:
            df = df[df["eval_dataset"] == eval_dataset]
        if df.empty:
            return []
        return sorted(df["aggregation"].unique())

    def get_pooling_strategies(
        self,
        train_dataset: str | None = None,
        probe: str | None = None,
        model: str | None = None,
        eval_dataset: str | None = None,
    ) -> list[str]:
        """Return pooling strategies available for the given filters."""
        df = self.all_metrics
        if train_dataset:
            df = df[df["train_dataset"] == train_dataset]
        if probe:
            df = df[df["probe"] == probe]
        if model:
            df = df[df["model"] == model]
        if eval_dataset:
            df = df[df["eval_dataset"] == eval_dataset]
        if df.empty:
            return []
        return sorted(df["pooling"].unique())

    def get_configs(
        self,
        train_dataset: str | None = None,
        probe: str | None = None,
        model: str | None = None,
        eval_dataset: str | None = None,
    ) -> list[tuple[str, str]]:
        """Return (pooling, aggregation) configs available for the given filters."""
        df = self.all_metrics
        if train_dataset:
            df = df[df["train_dataset"] == train_dataset]
        if probe:
            df = df[df["probe"] == probe]
        if model:
            df = df[df["model"] == model]
        if eval_dataset:
            df = df[df["eval_dataset"] == eval_dataset]
        if df.empty:
            return []
        combos = df[["pooling", "aggregation"]].drop_duplicates().sort_values(["pooling", "aggregation"])
        return [(row.pooling, row.aggregation) for row in combos.itertuples(index=False)]
