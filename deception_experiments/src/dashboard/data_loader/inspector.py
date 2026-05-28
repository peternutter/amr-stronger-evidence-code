"""Inspector mixin providing data exploration methods for DashboardDataLoader."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset


class InspectorMixin:
    """Mixin class providing data inspection methods.

    Requires: data_dir (Path)
    """

    data_dir: Path

    def get_dataset_names(self) -> list[str]:
        """Return list of dataset directories."""
        return sorted([p.name for p in self.data_dir.iterdir() if p.is_dir()])

    def get_models_for_dataset(self, dataset_name: str) -> list[str]:
        """Return models available for a dataset."""
        models_path = self.data_dir / dataset_name / "responses"
        if not models_path.is_dir():
            return []
        return sorted([p.name for p in models_path.iterdir() if p.is_dir() and (p / "metadata").is_dir()])

    def load_dataset_metadata(self, dataset_name: str, model_name: str) -> Dataset | None:
        """Load HuggingFace dataset metadata."""
        from datasets import Dataset

        metadata_path = self.data_dir / dataset_name / "responses" / model_name / "metadata"
        if not metadata_path.is_dir():
            return None
        return Dataset.load_from_disk(str(metadata_path))
