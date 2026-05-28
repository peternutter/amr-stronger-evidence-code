"""Judge mixin providing multi-judge analysis methods for DashboardDataLoader."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass


class JudgeMixin:
    """Mixin class providing multi-judge analysis methods.

    Requires: data_dir (Path), get_dataset_names, get_models_for_dataset, load_dataset_metadata
    """

    data_dir: Path

    def get_datasets_with_multi_judge(self) -> list[tuple[str, str, Path]]:
        """Find all datasets that have multi-judge result columns.

        Returns:
            List of (dataset_name, model_name, metadata_path) tuples.
        """
        from datasets import Dataset

        results = []
        for dataset_name in self.get_dataset_names():
            models = self.get_models_for_dataset(dataset_name)
            for model_name in models:
                metadata_path = self.data_dir / dataset_name / "responses" / model_name / "metadata"
                if not metadata_path.is_dir():
                    continue
                try:
                    ds = Dataset.load_from_disk(str(metadata_path))
                    if any(c.startswith("judge_results_") for c in ds.column_names):
                        results.append((dataset_name, model_name, metadata_path))
                except Exception:
                    continue
        return results

    def get_judge_names_for_dataset(self, dataset_name: str, model_name: str) -> list[str]:
        """Return list of judge names from judge_results_* columns.

        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model

        Returns:
            List of judge names (extracted from column names).
        """
        ds = self.load_dataset_metadata(dataset_name, model_name)
        if ds is None:
            return []

        judges = []
        for col in ds.column_names:
            if col.startswith("judge_results_"):
                judge_name = col.replace("judge_results_", "")
                judges.append(judge_name)
        return sorted(judges)

    def get_judge_labels_df(self, dataset_name: str, model_name: str) -> pd.DataFrame:
        """Get DataFrame with per-sample labels from all judges.

        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model

        Returns:
            DataFrame with columns: sample_index, primary_label, judge_X_label, ...
        """
        ds = self.load_dataset_metadata(dataset_name, model_name)
        if ds is None:
            return pd.DataFrame()

        data = {"sample_index": list(range(len(ds)))}

        if "label" in ds.column_names:
            data["primary_label"] = ds["label"]

        for col in ds.column_names:
            if col.startswith("judge_results_"):
                judge_name = col.replace("judge_results_", "")
                labels = []
                for result in ds[col]:
                    if result is None:
                        labels.append(None)
                    elif isinstance(result, dict):
                        labels.append(result.get("label"))
                    else:
                        labels.append(None)
                data[f"judge_{judge_name}"] = labels

        return pd.DataFrame(data)

    def get_multi_judge_summary(self, dataset_name: str, model_name: str) -> dict:
        """Get summary statistics for multi-judge evaluations.

        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model

        Returns:
            Dict with judges, agreement_rate, fleiss_kappa, etc.
        """
        from src.utils.judge_metrics import (
            check_data_completeness,
            compute_agreement_rate,
            compute_fleiss_kappa,
            compute_pairwise_kappa,
            get_disagreement_indices,
        )

        df = self.get_judge_labels_df(dataset_name, model_name)
        if df.empty:
            return {}

        judge_cols = [c for c in df.columns if c.startswith("judge_")]
        if not judge_cols:
            return {}

        labels_by_judge = {}
        for col in judge_cols:
            judge_name = col.replace("judge_", "")
            labels_by_judge[judge_name] = df[col].tolist()

        if "primary_label" in df.columns:
            labels_by_judge["primary"] = df["primary_label"].tolist()

        completeness = check_data_completeness(labels_by_judge)

        return {
            "judges": list(labels_by_judge.keys()),
            "n_samples": len(df),
            "completeness": completeness,
            "agreement_rate": compute_agreement_rate(labels_by_judge),
            "fleiss_kappa": compute_fleiss_kappa(labels_by_judge),
            "pairwise_kappa": compute_pairwise_kappa(labels_by_judge),
            "disagreement_indices": get_disagreement_indices(labels_by_judge),
        }

    def get_judge_reasoning_config(self, dataset_name: str, model_name: str, judge_name: str) -> dict | None:
        """Extract reasoning configuration for a specific judge.

        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model
            judge_name: Name of the judge

        Returns:
            Dict with reasoning metadata (thinking_budget, reasoning_level) or None if not found.
        """
        ds = self.load_dataset_metadata(dataset_name, model_name)
        if ds is None:
            return None

        col_name = f"judge_results_{judge_name}"
        if col_name not in ds.column_names:
            return None

        results = ds[col_name]
        if not results:
            return None

        # Get the first non-None result to check for reasoning_config
        for result in results:
            if result is not None and isinstance(result, dict):
                return result.get("reasoning_config")

        return None

    def is_judge_complete(self, dataset_name: str, model_name: str, judge_name: str) -> bool:
        """Check if a judge run has labels for all samples (is complete).

        A run is complete if every sample has a label (no None or error-only entries).

        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model
            judge_name: Name of the judge

        Returns:
            True if all samples have labels, False otherwise.
        """
        ds = self.load_dataset_metadata(dataset_name, model_name)
        if ds is None:
            return False

        col_name = f"judge_results_{judge_name}"
        if col_name not in ds.column_names:
            return False

        results = ds[col_name]
        if not results:
            return False

        # Check if all results have a label
        for result in results:
            if result is None or not isinstance(result, dict) or result.get("label") is None:
                return False

        return True

    # =========================================================================
    # Steering Comparison Methods
    # =========================================================================

    def get_steered_datasets(self, filter_vector: str | None = None) -> list[tuple[str, str, str, list[str]]]:
        """Find datasets with multiple steered model variants.

        Args:
            filter_vector: Optional vector name to filter by (e.g., 'instructed_pairs').
                          If None, returns all vectors grouped separately.

        Returns:
            List of (dataset_name, base_model, vector_name, [variant_names]) tuples.
        """
        import re

        # Group by (dataset, base_model, vector_name)
        results: dict[tuple[str, str, str], list[str]] = {}

        for dataset_name in self.get_dataset_names():
            models = self.get_models_for_dataset(dataset_name)

            for model in models:
                # Match pattern like "Model_steered-dataset-strength"
                match = re.match(r"(.+)_steered-(.+)-(-?\w+)$", model)
                if match:
                    base_model = match.group(1)
                    vector_name = match.group(2)

                    # Apply filter if specified
                    if filter_vector is not None and vector_name != filter_vector:
                        continue

                    key = (dataset_name, base_model, vector_name)
                    if key not in results:
                        results[key] = []
                    results[key].append(model)

        # Only include if we have multiple steered variants
        return [(ds, base, vec, sorted(vars)) for (ds, base, vec), vars in sorted(results.items()) if len(vars) >= 2]

    def get_steering_variants(self, dataset_name: str, base_model: str, vector_name: str | None = None) -> list[str]:
        """Get list of steering variants for a base model in a dataset.

        Args:
            dataset_name: Name of dataset
            base_model: Base model name (without _steered suffix)
            vector_name: Optional vector name to filter by (e.g., 'instructed_pairs')

        Returns:
            List of full model names for steered variants.
        """
        import re

        models = self.get_models_for_dataset(dataset_name)
        variants = []
        for m in models:
            if m.startswith(base_model) and "_steered-" in m:
                if vector_name is not None:
                    # Extract vector from model name and check it matches
                    match = re.match(r".+_steered-(.+)-(-?\w+)$", m)
                    if match and match.group(1) == vector_name:
                        variants.append(m)
                else:
                    variants.append(m)
        return sorted(variants)

    def get_steering_comparison_df(
        self, dataset_name: str, base_model: str, vector_name: str | None = None
    ) -> pd.DataFrame:
        """Get DataFrame comparing samples across steering variants.

        Args:
            dataset_name: Name of dataset
            base_model: Base model name
            vector_name: Optional vector name to filter by (e.g., 'instructed_pairs')

        Returns:
            DataFrame with columns: sample_index, prompt_text (truncated),
            response_{variant}, judge_label_{variant}, ground_truth for each variant.
        """
        import re

        variants = self.get_steering_variants(dataset_name, base_model, vector_name)
        if not variants:
            return pd.DataFrame()

        # Load metadata for first variant to get sample count
        first_ds = self.load_dataset_metadata(dataset_name, variants[0])
        if first_ds is None:
            return pd.DataFrame()

        n_samples = len(first_ds)

        # Initialize data dict
        data: dict[str, list] = {
            "sample_index": list(range(n_samples)),
        }

        # Add prompt text (truncated)
        if "prompt_text" in first_ds.column_names:
            data["prompt"] = [(p[:100] + "..." if len(p) > 100 else p) for p in first_ds["prompt_text"]]

        # Add ground truth label if available
        if "label" in first_ds.column_names:
            data["ground_truth"] = first_ds["label"]

        # Helper to parse steering strength from model name
        def get_strength(model_name: str) -> str:
            match = re.search(r"_steered-.+-(-?\w+)$", model_name)
            return match.group(1) if match else model_name

        # Try to load base model (without steering) first
        base_ds = self.load_dataset_metadata(dataset_name, base_model)
        if base_ds is not None and len(base_ds) == n_samples:
            # Response text (truncated)
            if "response_text" in base_ds.column_names:
                data["response_base"] = [
                    (r[:200] + "..." if r and len(r) > 200 else r) for r in base_ds["response_text"]
                ]

            # Judge label
            if "label" in base_ds.column_names:
                data["judge_label_base"] = [bool(lbl) if lbl is not None else None for lbl in base_ds["label"]]

        # Load each steered variant
        for variant in variants:
            strength = get_strength(variant)
            ds = self.load_dataset_metadata(dataset_name, variant)
            if ds is None:
                continue

            # Response text (truncated)
            if "response_text" in ds.column_names:
                data[f"response_{strength}"] = [
                    (r[:200] + "..." if r and len(r) > 200 else r) for r in ds["response_text"]
                ]

            # Judge label - prefer direct 'label' column, fallback to judge_response_parsed
            if "label" in ds.column_names:
                # Convert to bool (1=True=deceptive, 0=False=truthful)
                data[f"judge_label_{strength}"] = [bool(lbl) if lbl is not None else None for lbl in ds["label"]]
            elif "judge_response_parsed" in ds.column_names:
                labels = []
                for parsed in ds["judge_response_parsed"]:
                    if parsed is None:
                        labels.append(None)
                    elif isinstance(parsed, dict):
                        labels.append(parsed.get("label"))
                    else:
                        labels.append(None)
                data[f"judge_label_{strength}"] = labels

        return pd.DataFrame(data)
