"""Shared helpers for interactive dashboard components."""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import pandas as pd


class ResultScanner:
    """Scans a data directory to discover available experiment configurations."""

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)

    def scan(self) -> pd.DataFrame:
        """Return a DataFrame describing every experiment directory that has results."""
        results: list[dict[str, object]] = []

        for train_dataset_dir in self.data_dir.iterdir():
            if not train_dataset_dir.is_dir():
                continue

            train_dataset = train_dataset_dir.name

            for probe_eval_dir in train_dataset_dir.glob("probe-*-eval"):
                probe_match = re.match(r"probe-(.+)-eval", probe_eval_dir.name)
                if not probe_match:
                    continue
                probe = probe_match.group(1)

                for model_dir in probe_eval_dir.iterdir():
                    if not model_dir.is_dir():
                        continue

                    model = model_dir.name

                    for eval_dataset_dir in model_dir.iterdir():
                        if not eval_dataset_dir.is_dir():
                            continue

                        eval_dataset = eval_dataset_dir.name
                        result_files = list(eval_dataset_dir.glob("results_layer_*.pkl"))
                        if not result_files:
                            continue

                        results.append(
                            {
                                "train_dataset": train_dataset,
                                "probe": probe,
                                "model": model,
                                "eval_dataset": eval_dataset,
                                "path": eval_dataset_dir,
                            }
                        )

        return pd.DataFrame(results)


class ResultLoader:
    """Loads and normalizes per-layer probe evaluation files."""

    def __init__(self, result_dir: Path | str):
        self.result_dir = Path(result_dir)

    def load(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []

        for result_file in self.result_dir.glob("results_layer_*.pkl"):
            # Parse filename: results_layer_{layer}_{pooling}_{aggregation}.pkl
            match = re.match(
                r"results_layer_(\d+)_([a-zA-Z0-9_]+)_([a-z]+)\.pkl",
                result_file.name,
            )
            if not match:
                continue

            layer = int(match.group(1))
            pooling = match.group(2)
            aggregation = match.group(3)

            data = joblib.load(result_file)
            metrics = data.get("metrics", {})

            rows.append(
                {
                    "layer": layer,
                    "pooling": pooling,
                    "aggregation": aggregation,
                    "accuracy": metrics.get("accuracy", 0.0),
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1": metrics.get("f1", 0.0),
                    "roc_auc": metrics.get("roc_auc", 0.0),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["layer", "aggregation"]).reset_index(drop=True)
        return df


def ensure_path(path: Path | str | None) -> Path | None:
    """Return ``Path`` objects from DataFrame entries."""
    if path is None:
        return None
    return path if isinstance(path, Path) else Path(path)


def find_experiment_path(
    experiments: pd.DataFrame,
    *,
    train_dataset: str,
    probe: str,
    model: str,
    eval_dataset: str,
) -> Path | None:
    """Return the filesystem path for a specific experiment configuration."""
    if experiments.empty:
        return None

    matches = experiments[
        (experiments["train_dataset"] == train_dataset)
        & (experiments["probe"] == probe)
        & (experiments["model"] == model)
        & (experiments["eval_dataset"] == eval_dataset)
    ]

    if matches.empty:
        return None

    return ensure_path(matches.iloc[0]["path"])


def format_pooling_label(pooling: str, aggregation: str = "") -> str:
    """Return a human-readable label for pooling/aggregation combinations."""
    base = f"{pooling.replace('_', ' ').title()}"
    if aggregation:
        return f"{base}/{aggregation.title()}"
    return base


def list_pooling_options(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return dropdown-ready options for every pooling strategy present in ``df``."""
    if df.empty or "pooling" not in df.columns:
        return []

    combos = df[["pooling"]].drop_duplicates().sort_values(["pooling"])
    options: list[tuple[str, str]] = []
    for row in combos.itertuples(index=False):
        pooling = str(row.pooling)
        options.append((format_pooling_label(pooling), pooling))
    return options


def load_result_file(result_path: Path, layer: int, pooling: str, aggregation: str) -> dict | None:
    """Load a single results_layer file if it exists."""
    candidate = Path(result_path) / f"results_layer_{layer}_{pooling}_{aggregation}.pkl"
    if not candidate.exists():
        return None
    return joblib.load(candidate)
