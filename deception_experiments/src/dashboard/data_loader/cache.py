"""Cache mixin providing caching and loading methods for DashboardDataLoader."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    pass


def _compute_dir_fingerprint(eval_dir: Path) -> str | None:
    """Compute fingerprint for a single eval directory (for parallel execution)."""
    try:
        result_files = list(eval_dir.glob("results_layer_*.pkl"))
        if not result_files:
            return None
        return f"{eval_dir}:{len(result_files)}:{sum(f.stat().st_mtime for f in result_files)}"
    except Exception:
        return None


def _process_single_result_file(
    result_file: Path,
    train_dataset: str,
    probe: str,
    model: str,
    eval_dataset: str,
) -> dict | None:
    """Process a single result file and return metrics dict (for parallel execution)."""
    try:
        # Parse filename: results_layer_{layer}_{pooling}_{aggregation}.pkl
        stem = result_file.stem
        parts = stem.split("_")
        if len(parts) < 4:
            return None

        layer = int(parts[2])
        pooling = parts[3]
        aggregation = parts[4] if len(parts) > 4 else "legacy"

        data = joblib.load(result_file)
        metrics = data.get("metrics", {})

        return {
            "train_dataset": train_dataset,
            "probe": probe,
            "model": model,
            "eval_dataset": eval_dataset,
            "layer": layer,
            "pooling": pooling,
            "aggregation": aggregation,
            **metrics,
        }
    except Exception:
        return None


class CacheMixin:
    """Mixin class providing caching and data loading methods.

    Requires: data_dir (Path), allow_stale (bool), experiments (property)
    """

    data_dir: Path
    allow_stale: bool
    _experiments_df: pd.DataFrame | None
    _all_metrics_df: pd.DataFrame | None
    _cached_fingerprint: str | None
    _result_cache: dict
    _cache_was_stale: bool
    _stale_reason: str | None

    def _load_experiments(self) -> pd.DataFrame:
        """Load experiments DataFrame, using cache if available and fresh."""
        cache_path = self._get_experiments_cache_path()
        metadata = self._load_cache_metadata()

        if cache_path.exists() and metadata:
            current_fingerprint = self._compute_experiments_fingerprint()
            stored_fingerprint = metadata.get("fingerprint")

            if stored_fingerprint == current_fingerprint or self.allow_stale:
                if stored_fingerprint != current_fingerprint:
                    self._cache_was_stale = True
                    self._stale_reason = "Result files have changed since cache was built"

                try:
                    return pd.read_parquet(cache_path)
                except Exception:
                    pass
            else:
                self._cache_was_stale = True
                self._stale_reason = "Result files have changed since cache was built"

        return self._scan_experiments()

    def _scan_experiments(self) -> pd.DataFrame:
        """Scan data directory for experiment configurations (no caching)."""
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
                                "path": str(eval_dataset_dir),
                            }
                        )

        return pd.DataFrame(results)

    def _get_cache_dir(self) -> Path:
        """Get cache directory path."""
        return self.data_dir / ".dashboard_cache"

    def _get_experiments_cache_path(self) -> Path:
        """Get path for cached experiments DataFrame."""
        return self._get_cache_dir() / "experiments.parquet"

    def _get_metrics_cache_path(self) -> Path:
        """Get path for cached metrics DataFrame."""
        return self._get_cache_dir() / "all_metrics.parquet"

    def _get_cache_metadata_path(self) -> Path:
        """Get path for cache metadata file."""
        return self._get_cache_dir() / "cache_metadata.json"

    def _compute_experiments_fingerprint(self, show_progress: bool = False) -> str:
        """Compute fingerprint based on experiment directory structure.

        Uses parallel processing for speed on network filesystems.
        Caches the result to avoid recomputing within the same session.
        """
        if self._cached_fingerprint is not None:
            return self._cached_fingerprint

        eval_dirs: list[Path] = []
        for train_dir in self.data_dir.iterdir():
            if not train_dir.is_dir():
                continue
            for probe_eval_dir in train_dir.glob("probe-*-eval"):
                for model_dir in probe_eval_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    for eval_dir in model_dir.iterdir():
                        if eval_dir.is_dir():
                            eval_dirs.append(eval_dir)

        if not eval_dirs:
            self._cached_fingerprint = hashlib.md5(b"").hexdigest()[:16]
            return self._cached_fingerprint

        n_workers = min(32, len(eval_dirs))

        results = joblib.Parallel(n_jobs=n_workers)(
            joblib.delayed(_compute_dir_fingerprint)(d)
            for d in tqdm(eval_dirs, desc="Computing fingerprint", disable=not show_progress)
        )

        dir_info = [r for r in results if r is not None]

        dir_info.sort()
        content = "\n".join(dir_info)
        self._cached_fingerprint = hashlib.md5(content.encode()).hexdigest()[:16]
        return self._cached_fingerprint

    def _load_cache_metadata(self) -> dict[str, Any]:
        """Load cache metadata if it exists."""
        metadata_path = self._get_cache_metadata_path()
        if metadata_path.exists():
            try:
                return json.loads(metadata_path.read_text())
            except Exception:
                pass
        return {}

    def _save_cache_metadata(self, fingerprint: str, num_experiments: int, num_metrics: int) -> None:
        """Save cache metadata."""
        metadata = {
            "fingerprint": fingerprint,
            "created_at": datetime.now().isoformat(),
            "num_experiments": num_experiments,
            "num_metrics": num_metrics,
        }
        try:
            self._get_cache_dir().mkdir(parents=True, exist_ok=True)
            self._get_cache_metadata_path().write_text(json.dumps(metadata, indent=2))
        except Exception:
            pass

    def get_cache_status(self, show_progress: bool = False) -> dict[str, Any]:
        """Get detailed cache status information."""
        exp_cache = self._get_experiments_cache_path()
        metrics_cache = self._get_metrics_cache_path()
        metadata = self._load_cache_metadata()

        experiments_cached = exp_cache.exists()
        metrics_cached = metrics_cache.exists()

        cache_stale = False
        stale_reason = None

        if experiments_cached and metrics_cached and metadata:
            current_fingerprint = self._compute_experiments_fingerprint(show_progress=show_progress)
            stored_fingerprint = metadata.get("fingerprint")
            if stored_fingerprint != current_fingerprint:
                cache_stale = True
                stale_reason = "Result files have changed since cache was built"
        elif not experiments_cached or not metrics_cached:
            cache_stale = True
            stale_reason = "Cache files missing"

        return {
            "experiments_cached": experiments_cached,
            "metrics_cached": metrics_cached,
            "cache_stale": cache_stale,
            "stale_reason": stale_reason,
            "cache_time": metadata.get("created_at"),
            "num_experiments": metadata.get("num_experiments"),
            "num_metrics": metadata.get("num_metrics"),
        }

    def refresh_cache(self, show_progress: bool = True) -> None:
        """Force a fresh scan of all data and update the cache."""
        self._cached_fingerprint = None
        self._experiments_df = self._scan_experiments()
        self._all_metrics_df = self._load_all_metrics_from_files(show_progress=show_progress)
        self._save_to_cache(self._experiments_df, self._all_metrics_df)

    def _load_all_metrics_from_files(self, show_progress: bool = True) -> pd.DataFrame:
        """Helper to load all metrics from pkl files without checking cache."""
        file_tasks: list[tuple[Path, str, str, str, str]] = []
        for _, exp in self.experiments.iterrows():
            result_dir = Path(exp["path"])
            for result_file in result_dir.glob("results_layer_*.pkl"):
                file_tasks.append(
                    (
                        result_file,
                        exp["train_dataset"],
                        exp["probe"],
                        exp["model"],
                        exp["eval_dataset"],
                    )
                )

        if not file_tasks:
            return pd.DataFrame()

        print(f"   Processing {len(file_tasks):,} result files in parallel...")

        n_jobs = min(32, len(file_tasks)) if file_tasks else 1
        results = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
            joblib.delayed(_process_single_result_file)(result_file, train_dataset, probe, model, eval_dataset)
            for result_file, train_dataset, probe, model, eval_dataset in tqdm(
                file_tasks, desc="Loading metrics", disable=not show_progress
            )
        )

        rows = [r for r in results if r is not None]

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["train_dataset", "probe", "model", "eval_dataset", "layer", "aggregation"])
            df = df.reset_index(drop=True)

        return df

    def _save_to_cache(self, experiments_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
        """Save both DataFrames to cache and update metadata."""
        try:
            cache_dir = self._get_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)

            experiments_df.to_parquet(self._get_experiments_cache_path())
            metrics_df.to_parquet(self._get_metrics_cache_path())

            current_fingerprint = self._compute_experiments_fingerprint()
            self._save_cache_metadata(current_fingerprint, len(experiments_df), len(metrics_df))

            self._cache_was_stale = False
            self._stale_reason = None
        except Exception as e:
            print(f"Warning: Caching failed: {e}")

    def _load_all_metrics(self) -> pd.DataFrame:
        """Load metrics from all experiments into a single DataFrame.

        Uses disk caching to avoid re-loading ~40K+ pkl files on every startup.
        Cache is invalidated when result files change.
        """
        cache_path = self._get_metrics_cache_path()
        metadata = self._load_cache_metadata()
        current_fingerprint = self._compute_experiments_fingerprint(show_progress=True)

        if cache_path.exists() and metadata:
            stored_fingerprint = metadata.get("fingerprint")
            if stored_fingerprint == current_fingerprint or self.allow_stale:
                if stored_fingerprint != current_fingerprint:
                    self._cache_was_stale = True
                    self._stale_reason = "Result files have changed since cache was built"

                try:
                    return pd.read_parquet(cache_path)
                except Exception:
                    pass

        df = self._load_all_metrics_from_files()
        self._save_to_cache(self.experiments, df)

        return df

    def _get_experiment_path(
        self,
        train_dataset: str,
        probe: str,
        model: str,
        eval_dataset: str,
    ) -> Path | None:
        """Get filesystem path for an experiment."""
        matches = self.experiments[
            (self.experiments["train_dataset"] == train_dataset)
            & (self.experiments["probe"] == probe)
            & (self.experiments["model"] == model)
            & (self.experiments["eval_dataset"] == eval_dataset)
        ]
        if matches.empty:
            return None
        return Path(matches.iloc[0]["path"])

    def _load_result_file(
        self,
        result_path: Path,
        layer: int,
        pooling: str,
        aggregation: str = "legacy",
    ) -> dict[str, Any] | None:
        """Load and cache a single result file."""
        cache_key = f"{result_path}_{layer}_{pooling}_{aggregation}"
        if cache_key in self._result_cache:
            return self._result_cache[cache_key]

        candidate = result_path / f"results_layer_{layer}_{pooling}_{aggregation}.pkl"

        if not candidate.exists():
            self._result_cache[cache_key] = None
            return None

        data = joblib.load(candidate)
        self._result_cache[cache_key] = data
        return data
