import itertools
import time
from pathlib import Path
from typing import Any

import hydra
import joblib
import numpy as np
from joblib import Parallel, delayed
from omegaconf import DictConfig

from src.data import ActivationDataModule
from src.utils import RankedLogger, extras

log = RankedLogger(__name__, rank_zero_only=True)


def train_probe(
    probe,
    X: np.ndarray,
    y: np.ndarray,
    tune_hyperparameters: bool = False,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict:
    """Train a probe with optional hyperparameter tuning.

    Args:
        probe: Classifier instance or class with create_tuned_estimator method
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        tune_hyperparameters: If True, use GridSearchCV/CV variants for tuning
        n_splits: Number of cross-validation splits
        random_state: Random seed
        n_jobs: Number of parallel jobs for cross-validation

    Returns:
        Dictionary containing trained probe and cross-validation results
    """

    # Check for invalid values in the input data
    if not np.all(np.isfinite(X)):
        log.error("Input data X contains NaN or infinite values. Aborting training for this probe.")
        # Optionally, you could try to clean the data here, but for now, we'll just exit.
        # For example: X = np.nan_to_num(X)
        raise ValueError("Input data X contains NaN or infinite values.")

    if tune_hyperparameters:
        log.info("=" * 80)
        log.info("HYPERPARAMETER TUNING ENABLED")
        log.info("=" * 80)

        # Get the tuned estimator (GridSearchCV or *CV variant)
        if hasattr(probe, "create_tuned_estimator"):
            # Pass all probe parameters from config via get_params()
            probe_params = probe.get_params()
            # Set verbose to 0 to avoid excessive logging from saga solver
            if "verbose" in probe_params:
                probe_params["verbose"] = 0
            # Remove n_jobs if it exists, to avoid duplicate keyword argument
            if "n_jobs" in probe_params:
                del probe_params["n_jobs"]
            tuned_probe = probe.create_tuned_estimator(cv_folds=n_splits, n_jobs=n_jobs, **probe_params)
        else:
            log.warning(f"Probe {type(probe).__name__} doesn't support tuning. Using fixed params.")
            tuned_probe = probe

        log.info(f"Performing {n_splits}-fold CV with hyperparameter search...")
        if hasattr(tuned_probe, "param_grid"):
            log.info(f"Parameter grid: {tuned_probe.param_grid}")

        # Fit with hyperparameter search (CV happens internally)
        tuned_probe.fit(X, y)

        # Extract best parameters and scores
        best_params = {}
        cv_scores = []

        if hasattr(tuned_probe, "best_params_"):
            # GridSearchCV
            best_params = tuned_probe.best_params_
            best_score = tuned_probe.best_score_
            cv_results = tuned_probe.cv_results_
            best_index = tuned_probe.best_index_
            cv_scores = [cv_results[f"split{i}_test_score"][best_index] for i in range(tuned_probe.n_splits_)]

            log.info("\n" + "=" * 80)
            log.info("BEST HYPERPARAMETERS FOUND")
            log.info("=" * 80)
            for param, value in best_params.items():
                log.info(f"  {param}: {value}")
            log.info(f"\nBest CV Accuracy: {best_score:.4f} ± {cv_results['std_test_score'][best_index]:.4f}")
            log.info("=" * 80 + "\n")

        # Get the best estimator (already refitted on full data)
        final_probe = tuned_probe.best_estimator_ if hasattr(tuned_probe, "best_estimator_") else tuned_probe

        fold_results = {
            "accuracy": cv_scores,
            "best_params": best_params,
        }

        mean_results = {
            "accuracy": (np.mean(cv_scores), np.std(cv_scores)),
        }

    else:
        # No tuning - simple fit on all data
        log.info("Using fixed hyperparameters from config (no tuning)")
        log.info("Training probe on all data...")

        final_probe = probe
        final_probe.fit(X, y)

        log.info("✓ Training complete")

        fold_results = {"fixed_params": probe.get_params()}
        mean_results = {}

    return {
        "probe": final_probe,
        "fold_results": fold_results,
        "mean_results": mean_results,
        "n_samples": len(X),
        "n_features": X.shape[1],
    }


def train_layer_config(
    layer: int,
    pooling_strategy: str,
    activations_dir: Path,
    output_dir: Path,
    probe_target: str,
    probe_config: DictConfig,
    tune_hyperparameters: bool,
    use_mask: bool,
    n_splits: int,
    random_state: int,
    batch_size: int,
    num_workers: int,
    config_idx: int = 0,
    total_configs: int = 1,
) -> dict[str, Any]:
    """Train a probe for a single layer+pooling configuration.

    This function is designed to be called in parallel via joblib.

    Args:
        layer: Layer index to train on
        pooling_strategy: Pooling strategy to use
        activations_dir: Path to activations directory
        output_dir: Path to output directory for probes
        source: Activation source (e.g., 'prompt', 'completion')
        probe_target: Probe class target string for hydra instantiation
        probe_config: Probe configuration DictConfig
        tune_hyperparameters: Whether to tune hyperparameters
        n_splits: Number of CV splits
        random_state: Random seed
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        config_idx: Index of current config (for progress logging)
        total_configs: Total number of configs (for progress logging)

    Returns:
        Dictionary with training results and status
    """
    config_name = f"layer_{layer}_{pooling_strategy}"
    start_time = time.time()

    # Print progress (visible in logs)
    log.info(f"[{config_idx+1}/{total_configs}] Starting {config_name}...")

    try:
        # Load activation data
        datamodule = ActivationDataModule(
            data_dir=str(activations_dir),
            layer=layer,
            pooling_strategy=pooling_strategy,
            batch_size=batch_size,
            num_workers=num_workers,
            use_mask=use_mask,
        )
        datamodule.prepare_data()
        datamodule.setup(stage="train")

        X, y = datamodule.get_X_y()
        if X.size == 0:
            elapsed = time.time() - start_time
            log.info(f"[{config_idx+1}/{total_configs}] Skipped {config_name}: No data ({elapsed:.1f}s)")
            return {
                "config_name": config_name,
                "status": "skipped",
                "reason": "No data available",
            }
        # If only one class present, skip
        if len(np.unique(y)) < 2:
            elapsed = time.time() - start_time
            log.info(f"[{config_idx+1}/{total_configs}] Skipped {config_name}: Only one class ({elapsed:.1f}s)")
            return {
                "config_name": config_name,
                "status": "skipped",
                "reason": "Only one class present",
            }

        # Create probe using hydra instantiation
        probe = hydra.utils.instantiate(probe_config)

        # Train probe with optional hyperparameter tuning
        # Use n_jobs=1 for CV within each parallel worker to avoid nested parallelism
        results = train_probe(
            probe,
            X=X,
            y=y,
            tune_hyperparameters=tune_hyperparameters,
            n_splits=n_splits,
            random_state=random_state,
            n_jobs=1,
        )

        # Save probe
        probe_path = output_dir / f"probe_layer_{layer}_{pooling_strategy}.pkl"
        joblib.dump(results["probe"], probe_path)

        # Save full results
        results_path = output_dir / f"results_layer_{layer}_{pooling_strategy}.pkl"
        joblib.dump(results, results_path)

        elapsed = time.time() - start_time
        log.info(f"[{config_idx+1}/{total_configs}] Completed {config_name} ({elapsed:.1f}s)")

        return {
            "config_name": config_name,
            "status": "success",
            "probe_path": str(probe_path),
            "results_path": str(results_path),
            "n_samples": results["n_samples"],
            "n_features": results["n_features"],
            "mean_results": results["mean_results"],
            "elapsed_time": elapsed,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        log.error(f"[{config_idx+1}/{total_configs}] Failed {config_name}: {e} ({elapsed:.1f}s)")
        return {
            "config_name": config_name,
            "status": "error",
            "error": str(e),
        }


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train_probes",
)
def main(cfg: DictConfig):
    """Train probe for deception detection."""
    log.info("Starting probe training...")
    log.info(f"Model: {cfg.model.model_name_or_path}")
    log.info(f"Data: {cfg.data._target_}")

    # Initialize extras (logging, seeding, etc.)
    extras(cfg)

    # Find available layers from results directory
    activations_dir = Path(cfg.paths.activations_dir)
    available_layers = []
    for layer_dir in activations_dir.glob("layer_*"):
        if layer_dir.is_dir():
            layer_num = int(layer_dir.name.split("_")[1])
            available_layers.append(layer_num)
    available_layers.sort()
    log.info(f"Available layers in results directory: {available_layers}")

    # Handle requested layers
    if cfg.layer == "all":
        layers = available_layers
    else:
        if hasattr(cfg.layer, "__iter__") and not isinstance(cfg.layer, str):
            layers = list(cfg.layer)
        else:
            layers = [cfg.layer]
        if available_layers:
            max_layer = max(available_layers)
            layers = [(layer + max_layer + 1) % (max_layer + 1) for layer in layers]  # Handle negative layers
        # Filter to only keep layers that exist
        missing_layers = [layer_idx for layer_idx in layers if layer_idx not in available_layers]
        if missing_layers:
            log.warning(f"Requested layers not found in activations (will be skipped): {missing_layers}")
        layers = [layer_idx for layer_idx in layers if layer_idx in available_layers]
    log.info(f"Using layers for probe training: {layers}")

    # Handle requested pooling strategies
    if hasattr(cfg.pooling_strategy, "__iter__") and not isinstance(cfg.pooling_strategy, str):
        pooling_strategies = list(cfg.pooling_strategy)
    else:
        pooling_strategies = [cfg.pooling_strategy]
    log.info(f"Using pooling strategies: {pooling_strategies}")

    # Create output directory in probes_dir/{model_name}_probes/
    output_dir = Path(cfg.paths.probes_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Probes will be saved to: {output_dir}")

    # Get parallelization config
    n_workers = cfg.get("n_workers", min(len(layers), 20))
    log.info(f"Using {n_workers} parallel workers for layer training")

    # Generate all layer+pooling combinations
    combinations = list(itertools.product(layers, pooling_strategies))
    log.info(f"Total configurations to train: {len(combinations)}")

    # Train probes in parallel over layers
    log.info("\nStarting parallel probe training...")
    total_configs = len(combinations)
    results = Parallel(n_jobs=n_workers, verbose=0, backend="threading")(
        delayed(train_layer_config)(
            layer=layer,
            pooling_strategy=pooling_strategy,
            activations_dir=activations_dir,
            output_dir=output_dir,
            probe_target=cfg.probe._target_,
            probe_config=cfg.probe,
            tune_hyperparameters=cfg.get("tune_hyperparameters", False),
            use_mask=cfg.activation_datamodule.get("use_mask", False),
            n_splits=cfg.get("n_splits", 5),
            random_state=cfg.seed,
            batch_size=cfg.activation_datamodule.batch_size,
            num_workers=cfg.activation_datamodule.num_workers,
            config_idx=idx,
            total_configs=total_configs,
        )
        for idx, (layer, pooling_strategy) in enumerate(combinations)
    )

    # Summarize results
    successful = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed = [r for r in results if r["status"] == "error"]

    # Print final summary
    log.info("\n" + "=" * 80)
    log.info("✅ PROBE TRAINING COMPLETE")
    log.info("=" * 80)
    log.info(f"Model: {cfg.model.model_name_or_path}")
    log.info(f"Dataset: {cfg.data._target_}")
    log.info("\nResults:")
    log.info(f"  ✅ Successful: {len(successful)}")
    log.info(f"  ⏭️  Skipped: {len(skipped)}")
    log.info(f"  ❌ Failed: {len(failed)}")

    if skipped:
        log.info("\nSkipped configurations:")
        for r in skipped:
            log.info(f"  {r['config_name']}: {r['reason']}")

    if failed:
        log.warning("\nFailed configurations:")
        for r in failed:
            log.warning(f"  {r['config_name']}: {r['error']}")

    log.info(f"\nProbes saved to: {output_dir}")
    log.info("\nPer-configuration summary:")

    for r in successful:
        log.info(f"\n{r['config_name']}:")
        log.info(f"  Samples: {r['n_samples']}, Features: {r['n_features']}")
        for metric, (mean, std) in r["mean_results"].items():
            if std > 0:
                log.info(f"  {metric.capitalize():12s} {mean:.4f} ± {std:.4f}")
            else:
                log.info(f"  {metric.capitalize():12s} {mean:.4f}")

    log.info("\nTo load a probe:")
    log.info("  import joblib")
    log.info(f"  probe = joblib.load('{output_dir}/probe_layer_0_<pooling>.pkl')")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
