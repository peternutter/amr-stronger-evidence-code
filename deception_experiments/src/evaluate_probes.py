"""Probe evaluation with auto-discovery of trained probes.

This script discovers all trained probes for a (model, data_train) pair and evaluates
them on a given evaluation dataset in parallel using joblib.

Instead of sweeping over probe/layer/pooling/source externally, this script:
1. Takes model, data_train, and data_eval as inputs (via Hydra sweep)
2. Auto-discovers all trained probes by scanning probes-* directories
3. Evaluates all discovered configurations in parallel

Usage:
    # Single run
    uv run python -m src.evaluate_probes \
        model=llama3.1/70b \
        data@data_train=instructed_pairs \
        data@data_eval=deception_bench/L1_self

    # Multirun (sweep over model x data_train x data_eval only)
    uv run python -m src.evaluate_probes --multirun \
        sweeps=evaluate_probes_full
"""

import time
from dataclasses import dataclass
from pathlib import Path

import hydra
import joblib
from joblib import Parallel, delayed
from omegaconf import DictConfig

from src.utils import RankedLogger, extras
from src.utils.eval_utils import discover_model_variants, log_variant_info

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class ProbeConfig:
    """Configuration for a discovered probe."""

    probe_type: str  # e.g., "logistic_regression_sgd"
    layer: int
    pooling_strategy: str  # e.g., "last", "mean", "flat"
    probe_path: Path


def discover_probes(data_train_dir: Path, model_safe_name: str) -> list[ProbeConfig]:
    """Scan all probes-* directories for trained probes.

    Args:
        data_train_dir: The data directory for the training dataset
        model_safe_name: Safe name of the model (e.g., "Llama_Llama-3.1-70B-Instruct")

    Returns:
        List of ProbeConfig objects for all discovered probes
    """
    probes = []

    # Find all probes-* directories
    for probe_dir in sorted(data_train_dir.glob("probes-*")):
        if not probe_dir.is_dir():
            continue

        probe_type = probe_dir.name.replace("probes-", "")
        model_dir = probe_dir / model_safe_name

        if not model_dir.exists():
            log.debug(f"No probes found for {probe_type}/{model_safe_name}")
            continue

        # Scan for probe files: probe_layer_{layer}_{pooling}_{source}.pkl
        for probe_file in sorted(model_dir.glob("probe_layer_*.pkl")):
            # Parse filename
            stem = probe_file.stem  # e.g., "probe_layer_0_last_prompt"
            parts = stem.split("_")

            if len(parts) < 4:
                log.warning(f"Unexpected probe filename format: {probe_file}")
                continue

            try:
                layer = int(parts[2])
                pooling = parts[3]

                probes.append(
                    ProbeConfig(
                        probe_type=probe_type,
                        layer=layer,
                        pooling_strategy=pooling,
                        probe_path=probe_file,
                    )
                )
            except (ValueError, IndexError) as e:
                log.warning(f"Failed to parse probe filename {probe_file}: {e}")
                continue

    return probes


# apply_eval_mask_mode moved to src/utils/eval_utils.py


def evaluate_single_probe(
    probe_config: ProbeConfig,
    activations_dir: Path,
    results_base_dir: Path,
    data_eval_safe_name: str,
    model_safe_name: str,
    batch_size: int,
    num_workers: int,
    config_idx: int,
    total_configs: int,
    use_mask: bool,
    control_activations_dir: Path | None = None,
    eval_mask_mode: str = "all",
    metadata_dir: Path | None = None,
) -> dict:
    """Evaluate a single probe configuration.

    This function is designed to be called in parallel via joblib.
    """
    from src.utils.eval_utils import evaluate_probe_on_activations

    config_name = f"{probe_config.probe_type}/layer_{probe_config.layer}_{probe_config.pooling_strategy}"
    start_time = time.time()
    log.info(f"[{config_idx + 1}/{total_configs}] Starting {config_name}...")

    try:
        # Load probe
        probe = joblib.load(probe_config.probe_path)

        # Build results directory
        results_dir = results_base_dir / f"probe-{probe_config.probe_type}-eval" / model_safe_name / data_eval_safe_name

        # Use unified evaluation function
        results, sample_count = evaluate_probe_on_activations(
            probe=probe,
            activations_dir=activations_dir,
            results_dir=results_dir,
            layer=probe_config.layer,
            pooling_strategy=probe_config.pooling_strategy,
            model_name=model_safe_name,
            batch_size=batch_size,
            num_workers=num_workers,
            use_mask=use_mask,
            eval_mask_mode=eval_mask_mode,
            metadata_dir=metadata_dir,
            control_activations_dir=control_activations_dir,
        )

        elapsed = time.time() - start_time

        if results is None:
            log.info(f"[{config_idx + 1}/{total_configs}] Skipped {config_name}: No data ({elapsed:.1f}s)")
            return {"config_name": config_name, "status": "skipped", "reason": "No data available"}

        log.info(f"[{config_idx + 1}/{total_configs}] Completed {config_name} ({elapsed:.1f}s)")

        return {
            "config_name": config_name,
            "status": "success",
            "n_samples": sample_count,
            "elapsed_time": elapsed,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        log.error(f"[{config_idx + 1}/{total_configs}] Failed {config_name}: {e} ({elapsed:.1f}s)")
        return {"config_name": config_name, "status": "error", "error": str(e)}


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="evaluate_probes",
)
def main(cfg: DictConfig):
    """Evaluate probes with auto-discovery and parallel execution."""
    log.info("=" * 80)
    log.info("PROBE EVALUATION")
    log.info("=" * 80)
    log.info(f"Model: {cfg.model.model_name_or_path}")
    log.info(f"Model safe name: {cfg.model.safe_name}")
    log.info(f"Training data: {cfg.data_train.safe_name}")
    log.info(f"Evaluation data: {cfg.data_eval.safe_name}")

    extras(cfg)

    # Paths
    data_train_dir = Path(cfg.data_train.data_dir)
    data_eval_dir = Path(cfg.data_eval.data_dir)
    model_safe_name = cfg.model.safe_name

    # Control data for FPR calibration (optional)
    control_activations_dir = None
    if hasattr(cfg, "data_control") and cfg.data_control is not None:
        control_path = Path(cfg.paths.get("control_activations_dir", ""))
        if control_path and control_path.exists():
            control_activations_dir = control_path
            log.info(f"Control data: {cfg.data_control.safe_name}")
            log.info(f"Control activations dir: {control_activations_dir}")
        else:
            log.warning(f"Control activations not found at {control_path}, skipping Recall@1%FPR")

    log.info(f"Training data dir: {data_train_dir}")

    # Discover available probes
    log.info("\nDiscovering trained probes...")
    probes = discover_probes(data_train_dir, model_safe_name)

    if not probes:
        log.error(f"No trained probes found for model {model_safe_name} in {data_train_dir}")
        return None

    # Group probes by type for logging
    probes_by_type = {}
    for p in probes:
        probes_by_type.setdefault(p.probe_type, []).append(p)

    log.info(f"\nDiscovered {len(probes)} probe configurations")
    for probe_type, type_probes in sorted(probes_by_type.items()):
        layers = sorted({p.layer for p in type_probes})
        poolings = sorted({p.pooling_strategy for p in type_probes})
        log.info(f"  {probe_type}: {len(type_probes)} probes")
        log.info(f"    Layers: {len(layers)} ({min(layers)}-{max(layers)})")
        log.info(f"    Poolings: {poolings}")

    # Discover model variants (base + steered versions)
    responses_dir = data_eval_dir / "responses"
    model_variants = discover_model_variants(responses_dir, model_safe_name)

    if not model_variants:
        log.error(f"No model responses found in {responses_dir} for {model_safe_name}")
        return None

    log_variant_info(model_variants, model_safe_name)

    # Get parallelization config
    n_workers = cfg.get("n_workers", min(len(probes), 20))
    log.info(f"\nUsing {n_workers} parallel workers")

    # Get eval_mask_mode from data_eval config (for DeceptionBench)
    eval_mask_mode = cfg.data_eval.get("eval_mask_mode", "all")
    if eval_mask_mode != "all":
        log.info(f"Using eval_mask_mode: {eval_mask_mode}")

    # Aggregate results across all variants
    all_results = []

    # Evaluate each model variant
    for variant_name in model_variants:
        log.info(f"\n{'=' * 80}")
        log.info(f"Evaluating variant: {variant_name}")
        log.info("=" * 80)

        # Build activations directory for this variant
        activations_dir = responses_dir / variant_name

        # Metadata directory for json_field_indices (needed for response_only/reasoning_only)
        metadata_dir = activations_dir / "metadata" if eval_mask_mode != "all" else None

        # Run evaluations in parallel
        log.info(f"Starting parallel probe evaluation for {variant_name}...")
        total_configs = len(probes)

        results = Parallel(n_jobs=n_workers, verbose=0)(
            delayed(evaluate_single_probe)(
                probe_config=probe_config,
                activations_dir=activations_dir,
                results_base_dir=data_train_dir,
                data_eval_safe_name=cfg.data_eval.safe_name,
                model_safe_name=variant_name,  # Use variant name for results
                batch_size=cfg.get("batch_size", 8),
                num_workers=cfg.get("dataloader_num_workers", 0),
                config_idx=idx,
                total_configs=total_configs,
                use_mask=cfg.get("use_mask", True),
                control_activations_dir=control_activations_dir,
                eval_mask_mode=eval_mask_mode,
                metadata_dir=metadata_dir,
            )
            for idx, probe_config in enumerate(probes)
        )

        # Add variant name to each result
        for r in results:
            if r is not None:
                r["variant"] = variant_name
                all_results.append(r)

    # Summarize all results
    successful = [r for r in all_results if r["status"] == "success"]
    skipped = [r for r in all_results if r["status"] == "skipped"]
    failed = [r for r in all_results if r["status"] == "error"]

    log.info("\n" + "=" * 80)
    log.info("✅ PROBE EVALUATION COMPLETE")
    log.info("=" * 80)
    log.info(f"Model: {cfg.model.model_name_or_path}")
    log.info(f"Model variants evaluated: {len(model_variants)}")
    log.info(f"Training data: {cfg.data_train.safe_name}")
    log.info(f"Evaluation data: {cfg.data_eval.safe_name}")
    log.info("\nResults:")
    log.info(f"  ✅ Successful: {len(successful)}")
    log.info(f"  ⏭️  Skipped: {len(skipped)}")
    log.info(f"  ❌ Failed: {len(failed)}")

    if skipped:
        log.info("\nSkipped configurations:")
        for r in skipped[:10]:  # Limit output
            log.info(f"  {r.get('variant', 'N/A')}/{r['config_name']}: {r['reason']}")
        if len(skipped) > 10:
            log.info(f"  ... and {len(skipped) - 10} more")

    if failed:
        log.warning("\nFailed configurations:")
        for r in failed[:10]:  # Limit output
            log.warning(f"  {r.get('variant', 'N/A')}/{r['config_name']}: {r['error']}")
        if len(failed) > 10:
            log.warning(f"  ... and {len(failed) - 10} more")

    if successful:
        total_time = sum(r["elapsed_time"] for r in successful)
        avg_time = total_time / len(successful)
        log.info(f"\nTiming: {total_time:.1f}s total, {avg_time:.2f}s avg per config")

    log.info("=" * 80)
    return None


if __name__ == "__main__":
    main()
