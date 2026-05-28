"""Evaluate pre-trained Apollo probes on datasets.

This script loads Apollo probes (from ApolloResearch detector.pt files) via Hydra
config and evaluates them on specified datasets. Unlike evaluate_probes.py which
auto-discovers trained probes, this script explicitly instantiates probes from config.

Usage:
    # Single run
    uv run python -m src.evaluate_probes_apollo \
        probe=apollo/instructed_pairs \
        model=llama3.3/70b \
        data@data_eval=instructed_pairs

    # Multirun (sweep over probes x datasets)
    uv run python -m src.evaluate_probes_apollo --multirun \
        sweeps=evaluate_probes_apollo
"""

import time
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.probes import ApolloProbe
from src.utils import RankedLogger, extras
from src.utils.eval_utils import discover_model_variants, log_variant_info

log = RankedLogger(__name__, rank_zero_only=True)


def evaluate_single_variant(
    probe: ApolloProbe,
    variant_name: str,
    activations_dir: Path,
    layer: int,
    results_base_dir: Path,
    probe_name: str,
    data_eval_safe_name: str,
    control_activations_dir: Path | None,
    cfg: DictConfig,
    eval_mask_mode: str = "all",
    metadata_dir: Path | None = None,
) -> tuple[dict, int]:
    """Evaluate a single model variant with the Apollo probe.

    Returns:
        Tuple of (results dict, sample count)
    """
    from src.utils.eval_utils import evaluate_probe_on_activations

    log.info(f"\nLoading evaluation activations for {variant_name}...")

    # Build results directory
    results_dir = results_base_dir / probe_name / f"probe-{probe_name}-eval" / variant_name / data_eval_safe_name

    # Use unified evaluation function
    results, sample_count = evaluate_probe_on_activations(
        probe=probe,
        activations_dir=activations_dir,
        results_dir=results_dir,
        layer=layer,
        pooling_strategy="flat",
        model_name=variant_name,
        batch_size=cfg.get("batch_size", 8),
        num_workers=cfg.get("dataloader_num_workers", 0),
        use_mask=cfg.get("use_mask", True),
        eval_mask_mode=eval_mask_mode,
        metadata_dir=metadata_dir,
        control_activations_dir=control_activations_dir,
        extra_result_fields={"probe_name": probe_name},
    )

    if results is None:
        log.warning(f"No activations found for {variant_name}!")
        return {}, 0

    log.info(f"  Evaluated {sample_count} samples")
    return results, sample_count


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="evaluate_probes_apollo",
)
def main(cfg: DictConfig):
    """Evaluate Apollo probes on all model variants."""
    log.info("=" * 80)
    log.info("APOLLO PROBE EVALUATION")
    log.info("=" * 80)
    log.info(f"Probe: {cfg.probe.detector_name}")
    log.info(f"Model: {cfg.model.model_name_or_path}")
    log.info(f"Evaluation data: {cfg.data_eval.safe_name}")

    extras(cfg)

    start_time = time.time()

    # Load Apollo probe via Hydra instantiate
    log.info("\nLoading Apollo probe...")
    probe: ApolloProbe = hydra.utils.instantiate(cfg.probe)
    log.info(f"  Loaded: {probe}")
    log.info(f"  Layers: {probe.layers}")
    log.info(f"  Normalize: {probe.normalize}")

    # Get the layer from the probe (Apollo probes specify their own layer)
    if len(probe.layers) != 1:
        raise ValueError(f"Expected single-layer probe, got layers: {probe.layers}")
    layer = probe.layers[0]
    log.info(f"  Using layer: {layer}")

    # Paths
    model_safe_name = cfg.model.safe_name
    data_eval_safe_name = cfg.data_eval.safe_name
    data_eval_dir = Path(cfg.data_eval.data_dir)

    # Control data for FPR calibration (optional)
    control_activations_dir = None
    if hasattr(cfg, "data_control") and cfg.data_control is not None:
        control_path = Path(cfg.paths.get("control_activations_dir", ""))
        if control_path and control_path.exists():
            control_activations_dir = control_path
            log.info(f"Control data: {cfg.data_control.safe_name}")

    # Discover model variants (base + steered versions)
    responses_dir = data_eval_dir / "responses"
    model_variants = discover_model_variants(responses_dir, model_safe_name)

    if not model_variants:
        log.error(f"No model responses found in {responses_dir} for {model_safe_name}")
        return None

    log_variant_info(model_variants, model_safe_name)

    # Results setup
    probe_name = f"apollo_{cfg.probe.detector_name}"
    results_base_dir = Path(cfg.paths.data_dir)

    # Get eval_mask_mode from data_eval config (for DeceptionBench)
    eval_mask_mode = cfg.data_eval.get("eval_mask_mode", "all")
    if eval_mask_mode != "all":
        log.info(f"Using eval_mask_mode: {eval_mask_mode}")

    # Parallelization config
    n_workers = cfg.get("n_workers", min(len(model_variants), 4))
    use_parallel = len(model_variants) > 1 and n_workers > 1

    if use_parallel:
        log.info(f"\nUsing {n_workers} parallel workers for {len(model_variants)} variants")
        from joblib import Parallel, delayed

        # Build variant configs for parallel execution
        variant_configs = []
        for variant_name in model_variants:
            activations_dir = responses_dir / variant_name
            metadata_dir = activations_dir / "metadata" if eval_mask_mode != "all" else None
            variant_configs.append((variant_name, activations_dir, metadata_dir))

        # Run evaluations in parallel
        results = Parallel(n_jobs=n_workers, verbose=0)(
            delayed(evaluate_single_variant)(
                probe=probe,
                variant_name=variant_name,
                activations_dir=activations_dir,
                layer=layer,
                results_base_dir=results_base_dir,
                probe_name=probe_name,
                data_eval_safe_name=data_eval_safe_name,
                control_activations_dir=control_activations_dir,
                cfg=cfg,
                eval_mask_mode=eval_mask_mode,
                metadata_dir=metadata_dir,
            )
            for variant_name, activations_dir, metadata_dir in variant_configs
        )

        # Aggregate results
        all_results = {}
        total_samples = 0
        for variant_name, (variant_result, sample_count) in zip(
            [vc[0] for vc in variant_configs], results, strict=False
        ):
            if variant_result:
                all_results[variant_name] = variant_result
                total_samples += sample_count

    else:
        # Sequential evaluation for single variant
        all_results = {}
        total_samples = 0

        for variant_name in model_variants:
            log.info(f"\n{'=' * 80}")
            log.info(f"Evaluating variant: {variant_name}")
            log.info("=" * 80)

            activations_dir = responses_dir / variant_name
            metadata_dir = activations_dir / "metadata" if eval_mask_mode != "all" else None

            variant_results, sample_count = evaluate_single_variant(
                probe=probe,
                variant_name=variant_name,
                activations_dir=activations_dir,
                layer=layer,
                results_base_dir=results_base_dir,
                probe_name=probe_name,
                data_eval_safe_name=data_eval_safe_name,
                control_activations_dir=control_activations_dir,
                cfg=cfg,
                eval_mask_mode=eval_mask_mode,
                metadata_dir=metadata_dir,
            )

            if variant_results:
                all_results[variant_name] = variant_results
                total_samples += sample_count

    elapsed = time.time() - start_time

    log.info("\n" + "=" * 80)
    log.info("✅ APOLLO PROBE EVALUATION COMPLETE")
    log.info("=" * 80)
    log.info(f"Probe: {cfg.probe.detector_name}")
    log.info(f"Model variants evaluated: {len(model_variants)}")
    log.info(f"Evaluation data: {data_eval_safe_name}")
    log.info(f"Total samples evaluated: {total_samples}")
    log.info(f"Time: {elapsed:.1f}s")
    log.info("=" * 80)

    return all_results


if __name__ == "__main__":
    main()
