"""Steering utilities for activation manipulation experiments.

Provides tools to:
1. Extract steering direction vectors from trained probes
2. Apply steering hooks during model forward passes

Steering Operations:
- strength > 0: Add direction to activations (steer towards deceptive)
- strength < 0: Subtract direction from activations (steer towards honest)
- strength == 0: Project out direction (remove the component entirely)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import torch

if TYPE_CHECKING:
    from src.probes.lr_probe import LogisticRegressionProbe

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def load_probe_direction(
    probe_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """Load steering direction from a trained LogisticRegressionProbe.

    Automatically handles normalization: if the probe was trained with
    StandardScaler, the coefficient is transformed back to original space.

    Args:
        probe_path: Path to the saved probe .pkl file
        device: Device to place the direction tensor on

    Returns:
        Tuple of (direction tensor [1, hidden_dim], probe metadata dict)
    """
    probe_path = Path(probe_path)
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe not found: {probe_path}")

    probe: LogisticRegressionProbe = joblib.load(probe_path)

    # Validate probe type - support both pure LR and SGD versions
    # Both use LogisticRegressionProbe or consistent API
    SUPPORTED_PROBES = ["LogisticRegressionProbe", "SGDProbe", "ApolloProbe"]
    probe_type_name = type(probe).__name__
    if probe_type_name not in SUPPORTED_PROBES:
        log.warning(f"Probe type {probe_type_name} not explicitly tested for steering, but attempting...")

    if not hasattr(probe, "model_") or probe.model_ is None:
        raise ValueError(f"Probe at {probe_path} is not fitted (no model_ attribute)")

    # Extract coefficient (shape: [1, hidden_dim] for binary classification)
    # Some probes might have coef_ directly, others might need access via model_
    if hasattr(probe.model_, "coef_"):
        coef = probe.model_.coef_  # numpy array [1, hidden_dim]
    elif hasattr(probe, "coef_"):
        coef = probe.coef_
    else:
        raise ValueError(f"Could not find coefficients in probe {probe_type_name}")

    # Un-normalize if scaler was used
    if hasattr(probe, "scaler_") and probe.scaler_ is not None:
        # StandardScaler transforms X as: X_scaled = (X - mean) / scale
        # The decision boundary w @ X_scaled = 0 becomes:
        # w @ ((X - mean) / scale) = 0
        # w / scale @ X = w @ mean / scale (constant, absorbed by intercept)
        # So the direction in original space is: w / scale
        scale = probe.scaler_.scale_  # [hidden_dim]
        coef = coef / scale
        log.info("Applied inverse scaling to probe direction")

    # Normalize to unit vector for consistent steering strength
    coef = coef / np.linalg.norm(coef)

    direction = torch.from_numpy(coef.astype(np.float32)).to(device)  # [1, hidden_dim]

    metadata = {
        "probe_path": str(probe_path),
        "probe_type": type(probe).__name__,
        "C": getattr(probe, "C", None),
        "normalized": probe.scaler_ is not None,
    }

    log.info(f"Loaded steering direction from {probe_path.name}, shape={direction.shape}")

    return direction, metadata


def load_apollo_probe_direction(
    detector_path: str | Path,
    layer: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """Load steering direction from an Apollo detector.pt file.

    Apollo detectors store directions per layer as a tensor.

    Args:
        detector_path: Path to the detector.pt file
        layer: Layer index to extract direction from
        device: Device to place the direction tensor on

    Returns:
        Tuple of (direction tensor [1, hidden_dim], probe metadata dict)
    """
    from src.probes import ApolloProbe

    detector_path = Path(detector_path)
    config_path = detector_path.parent / "cfg.yaml"

    probe = ApolloProbe.load(
        detector_path,
        config_path if config_path.exists() else None,
        auto_download=False,
    )

    # Apollo probes store directions as [n_layers, hidden_dim]
    # Find the index for the requested layer
    if layer not in probe.layers:
        raise ValueError(f"Layer {layer} not found in Apollo probe. Available layers: {probe.layers}")

    layer_idx = probe.layers.index(layer)
    direction = probe.directions[layer_idx : layer_idx + 1]  # [1, hidden_dim]

    # Normalize to unit vector
    direction = direction / torch.norm(direction)

    # Move to device
    direction = direction.to(device)

    metadata = {
        "probe_path": str(detector_path),
        "probe_type": "ApolloProbe",
        "layers": probe.layers,
        "selected_layer": layer,
        "normalized": probe.normalize,
    }

    log.info(f"Loaded Apollo steering direction from {detector_path.name}, layer={layer}")

    return direction, metadata


def build_probe_path(
    data_dir: Path | str,
    source_data: str,
    model_safe_name: str,
    probe_type: str,
    layer: int,
    pooling_strategy: str,
) -> Path:
    """Build the path to a probe file based on training parameters.

    Args:
        data_dir: Base data directory (e.g., data/)
        source_data: Dataset the probe was trained on (e.g., instructed_pairs)
        model_safe_name: Model identifier (e.g., qwen2.5_0.5b)
        probe_type: Probe type identifier (e.g., logistic_regression)
        layer: Layer index the probe was trained on
        pooling_strategy: Pooling strategy used (e.g., last, mean)

    Returns:
        Path to the probe .pkl file
    """
    data_dir = Path(data_dir)
    probe_dir = data_dir / source_data / f"probes-{probe_type}" / model_safe_name
    probe_file = probe_dir / f"probe_layer_{layer}_{pooling_strategy}.pkl"
    return probe_file


def build_apollo_probe_path(data_dir: Path | str, source_data: str) -> Path:
    """Build the path to an Apollo detector file.

    Apollo probes are stored in a different format (detector.pt) and location
    (data/apollo_detectors/{detector_name}/).

    Args:
        data_dir: Base data directory (e.g., data/)
        source_data: Apollo source data identifier (e.g., apollo_instructed_pairs)

    Returns:
        Path to the detector.pt file
    """
    data_dir = Path(data_dir)
    # Extract detector name from source_data (e.g., apollo_instructed_pairs -> instructed_pairs)
    detector_name = source_data.replace("apollo_", "", 1)
    return data_dir / "apollo_detectors" / detector_name / "detector.pt"


def is_apollo_probe(source_data: str) -> bool:
    """Check if the source_data refers to an Apollo probe."""
    return source_data.startswith("apollo_")


def create_steering_hook(
    direction: torch.Tensor,
    strength: float,
    layer_idx: int,
) -> tuple[callable, str]:
    """Create a forward hook that applies steering to activations.

    Args:
        direction: Unit direction vector [1, hidden_dim]
        strength: Steering coefficient
            - > 0: Add direction (steer towards the probe's positive class)
            - < 0: Subtract direction (steer away from positive class)
            - == 0: Project out direction (remove component entirely)
        layer_idx: Layer index (for logging)

    Returns:
        Tuple of (hook function, operation description string)
    """
    if strength == 0:
        # Project out: remove the component along direction
        # h_new = h - (h @ d^T) * d where d is unit direction

        def project_out_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output

            # Move direction to same device as hidden_state (for multi-GPU models)
            dir_local = direction.to(hidden_state.device, dtype=hidden_state.dtype)
            dir_t = dir_local.t()  # [hidden_dim, 1]

            # hidden_state: [batch, seq_len, hidden_dim]
            # direction: [1, hidden_dim]
            projection = hidden_state @ dir_t  # [batch, seq_len, 1]
            projected_out = hidden_state - projection * dir_local  # [batch, seq_len, hidden_dim]

            if isinstance(output, tuple):
                return (projected_out,) + output[1:]
            return projected_out

        operation = "project_out"
        return project_out_hook, operation

    else:
        # Add/subtract: h_new = h + strength * direction
        # Positive strength steers towards deceptive (probe's class 1)
        # Negative strength steers towards honest (probe's class 0)
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output

            # Move direction to same device as hidden_state (for multi-GPU models)
            dir_local = direction.to(hidden_state.device, dtype=hidden_state.dtype)

            # hidden_state: [batch, seq_len, hidden_dim]
            # direction: [1, hidden_dim] -> broadcast to all positions
            steered = hidden_state + strength * dir_local

            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered

        operation = f"add_{strength}" if strength > 0 else f"subtract_{abs(strength)}"
        return steering_hook, operation


def setup_steering(cfg, model) -> tuple[torch.utils.hooks.RemovableHandle | None, dict | None]:
    """Setup steering hook if enabled in config.

    Loads a probe direction and registers a forward hook to steer activations
    during generation. Also modifies cfg.paths.results_dir to include steering info.

    Args:
        cfg: Hydra DictConfig with optional 'steering' section
        model: LocalModel to register the hook on

    Returns:
        Tuple of (hook handle or None, steering config dict or None)
    """
    if not cfg.get("steering") or not cfg.steering.get("enabled", False):
        return None, None

    steering_cfg = cfg.steering
    log.info("=" * 80)
    log.info("STEERING MODE ENABLED")
    log.info(f"  Source data: {steering_cfg.source_data}")
    log.info(f"  Layer: {steering_cfg.layer}")
    log.info(f"  Pooling: {steering_cfg.pooling_strategy}")
    log.info(f"  Strength: {steering_cfg.strength}")
    log.info("=" * 80)

    # Handle Apollo probes vs standard probes
    if is_apollo_probe(steering_cfg.source_data):
        # Apollo probes use a different path structure and loading mechanism
        probe_path = build_apollo_probe_path(
            data_dir=Path(cfg.paths.root_dir) / "data",
            source_data=steering_cfg.source_data,
        )

        if not probe_path.exists():
            detector_name = steering_cfg.source_data.replace("apollo_", "", 1)
            raise FileNotFoundError(
                f"Apollo detector not found: {probe_path}\n"
                f"Download with: ApolloProbe.download_from_github('{detector_name}', ...)"
            )

        log.info(f"Loading Apollo detector from: {probe_path}")
        direction, probe_meta = load_apollo_probe_direction(probe_path, layer=steering_cfg.layer, device=model.device)
    else:
        # Standard probe path
        probe_path = build_probe_path(
            data_dir=Path(cfg.paths.root_dir) / "data",
            source_data=steering_cfg.source_data,
            model_safe_name=model.safe_name,
            probe_type=steering_cfg.probe_type,
            layer=steering_cfg.layer,
            pooling_strategy=steering_cfg.pooling_strategy,
        )

        if not probe_path.exists():
            raise FileNotFoundError(
                f"Steering probe not found: {probe_path}\n"
                f"Run train_probes with source_data={steering_cfg.source_data} first."
            )

        # Load probe direction
        direction, probe_meta = load_probe_direction(probe_path, device=model.device)

    # Register steering hook
    hook_handle = model.register_steering_hook(
        layer=steering_cfg.layer,
        direction=direction,
        strength=steering_cfg.strength,
    )

    # Store steering config for metadata
    steering_config = {
        "source_data": steering_cfg.source_data,
        "probe_type": steering_cfg.probe_type,
        "layer": steering_cfg.layer,
        "pooling_strategy": steering_cfg.pooling_strategy,
        "strength": steering_cfg.strength,
        "probe_path": str(probe_path),
        "probe_meta": probe_meta,
    }

    # Modify output directory to include steering info
    strength_str = f"{steering_cfg.strength}".replace(".", "_").replace("-", "neg")
    steering_suffix = f"_steered-{steering_cfg.source_data}-{strength_str}"
    original_results_dir = Path(cfg.paths.results_dir)
    output_dir_name = original_results_dir.name + steering_suffix
    cfg.paths.results_dir = str(original_results_dir.parent / output_dir_name)
    log.info(f"Steering output directory: {cfg.paths.results_dir}")

    return hook_handle, steering_config
