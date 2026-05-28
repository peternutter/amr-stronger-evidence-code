"""
Wrapper for ApolloResearch detector.pt files.

This module provides a probe class that loads pre-trained detectors from the
ApolloResearch/deception-detection repository and provides an sklearn-compatible
interface for use with the existing evaluation infrastructure.

The ApolloResearch LogisticRegressionDetector stores:
- layers: List of model layers (e.g., [22])
- directions: Probe weights tensor [n_layers, embedding_dim]
- scaler_mean, scaler_scale: Normalization tensors
- normalize: Boolean flag
- reg_coeff: Regularization coefficient (for reference)

Usage:
    from src.probes import ApolloProbe

    probe = ApolloProbe.load("path/to/detector.pt", "path/to/cfg.yaml")

    # Use like any sklearn probe
    probs = probe.predict_proba(activations)  # [n_samples, 2]
    preds = probe.predict(activations)        # [n_samples]
"""

import pickle
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin

# GitHub raw content URLs for Apollo detectors
APOLLO_GITHUB_BASE = "https://raw.githubusercontent.com/ApolloResearch/deception-detection/main/example_results"
AVAILABLE_DETECTORS = ["instructed_pairs", "roleplaying", "descriptive", "followup", "sae_rp"]


class ApolloProbe(BaseEstimator, ClassifierMixin):
    """
    Wrapper for ApolloResearch detector.pt files.

    Loads a pre-trained detector from the ApolloResearch repository and provides
    an sklearn-compatible interface for evaluation.

    The detector uses a linear probe (direction vector) with optional normalization
    to score activations. Scores are converted to probabilities using sigmoid.

    Attributes:
        layers: List of layer indices the detector was trained on
        directions: Weight tensor [n_layers, embedding_dim]
        scaler_mean: Mean for normalization [n_layers, embedding_dim]
        scaler_scale: Scale for normalization [n_layers, embedding_dim]
        normalize: Whether to apply normalization
        reg_coeff: Regularization coefficient used during training
        config: Optional config dict from cfg.yaml
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        layers: list[int] | None = None,
        directions: torch.Tensor | None = None,
        scaler_mean: torch.Tensor | None = None,
        scaler_scale: torch.Tensor | None = None,
        normalize: bool = True,
        reg_coeff: float = 1.0,
        config: dict[str, Any] | None = None,
    ):
        self.layers = layers or []
        self.directions = directions
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale
        self.normalize = normalize
        self.reg_coeff = reg_coeff
        self.config = config or {}
        self.classes_ = np.array([0, 1])  # Binary classification

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    @classmethod
    def download_from_github(
        cls,
        detector_name: str,
        output_dir: str | Path,
        force: bool = False,
    ) -> tuple[Path, Path]:
        """
        Download a detector from the ApolloResearch GitHub repository.

        Args:
            detector_name: Name of detector (e.g., 'instructed_pairs', 'roleplaying')
            output_dir: Directory to save files to
            force: If True, re-download even if files exist

        Returns:
            Tuple of (detector_path, config_path)

        Raises:
            ValueError: If detector_name is not available
        """
        if detector_name not in AVAILABLE_DETECTORS:
            raise ValueError(f"Unknown detector: {detector_name}. " f"Available: {AVAILABLE_DETECTORS}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        detector_path = output_dir / "detector.pt"
        config_path = output_dir / "cfg.yaml"

        # Download detector.pt
        if force or not detector_path.exists():
            detector_url = f"{APOLLO_GITHUB_BASE}/{detector_name}/detector.pt"
            print(f"Downloading {detector_url}...")
            urllib.request.urlretrieve(detector_url, detector_path)
            print(f"  Saved to {detector_path}")

        # Download cfg.yaml
        if force or not config_path.exists():
            config_url = f"{APOLLO_GITHUB_BASE}/{detector_name}/cfg.yaml"
            print(f"Downloading {config_url}...")
            urllib.request.urlretrieve(config_url, config_path)
            print(f"  Saved to {config_path}")

        return detector_path, config_path

    @classmethod
    def load(
        cls,
        detector_path: str | Path,
        config_path: str | Path | None = None,
        detector_name: str | None = None,
        auto_download: bool = True,
    ) -> "ApolloProbe":
        """
        Load a pre-trained Apollo detector from a .pt file.

        If the detector file doesn't exist and auto_download is True, will attempt
        to download from GitHub.

        Args:
            detector_path: Path to detector.pt file (pickle format)
            config_path: Optional path to cfg.yaml with training config
            detector_name: Name of detector for auto-download (e.g., 'instructed_pairs', 'roleplaying')
            auto_download: If True, download from GitHub if file doesn't exist

        Returns:
            Loaded ApolloProbe instance

        Raises:
            FileNotFoundError: If detector file doesn't exist and can't be downloaded
            ValueError: If detector format is invalid
        """
        detector_path = Path(detector_path)

        # Auto-download if file doesn't exist
        if not detector_path.exists() and auto_download:
            if detector_name is None:
                raise FileNotFoundError(
                    f"Detector file not found: {detector_path}\n"
                    f"Specify detector_name to auto-download from GitHub.\n"
                    f"Available detectors: {AVAILABLE_DETECTORS}"
                )
            if detector_name not in AVAILABLE_DETECTORS:
                raise ValueError(f"Unknown detector: {detector_name}. " f"Available: {AVAILABLE_DETECTORS}")
            print(f"Detector not found locally, downloading '{detector_name}' from GitHub...")
            cls.download_from_github(detector_name, detector_path.parent)

        if not detector_path.exists():
            raise FileNotFoundError(f"Detector file not found: {detector_path}")

        # Load detector (pickle format used by ApolloResearch)
        with open(detector_path, "rb") as f:
            data = pickle.load(f)

        # Validate required fields
        required_fields = ["layers", "directions"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Detector missing required field: {field}")

        # Load optional config
        config = {}
        if config_path is not None:
            config_path = Path(config_path)
            # Auto-download config if detector was downloaded
            if not config_path.exists() and auto_download:
                detector_name = config_path.parent.name
                if detector_name in AVAILABLE_DETECTORS:
                    config_url = f"{APOLLO_GITHUB_BASE}/{detector_name}/cfg.yaml"
                    print(f"Downloading config from {config_url}...")
                    urllib.request.urlretrieve(config_url, config_path)

            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)

        # IMPORTANT: Do NOT apply normalization when using Apollo's probe on our activations.
        # Apollo's scaler was fitted on THEIR activation extraction pipeline. Our activations
        # have different statistics (different tokenization, padding, etc.), so applying their
        # scaler produces a systematic offset that shifts all scores negative.
        #
        # Without normalization:
        #   - Honest scores: ~-0.1 (negative)
        #   - Deceptive scores: ~+0.6 (positive)
        #   - Threshold of 0 works correctly
        #
        # With normalization (wrong for our data):
        #   - All scores: ~-7 (strongly negative)
        #   - Threshold of 0 catches nothing
        return cls(
            layers=data["layers"],
            directions=data["directions"],
            scaler_mean=data.get("scaler_mean"),
            scaler_scale=data.get("scaler_scale"),
            normalize=False,  # Don't apply their scaler to our activations
            reg_coeff=data.get("reg_coeff", 1.0),
            config=config,
        )

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision scores by projecting activations onto the direction.

        For multi-layer detectors, scores are averaged across layers.

        Args:
            X: Feature matrix [n_samples, n_features] or [n_samples, n_layers, n_features]

        Returns:
            Decision scores [n_samples]
        """
        if self.directions is None:
            raise ValueError("Probe has no directions. Was it loaded correctly?")

        # Convert to tensor for computation
        device = self.directions.device
        dtype = self.directions.dtype

        X_tensor = torch.tensor(X, dtype=dtype, device=device)

        # Handle different input shapes
        # Expected: [n_samples, n_features] for single layer
        # Or: [n_samples, n_layers, n_features] for multi-layer
        if X_tensor.ndim == 2:
            # Single layer case: [n_samples, n_features]
            n_layers = len(self.layers)

            if n_layers == 1:
                # Reshape to [n_samples, 1, n_features] for consistency
                X_tensor = X_tensor.unsqueeze(1)
            else:
                # Activations should already include all layers
                raise ValueError(
                    f"Detector expects {n_layers} layers but got 2D input. "
                    f"Expected shape: [n_samples, {n_layers}, n_features]"
                )
        elif X_tensor.ndim == 3:
            n_layers_in = X_tensor.shape[1]
            if n_layers_in != len(self.layers):
                raise ValueError(f"Layer mismatch: detector expects {len(self.layers)} layers, " f"got {n_layers_in}")
        else:
            raise ValueError(f"Invalid input shape: {X_tensor.shape}")

        # Apply normalization if enabled
        if self.normalize and self.scaler_mean is not None and self.scaler_scale is not None:
            X_tensor = (X_tensor - self.scaler_mean.to(device)) / self.scaler_scale.to(device)

        # Compute scores: dot product with direction, averaged over layers
        # directions: [n_layers, n_features]
        # X_tensor: [n_samples, n_layers, n_features]
        # Result: [n_samples, n_layers] -> mean -> [n_samples]
        scores_by_layer = torch.einsum("slf,lf->sl", X_tensor, self.directions)
        scores = scores_by_layer.mean(dim=-1)

        return scores.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix [n_samples, n_features]

        Returns:
            Predicted labels [n_samples]
        """
        scores = self.decision_function(X)
        # Apollo detectors: positive score = deceptive
        return (scores > 0).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using sigmoid transformation.

        Args:
            X: Feature matrix [n_samples, n_features]

        Returns:
            Class probabilities [n_samples, 2] where [:, 1] is P(deceptive)
        """
        scores = self.decision_function(X)

        # Convert to probabilities using sigmoid
        # Positive score -> higher probability of deceptive (class 1)
        proba_positive = 1 / (1 + np.exp(-scores))
        proba_negative = 1 - proba_positive

        return np.column_stack([proba_negative, proba_positive])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fitting is not supported for pre-trained Apollo probes.

        This method exists for sklearn API compatibility but raises an error
        since Apollo probes are evaluation-only.

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError(
            "ApolloProbe is evaluation-only. Use ApolloProbe.load() to load "
            "a pre-trained detector, or use LogisticRegressionProbe for training."
        )

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            "layers": self.layers,
            "directions": self.directions,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "normalize": self.normalize,
            "reg_coeff": self.reg_coeff,
            "config": self.config,
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @property
    def safe_name(self) -> str:
        """Return a safe name for file paths."""
        return "apollo_probe"

    def __repr__(self) -> str:
        return f"ApolloProbe(layers={self.layers}, " f"normalize={self.normalize}, " f"reg_coeff={self.reg_coeff})"
