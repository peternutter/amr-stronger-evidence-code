"""
Utility functions for extracting, saving and loading model activations/features.
"""

from pathlib import Path
from typing import Literal

import torch

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extract_activations(
    outputs: dict,
    layer: str | int = "last_hidden_state",
    pooling_strategy: Literal["last", "min", "max", "mean"] | None = None,
    attention_mask: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Extract activations/features from the model outputs.

    If `attention_mask` is provided, the pooling strategy will be applied
    to the non-masked tokens. Otherwise, the pooling strategy will be applied to all tokens.

    Args:
        outputs: The model outputs.

        layer (str | int, optional):
            The layer from which to extract activations.

            If a string is provided, it will be used as the name of the attribute to extract
            from the outputs (e.g. `last_hidden_state` or `logits`).

            If an integer is provided, it will be used as the index of the `hidden_states`
            attribute of the outputs.

        pooling_strategy (str | None, optional):
            The strategy to use for reducing the hidden states over the sequence length.
            Must be either "last", "min", "max", "mean" or None.
            If None, returns a list of tensors with trimmed sequences (padding removed).

        attention_mask (torch.Tensor of shape (batch_size, sequence_length), optional):
            The attention mask used to generate the outputs. This is used for the pooling strategy
            and for trimming padding when pooling_strategy is None.

    Returns:
        list[torch.Tensor]:
            If pooling_strategy is provided, returns a tensor of shape (batch_size, feature_size).
            If pooling_strategy is None, returns a list of tensors, each with shape
            (sequence_length, feature_size) where sequence_length is trimmed to remove padding.
    """
    # Extract activations from the specified layer
    if isinstance(layer, str):
        hidden_state = getattr(outputs, layer)
    elif isinstance(layer, int):
        if isinstance(outputs, dict):
            hidden_state = outputs["hidden_states"][layer]
        else:
            hidden_state = outputs.hidden_states[layer]
    else:
        raise ValueError(f"Invalid feature layer: {layer}")

    if hidden_state is None:
        raise ValueError(f"Model outputs do not contain the feature layer: {layer}")
    if torch.isnan(hidden_state).any():
        raise ValueError("Extracted features contain NaN values.")

    # Clone hidden_state
    # This avoids RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
    hidden_state = hidden_state.clone()

    if attention_mask is None:
        attention_mask = torch.ones(hidden_state.shape[:2], dtype=torch.int)

    # Apply pooling strategy over sequence length
    if pooling_strategy is None:
        # Return list of tensors with trimmed sequences (no padding)
        trimmed_sequences = []
        for i in range(hidden_state.shape[0]):  # Iterate over batch
            mask = attention_mask[i].bool()

            # If all tokens are masked, skip this sequence
            if not mask.any():
                log.warning("All tokens are masked for a sequence; returning empty tensor for this sequence.")
                continue

            # Find first and last non-masked positions
            non_masked_indices = torch.where(mask)[0]
            start_idx = non_masked_indices[0].item()
            end_idx = non_masked_indices[-1].item() + 1  # +1 for exclusive end

            # Extract only the non-padded tokens
            trimmed_seq = hidden_state[i, start_idx:end_idx, :]  # shape: (sequence_i_length, feature_size)
            trimmed_sequences.append(trimmed_seq)

        return trimmed_sequences  # List of tensors with shape (sequence_i_length, feature_size)

    pooled_states = []
    for i in range(hidden_state.shape[0]):
        mask = attention_mask[i].bool()

        if not mask.any():
            # If all tokens are masked, use zeros
            pooled_states.append(torch.zeros(hidden_state.shape[2], device=hidden_state.device))
            continue

        if pooling_strategy == "last":
            # Extract features of the last non-masked tokens (handle left/right padding)
            non_masked_indices = torch.where(mask)[0]
            last_idx = non_masked_indices[-1].item()
            pooled_states.append(hidden_state[i, last_idx, :])
        elif pooling_strategy == "min":
            # Compute min of features over non-masked tokens (handle left/right padding)
            non_masked_tokens = hidden_state[i, mask, :]
            pooled_states.append(non_masked_tokens.min(dim=0).values)
        elif pooling_strategy == "max":
            # Compute max of features over non-masked tokens (handle left/right padding)
            non_masked_tokens = hidden_state[i, mask, :]
            pooled_states.append(non_masked_tokens.max(dim=0).values)
        elif pooling_strategy == "mean":
            # Compute mean of features over non-masked tokens (handle left/right padding)
            non_masked_tokens = hidden_state[i, mask, :]
            pooled_states.append(non_masked_tokens.mean(dim=0))
        else:
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}")

    return pooled_states  # List of tensors with shape: (feature_size)


def load_activations(
    activation_dir: Path,
    layer: str | int | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Load activation files from directory.

    Args:
        activation_dir: Directory containing activation files
        layer: Specific layer to load (e.g., -1, "-1", 0, "0"), or None to load all
               Negative indices count from the end (e.g., -1 = last layer)

    Returns:
        Dictionary mapping layer names to {"activations": tensor, "labels": tensor}
    """
    activation_files = sorted(activation_dir.glob("layer_*.pt"))

    if not activation_files:
        raise ValueError(f"No activation files found in {activation_dir}")

    # Convert layer to string and handle negative indices
    target_layer = None
    if layer is not None:
        layer_str = str(layer)

        # If negative index, convert to positive
        if layer_str.startswith("-") or (isinstance(layer, int) and layer < 0):
            # Get all layer numbers
            layer_numbers = []
            for f in activation_files:
                layer_num = int(f.stem.replace("layer_", ""))
                layer_numbers.append(layer_num)

            # Convert negative index to positive
            if isinstance(layer, int):
                layer_idx = layer
            else:
                layer_idx = int(layer_str)

            if layer_idx < 0:
                # Python-style negative indexing
                num_layers = len(layer_numbers)
                positive_idx = num_layers + layer_idx
                if positive_idx < 0 or positive_idx >= num_layers:
                    raise ValueError(f"Layer index {layer_idx} out of range for {num_layers} layers")
                target_layer = str(sorted(layer_numbers)[positive_idx])
                log.info(f"Converting layer index {layer_idx} to layer {target_layer} ({num_layers} layers total)")
            else:
                target_layer = layer_str
        else:
            target_layer = layer_str

    data = {}
    for file_path in activation_files:
        # Extract layer name from filename (e.g., "layer_0.pt" -> "0")
        layer_name = file_path.stem.replace("layer_", "")

        # Skip if specific layer requested and this isn't it
        if target_layer is not None and layer_name != target_layer:
            continue

        log.info(f"Loading {file_path.name}...")
        loaded = torch.load(file_path, map_location="cpu")

        data[layer_name] = {
            "activations": loaded["activations"],
            "labels": loaded["labels"],
            "model_name": loaded.get("model_name", "unknown"),
        }

    if not data:
        available_layers = [f.stem.replace("layer_", "") for f in activation_files]
        raise ValueError(
            f"No data loaded. Layer {layer} (target: {target_layer}) not found in {activation_dir}\n"
            f"Available layers: {', '.join(sorted(available_layers, key=int))}"
        )

    return data
