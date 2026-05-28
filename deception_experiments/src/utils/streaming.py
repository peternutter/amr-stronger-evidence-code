"""Streaming utilities for memory-efficient activation extraction.

This module provides utilities to stream large activation tensors to disk
during extraction, avoiding memory accumulation that can cause OOM errors.
"""

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from datasets import Dataset

from src.utils import RankedLogger, extract_activations

log = RankedLogger(__name__, rank_zero_only=True)


class StreamingActivationWriter:
    """Writes activations to disk incrementally to avoid memory accumulation.

    Uses async I/O via ThreadPoolExecutor so disk writes don't block GPU compute.
    Saves each batch to temporary torch files, then consolidates into HuggingFace Dataset.
    """

    def __init__(self, output_dir: Path, layers: list[int], max_workers: int = 4):
        self.output_dir = output_dir
        self.layers = layers
        self.tmp_dir = output_dir / ".tmp_activations"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        # Create per-layer temp directories
        self.layer_dirs = {}
        for layer in layers:
            layer_dir = self.tmp_dir / f"layer_{layer}"
            layer_dir.mkdir(exist_ok=True)
            self.layer_dirs[layer] = layer_dir

        # Thread pool for async writes
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_futures = []

        # Track sample indices for metadata
        self.sample_offset = 0
        self.batch_count = 0

    def _save_batch_sync(
        self,
        layer_idx: int,
        batch_idx: int,
        activations: list[torch.Tensor],
        sample_indices: list[int],
        detection_masks: list[list[bool]],
    ) -> None:
        """Synchronous save - called in thread pool."""
        batch_file = self.layer_dirs[layer_idx] / f"batch_{batch_idx:05d}.pt"
        # Use torch.save for native bfloat16 support
        torch.save(
            {
                "activations": activations,  # List of tensors (variable length)
                "sample_indices": sample_indices,
                "detection_masks": detection_masks,
            },
            batch_file,
        )

    def write_batch(
        self,
        outputs: dict,
        batch_masks: list[list[bool]],
    ) -> None:
        """Write a batch's activations to disk asynchronously.

        Args:
            outputs: Dict with 'layer_activations', 'attention_mask', etc.
            batch_masks: Detection masks for this batch
        """
        layer_acts = outputs.get("layer_activations", {})
        attention_mask = outputs["attention_mask"]
        batch_size = attention_mask.shape[0]

        sample_indices = list(range(self.sample_offset, self.sample_offset + batch_size))

        for layer_idx in self.layers:
            if layer_idx not in layer_acts:
                continue

            # Extract per-sample activations (unpadded)
            batch_activations = extract_activations(
                {"hidden_states": [layer_acts[layer_idx]], "attention_mask": attention_mask},
                layer=0,
                pooling_strategy=None,
                attention_mask=attention_mask,
            )

            # Keep as tensors (no numpy conversion needed)
            activations_tensors = []
            for acts in batch_activations:
                # Ensure on CPU
                if isinstance(acts, torch.Tensor):
                    acts = acts.cpu()
                activations_tensors.append(acts)

            # Submit async write
            future = self.executor.submit(
                self._save_batch_sync,
                layer_idx,
                self.batch_count,
                activations_tensors,
                sample_indices,
                batch_masks,
            )
            self.pending_futures.append(future)

        self.sample_offset += batch_size
        self.batch_count += 1

    def wait_for_pending(self) -> None:
        """Wait for all pending writes to complete."""
        for future in self.pending_futures:
            future.result()  # Raises if any write failed
        self.pending_futures.clear()

    def consolidate_layer(self, layer_idx: int, layer_base: dict) -> Dataset:
        """Consolidate all batch files for a layer into a HuggingFace Dataset."""
        layer_dir = self.layer_dirs[layer_idx]
        batch_files = sorted(layer_dir.glob("batch_*.pt"))

        all_activations = []
        all_sample_indices = []
        all_detection_masks = []

        for batch_file in batch_files:
            data = torch.load(batch_file, weights_only=False)

            all_activations.extend(data["activations"])
            all_sample_indices.extend(data["sample_indices"])
            all_detection_masks.extend(data["detection_masks"])

        # Compute completion_start_index from detection_masks (count of False values = prompt tokens)
        all_completion_start_indices = [sum(1 for m in mask if not m) for mask in all_detection_masks]

        # Build dataset
        layer_dataset = {key: list(value) for key, value in layer_base.items()}
        layer_dataset["activations"] = all_activations
        layer_dataset["completion_start_index"] = all_completion_start_indices

        return Dataset.from_dict(layer_dataset)

    def cleanup(self) -> None:
        """Clean up temporary files and shut down executor."""
        self.executor.shutdown(wait=True)
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
