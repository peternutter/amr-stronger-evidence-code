#!/usr/bin/env python3
"""Simple script to explore dataset structure and sample content.

This is a utility script for debugging/exploration - not part of the main pipeline.

Usage:
    python scripts/explore_dataset.py --data instructed_pairs --model qwen2.5/0.5b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from lightning import LightningDataModule


def _format_value(value, max_str_len: int) -> str:
    """Internal helper to format a single value."""

    # 1. Handle Tensors and numpy arrays
    if isinstance(value, np.ndarray | torch.Tensor):
        details = [f"shape={list(value.shape)}"]
        if hasattr(value, "dtype"):
            # Clean up dtype name (e.g., 'torch.float32' -> 'float32')
            details.append(f"dtype={str(value.dtype).split('.')[-1]}")
        if hasattr(value, "device"):
            details.append(f"device='{value.device}'")
        return f"<{type(value).__name__} {', '.join(details)}>"

    # 2. Handle Strings
    if isinstance(value, str):
        if len(value) > max_str_len:
            return f"'{value[:max_str_len-40]}[...]{value[-40:]}' [len={len(value)}]"
        return f"'{value}' [len={len(value)}]"

    # 3. Handle all other types
    return repr(value) + f" [type={type(value).__name__}]"


def pretty_format(data, indent=0, max_str_len=120) -> str:
    """Recursively formats a nested data structure into a string."""
    indent_str = "  " * indent
    next_indent_str = "  " * (indent + 1)
    output = []

    if isinstance(data, dict):
        output.append(f"{indent_str}{{")
        for k, v in data.items():
            if isinstance(v, dict | list | tuple):
                output.append(f"{next_indent_str}{repr(k)}:")
                output.append(pretty_format(v, indent + 1, max_str_len))
            else:
                formatted_v = _format_value(v, max_str_len)
                output.append(f"{next_indent_str}{repr(k)}: {formatted_v},")
        output.append(f"{indent_str}}}")

    elif isinstance(data, list | tuple):
        brackets = ("[", "]") if isinstance(data, list) else ("(", ")")
        output.append(f"{indent_str}{brackets[0]}")
        for item in data:
            if isinstance(item, dict | list | tuple):
                output.append(pretty_format(item, indent + 1, max_str_len))
            else:
                output.append(f"{next_indent_str}{_format_value(item, max_str_len)},")
        output.append(f"{indent_str}{brackets[1]}")

    else:
        output.append(f"{indent_str}{_format_value(data, max_str_len)}")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Explore dataset structure")
    parser.add_argument("--data", type=str, required=True, help="Data config name (e.g., 'instructed_pairs')")
    parser.add_argument("--model", type=str, default="qwen2.5/0.5b", help="Model config for tokenizer")
    args = parser.parse_args()

    # Initialize hydra config for data
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg: DictConfig = hydra.compose(
            config_name="calculate_activations",
            overrides=[f"data={args.data}", f"model={args.model}"],
        )

    print(f"Data config:\n{OmegaConf.to_yaml(cfg.data)}")

    # Load just the tokenizer (don't load the full model)
    model_path = cfg.model.model_name_or_path
    print(f"\nLoading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Instantiate datamodule
    print(f"\nInstantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Set tokenizer on datamodule
    if hasattr(datamodule, "tokenizer") and datamodule.tokenizer is None:
        datamodule.tokenizer = tokenizer

    print(f"Datamodule has completions: {datamodule.has_completions}")
    print(f"Datamodule has labels: {datamodule.has_labels}")
    print(f"Datamodule has judge prompts: {datamodule.has_judge_prompts}")

    def print_column_and_types(include_completions: bool = False):
        suffix = " with completions" if include_completions else ""

        print("=" * 40)
        print(f"Preparing and setting up datamodule{suffix}...")

        datamodule.include_completions = include_completions
        datamodule.prepare_data()
        datamodule.setup(stage="predict")

        print(f"Dataset sample:\n{pretty_format(datamodule.dataset[0])}")

        dataloader = datamodule.predict_dataloader()
        batch = next(iter(dataloader))
        print(f"Sample batch from predict dataloader{suffix}:\n{pretty_format(batch)}")

    print_column_and_types(include_completions=False)
    if datamodule.has_completions:
        print_column_and_types(include_completions=True)


if __name__ == "__main__":
    main()
