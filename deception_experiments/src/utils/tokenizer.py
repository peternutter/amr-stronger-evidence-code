"""Shared tokenizer utilities for model family detection and caching."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


@functools.lru_cache(maxsize=8)
def get_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """Get cached tokenizer for model family.

    Args:
        model_name: Model name (can use underscores or slashes).

    Returns:
        Cached tokenizer instance for the model family.
    """
    from transformers import AutoTokenizer

    # Normalize model name
    normalized = model_name.replace("_", "/")

    # Map to family-specific tokenizer
    if "Llama" in normalized:
        tokenizer_name = "meta-llama/Llama-3.3-70B-Instruct"
    elif "Qwen" in normalized:
        tokenizer_name = "Qwen/Qwen2.5-72B-Instruct"
    else:
        tokenizer_name = normalized

    return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
