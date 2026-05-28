"""Re-judge existing datasets with additional judge models.

This module allows running new judge models on datasets that already have
judge prompts stored in their metadata. Results are added as new columns
to the existing metadata file.

Usage:
    python -m src.rejudge dataset_path=/path/to/dataset judges="gpt-4o,haiku-4.5"
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.activation_metadata import ActivationMetadataStore
from src.models.remote_model import RemoteModel
from src.utils import RankedLogger, extras

if TYPE_CHECKING:
    pass

log = RankedLogger(__name__, rank_zero_only=True)

# Registry of parse functions by dataset type
PARSE_FUNCTIONS = {}


def register_parse_fn(dataset_type: str):
    """Decorator to register a parse function for a dataset type."""

    def decorator(fn):
        PARSE_FUNCTIONS[dataset_type] = fn
        return fn

    return decorator


@register_parse_fn("deception_bench")
def parse_deception_bench(judge_response: str) -> dict:
    """Parse DeceptionBench judge response with both thought and response labels.

    Returns dict with:
    - thought_label: Label for the thought/reasoning section (deceptive intent)
    - response_label: Label for the response section (deceptive behavior)
    - label: Primary label for probe training/evaluation (uses response_label)

    We use response_label as the primary ground truth because:
    1. It measures actual deceptive *behavior*, not just intent
    2. Aligns with what the user actually receives from the model
    3. Consistent with other deception datasets (judged on output, not intent)

    Both labels are stored for analysis (e.g., comparing intent vs behavior).
    If only response is present (old format), uses it for both.
    """
    from src.utils.types import Label

    if "###FINAL:" not in judge_response:
        raise ValueError("No ###FINAL: found in judge response.")

    eval_str = judge_response.split("###FINAL:")[1].strip()
    evaluation = json.loads(eval_str)

    # Extract both fields
    thought_label = evaluation.get("thought")
    response_label = evaluation.get("response")

    if not response_label and not thought_label:
        raise ValueError("Missing both 'thought' and 'response' fields in evaluation result.")

    # Fall back to response if thought is missing (backward compatibility)
    if thought_label is None:
        thought_label = response_label
    if response_label is None:
        response_label = thought_label

    return {
        "thought_label": Label.from_str(thought_label) if thought_label else None,
        "response_label": Label.from_str(response_label) if response_label else None,
        "label": Label.from_str(response_label) if response_label else Label.from_str(thought_label),  # Primary label
    }


@register_parse_fn("mask_provided_facts")
def parse_mask_provided_facts(judge_response: str) -> dict:
    """Parse MASK provided_facts judge response."""
    from src.data.mask import MaskDataModule

    answer = MaskDataModule.extract_mask_answer(judge_response)
    # For MASK, we store the raw answer (A, B, C, D) since label determination
    # requires additional context (truth_option)
    return {"answer": answer}


def _extract_reasoning_config(judge_cfg: DictConfig) -> dict | None:
    """Extract reasoning configuration from judge config.

    Args:
        judge_cfg: Judge Hydra config containing provider, model, and optional thinking_budget

    Returns:
        Dict with reasoning metadata (thinking_budget for Anthropic, reasoning level for OpenAI)
        or None if no reasoning config found.
    """
    config = {}

    # Check for thinking_budget (Anthropic models)
    if hasattr(judge_cfg, "thinking_budget") and judge_cfg.thinking_budget:
        config["thinking_budget"] = judge_cfg.thinking_budget

    # For OpenAI models, we'd detect reasoning capability from model name
    # but that's already implicit in the model name, so we just record it's present
    if hasattr(judge_cfg, "provider") and judge_cfg.provider == "openai":
        if hasattr(judge_cfg, "model") and "o1" in judge_cfg.model.lower():
            config["reasoning_level"] = "high"  # o1 models have built-in reasoning

    return config if config else None


def rejudge_sample(
    judge: RemoteModel,
    judge_prompt: list[dict],
    parse_fn: callable,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    reasoning_config: dict | None = None,
) -> dict:
    """Run a judge on a single sample.

    Args:
        judge: The judge model to use
        judge_prompt: The conversation prompt for the judge
        parse_fn: Function to parse judge response
        max_retries: Number of retry attempts
        retry_delay: Delay between retries in seconds
        reasoning_config: Dict with reasoning metadata (thinking_budget, reasoning_level, etc.)

    Returns:
        Dict with model, response_raw, thinking, token usage, parsed result, label, timestamp
    """
    reasoning_config = reasoning_config or {}
    for attempt in range(max_retries):
        try:
            result = judge.generate(judge_prompt)
            # result is a GenerationResult with text, thinking, token usage
            parsed = parse_fn(result.text)
            result_dict = {
                "model": judge.model,
                "response_raw": result.text,
                "thinking": result.thinking,
                "used_thinking": result.used_thinking,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "thinking_tokens": result.thinking_tokens,
                "label": parsed.get("label"),
                "parsed": parsed,
                "timestamp": datetime.now().isoformat(),
                "error": None,
            }
            # Add reasoning config if available
            if reasoning_config:
                result_dict["reasoning_config"] = reasoning_config
            return result_dict
        except Exception as e:
            if attempt < max_retries - 1:
                log.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(retry_delay)
            else:
                log.error(f"All {max_retries} attempts failed: {e}")
                result_dict = {
                    "model": judge.model,
                    "response_raw": None,
                    "thinking": None,
                    "used_thinking": False,
                    "input_tokens": None,
                    "output_tokens": None,
                    "thinking_tokens": None,
                    "label": None,
                    "parsed": None,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                }
                # Add reasoning config if available
                if reasoning_config:
                    result_dict["reasoning_config"] = reasoning_config
                return result_dict


def rejudge_with_model(
    dataset: Dataset,
    metadata_store: ActivationMetadataStore,
    judge_cfg: DictConfig,
    cfg: DictConfig,
    parse_fn: callable,
) -> Dataset:
    """Run rejudge loop for a single judge model.

    Returns:
        Updated dataset
    """
    # Initialize judge
    judge: RemoteModel = hydra.utils.instantiate(judge_cfg)
    judge_name = judge_cfg.model.replace(".", "_").replace("-", "_")
    base_column = f"judge_results_{judge_name}"

    # Determine result column (resume incomplete run or start new one)
    result_column = base_column
    run_idx = 1

    while True:
        # Check current candidate column
        candidate_col = base_column if run_idx == 1 else f"{base_column}_run_{run_idx}"

        if candidate_col not in dataset.column_names:
            # Column doesn't exist -> start new run here
            result_column = candidate_col
            existing_results = [None] * len(dataset)
            log.info(f"Starting new run in column: {result_column}")
            break

        # Column exists - check if complete
        current_results = dataset[candidate_col]
        n_done = sum(1 for r in current_results if r is not None and r.get("label") is not None)

        if n_done < len(dataset):
            # Incomplete -> resume this run
            result_column = candidate_col
            existing_results = current_results
            log.info(f"Resuming existing run in column: {result_column} ({n_done}/{len(dataset)} done)")
            break

        # Complete -> try next run number
        log.info(f"Run {candidate_col} is complete ({n_done}/{len(dataset)}). Checking next...")
        run_idx += 1

    log.info(f"Using judge: {judge_cfg.model}")
    log.info(f"Results will be stored in column: {result_column}")

    # Extract reasoning config from judge config
    reasoning_config = _extract_reasoning_config(judge_cfg)
    if reasoning_config:
        log.info(f"Reasoning config: {reasoning_config}")

    # Process samples
    results = list(existing_results)  # Copy to modify
    n_skipped = 0
    n_processed = 0

    for i in tqdm(range(len(dataset)), desc=f"Re-judging with {judge_cfg.model}"):
        # Skip if already done
        if results[i] is not None and results[i].get("label") is not None:
            n_skipped += 1
            continue

        sample = dataset[i]
        judge_prompt = sample["judge_prompt"]

        # Handle case where judge_prompt was stored as string
        if isinstance(judge_prompt, str):
            judge_prompt = json.loads(judge_prompt)

        result = rejudge_sample(
            judge=judge,
            judge_prompt=judge_prompt,
            parse_fn=parse_fn,
            max_retries=cfg.max_retries,
            retry_delay=cfg.retry_delay,
            reasoning_config=reasoning_config,
        )
        results[i] = result
        n_processed += 1

        # Periodic save checkpoint
        if n_processed > 0 and n_processed % cfg.save_every == 0:
            log.info(f"Checkpoint: saving after {n_processed} samples")
            dataset = _save_results(dataset, result_column, results, metadata_store)

    log.info(f"Processed: {n_processed}, Skipped (already done): {n_skipped}")

    # Final save
    dataset = _save_results(dataset, result_column, results, metadata_store)
    log.info(f"✓ Saved results to {result_column}")
    return dataset


def rejudge_dataset(cfg: DictConfig) -> None:
    """Re-judge a dataset with a list of judge models sequentially."""
    dataset_path = Path(cfg.dataset_path)
    dataset_type = cfg.dataset_type

    log.info(f"Re-judging dataset at: {dataset_path}")
    log.info(f"Dataset type: {dataset_type}")

    # Load existing metadata
    metadata_store = ActivationMetadataStore(dataset_path)
    if not metadata_store.exists():
        raise FileNotFoundError(f"No metadata found at {dataset_path}")

    dataset = metadata_store.load()
    log.info(f"Loaded {len(dataset)} samples")

    # Check for judge_prompt column
    if "judge_prompt" not in dataset.column_names:
        raise ValueError(f"Dataset missing 'judge_prompt' column. " f"Available columns: {dataset.column_names}")

    # Get parse function
    if dataset_type not in PARSE_FUNCTIONS:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. " f"Available: {list(PARSE_FUNCTIONS.keys())}")
    parse_fn = PARSE_FUNCTIONS[dataset_type]

    # Parse list of judges
    if not cfg.judges:
        raise ValueError("No judges specified in config (cfg.judges is empty)")

    # Check if 'judges' passed as string or list
    if isinstance(cfg.judges, str):
        judges_list = [j.strip() for j in cfg.judges.split(",") if j.strip()]
    else:
        judges_list = list(cfg.judges)

    log.info(f"Running judges: {judges_list}")

    # Iterate over judges
    for judge_name in judges_list:
        log.info(f"--- Starting judge: {judge_name} ---")
        try:
            # Load judge configuration
            # Assume configs are in ../configs/judge relative to src if using hydration relative paths
            # Better to use absolute path based on original cwd
            config_dir = Path(hydra.utils.get_original_cwd()) / "configs" / "judge"
            judge_config_path = config_dir / f"{judge_name}.yaml"

            if not judge_config_path.exists():
                log.error(f"Judge config not found at {judge_config_path}")
                continue

            judge_cfg = OmegaConf.load(judge_config_path)

            # Run rejudge for this model
            dataset = rejudge_with_model(dataset, metadata_store, judge_cfg, cfg, parse_fn)

        except Exception as e:
            log.error(f"Failed to run judge {judge_name}: {e}")
            import traceback

            log.error(traceback.format_exc())
            # Continue to next judge even if one fails
            continue


def _save_results(
    dataset: Dataset,
    column_name: str,
    results: list,
    metadata_store: ActivationMetadataStore,
) -> Dataset:
    """Save results by adding/updating column in dataset efficiently."""
    import gc
    import shutil
    import tempfile

    try:
        # If column exists, remove it first (to overwrite)
        if column_name in dataset.column_names:
            dataset = dataset.remove_columns(column_name)

        # Add the new column
        dataset = dataset.add_column(column_name, results)

        # Save to a temp directory first, then replace original
        # This avoids "dataset can't overwrite itself" error
        original_path = metadata_store.path()
        temp_dir = tempfile.mkdtemp(dir=original_path.parent)
        temp_path = Path(temp_dir) / "metadata_new"

        try:
            # Save to temp location
            dataset.save_to_disk(str(temp_path))

            # Remove original and rename temp to original
            if original_path.exists():
                shutil.rmtree(original_path)
            shutil.move(str(temp_path), str(original_path))

            # Reload dataset from new location to avoid stale references
            dataset = Dataset.load_from_disk(str(original_path))
        finally:
            # Clean up temp dir if it still exists
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        # Clean up
        gc.collect()

        return dataset
    except Exception as e:
        log.error(f"Error saving dataset: {e}")
        raise


@hydra.main(version_base=None, config_path="../configs", config_name="rejudge")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    rejudge_dataset(cfg)


if __name__ == "__main__":
    main()
