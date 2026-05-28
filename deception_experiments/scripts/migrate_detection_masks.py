#!/usr/bin/env python3
"""Migrate detection masks for activation datasets.

This unified script can:
1. Process a specific dataset (provide dataset path)
2. Process all datasets (--all flag)
3. Preview changes without applying them (--dry-run flag)

Supports both prefill (instructed_pairs, roleplaying/offpolicy_train) and
generation (roleplaying/plain) datasets.
"""

import argparse
import random
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
from datasets import load_from_disk
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig
from src.utils import RankedLogger, default_collate_fn, extras
from tqdm import tqdm

rank_zero_only.rank = 0

log = RankedLogger(__name__, rank_zero_only=True)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def get_model_map():
    """Build mapping from safe_name to model config path."""
    model_map = {}
    model_configs_dir = CONFIGS_DIR / "model"

    for yaml_path in model_configs_dir.rglob("*.yaml"):
        try:
            content = yaml_path.read_text()
            # Extract safe_name using regex
            match = re.search(r"^safe_name:\s*(.+)$", content, re.MULTILINE)
            if match:
                safe_name = match.group(1).strip()
                # Config path relative to configs/model, without .yaml
                rel_path = yaml_path.relative_to(model_configs_dir).with_suffix("")
                config_name = str(rel_path)
                model_map[safe_name] = config_name
        except Exception as e:
            print(f"Warning: Failed to parse {yaml_path}: {e}")

    return model_map


def infer_dataset_config_args(dataset_dir_name):
    """Infer Hydra arguments for a given dataset directory name."""
    if dataset_dir_name == "instructed_pairs":
        return "data=instructed_pairs"

    if dataset_dir_name == "alpaca":
        return "data=alpaca"

    # Handle instructed stress test variants: instructed_sarcasm, instructed_alien, etc.
    if dataset_dir_name.startswith("instructed_"):
        variant = dataset_dir_name.replace("instructed_", "")
        # These are stored under configs/data/instructed/<variant>.yaml
        config_path = CONFIGS_DIR / "data" / "instructed" / f"{variant}.yaml"
        if config_path.exists():
            return f"data=instructed/{variant}"
        # Fallback: try as a direct variant name
        return f"data=instructed/.base prompt_variant={variant}"

    parts = dataset_dir_name.split("-")

    if len(parts) < 2:
        return None

    dataset_type = parts[0]

    if dataset_type == "mask":
        # Format: mask-<subset>-<variant>
        if len(parts) >= 3:
            subset = parts[1]
            variant = "-".join(parts[2:])

            # Special case for "labeled" which uses pressure config + override
            if variant == "labeled":
                return f"data=mask/{subset}/.base"

            # Special case for "None" variant (provided_facts)
            if variant == "None":
                return f"data=mask/{subset}"

            # General case: try to find matching config
            config_path = CONFIGS_DIR / "data" / "mask" / subset / f"{variant}.yaml"
            if config_path.exists():
                return f"data=mask/{subset}/{variant}"

            # Fallback for belief variants
            return f"data=mask/{subset}/pressure prompt_variant={variant}"

    elif dataset_type == "roleplaying":
        # Format: roleplaying-<variant>
        if len(parts) >= 2:
            variant = "-".join(parts[1:])
            config_path = CONFIGS_DIR / "data" / "roleplaying" / f"{variant}.yaml"
            if config_path.exists():
                return f"data=roleplaying/{variant}"
            return f"data=roleplaying/plain prompt_variant={variant}"

    elif dataset_type == "deception_bench":
        # Format: deception_bench-<variant>
        if len(parts) >= 2:
            variant = "-".join(parts[1:])
            config_path = CONFIGS_DIR / "data" / "deception_bench" / f"{variant}.yaml"
            if config_path.exists():
                return f"data=deception_bench/{variant}"
            return f"data=deception_bench/L1_other prompt_variant={variant}"

    elif dataset_type == "instructed_pairs":
        return "data=instructed_pairs"

    return None


def run_single_migration(script_path, data_args, model_arg, dry_run=False):
    """Run migration for a single dataset configuration by calling the original processing logic."""
    # Build command to call the script with Hydra args
    cmd = [sys.executable, str(script_path), *data_args.split()]
    if model_arg:
        cmd.append(model_arg)
    if dry_run:
        cmd.append("--dry-run")

    cmd_str = " ".join(cmd)

    if dry_run:
        print(f"[DRY-RUN] Would run: {cmd_str}")
        return (True, cmd_str, None)

    print(f"Running: {cmd_str}")
    try:
        subprocess.check_call(cmd, stderr=subprocess.STDOUT)
        return (True, cmd_str, None)
    except subprocess.CalledProcessError as e:
        error_msg = f"Exit code {e.returncode}"
        print(f"FAILED: {cmd_str} - {error_msg}")
        return (False, cmd_str, error_msg)
    except Exception as e:
        error_msg = str(e)
        print(f"FAILED: {cmd_str} - {error_msg}")
        return (False, cmd_str, error_msg)


def process_all_datasets(dry_run=False, jobs=20, model_filter=None):
    """Discover and process all datasets with activations.

    Args:
        dry_run: If True, preview changes without applying.
        jobs: Number of parallel jobs.
        model_filter: If provided, only process datasets for this model safe_name (substring match).
    """
    if not DATA_DIR.exists():
        print(f"Error: Data directory not found at {DATA_DIR}")
        return False

    print("Scanning for datasets...")
    model_map = get_model_map()
    print(f"Found {len(model_map)} model configs.")

    # Get path to this script for subprocess calls
    script_path = Path(__file__).resolve()

    tasks = []

    # Scan data directory
    for dataset_path in sorted(DATA_DIR.iterdir()):
        if not dataset_path.is_dir():
            continue

        dataset_name = dataset_path.name
        responses_dir = dataset_path / "responses"

        if not responses_dir.exists():
            continue

        # Check for models with activations
        for model_path in responses_dir.iterdir():
            if not model_path.is_dir():
                continue

            model_safe_name = model_path.name

            # Check for layers or metadata
            layers = list(model_path.glob("layer_*"))
            metadata = model_path / "metadata"
            if not layers and not metadata.exists():
                continue

            # Filter by model if specified
            if model_filter and model_filter.lower() not in model_safe_name.lower():
                continue

            # We found a dataset + model with activations
            data_args = infer_dataset_config_args(dataset_name)
            if not data_args:
                print(f"Warning: Could not infer config for {dataset_name}")
                continue

            # Find model config
            model_config = model_map.get(model_safe_name)
            if not model_config:
                print(f"Warning: Could not find config for model '{model_safe_name}'")
                # If default model (Qwen2.5-0.5B-Instruct), allow without model arg
                if "Qwen2.5-0.5B-Instruct" in model_safe_name:
                    model_arg = ""
                else:
                    print(f"Skipping {dataset_name}/{model_safe_name} due to missing model config mapping.")
                    continue
            else:
                model_arg = f"model={model_config}"

            tasks.append((data_args, model_arg))

    print(f"Found {len(tasks)} datasets to process.")

    if not tasks:
        return True

    # Run tasks and collect results
    results = []
    if dry_run:
        for data_args, model_arg in tasks:
            results.append(run_single_migration(script_path, data_args, model_arg, dry_run=True))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(run_single_migration, script_path, data_args, model_arg)
                for data_args, model_arg in tasks
            ]
            for future in futures:
                results.append(future.result())

    # Summarize results
    succeeded = [(cmd, err) for success, cmd, err in results if success]
    failed = [(cmd, err) for success, cmd, err in results if not success]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total datasets: {len(results)}")
    print(f"Succeeded:     {len(succeeded)}")
    print(f"Failed:        {len(failed)}")

    if failed:
        print("\n" + "-" * 60)
        print("FAILED DATASETS:")
        print("-" * 60)
        for cmd, err in failed:
            print(f"  Command: {cmd}")
            print(f"    Error: {err}")
            print()
        return False
    else:
        print("\nAll datasets processed successfully!")
        return True


@hydra.main(version_base=None, config_path="../configs", config_name="calculate_activations")
def process_single_dataset(cfg: DictConfig):
    """Process detection masks for a single dataset."""

    def get_trimmed_tokens(metadata_dataset, sample_idx):
        """Helper to get trimmed tokens for a sample (already sliced in new format)."""
        input_ids = metadata_dataset[sample_idx]["input_ids"]

        if hasattr(input_ids, "tolist"):
            return input_ids.tolist()
        else:
            return list(input_ids)

    def _calculate_prompt_length(tokenizer, conversation):
        """Calculate prompt length matching default_collate_fn logic.

        Handles stripping EOT suffixes for assistant prefixes.
        """
        # Apply chat template
        has_assistant = conversation and conversation[-1].get("role") == "assistant"

        prompt_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=not has_assistant,
        )

        # Strip EOT suffix if last message is assistant with content (answer_prefix)
        last_is_assistant_prefix = has_assistant and conversation[-1].get("content")

        if last_is_assistant_prefix:
            eot_suffixes = [
                "<|im_end|>\n",
                "<|im_end|>",
                "<|eot_id|>",  # Llama 3
                "<end_of_turn>\n",
                "</s>",
            ]
            for suffix in eot_suffixes:
                if prompt_text.endswith(suffix):
                    prompt_text = prompt_text.removesuffix(suffix)
                    break

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        return len(prompt_ids)

    data_dir = Path(cfg.paths.results_dir)
    log.info(f"Scanning {data_dir} for datasets...")

    # Find all layer directories
    layer_dirs = sorted(data_dir.glob("layer_*"))
    metadata_dir = data_dir / "metadata"

    if not layer_dirs and not metadata_dir.exists():
        log.warning("No layer datasets or metadata found!")
        return

    # Load tokenizer
    from transformers import AutoTokenizer

    tokenizer_path = cfg.model.model_name_or_path
    log.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token if tokenizer.eos_token else tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        )

    # Set seed for reproducibility
    extras(cfg)
    random.seed(42)

    # Instantiate datamodule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.tokenizer = tokenizer
    datamodule.prepare_data()
    datamodule.setup(stage="predict")

    # Always use metadata as reference if it exists
    if metadata_dir.exists():
        log.info("Using metadata as reference")
        ref_dataset = load_from_disk(str(metadata_dir))
    elif layer_dirs:
        reference_layer = layer_dirs[0]
        log.info(f"Using reference layer: {reference_layer.name}")
        ref_dataset = load_from_disk(str(reference_layer))
    else:
        log.error("No reference dataset found (neither metadata nor layers).")
        return

    if "sample_index" not in ref_dataset.column_names:
        log.error("Reference dataset missing 'sample_index'. Cannot proceed.")
        return

    # Check if masks already exist for comparison
    old_masks = None
    if "detection_mask" in ref_dataset.column_names:
        log.info("Existing masks found - will compare before/after")
        old_masks = [list(mask) if hasattr(mask, "__iter__") else mask for mask in ref_dataset["detection_mask"]]

    # Calculate masks for all samples
    log.info(f"Calculating masks for {len(ref_dataset)} samples...")
    all_masks = []

    # Get all sample indices
    sample_indices = ref_dataset["sample_index"]
    if hasattr(sample_indices, "tolist"):
        sample_indices = sample_indices.tolist()
    elif not isinstance(sample_indices, list):
        sample_indices = list(sample_indices)

    batch_size = 100
    for i in tqdm(range(0, len(ref_dataset), batch_size), desc="Computing masks"):
        batch_slice = ref_dataset[i : i + batch_size]
        batch_indices = batch_slice["sample_index"]

        # Ensure batch_indices is a list
        if hasattr(batch_indices, "tolist"):
            batch_indices = batch_indices.tolist()
        elif not isinstance(batch_indices, list):
            batch_indices = [batch_indices] if not hasattr(batch_indices, "__iter__") else list(batch_indices)

        # Get original samples
        original_samples = [datamodule.dataset[idx] for idx in batch_indices]

        # Calculate masks based on dataset type
        if datamodule.has_completions:
            # Prefill dataset: mask only the assistant response tokens
            if not metadata_dir.exists():
                raise ValueError("Metadata required for prefill datasets to get completion_attention_mask")

            metadata_dataset = load_from_disk(str(metadata_dir))

            for sample_idx in batch_indices:
                # Get conversation from metadata if available
                if "conversation" in metadata_dataset.column_names:
                    conversation = metadata_dataset[sample_idx]["conversation"]
                else:
                    if sample_idx == batch_indices[0]:
                        log.warning(
                            "Conversations not found in metadata. Using datamodule.dataset which may "
                            "produce non-deterministic results if use_random_prefixes=True."
                        )
                    conversation = datamodule.dataset[sample_idx]["conversation"]

                # Get tokenized sequence from metadata
                input_ids = metadata_dataset[sample_idx]["input_ids"]
                attention_mask = metadata_dataset[sample_idx]["attention_mask"]

                if hasattr(input_ids, "tolist"):
                    input_ids = input_ids.tolist()
                if hasattr(attention_mask, "tolist"):
                    attention_mask = attention_mask.tolist()

                # Calculate where assistant content starts
                prefix_conversation = conversation[:-1]

                prefix_with_prompt = tokenizer.apply_chat_template(
                    prefix_conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prefix_ids = tokenizer(prefix_with_prompt, add_special_tokens=False)["input_ids"]
                training_start_pos = len(prefix_ids)

                # Build mask
                mask = [i >= training_start_pos for i in range(len(input_ids))]
                all_masks.append(mask)

        else:
            # Generation dataset: mask only the generated tokens
            if metadata_dir.exists():
                metadata_dataset = load_from_disk(str(metadata_dir))
                for sample_idx in batch_indices:
                    input_ids = metadata_dataset[sample_idx]["input_ids"]

                    if hasattr(input_ids, "__len__"):
                        seq_len = len(input_ids)
                    else:
                        seq_len = len(list(input_ids))

                    # For generation datasets, the prompt includes the answer_prefix.
                    # Use the ORIGINAL conversation from datamodule (NOT metadata which has full response)
                    # This matches how default_collate_fn tokenizes the prompt before generation.

                    # Get original conversation structure from datamodule
                    original_conversation = datamodule.dataset[sample_idx].get("conversation", [])

                    prompt_len = _calculate_prompt_length(tokenizer, original_conversation)

                    # Build mask: False for prompt, True for generated completion (excluding EOS)
                    eos_token_id = tokenizer.eos_token_id
                    mask = []
                    for i in range(seq_len):
                        if i < prompt_len:
                            mask.append(False)  # Prompt token
                        elif hasattr(input_ids, "__getitem__") and input_ids[i] == eos_token_id:
                            mask.append(False)  # EOS token - exclude from mask
                        else:
                            mask.append(True)  # Generated completion token
                    all_masks.append(mask)
            else:
                # Fallback: compute masks on the fly
                log.warning("No metadata found, computing masks from collate function")
                batch_out = default_collate_fn(original_samples, tokenizer)
                masks = batch_out.get("detection_mask", [])
                all_masks.extend([m.numpy().tolist() for m in masks])

    log.info(f"Computed {len(all_masks)} masks.")

    # Compare with old masks if they existed
    if old_masks is not None:
        log.info("\n" + "=" * 80)
        log.info("COMPARISON: Old masks vs New masks")
        log.info("=" * 80)

        differences_found = 0
        samples_to_show = []

        for i in range(len(all_masks)):
            old_mask = old_masks[i]
            new_mask = all_masks[i]

            # Handle length differences
            if len(old_mask) != len(new_mask):
                differences_found += 1
                if len(samples_to_show) < 3:
                    samples_to_show.append((i, "length", len(old_mask), len(new_mask)))
                continue

            # Check for content differences
            if old_mask != new_mask:
                differences_found += 1
                if len(samples_to_show) < 3:
                    changed_indices = [j for j in range(len(old_mask)) if old_mask[j] != new_mask[j]]
                    samples_to_show.append((i, "content", old_mask, new_mask, changed_indices))

        log.info(f"Differences found: {differences_found}/{len(all_masks)} samples")

        if differences_found == 0:
            log.info("✓ Masks are identical - no changes!")
        else:
            log.info(f"Showing first {len(samples_to_show)} samples with differences:\n")

            if metadata_dir.exists():
                metadata_dataset = load_from_disk(str(metadata_dir))

                for diff in samples_to_show:
                    if diff[1] == "length":
                        i, _, old_len, new_len = diff
                        log.info(f"Sample {i}: Length changed {old_len} → {new_len}")
                    else:
                        i, _, old_mask, new_mask, changed_indices = diff
                        sample_idx = ref_dataset[i]["sample_index"]
                        if hasattr(sample_idx, "item"):
                            sample_idx = sample_idx.item()
                        sample_idx = int(sample_idx)

                        trimmed_tokens = get_trimmed_tokens(metadata_dataset, sample_idx)

                        log.info(f"\nSample {i} (sample_index={sample_idx}):")
                        log.info(f"  {len(changed_indices)} tokens changed training status")

                        # Show changed tokens
                        for idx in changed_indices[:10]:
                            if idx < len(trimmed_tokens):
                                token_text = tokenizer.decode([trimmed_tokens[idx]], skip_special_tokens=False)
                                old_status = "TRAIN" if old_mask[idx] else "skip"
                                new_status = "TRAIN" if new_mask[idx] else "skip"
                                log.info(f"    Token {idx}: {old_status} → {new_status}: {token_text!r}")

                        if len(changed_indices) > 10:
                            log.info(f"    ... and {len(changed_indices) - 10} more")

        log.info("\n" + "=" * 80 + "\n")

    # Preview training masks for first few samples
    log.info("=" * 80)
    log.info("VALIDATION: Preview of training masks")
    log.info("=" * 80)

    if metadata_dir.exists() and layer_dirs:
        metadata_dataset = load_from_disk(str(metadata_dir))

        for row_idx in range(min(3, len(all_masks))):
            sample_idx = ref_dataset[row_idx]["sample_index"]
            if hasattr(sample_idx, "item"):
                sample_idx = sample_idx.item()
            sample_idx = int(sample_idx)

            trimmed_tokens = get_trimmed_tokens(metadata_dataset, sample_idx)
            mask = all_masks[row_idx]

            # Decode trained vs untrained
            trained_tokens = [t for t, m in zip(trimmed_tokens, mask, strict=False) if m]
            untrained_tokens = [t for t, m in zip(trimmed_tokens, mask, strict=False) if not m]

            trained_text = tokenizer.decode(trained_tokens, skip_special_tokens=False)
            untrained_text = tokenizer.decode(untrained_tokens, skip_special_tokens=False)

            log.info(f"\nSample {row_idx} (sample_index={sample_idx}):")
            log.info(
                f"  Training on {len(trained_tokens)}/{len(mask)} tokens ({len(trained_tokens) / len(mask) * 100:.1f}%)"
            )
            log.info(f"\n  ❌ NOT trained:\n  {untrained_text}")
            log.info(f"\n  ✓ TRAINED:\n  {trained_text}")
            log.info("-" * 80)

    log.info("\n" + "=" * 80 + "\n")

    # Check for dry-run mode
    import os

    if os.environ.get("MIGRATE_DRY_RUN") == "1":
        log.info("DRY-RUN mode: Skipping actual updates")
        log.info("✓ Validation complete! Use without --dry-run to apply changes.")
        return

    # Helper function for atomic dataset updates
    def update_dataset_with_masks(dataset_path: Path, masks: list) -> None:
        """Atomically update dataset with detection masks."""
        dataset = load_from_disk(str(dataset_path))

        if "detection_mask" in dataset.column_names:
            dataset = dataset.remove_columns(["detection_mask"])

        dataset = dataset.add_column("detection_mask", masks)

        # Use temp directory for atomic write
        temp_dir = dataset_path.parent / f"{dataset_path.name}_temp"
        dataset.save_to_disk(str(temp_dir))
        shutil.rmtree(str(dataset_path))
        temp_dir.rename(dataset_path)

    # Apply masks to all layer datasets
    log.info("Applying masks to datasets...")
    for layer_dir in layer_dirs:
        log.info(f"Updating {layer_dir.name}...")
        update_dataset_with_masks(layer_dir, all_masks)

    # Apply masks to metadata
    if metadata_dir.exists():
        log.info("Updating metadata...")
        update_dataset_with_masks(metadata_dir, all_masks)

    log.info("✓ Detection masks applied successfully!")


def main():
    """Main entry point - parse arguments and dispatch to appropriate handler."""
    parser = argparse.ArgumentParser(
        description="Migrate detection masks for activation datasets",
        epilog="""
Examples:
  # Process all datasets
  python scripts/migrate_detection_masks.py --all

  # Preview changes for all datasets
  python scripts/migrate_detection_masks.py --all --dry-run

  # Process specific dataset
  python scripts/migrate_detection_masks.py data=instructed_pairs model=qwen2.5/0.5b

  # Preview changes for specific dataset
  python scripts/migrate_detection_masks.py data=instructed_pairs --dry-run
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true", help="Process all datasets with activations")
    parser.add_argument(
        "--model-filter",
        type=str,
        default=None,
        help="Only process datasets for models matching this substring (with --all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying them")
    parser.add_argument("--jobs", "-j", type=int, default=20, help="Number of parallel jobs (with --all)")

    # Parse only known args to allow Hydra args to pass through
    args, remaining = parser.parse_known_args()

    if args.all:
        # Process all datasets
        success = process_all_datasets(dry_run=args.dry_run, jobs=args.jobs, model_filter=args.model_filter)
        sys.exit(0 if success else 1)
    else:
        # Process single dataset using Hydra
        # Set dry_run via environment variable to avoid Hydra conflict
        if args.dry_run:
            import os

            os.environ["MIGRATE_DRY_RUN"] = "1"
        # Put remaining args back for Hydra (excluding --dry-run)
        sys.argv = [sys.argv[0]] + remaining
        process_single_dataset()


if __name__ == "__main__":
    main()
