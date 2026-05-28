#!/usr/bin/env python3
"""Unified activation migration script.

Migrates all activation datasets to the unified format defined in activation_schema.py.

Handles all categories:
- Category A (Split): Reconstruct from activations + completion_activations
- Category B (Correct): Already correct, verify only
- Category C (Shifted): Move completion_activations -> activations
- Category D (No Mask): Backfill detection_mask from completion_start_index

Usage:
    # Dry run single dataset
    python scripts/migrate_activations.py <path> --dry-run

    # Migrate single dataset
    python scripts/migrate_activations.py <path>

    # Migrate all datasets
    python scripts/migrate_activations.py --all --dry-run
    python scripts/migrate_activations.py --all
"""

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.activation_schema import (
    ACTIVATIONS,
    COMPLETION_START_INDEX,
    DETECTION_MASK,
    ActivationSchemaError,
    get_sample_info,
    validate_dataset,
)


def detect_category(dataset: Dataset) -> str:
    """Detect which category a dataset belongs to.

    Returns:
        "A" (split), "B" (correct), "C" (shifted), "D" (no mask)
    """
    cols = set(dataset.column_names)
    sample = dataset[0]

    has_activations = ACTIVATIONS in cols
    has_completion_activations = "completion_activations" in cols
    has_detection_mask = DETECTION_MASK in cols

    # Category B: Already in correct format
    if has_activations and has_detection_mask and not has_completion_activations:
        try:
            validate_dataset(dataset, check_all=False)
            return "B"
        except ActivationSchemaError:
            pass  # Not valid, continue checking

    # Category D: Missing detection_mask
    if not has_detection_mask:
        return "D"

    # Get lengths for categorization
    acts_len = len(sample[ACTIVATIONS]) if has_activations else 0
    comp_acts_len = len(sample["completion_activations"]) if has_completion_activations else 0
    mask_len = len(sample[DETECTION_MASK])

    # Category A: Split storage (activations + completion_activations = mask_len)
    if has_activations and has_completion_activations:
        if acts_len + comp_acts_len == mask_len:
            return "A"

    # Category C: Full sequence in completion_activations
    if has_completion_activations and comp_acts_len == mask_len:
        return "C"

    raise ValueError(
        f"Cannot determine category. "
        f"activations={acts_len}, completion_activations={comp_acts_len}, mask={mask_len}"
    )


def fix_category_a(dataset: Dataset) -> Dataset:
    """Fix Category A: Reconstruct full sequence from split storage."""

    def reconstruct(batch):
        fixed_activations = []
        completion_start_indices = []

        for i in range(len(batch[ACTIVATIONS])):
            prompt_acts = np.array(batch[ACTIVATIONS][i])
            comp_acts = np.array(batch["completion_activations"][i])
            mask = np.array(batch[DETECTION_MASK][i], dtype=bool)

            # Reconstruct: full[~mask] = prompt, full[mask] = completion
            full_acts = np.zeros((len(mask), prompt_acts.shape[-1]), dtype=prompt_acts.dtype)
            full_acts[~mask] = prompt_acts
            full_acts[mask] = comp_acts

            fixed_activations.append(full_acts)
            completion_start_indices.append(len(prompt_acts))

        return {
            ACTIVATIONS: fixed_activations,
            COMPLETION_START_INDEX: completion_start_indices,
        }

    fixed = dataset.map(reconstruct, batched=True, remove_columns=["completion_activations"])
    return fixed


def fix_category_c(dataset: Dataset) -> Dataset:
    """Fix Category C: Move completion_activations -> activations."""

    def move_and_mark(batch):
        completion_start_indices = []

        for mask in batch[DETECTION_MASK]:
            mask_arr = np.array(mask, dtype=bool)
            # completion_start_index = first True position
            completion_start_indices.append(int((~mask_arr).sum()))

        return {
            ACTIVATIONS: batch["completion_activations"],
            COMPLETION_START_INDEX: completion_start_indices,
        }

    # Remove old activations column if it exists and has different length
    remove_cols = ["completion_activations"]
    if ACTIVATIONS in dataset.column_names:
        sample = dataset[0]
        if len(sample[ACTIVATIONS]) != len(sample[DETECTION_MASK]):
            remove_cols.append(ACTIVATIONS)

    fixed = dataset.map(move_and_mark, batched=True, remove_columns=remove_cols)
    return fixed


def fix_category_d(dataset: Dataset) -> Dataset:
    """Fix Category D: Backfill detection_mask from activations/completion_start_index."""

    def backfill_mask(batch):
        detection_masks = []
        completion_start_indices = []

        # Check what we have to work with
        has_comp_start = COMPLETION_START_INDEX in batch
        has_comp_acts = "completion_activations" in batch

        for i in range(len(batch[ACTIVATIONS])):
            acts = batch[ACTIVATIONS][i]
            acts_len = len(acts)

            if has_comp_start:
                comp_start = batch[COMPLETION_START_INDEX][i]
            elif has_comp_acts:
                # Infer from activations vs completion_activations
                comp_len = len(batch["completion_activations"][i])
                comp_start = acts_len - comp_len
            else:
                # No info - assume all completion (conservative)
                comp_start = 0

            # Create mask: False for prompt, True for completion
            mask = [False] * comp_start + [True] * (acts_len - comp_start)

            detection_masks.append(mask)
            completion_start_indices.append(comp_start)

        result = {
            DETECTION_MASK: detection_masks,
        }

        if not has_comp_start:
            result[COMPLETION_START_INDEX] = completion_start_indices

        return result

    # Remove completion_activations if present
    remove_cols = []
    if "completion_activations" in dataset.column_names:
        remove_cols.append("completion_activations")

    fixed = dataset.map(backfill_mask, batched=True, remove_columns=remove_cols if remove_cols else None)
    return fixed


def verify_reconstruction(dataset: Dataset, category: str, num_samples: int = 10) -> tuple[bool, str]:
    """Verify that samples can be reconstructed correctly.

    Returns:
        (success, message)
    """
    samples_to_check = min(num_samples, len(dataset))

    for i in range(samples_to_check):
        sample = dataset[i]

        if category == "A":
            prompt_len = len(sample[ACTIVATIONS])
            comp_len = len(sample["completion_activations"])
            mask_len = len(sample[DETECTION_MASK])

            if prompt_len + comp_len != mask_len:
                return False, f"Sample {i}: Length mismatch: {prompt_len} + {comp_len} != {mask_len}"

            # Check feature dimensions match
            prompt_dim = np.array(sample[ACTIVATIONS]).shape[-1]
            comp_dim = np.array(sample["completion_activations"]).shape[-1]
            if prompt_dim != comp_dim:
                return False, f"Sample {i}: Feature dimension mismatch: {prompt_dim} != {comp_dim}"

        elif category == "C":
            comp_len = len(sample["completion_activations"])
            mask_len = len(sample[DETECTION_MASK])

            if comp_len != mask_len:
                return False, f"Sample {i}: Length mismatch: {comp_len} != {mask_len}"

        elif category == "D":
            # Just need activations to exist
            if ACTIVATIONS not in sample:
                return False, f"Sample {i}: Missing {ACTIVATIONS} column"

    return True, f"All {samples_to_check} samples verified"


def migrate_single_layer(args: tuple[Path, str]) -> tuple[str, str | None]:
    """Migrate a single layer. Top-level function for multiprocessing.

    Args:
        args: Tuple of (layer_path, category)

    Returns:
        (layer_name, error_msg or None)
    """
    layer_path, category = args
    try:
        dataset = load_from_disk(str(layer_path))

        if category == "A":
            fixed = fix_category_a(dataset)
        elif category == "C":
            fixed = fix_category_c(dataset)
        elif category == "D":
            fixed = fix_category_d(dataset)
        else:
            raise ValueError(f"Unexpected category: {category}")

        # Save to temp location first, then move
        temp_path = layer_path.parent / f"{layer_path.name}_temp"
        fixed.set_format(type="torch")
        fixed.save_to_disk(str(temp_path))

        # Remove old and rename temp to original
        shutil.rmtree(layer_path)
        temp_path.rename(layer_path)

        return (layer_path.name, None)
    except Exception as e:
        return (layer_path.name, str(e))


def process_dataset(model_dir: Path, dry_run: bool = False, n_workers: int = 8) -> dict:
    """Process a single model's activation dataset.

    Returns:
        Dictionary with results
    """
    result = {
        "path": str(model_dir),
        "success": False,
        "category": None,
        "message": "",
        "layers": 0,
    }

    # Find layer directories
    layer_dirs = sorted([p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("layer_")])

    if not layer_dirs:
        result["message"] = "No layer_* directories found"
        return result

    result["layers"] = len(layer_dirs)

    # Detect category from first layer
    try:
        first_layer = load_from_disk(str(layer_dirs[0]))
    except Exception as e:
        result["message"] = f"Failed to load layer_0: {e}"
        return result

    try:
        category = detect_category(first_layer)
    except Exception as e:
        result["message"] = f"Failed to detect category: {e}"
        return result

    result["category"] = category

    # Category B is already correct
    if category == "B":
        result["success"] = True
        result["message"] = "Already in correct format"
        return result

    # Verify reconstruction
    success, msg = verify_reconstruction(first_layer, category)
    if not success:
        result["message"] = f"Reconstruction verification failed: {msg}"
        return result

    if dry_run:
        result["success"] = True
        result["message"] = f"Ready for migration ({msg})"
        return result

    # Create backup
    backup_dir = model_dir.parent / f"{model_dir.name}_backup"
    if not backup_dir.exists():
        shutil.copytree(model_dir, backup_dir)

    # Process all layers in parallel using top-level function
    actual_workers = min(n_workers, len(layer_dirs))
    errors = []

    # Prepare args for parallel processing
    layer_args = [(lp, category) for lp in layer_dirs]

    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        futures = {executor.submit(migrate_single_layer, args): args[0] for args in layer_args}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Migrating {model_dir.name}", leave=False):
            layer_name, error = future.result()
            if error:
                errors.append(f"{layer_name}: {error}")

    if errors:
        result["message"] = f"Failed on {len(errors)} layers: {errors[0]}"
        return result

    result["success"] = True
    result["message"] = f"Migrated {len(layer_dirs)} layers (parallel, {actual_workers} workers)"
    return result


def find_all_datasets(data_base: Path) -> list[Path]:
    """Find all activation dataset directories."""
    datasets = []

    # Find all directories containing layer_0
    for layer0 in data_base.rglob("layer_0"):
        if layer0.is_dir():
            model_dir = layer0.parent
            # Verify it's a proper activation directory
            if (model_dir / "layer_0").is_dir():
                datasets.append(model_dir)

    return sorted(datasets)


def print_sample_info(model_dir: Path):
    """Print sample info for debugging."""
    layer0_path = model_dir / "layer_0"
    if not layer0_path.exists():
        print("  No layer_0 found")
        return

    try:
        dataset = load_from_disk(str(layer0_path))
        print(f"  Columns: {dataset.column_names}")
        print(f"  Samples: {len(dataset)}")
        print("  Sample 0:")
        print(get_sample_info(dataset[0]))
    except Exception as e:
        print(f"  Error loading: {e}")


def main():
    parser = argparse.ArgumentParser(description="Migrate activation datasets to unified format")
    parser.add_argument("path", type=str, nargs="?", help="Path to activation directory")
    parser.add_argument("--all", action="store_true", help="Migrate all datasets")
    parser.add_argument("--dry-run", action="store_true", help="Verify without making changes")
    parser.add_argument(
        "--data-base",
        type=str,
        default="data",
        help="Base directory for --all mode",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed info")
    args = parser.parse_args()

    if not args.path and not args.all:
        parser.error("Either provide a path or use --all")

    mode = "DRY RUN" if args.dry_run else "MIGRATION"
    print("=" * 80)
    print(f"Activation {mode}")
    print("=" * 80)

    if args.all:
        data_base = Path(args.data_base)
        datasets = find_all_datasets(data_base)
        print(f"Found {len(datasets)} datasets to process\n")
    else:
        datasets = [Path(args.path)]

    # Track results
    results = {"A": 0, "B": 0, "C": 0, "D": 0, "failed": 0}
    failed_datasets = []

    for model_dir in tqdm(datasets, desc="Processing datasets"):
        if args.verbose:
            print(f"\n{model_dir}")
            print_sample_info(model_dir)

        try:
            result = process_dataset(model_dir, dry_run=args.dry_run)
        except Exception as e:
            result = {
                "success": False,
                "category": None,
                "message": f"Crashed: {type(e).__name__}: {e}",
            }

        if result["success"]:
            if result["category"]:
                results[result["category"]] += 1
            status = "✓"
        else:
            results["failed"] += 1
            failed_datasets.append((model_dir, result["message"]))
            status = "✗"

        if args.verbose or not result["success"]:
            print(f"{status} {model_dir.name}: Category {result['category']} - {result['message']}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Category A (Split):     {results['A']} datasets")
    print(f"Category B (Correct):   {results['B']} datasets")
    print(f"Category C (Shifted):   {results['C']} datasets")
    print(f"Category D (No Mask):   {results['D']} datasets")
    print(f"Failed:                 {results['failed']} datasets")

    if failed_datasets:
        print("\nFailed datasets:")
        for path, msg in failed_datasets:
            print(f"  {path}: {msg}")

    if args.dry_run:
        total_ready = results["A"] + results["C"] + results["D"]
        print(f"\n✓ {total_ready} datasets ready for migration, {results['B']} already correct")
        print("\nTo execute migration:")
        if args.all:
            print("  python scripts/migrate_activations.py --all")
        else:
            print(f"  python scripts/migrate_activations.py {args.path}")
    else:
        total_migrated = results["A"] + results["C"] + results["D"]
        print(f"\n✓ {total_migrated} datasets migrated, {results['B']} already correct")
        print("\nBackups saved at: <dataset>_backup/")

    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
