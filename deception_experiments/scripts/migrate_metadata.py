#!/usr/bin/env python3
"""Migrate metadata to unified token storage format.

This script updates metadata files to use the simplified structure:
1. `input_ids`: Full sequence token IDs (renamed from `completion_input_ids` if present)
2. `attention_mask`: Full sequence attention mask (renamed from `completion_attention_mask`)
3. `completion_start_index`: Index where completion starts (derived if missing)
4. Verify alignment with layer activations.

Usage:
    # Dry run for single dataset
    python scripts/migrate_metadata.py data/deception_bench-L1_self/responses/Llama... --dry-run

    # Migrate single dataset
    python scripts/migrate_metadata.py data/deception_bench-L1_self/responses/Llama...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_all_model_dirs(data_base: Path) -> list[Path]:
    """Find all model directories containing metadata."""
    model_dirs = []
    for metadata_path in data_base.rglob("metadata"):
        if metadata_path.is_dir():
            model_dirs.append(metadata_path.parent)
    return sorted(set(model_dirs))


def process_model_dir(model_dir: Path, dry_run: bool = False) -> dict:
    """Migrate metadata in a model directory."""
    result = {
        "path": str(model_dir),
        "success": False,
        "message": "",
    }

    metadata_path = model_dir / "metadata"
    if not metadata_path.exists():
        result["message"] = "No metadata directory found"
        return result

    try:
        metadata = load_from_disk(str(metadata_path))
    except Exception as e:
        result["message"] = f"Failed to load metadata: {e}"
        return result

    # Load layer data to get reference lengths (optional - used for verification)
    layer_dirs = sorted([p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("layer_")])
    sample_ref = {}  # Empty if no layers - migration still works via mask slicing

    if layer_dirs:
        try:
            # Load first layer to get lengths and completion_start_indices
            layer_ds = load_from_disk(str(layer_dirs[0]))

            # Build map: sample_index -> (activation_len, completion_start_index)
            for i in range(len(layer_ds)):
                sample = layer_ds[i]
                idx = sample.get("sample_index", i)
                act_len = len(sample["activations"])
                csi = sample.get("completion_start_index", 0)
                sample_ref[idx] = (act_len, csi)

        except Exception as e:
            # Continue without layer reference - mask slicing still works
            print(f"  Warning: Could not load layer data: {e}")

    s0 = metadata[0]
    idx0 = s0.get("sample_index", 0)

    if dry_run:
        has_legacy_format = "completion_input_ids" in s0
        has_layer_ref = len(sample_ref) > 0

        if idx0 in sample_ref:
            ref_len, _ = sample_ref[idx0]

            # Simulate extraction
            if has_legacy_format:
                c_ids = s0["completion_input_ids"]
                c_mask = s0.get("completion_attention_mask")
                mask_arr = np.array(c_mask) if c_mask else None
                if mask_arr is not None:
                    non_zero = np.where(mask_arr)[0]
                    if len(non_zero) > 0:
                        sliced_len = non_zero[-1] + 1 - non_zero[0]
                    else:
                        sliced_len = len(c_ids)
                else:
                    sliced_len = len(c_ids)
            else:
                sliced_len = len(s0.get("input_ids", []))

            diff = sliced_len - ref_len
            status_msg = f"Sample 0: Meta sliced={sliced_len} vs Layer={ref_len}"
            if diff != 0:
                print(f"  MISMATCH: {status_msg} (Diff: {diff}). Will adjust.")
            else:
                print(f"  OK: {status_msg}. Perfect alignment.")
        elif has_legacy_format:
            has_det_mask = "detection_mask" in s0
            print(
                f"  MASK-ONLY: No layer ref. CSI from detection_mask={has_det_mask}. Will slice using attention_mask."
            )
        else:
            print("  SKIP: Already has input_ids format, no layer reference to verify.")

        result["success"] = True
        result["message"] = f"Dry run successful. Ready to migrate. (layers={'yes' if has_layer_ref else 'no'})"
        return result

    # TRANSFORM
    def transform_batch(batch):
        out_input_ids = []
        out_mask = []
        out_csi = []

        for i in range(len(batch["sample_index"])):
            idx = batch["sample_index"][i]

            # Get source IDs and mask
            if "completion_input_ids" in batch:
                src_ids = list(batch["completion_input_ids"][i])
                src_mask = list(batch["completion_attention_mask"][i])
            else:
                # Already has input_ids format
                src_ids = list(batch["input_ids"][i])
                src_mask = list(batch["attention_mask"][i])

            # Slice using mask boundaries - matches extract_activations logic exactly
            mask_arr = np.array(src_mask)
            non_zero = np.where(mask_arr)[0]
            if len(non_zero) > 0:
                start = non_zero[0]
                end = non_zero[-1] + 1
                valid_ids = src_ids[start:end]
                valid_mask = src_mask[start:end]
            else:
                valid_ids = src_ids
                valid_mask = src_mask

            # Align with layer reference length if available
            if idx in sample_ref:
                target_len, target_csi = sample_ref[idx]
                curr_len = len(valid_ids)
                if curr_len > target_len:
                    valid_ids = valid_ids[:target_len]
                    valid_mask = valid_mask[:target_len]
                elif curr_len < target_len:
                    diff = target_len - curr_len
                    valid_ids = [0] * diff + valid_ids
                    valid_mask = [0] * diff + valid_mask
                out_csi.append(target_csi)
            else:
                # Derive CSI from detection_mask: count of False = prompt length
                if "detection_mask" in batch:
                    det_mask = batch["detection_mask"][i]
                    csi = sum(1 for x in det_mask if not x)
                    out_csi.append(csi)
                else:
                    out_csi.append(0)

            out_input_ids.append(valid_ids)
            out_mask.append(valid_mask)

        return {"input_ids": out_input_ids, "attention_mask": out_mask, "completion_start_index": out_csi}

    # Remove old columns
    removals = ["completion_input_ids", "completion_attention_mask"]
    removals = [c for c in removals if c in metadata.column_names]

    new_metadata = metadata.map(transform_batch, batched=True, remove_columns=removals)
    import shutil

    # Save to temp path
    temp_path = str(metadata_path) + "_temp"
    new_metadata.save_to_disk(temp_path)

    # Move to original path
    shutil.rmtree(str(metadata_path))
    shutil.move(temp_path, str(metadata_path))

    result["success"] = True
    result["message"] = f"Migrated {len(metadata)} samples"
    return result


def main():
    parser = argparse.ArgumentParser(description="Migrate metadata to unified format")
    parser.add_argument("path", type=str, nargs="?", help="Path to model directory")
    parser.add_argument("--all", action="store_true", help="Process all datasets")
    parser.add_argument("--dry-run", action="store_true", help="Verify without making changes")
    parser.add_argument("--data-base", type=str, default="data", help="Base directory for --all")
    args = parser.parse_args()

    if not args.path and not args.all:
        parser.error("Either provide a path or use --all")

    print("=" * 80)
    print(f"Metadata Migration - {'DRY RUN' if args.dry_run else 'EXECUTION'}")
    print("=" * 80)

    if args.all:
        model_dirs = find_all_model_dirs(Path(args.data_base))
    else:
        model_dirs = [Path(args.path)]

    for model_dir in tqdm(model_dirs, desc="Processing"):
        res = process_model_dir(model_dir, dry_run=args.dry_run)
        status = "✓" if res["success"] else "✗"
        print(f"{status} {model_dir.name}: {res['message']}")


if __name__ == "__main__":
    main()
