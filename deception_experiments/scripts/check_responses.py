#!/usr/bin/env python3
"""
Script to check which datasets have responses generated for which models and how many layers.
Also checks if belief datasets have generated metadata (important for MASK evaluation).
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path


def get_layer_count(model_path: Path) -> int:
    """Count the number of layer_* directories in a model response folder."""
    if not model_path.exists():
        return 0
    return len([d for d in model_path.iterdir() if d.is_dir() and d.name.startswith("layer_")])


def has_metadata(model_path: Path) -> bool:
    """Check if a model response directory has metadata files."""
    if not model_path.exists():
        return False
    # Check for metadata/ subdirectory (used by belief datasets)
    if (model_path / "metadata").is_dir():
        return True
    # Check for metadata.json or any .parquet/.json files in the root
    metadata_patterns = ["metadata.json", "*.parquet", "dataset_info.json"]
    for pattern in metadata_patterns:
        if list(model_path.glob(pattern)):
            return True
    return False


def check_responses(data_dir: Path, model_filter: str | None = None) -> dict:
    """Check all datasets for model responses, layer counts, and metadata status."""
    results = defaultdict(dict)

    # Find all dataset directories (those with a 'responses' subfolder)
    for dataset_dir in sorted(data_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        responses_dir = dataset_dir / "responses"
        if not responses_dir.exists():
            continue

        dataset_name = dataset_dir.name
        is_belief_dataset = "belief_" in dataset_name

        # Check each model in the responses directory
        for model_dir in sorted(responses_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            # Apply model filter if provided
            if model_filter and model_filter.lower() not in model_name.lower():
                continue

            layer_count = get_layer_count(model_dir)
            has_meta = has_metadata(model_dir)

            # For belief datasets, include if has metadata (layers not expected)
            # For other datasets, include if has layers
            if layer_count > 0 or (is_belief_dataset and has_meta):
                results[dataset_name][model_name] = {
                    "layers": layer_count,
                    "has_metadata": has_meta,
                }

    return results


def print_results(results: dict) -> None:
    """Print the results in a formatted table."""
    if not results:
        print("No matching datasets/models found.")
        return

    # Print header
    print("\n" + "=" * 80)
    print("DATASET RESPONSE SUMMARY")
    print("=" * 80 + "\n")

    for dataset_name, models in sorted(results.items()):
        # Check if this is a belief dataset (important for metadata)
        is_belief_dataset = "belief_" in dataset_name
        print(f"📁 {dataset_name}")
        print("-" * 60)

        for model_name, info in sorted(models.items()):
            layer_count = info["layers"]
            has_meta = info["has_metadata"]

            # Determine expected layers based on model name
            if "70B" in model_name or "72B" in model_name:
                expected = 80
            elif "32B" in model_name:
                expected = 64
            elif "14B" in model_name:
                expected = 48
            elif "8B" in model_name:
                expected = 32
            elif "7B" in model_name:
                expected = 28
            elif "1.5B" in model_name:
                expected = 28
            elif "0.5B" in model_name:
                expected = 24
            else:
                expected = "?"

            status = "✅" if layer_count >= expected else "⚠️ "

            # Show metadata status for belief datasets
            if is_belief_dataset:
                # For belief datasets, layers are not expected - only metadata matters
                meta_status = "✅ has metadata" if has_meta else "❌ NO METADATA"
                if layer_count > 0:
                    print(f"  {status} {model_name}: {layer_count} layers [{meta_status}]")
                else:
                    print(f"  {'✅' if has_meta else '❌'} {model_name}: [{meta_status}]")
            else:
                print(f"  {status} {model_name}: {layer_count} layers (expected: {expected})")

        print()

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80 + "\n")

    # Create a compact summary
    print(f"{'Dataset':<45} | Models with responses")
    print("-" * 80)

    for dataset_name, models in sorted(results.items()):
        is_belief = "belief_" in dataset_name
        model_parts = []
        for m, info in sorted(models.items()):
            short_name = m.split("_")[-1]
            if is_belief:
                # For belief datasets, show metadata status
                layer_str = f"{short_name}:{'✅meta' if info['has_metadata'] else '❌meta'}"
            else:
                layer_str = f"{short_name}:{info['layers']}"
            model_parts.append(layer_str)
        model_summary = ", ".join(model_parts)
        print(f"{dataset_name:<45} | {model_summary}")


def main():
    parser = argparse.ArgumentParser(description="Check model responses and metadata status.")
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Path to the data directory (default: ../data relative to script)",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Filter output by model name (case-insensitive substring match)"
    )

    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    print(f"Scanning: {args.data_dir}")
    if args.model:
        print(f"Filtering for model: {args.model}")

    results = check_responses(args.data_dir, model_filter=args.model)
    print_results(results)


if __name__ == "__main__":
    main()
