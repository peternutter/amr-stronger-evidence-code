#!/usr/bin/env python3
"""Precompute dashboard cache for fast app startup.

Run this after generating new experiment results:
    python scripts/precompute_dashboard_cache.py [--data-dir PATH]

Or with uv:
    uv run python scripts/precompute_dashboard_cache.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.plots.data_loader import DashboardDataLoader  # noqa: E402


def get_default_data_dir() -> Path:
    """Get the default data directory based on environment."""
    cluster_work_dir = os.environ.get("CLUSTER_WORK_DIR")
    if cluster_work_dir:
        return Path(cluster_work_dir) / "data"
    else:
        return Path(__file__).parent.parent / "data"


def main():
    parser = argparse.ArgumentParser(
        description="Precompute dashboard cache for fast app startup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default data directory
    python scripts/precompute_dashboard_cache.py

    # Specify custom data directory
    python scripts/precompute_dashboard_cache.py --data-dir /path/to/data

    # Force rebuild even if cache exists
    python scripts/precompute_dashboard_cache.py --force

    # Check cache status without rebuilding
    python scripts/precompute_dashboard_cache.py --check
""",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to data directory (default: auto-detect from CLUSTER_WORK_DIR or ./data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild cache even if it exists and is up-to-date",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check cache status, don't rebuild",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or get_default_data_dir()

    if not data_dir.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        sys.exit(1)

    print(f"📁 Data directory: {data_dir}")
    print("   Checking cache status (parallel fingerprint computation)...")

    loader = DashboardDataLoader(data_dir)
    cache_status = loader.get_cache_status(show_progress=True)

    # Display cache status
    print("\n📊 Cache Status:")
    print(f"   Experiments cache: {'✅ exists' if cache_status['experiments_cached'] else '❌ missing'}")
    print(f"   Metrics cache:     {'✅ exists' if cache_status['metrics_cached'] else '❌ missing'}")

    if cache_status["cache_stale"]:
        print(f"   ⚠️  Cache is STALE - {cache_status['stale_reason']}")
    elif cache_status["experiments_cached"] and cache_status["metrics_cached"]:
        print("   ✅ Cache is up-to-date")

    if cache_status["cache_time"]:
        print(f"   Last built: {cache_status['cache_time']}")

    if cache_status["num_experiments"] is not None:
        print(f"   Cached experiments: {cache_status['num_experiments']}")
    if cache_status["num_metrics"] is not None:
        print(f"   Cached metric records: {cache_status['num_metrics']:,}")

    # Check-only mode
    if args.check:
        if cache_status["cache_stale"] or not cache_status["experiments_cached"] or not cache_status["metrics_cached"]:
            print("\n💡 Run without --check to rebuild the cache")
            sys.exit(1)  # Exit with error so CI/scripts can detect stale cache
        sys.exit(0)

    # Clear existing cache if force rebuild
    if args.force:
        cache_dir = data_dir / ".dashboard_cache"
        if cache_dir.exists():
            print("\n🗑️  Clearing existing cache...")
            import shutil

            shutil.rmtree(cache_dir)

    # Skip if cache is up-to-date and not forcing
    if (
        not args.force
        and not cache_status["cache_stale"]
        and cache_status["experiments_cached"]
        and cache_status["metrics_cached"]
    ):
        print("\n✅ Cache is already up-to-date. Use --force to rebuild anyway.")
        sys.exit(0)

    # Build cache
    print("\n🔨 Building cache...")
    start = time.time()

    print("   Scanning experiments...")
    experiments = loader.experiments
    print(f"   Found {len(experiments)} experiment configurations")

    if len(experiments) == 0:
        print("   No experiments found")
        sys.exit(0)

    # Count total result files
    total_files = sum(len(list(Path(p).glob("results_layer_*.pkl"))) for p in experiments["path"].unique())
    print(f"   Total result files: {total_files:,}")

    print("   Loading all metrics (this may take a while)...")
    metrics = loader.all_metrics
    print(f"   Loaded {len(metrics):,} metric records")

    elapsed = time.time() - start
    print(f"\n✅ Cache built in {elapsed:.1f}s")

    # Show cache file sizes
    cache_dir = data_dir / ".dashboard_cache"
    if cache_dir.exists():
        print(f"   Cache location: {cache_dir}")
        for cache_file in sorted(cache_dir.iterdir()):
            if cache_file.is_file():
                size_mb = cache_file.stat().st_size / (1024 * 1024)
                print(f"   - {cache_file.name}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
