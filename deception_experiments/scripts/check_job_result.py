#!/usr/bin/env python3
"""Check submitit job result from pickle file.

Usage:
    python scripts/check_job_result.py <path_to_result.pkl>
    python scripts/check_job_result.py <path_to_multirun_dir>  # checks all result.pkl files
    python scripts/check_job_result.py <path_to_multirun_dir> --failed-only  # only show failed jobs
    python scripts/check_job_result.py <path_to_multirun_dir> --summary  # show summary grouped by error
"""

import argparse
import pickle
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path


def load_overrides_from_submitted(submitted_path: Path) -> list[str] | None:
    """Load overrides from a submitted.pkl file."""
    try:
        with open(submitted_path, "rb") as f:
            data = pickle.load(f)
        if hasattr(data, "args") and data.args:
            return data.args[0]
    except Exception as e:
        print(f"Warning: Failed to load overrides from {submitted_path}: {e}", file=sys.stderr)
    return None


def check_job_status(pkl_path: Path) -> tuple[bool, str, dict]:
    """Check job status without printing.

    Returns:
        Tuple of (is_failed, status_message, details_dict)
    """
    try:
        with open(pkl_path, "rb") as f:
            result = pickle.load(f)

        # Check if it's a tuple (Hydra multirun format: ('success'/'error', JobReturn))
        if isinstance(result, tuple) and len(result) == 2:
            status_str, job_return = result

            # Extract job info if available
            overrides = None
            if hasattr(job_return, "overrides"):
                overrides = job_return.overrides

            # Check the JobReturn status
            if hasattr(job_return, "status"):
                status = job_return.status
                # Check if status indicates failure (JobStatus.FAILED = 2)
                if hasattr(status, "value") and status.value == 2:
                    error_info = {"overrides": overrides}
                    if hasattr(job_return, "_return_value") and job_return._return_value is not None:
                        error = job_return._return_value
                        error_info["error_type"] = type(error).__name__
                        error_info["error_message"] = str(error)
                        error_str = str(error)
                        if "out of memory" in error_str.lower() or "oom" in error_str.lower():
                            error_info["is_oom"] = True
                    return True, "failed", error_info
                elif hasattr(status, "value") and status.value == 1:
                    return_val = None
                    if hasattr(job_return, "_return_value"):
                        return_val = job_return._return_value
                    return False, "completed", {"overrides": overrides, "return_value": return_val}

            # Fallback: check status string
            if status_str == "success":
                return False, "completed", {"overrides": overrides}
            else:
                return True, "failed", {"overrides": overrides}

        # Check if it's a direct exception
        if isinstance(result, BaseException):
            return (
                True,
                "exception",
                {
                    "error_type": type(result).__name__,
                    "error_message": str(result),
                    "traceback": result.__traceback__,
                },
            )
        elif result is None:
            return False, "completed", {"return_value": None}
        else:
            return False, "completed", {"return_value": result}

    except Exception as e:
        return True, "load_error", {"error": str(e)}


def print_job_result(pkl_path: Path, is_failed: bool, status: str, details: dict, show_overrides: bool = True) -> None:
    """Print job result details."""
    print(f"\n{'='*60}")
    print(f"📁 File: {pkl_path.name}")
    print(f"   Path: {pkl_path}")
    print("=" * 60)

    overrides = details.get("overrides")

    if status == "load_error":
        print(f"⚠️ Error loading pickle: {details.get('error')}")
    elif status == "exception":
        print("❌ JOB FAILED with exception:")
        print(f"   Type: {details.get('error_type')}")
        print(f"   Message: {details.get('error_message')}")
        if show_overrides and overrides:
            print(f"\n   Overrides: {overrides}")
        print()
        print("Full traceback:")
        print("-" * 40)
        tb = details.get("traceback")
        if tb:
            error = Exception(details.get("error_message"))
            tb_lines = traceback.format_exception(type(error), error, tb)
            print("".join(tb_lines))
        else:
            print(f"   {details.get('error_type')}: {details.get('error_message')}")
    elif status == "cancelled":
        print("🚫 JOB CANCELLED/PENDING (no result file)")
        if show_overrides and overrides:
            print(f"   Overrides: {overrides}")
    elif is_failed:
        print("❌ JOB FAILED")
        if show_overrides and overrides:
            print(f"   Overrides: {overrides}")
        if details.get("error_type"):
            print(f"\n   Error Type: {details.get('error_type')}")
            print(f"   Error Message: {details.get('error_message')}")
            if details.get("is_oom"):
                print("\n   💥 OUT OF MEMORY ERROR DETECTED!")
                error_str = details.get("error_message", "")
                if "GPU" in error_str:
                    gpu_info = re.search(
                        r"GPU \d+ has a total capacity of [\d.]+ GiB of which [\d.]+ GiB is free", error_str
                    )
                    if gpu_info:
                        print(f"   {gpu_info.group()}")
                    alloc_info = re.search(r"Tried to allocate [\d.]+ [GMK]iB", error_str)
                    if alloc_info:
                        print(f"   {alloc_info.group()}")
    else:
        return_val = details.get("return_value")
        job_info = f"\n   Overrides: {overrides}" if (show_overrides and overrides) else ""
        print(f"✅ JOB COMPLETED{job_info}")
        if return_val is not None:
            print(f"   Type: {type(return_val).__name__}")
            if hasattr(return_val, "__len__"):
                print(f"   Length: {len(return_val)}")
            result_str = str(return_val)
            if len(result_str) > 500:
                print(f"   Value: {result_str[:500]}...")
            else:
                print(f"   Value: {return_val}")


def extract_key_config(overrides: list[str] | None) -> str:
    """Extract key config info from overrides for summary display."""
    if not overrides:
        return "unknown"

    key_parts = []
    for override in overrides:
        if override.startswith("model="):
            key_parts.append(override.split("=")[1].split("/")[-1])
        elif override.startswith("data="):
            key_parts.append(override.split("=")[1].split("/")[-1])
        elif override.startswith("pooling_strategy="):
            key_parts.append(override.split("=")[1])
    return " | ".join(key_parts) if key_parts else str(overrides[:3])


def print_summary(results: list, cancelled_jobs: list) -> None:
    """Print a summary of failures grouped by error type."""
    print("\n" + "=" * 60)
    print("📊 FAILURE SUMMARY")
    print("=" * 60)

    # Group by error type
    error_groups = defaultdict(list)
    for pkl_file, is_failed, status, details in results:
        if is_failed:
            error_type = details.get("error_type", "unknown")
            error_msg = details.get("error_message", "")

            # Try to categorize common errors
            if "duplicate key" in error_msg.lower():
                category = "Config: Duplicate Key"
            elif "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                category = "OOM: Out of Memory"
            elif "not found" in error_msg.lower():
                category = "Missing: File/Config Not Found"
            elif status == "load_error":
                category = "Load Error"
            else:
                category = f"{error_type}"

            config_str = extract_key_config(details.get("overrides"))
            error_groups[category].append((pkl_file.name, config_str, error_msg[:100]))

    # Add cancelled jobs as a category
    if cancelled_jobs:
        error_groups["Cancelled/Pending"] = [
            (job_id, extract_key_config(overrides), "No result file found") for job_id, overrides in cancelled_jobs
        ]

    # Print grouped summary
    for category, jobs in sorted(error_groups.items()):
        print(f"\n🔴 {category} ({len(jobs)} jobs):")
        for job_name, config, _ in jobs[:10]:  # Show first 10
            print(f"   • {job_name}: {config}")
        if len(jobs) > 10:
            print(f"   ... and {len(jobs) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Check submitit job results from pickle files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/check_job_result.py path/to/result.pkl
    python scripts/check_job_result.py path/to/multirun_dir
    python scripts/check_job_result.py path/to/multirun_dir --failed-only
    python scripts/check_job_result.py path/to/multirun_dir --summary
""",
    )
    parser.add_argument("path", type=Path, help="Path to a result.pkl file or multirun directory")
    parser.add_argument(
        "-f",
        "--failed-only",
        action="store_true",
        help="Only show failed jobs (useful for quickly finding errors)",
    )
    parser.add_argument(
        "-s",
        "--summary",
        action="store_true",
        help="Show a summary of failures grouped by error type",
    )
    args = parser.parse_args()

    path = args.path
    failed_only = args.failed_only
    show_summary = args.summary

    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)

    if path.is_file() and path.suffix == ".pkl":
        # Single pickle file
        is_failed, status, details = check_job_status(path)
        if not failed_only or is_failed:
            print_job_result(path, is_failed, status, details)
    elif path.is_dir():
        # Directory - find all result.pkl files
        result_files = sorted(path.glob("*_result.pkl"))

        if not result_files:
            # Try looking in subdirectories
            result_files = sorted(path.glob("**/*_result.pkl"))

        # Also find submitted.pkl files (for detecting cancelled jobs)
        submitted_files = sorted(path.glob("*_submitted.pkl"))
        if not submitted_files:
            submitted_files = sorted(path.glob("**/*_submitted.pkl"))

        if not result_files and not submitted_files:
            print(f"No result.pkl or submitted.pkl files found in {path}")
            sys.exit(1)

        # Build a set of job IDs that have results
        result_job_ids = set()
        for rf in result_files:
            # Extract job ID from filename like "1203613_8_0_result.pkl"
            match = re.match(r"(\d+_\d+)", rf.name)
            if match:
                result_job_ids.add(match.group(1))

        # Find cancelled/pending jobs (submitted but no result)
        # Also build a mapping from job_id to submitted file for enriching failed jobs
        cancelled_jobs = []
        submitted_by_job_id = {}
        for sf in submitted_files:
            # Extract job ID from filename like "1203613_8_submitted.pkl"
            match = re.match(r"(\d+_\d+)", sf.name)
            if match:
                job_id = match.group(1)
                submitted_by_job_id[job_id] = sf
                if job_id not in result_job_ids:
                    overrides = load_overrides_from_submitted(sf)
                    cancelled_jobs.append((job_id, overrides))

        # Check all jobs first
        results = []
        for pkl_file in result_files:
            is_failed, status, details = check_job_status(pkl_file)

            # If this is a failed job and we don't have overrides, try to load from submitted.pkl
            if is_failed and not details.get("overrides"):
                match = re.match(r"(\d+_\d+)", pkl_file.name)
                if match:
                    job_id = match.group(1)
                    if job_id in submitted_by_job_id:
                        details["overrides"] = load_overrides_from_submitted(submitted_by_job_id[job_id])

            results.append((pkl_file, is_failed, status, details))

        # Count stats
        total = len(results)
        failed_count = sum(1 for _, is_failed, _, _ in results if is_failed)
        success_count = total - failed_count
        cancelled_count = len(cancelled_jobs)

        print(f"Found {total} completed jobs: {success_count} succeeded, {failed_count} failed")
        if cancelled_count > 0:
            print(f"Found {cancelled_count} cancelled/pending jobs (submitted but no result)")

        if show_summary:
            # Just show summary
            print_summary(results, cancelled_jobs)
        elif failed_only:
            print("\nShowing only failed/cancelled jobs:\n")
            for pkl_file, is_failed, status, details in results:
                if is_failed:
                    print_job_result(pkl_file, is_failed, status, details)
            # Also show cancelled jobs
            for job_id, overrides in cancelled_jobs:
                print(f"\n{'='*60}")
                print(f"🚫 CANCELLED/PENDING: Job {job_id}")
                print("=" * 60)
                if overrides:
                    print(f"   Overrides: {overrides}")
        else:
            for pkl_file, is_failed, status, details in results:
                print_job_result(pkl_file, is_failed, status, details)
            # Also show cancelled jobs
            for job_id, overrides in cancelled_jobs:
                print(f"\n{'='*60}")
                print(f"🚫 CANCELLED/PENDING: Job {job_id}")
                print("=" * 60)
                if overrides:
                    print(f"   Overrides: {overrides}")
    else:
        print(f"Error: Path must be a .pkl file or directory: {path}")
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()
