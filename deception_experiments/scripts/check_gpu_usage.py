#!/usr/bin/env python3
"""
Script to check GPU usage (VRAM and utilization) for SLURM jobs.
Takes a job ID (or job array ID) and queries nvidia-smi on each node.

Usage:
    uv run python scripts/check_gpu_usage.py 1195820
    uv run python scripts/check_gpu_usage.py  # checks all your running jobs
"""

import subprocess
import sys


def get_running_jobs(job_id_filter=None):
    """Get list of running jobs, optionally filtered by job ID."""
    # Use %A for actual job ID (works with srun --overlap), %i for array format
    cmd = ["squeue", "-u", subprocess.getoutput("whoami"), "-o", "%A %i %.30j %.2t %R", "-h"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    jobs = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 5:
            actual_jid = parts[0]  # Actual job ID for srun
            array_jid = parts[1]  # Array format for display
            name = parts[2]
            state = parts[3]
            node = parts[4]

            # Filter by job ID if provided (match against array format or actual)
            if job_id_filter:
                base_job_id = job_id_filter.split("_")[0]
                if not (array_jid.startswith(base_job_id) or actual_jid.startswith(base_job_id)):
                    continue

            # Only include running jobs
            if state == "R":
                jobs.append({"job_id": actual_jid, "array_id": array_jid, "node": node, "name": name, "state": state})

    return jobs


def get_gpu_info(job_id):
    """Get GPU info from a job using srun --overlap --jobid."""
    cmd = [
        "srun",
        "--overlap",
        "--jobid=" + job_id,
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append(
                    {
                        "gpu_id": int(parts[0]),
                        "name": parts[1],
                        "memory_used": int(parts[2]),
                        "memory_total": int(parts[3]),
                        "utilization": int(parts[4]),
                    }
                )
        return gpus
    except subprocess.TimeoutExpired:
        return []
    except Exception as e:
        print(f"  Error querying job {job_id}: {e}")
        return []


def format_memory(used, total):
    """Format memory usage with percentage."""
    pct = (used / total) * 100 if total > 0 else 0
    return f"{used}/{total} MiB ({pct:.1f}%)"


def main():
    job_id = sys.argv[1] if len(sys.argv) > 1 else None

    if job_id:
        print("Checking GPU usage for job: %s" % job_id)
    else:
        print("Checking GPU usage for all your running jobs")

    print("=" * 80)

    jobs = get_running_jobs(job_id)

    if not jobs:
        print("No running jobs found.")
        return

    print("Found %d running job(s)\n" % len(jobs))

    # Query each job
    for job in jobs:
        jid = job["job_id"]
        array_id = job["array_id"]
        node = job["node"]
        print(f"Job: {jid} [{array_id}] (node: {node})")
        print("-" * 60)

        gpus = get_gpu_info(jid)

        if not gpus:
            print("   Could not query GPU info")
        else:
            total_used = 0
            total_mem = 0
            total_util = 0

            for gpu in gpus:
                mem_pct = (gpu["memory_used"] / gpu["memory_total"]) * 100
                bar_len = int(mem_pct / 5)  # 20 char bar
                bar = "#" * bar_len + "-" * (20 - bar_len)

                print(f"   GPU {gpu['gpu_id']}: {gpu['name']}")
                print(f"      VRAM: [{bar}] {format_memory(gpu['memory_used'], gpu['memory_total'])}")
                print(f"      Util: {gpu['utilization']}%")

                total_used += gpu["memory_used"]
                total_mem += gpu["memory_total"]
                total_util += gpu["utilization"]

            avg_util = total_util / len(gpus)
            print(f"\n   Summary: {format_memory(total_used, total_mem)} total, {avg_util:.1f}% avg util")

        print()

    # Print overall summary
    print("=" * 80)
    print("Total jobs: %d" % len(jobs))


if __name__ == "__main__":
    main()
