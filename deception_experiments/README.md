# LLM Deception Detection

Research project for detecting deception in large language models using fine-tuned probes and behavioral analysis.

## Environment Setup

### 1. Install Prerequisites

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Configure Cache Directories (Cluster Only)

**On Cluster A**, add these exports to your shell config before running anything else:

```bash
# Add to ~/.zshrc (or ~/.bashrc if using bash)
export HF_HOME=/path/to/cluster/project/llm-deception-data/.cache/huggingface
export UV_CACHE_DIR=/path/to/cluster/project/llm-deception-data/.cache/uv

# Then reload
source ~/.zshrc  # or source ~/.bashrc
```

**On Cluster B**, add these exports to your shell config before running anything else:

```bash
# Add to ~/.zshrc (or ~/.bashrc if using bash)
export HF_HOME=$SCRATCH/.cache/huggingface
export UV_CACHE_DIR=$SCRATCH/.cache/uv
export WANDB_DIR=$SCRATCH/.wandb

export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OMP_NUM_THREADS=16

# Then reload
source ~/.zshrc  # or source ~/.bashrc
```

**Why?** Cluster home directories have limited quota. This ensures all downloads (models, datasets, packages) go to project space with larger quota.

**Verify:**

```bash
uv cache dir
echo $HF_HOME
```

### 3. Clone Project

```bash
# Clone repository
git clone https://github.com/peternutter/amr-stronger-evidence-code.git
cd amr-stronger-evidence-code/deception_experiments
```

### 4. Configure Environment (.env file)

```bash
cp .env.template .env
# Edit .env with your API keys and paths
```

**Important environment variables:**

- `CLUSTER_WORK_DIR`: Set to `/path/to/cluster/project/llm-deception-data` for cluster runs (leave unset for local)
- `OPENAI_API_KEY`: Required for judge evaluations
- `PROJECT_ROOT`: Repo location (automatically set to current directory if not specified)

**Example `.env` for local development:**

```bash
OPENAI_API_KEY=your_openai_key_here
# CLUSTER_WORK_DIR is not set - uses local paths
```

**How it works:** If `CLUSTER_WORK_DIR` is set, data and logs go to cluster storage (large quota). Otherwise, they go to your local repo directory.

**Note:** `.env` is automatically loaded for all Hydra runs via `extras.load_env: true` in the config.

### 5. Install Dependencies

```bash
# Create venv and install dependencies
uv sync
```

### 6. Install Pre-commit Hooks (Optional)

```bash
# Install git hooks (one-time)
uv run pre-commit install
```

Auto-runs on commit: formatting, linting, lockfile check. Skip with `git commit --no-verify`.

## Usage Cheatsheet

### Package Management (uv)

```bash
# Run commands (no venv activation needed)
uv run python -m src.train
uv run pytest
uv run pytest tests/ --runslow    # To also run slow tests

# Add/remove packages
uv add torch transformers
uv add --dev pytest         # dev dependency
uv remove torch

# Update dependencies
uv lock --upgrade           # all packages
uv lock --upgrade torch     # specific package
uv sync                     # sync after changes
```

### Hydra Configuration

```bash
# Basic usage (calculate activations)
uv run python -m src.calculate_activations

# Override values
uv run python -m src.calculate_activations model=qwen2.5/1.5b data=mask

# Debug config
uv run python -m src.calculate_activations extras.print_config=true
HYDRA_FULL_ERROR=1 uv run python -m src.calculate_activations trainer=mps

# Test launcher config without submitting
uv run python -m src.calculate_activations hydra/launcher=cluster_b/default --cfg all --resolve

# Multirun/sweeps
uv run python -m src.calculate_activations --multirun model=qwen2.5/1.5b,qwen2.5/14b
```

**Config structure:**

```text
configs/
├── calculate_activations.yaml  # Main activation extraction config
├── train_probes.yaml           # Probe training config
├── evaluate_probes.yaml        # Probe evaluation config
├── mask_evaluate.yaml          # MASK evaluation config
├── data/                       # Dataset configs
├── model/                      # Model configs
├── trainer/                    # Trainer configs (gpu/cpu/mps)
└── hydra/launcher/             # Launcher configs
```

### Running on Cluster A

**Setup (every session):**

```bash
module load gcc/12.2.0 python/3.11.6 cuda/12.1.1
```

**Submit jobs:**

```bash
# Single job (--multirun required to activate launcher)
uv run python -m src.calculate_activations --multirun hydra/launcher=cluster_a/default

# Model sweep
uv run python -m src.calculate_activations --multirun \
  hydra/launcher=cluster_a/default \
  model=qwen2.5/1.5b,qwen2.5/14b,qwen2.5/72b

# Customize resources
uv run python -m src.calculate_activations --multirun \
  hydra/launcher=cluster_a/default \
  hydra.launcher.timeout_min=120 \
  hydra.launcher.cpus_per_task=8 \
  hydra.launcher.mem_per_cpu=32G
```

**Launchers:** Use `cluster_a`, `cluster_b`, `cpu`, or `default` depending on your needs.

**Monitor:**

```bash
myjobs                    # check status
scancel <job_id>          # cancel job
scancel -u $USER          # cancel all jobs
```

**Interactive debugging:**

```bash
srun --account=ls_account --partition=gpu.4h --gpus=rtx_4090:1 --cpus-per-task=4 --mincpus=4 --mem-per-cpu=16G --time=30 --pty zsh
module load stack/2025-06 gcc/12.2.0 zsh/5.9
zsh
uv run python -m src.calculate_activations
```

**Note:** Logging sometimes doesn't work as expected with the interactive mode. Also don't forget to cancel the job after you're done.

See [Experiments](experiments.md) for complete workflows.

### Running on Cluster B

**Submit jobs:**

```bash
# Single job (--multirun required to activate launcher)
uv run python -m src.calculate_activations --multirun hydra/launcher=cluster_b/default

# Model sweep
uv run python -m src.calculate_activations --multirun \
  hydra/launcher=cluster_b/default \
  model=qwen2.5/1.5b,qwen2.5/14b,qwen2.5/72b

# Customize resources
uv run python -m src.calculate_activations --multirun \
  hydra/launcher=cluster_b/default \
  hydra.launcher.timeout_min=120 \
  hydra.launcher.cpus_per_task=8 \
  hydra.launcher.mem_per_cpu=32G
```

**Monitor:**

```bash
squeue --me               # check status
scancel <job_id>          # cancel job
scancel -u $USER          # cancel all jobs
srun --overlap --jobid=1192501_0 nvidia-smi
srun --overlap --jobid=1192501_0 --pty bash
uv run nvitop

```

**Interactive debugging:**

```bash
srun --account=a-00 --partition=debug --time=1:30:00 --pty zsh
uv run python -m src.calculate_activations
```

**Note:** Logging sometimes doesn't work as expected with the interactive mode. Also don't forget to cancel the job after you're done.

See [Experiments](experiments.md) for complete workflows.

### Testing

```bash
uv run pytest                         # all tests
uv run pytest -q                      # less verbose
uv run pytest tests/test_data.py      # specific file
uv run pytest --cov=src               # with coverage
```

### Dashboard

The Streamlit dashboard visualizes probe evaluation results. For fast startup, precompute the cache first.

**Precompute cache (recommended):**

```bash
# Build cache from all experiment results (first time: slow, ~few minutes)
uv run python scripts/precompute_dashboard_cache.py

# Check cache status without rebuilding
uv run python scripts/precompute_dashboard_cache.py --check

# Force rebuild even if cache exists
uv run python scripts/precompute_dashboard_cache.py --force

# Use custom data directory
uv run python scripts/precompute_dashboard_cache.py --data-dir /path/to/data
```

**Run dashboard:**

```bash
# Start Streamlit app
uv run streamlit run src/app.py

# Or with custom data directory
uv run streamlit run src/app.py -- --data-dir /path/to/data
```

**How it works:**

- The cache stores aggregated metrics in parquet format (much faster than loading 40K+ pkl files)
- Cache automatically invalidates when result files change
- Dashboard shows a warning if cache is stale
- First cache build is slow; subsequent app launches are < 1 second

**Cache location:** `data/.dashboard_cache/` (or `$CLUSTER_WORK_DIR/data/.dashboard_cache/` on cluster)

### Pre-commit

```bash
uv run pre-commit run --all-files     # manual run
uv run pre-commit run ruff            # specific hook
uv run pre-commit autoupdate          # update hooks
git commit --no-verify                # skip (emergency)
```
