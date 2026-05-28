# Experiment Workflows

This document contains all commands to reproduce experimental results.

Each section shows commands for cluster sweeps first, followed by **Quick Test** commands at the end (for local testing with small samples).

## 1. Extract Activations for Probe Training

Extract activations from prompts and completions for all datasets. Depending on the dataset, this might need generation and/or labeling which will be handled automatically.

```bash
# Full sweep across all models and datasets
uv run python -m src.calculate_activations --multirun sweeps=activations_full
```

###  MASK (separate judge evaluation)

Aggregate belief prompts and determine deception for subsets with belief+pressure pattern. Sweeps over models (qwen 1.5b, 14b, 72b) and subsets (known_facts, disinformation, statistics, continuations). Needs to be run before probe training/evaluation.

```bash
# Evaluate all MASK subsets
uv run python -m src.mask_evaluate --multirun sweeps=mask_eval_full
```

## 2. Train Probes

Train probes on extracted activations from probe training datasets.

```bash
# Full sweep across all models, datasets, layers, and pooling strategies
uv run python -m src.train_probes --multirun sweeps=train_probes_full
```

## 3. Probe Evaluation

Evaluate trained probes across all model and dataset combinations.

**Note:** The evaluation includes **Recall @ 1% FPR** metric which requires control data (alpaca) activations. Make sure to compute alpaca activations first (included in `activations_full` sweep).

```bash
# Full sweep for probe evaluation
uv run python -m src.evaluate_probes --multirun sweeps=evaluate_probes_full
```

Evaluate pre-trained Apollo probes on all datasets.

```bash
uv run python -m src.evaluate_probes_apollo --multirun sweeps=evaluate_probes_apollo
```

---

## Debug Sweeps (Quick Multi-Run Testing)

These commands run small sweeps with `max_samples=4` and `qwen2.5/0.5b` on CPU for fast testing.

### 1. Debug Activation Extraction

Test activation extraction on subset of datasets with 4 samples each.

```bash
uv run python -m src.calculate_activations --multirun sweeps=activations_debug
```

#### Debug MASK Evaluation

Test MASK evaluation with known_facts subset. Needs to be run before probe training/evaluation.

```bash
uv run python -m src.mask_evaluate --multirun sweeps=mask_eval_debug
```

### 2. Debug Probe Training

Test probe training with both pooling strategies (last, null) and activation sources (prompt, completion).

```bash
uv run python -m src.train_probes --multirun sweeps=train_probes_debug
```

### 3. Debug Probe Evaluation

Test probe evaluation across train/eval dataset combinations.

```bash
uv run python -m src.evaluate_probes --multirun sweeps=evaluate_probes_debug
```
