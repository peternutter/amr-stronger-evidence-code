#!/usr/bin/env python3
"""Export all publication-ready plots and tables.

This script defines all exports in a declarative way at the bottom.
Run: uv run python scripts/export_all.py

The export functions abstract away the complexity - just call:
    heatmap("accuracy", train_datasets=[...], eval_datasets=[...])
    table("calibrated", eval_datasets=[...])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from src.dashboard.data_loader import DashboardDataLoader
from src.plots.metrics_shared import compute_all_classification_metrics

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths (from environment, matching Hydra configs)
_CLUSTER_WORK_DIR = os.environ.get("CLUSTER_WORK_DIR", "")
DATA_DIR = Path(f"{_CLUSTER_WORK_DIR}/data") if _CLUSTER_WORK_DIR else Path("./data")
OUTPUT_DIR = DATA_DIR / "exports"

# Default model config
MODEL = "Llama_Llama-3.3-70B-Instruct"
LAYER = 22
POOLING = "flat"
AGGREGATION = "mean"

# =============================================================================
# DATASET NAME MAPPING
# =============================================================================

DATASET_NAMES = {
    # Roleplaying
    "roleplaying-plain": "Roleplaying",
    "roleplaying-actor": "Roleplaying (Actor)",
    "roleplaying-offpolicy_train": "Roleplaying (Off-Policy)",
    "roleplaying-recital": "Roleplaying (Recital)",
    # Instructed
    "instructed_alien": "Instructed (Alien)",
    "instructed_peasant": "Instructed (Peasant)",
    "instructed_sarcasm": "Instructed (Sarcasm)",
    "instructed_counterfactual": "Instructed (Counterfact.)",
    "instructed_wrong_answers": "Instructed (Wrong Ans.)",
    "instructed_pairs": "Instructed Pairs",
    # DeceptionBench
    "deception_bench-L1_self": "DB L1 Self",
    "deception_bench-L1_other": "DB L1 Other",
    "deception_bench-L2_self_pressure": "DB L2 Self Pres.",
    "deception_bench-L2_self_reward": "DB L2 Self Rew.",
    "deception_bench-L2_other_pressure": "DB L2 Other Pres.",
    "deception_bench-L2_other_reward": "DB L2 Other Rew.",
    # MASK
    "mask-continuations-labeled": "MASK Contin.",
    "mask-disinformation-labeled": "MASK Disinfo.",
    "mask-known_facts-labeled": "MASK Known",
    "mask-provided_facts-None": "MASK Provided",
    "mask-statistics-labeled": "MASK Stats",
    # Other
    "insider_trading-onpolicy": "Insider Trading",
    "sandbagging_v2-wmdp_mmlu": "Sandbagging",
    "alpaca": "Alpaca (Control)",
}

TRAIN_NAMES = {
    "instructed_pairs": "Ours (Instr. Pairs)",
    "roleplaying-offpolicy_train": "Ours (RP Off-Policy)",
    "apollo_instructed_pairs": "Apollo (Instr. Pairs)",
    "apollo_roleplaying": "Apollo (Roleplaying)",
}

# Train dataset -> probe mapping
PROBE_FOR_TRAIN = {
    "instructed_pairs": "logistic_regression",
    "roleplaying-offpolicy_train": "logistic_regression",
    "apollo_instructed_pairs": "apollo_instructed_pairs",
    "apollo_roleplaying": "apollo_roleplaying",
}

# =============================================================================
# FORMATTING
# =============================================================================


def fmt_pct(val: float) -> str:
    """Format as percentage with 1 decimal."""
    if val is None or np.isnan(val):
        return "—"
    return f"{val * 100:.1f}"


def fmt_int(val: float) -> str:
    """Format as integer."""
    if val is None or np.isnan(val):
        return "—"
    return str(int(val))


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

_loader: DashboardDataLoader | None = None


def _get_loader() -> DashboardDataLoader:
    global _loader
    if _loader is None:
        _loader = DashboardDataLoader(DATA_DIR)
    return _loader


def _get_metrics(
    train_dataset: str,
    eval_dataset: str,
    use_calibrated: bool = True,
) -> dict | None:
    """Get metrics for a train/eval pair."""
    loader = _get_loader()
    probe = PROBE_FOR_TRAIN.get(train_dataset, "logistic_regression")

    result_data = loader.get_confusion_data(
        train_dataset=train_dataset,
        probe=probe,
        model=MODEL,
        eval_dataset=eval_dataset,
        layer=LAYER,
        pooling=POOLING,
        aggregation=AGGREGATION,
    )

    if result_data is None:
        return None

    y_true = np.array(result_data.get("ground_truth", []))
    raw_scores = result_data.get("raw_scores") or result_data.get("logits_honest")
    calibrated_threshold = result_data.get("calibrated_threshold")

    if raw_scores is None or len(y_true) == 0:
        return None

    raw_scores = np.array(raw_scores)
    threshold = calibrated_threshold if (use_calibrated and calibrated_threshold) else 0.0
    y_pred = (raw_scores > threshold).astype(int)

    return compute_all_classification_metrics(y_true, y_pred)


def _get_roc_auc(
    train_dataset: str,
    eval_dataset: str,
) -> float | None:
    """Get ROC AUC for a train/eval pair (threshold-independent)."""
    loader = _get_loader()
    probe = PROBE_FOR_TRAIN.get(train_dataset, "logistic_regression")
    df = loader.all_metrics

    mask = (
        (df["train_dataset"] == train_dataset)
        & (df["probe"] == probe)
        & (df["model"] == MODEL)
        & (df["eval_dataset"] == eval_dataset)
        & (df["layer"] == LAYER)
        & (df["pooling"] == POOLING)
        & (df["aggregation"] == AGGREGATION)
    )

    filtered = df[mask]
    if len(filtered) == 0:
        return None

    return filtered.iloc[0].get("roc_auc")


# =============================================================================
# PUBLIC API
# =============================================================================


def heatmap(
    metric: Literal["accuracy", "recall", "f1", "precision"],
    train_datasets: list[str],
    eval_datasets: list[str],
    use_calibrated: bool = True,
    filename: str | None = None,
):
    """Export a cross-dataset heatmap.

    Args:
        metric: Metric to visualize (accuracy, recall, f1, precision)
        train_datasets: List of training datasets (rows)
        eval_datasets: List of evaluation datasets (columns)
        use_calibrated: Use calibrated threshold (True) or default (False)
        filename: Output filename (auto-generated if None)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        suffix = "calibrated" if use_calibrated else "default"
        filename = f"heatmap_{metric}_{suffix}.pdf"

    # Build matrix
    matrix = np.full((len(train_datasets), len(eval_datasets)), np.nan)
    for i, train_ds in enumerate(train_datasets):
        for j, eval_ds in enumerate(eval_datasets):
            m = _get_metrics(train_ds, eval_ds, use_calibrated)
            if m and metric in m:
                matrix[i, j] = m[metric]

    # Create figure
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(12, max(3, len(train_datasets) * 0.8)))

    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad(color="lightgray")
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.set_ylabel(metric.title(), rotation=270, labelpad=15)

    # Labels
    row_labels = [TRAIN_NAMES.get(ds, ds) for ds in train_datasets]
    col_labels = [DATASET_NAMES.get(ds, ds) for ds in eval_datasets]

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(row_labels)

    # Annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            if np.isnan(val):
                text, color = "—", "gray"
            else:
                text = fmt_pct(val)
                color = "white" if val < 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9, fontweight="medium")

    ax.set_xlabel("Evaluation Dataset")
    ax.set_ylabel("Training Dataset / Probe")
    threshold_label = "Calibrated" if use_calibrated else "Default"
    ax.set_title(f"{metric.title()} ({threshold_label} Threshold)", pad=15)

    # Grid
    ax.set_xticks(np.arange(len(col_labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {filename}")


def roc_auc_heatmap(
    train_datasets: list[str],
    eval_datasets: list[str],
    filename: str | None = None,
):
    """Export a ROC AUC heatmap (threshold-independent).

    Args:
        train_datasets: List of training datasets (rows)
        eval_datasets: List of evaluation datasets (columns)
        filename: Output filename (auto-generated if None)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = "heatmap_roc_auc.pdf"

    # Build matrix
    matrix = np.full((len(train_datasets), len(eval_datasets)), np.nan)
    for i, train_ds in enumerate(train_datasets):
        for j, eval_ds in enumerate(eval_datasets):
            auc = _get_roc_auc(train_ds, eval_ds)
            if auc is not None:
                matrix[i, j] = auc

    # Create figure
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(12, max(3, len(train_datasets) * 0.8)))

    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad(color="lightgray")
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0.5, vmax=1)  # ROC AUC: 0.5-1

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.set_ylabel("ROC AUC", rotation=270, labelpad=15)

    # Labels
    row_labels = [TRAIN_NAMES.get(ds, ds) for ds in train_datasets]
    col_labels = [DATASET_NAMES.get(ds, ds) for ds in eval_datasets]

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(row_labels)

    # Annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            if np.isnan(val):
                text, color = "—", "gray"
            else:
                text = fmt_pct(val)
                color = "white" if val < 0.75 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9, fontweight="medium")

    ax.set_xlabel("Evaluation Dataset")
    ax.set_ylabel("Training Dataset / Probe")
    ax.set_title("ROC AUC", pad=15)

    # Grid
    ax.set_xticks(np.arange(len(col_labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {filename}")


def table(
    name: str,
    train_dataset: str,
    eval_datasets: list[str],
    use_calibrated: bool = True,
    include_counts: bool = True,
):
    """Export a metrics table as LaTeX.

    Args:
        name: Output filename (without extension)
        train_dataset: Training dataset to use
        eval_datasets: List of evaluation datasets (rows)
        use_calibrated: Use calibrated threshold
        include_counts: Include TP/TN/FP/FN columns
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for eval_ds in eval_datasets:
        m = _get_metrics(train_dataset, eval_ds, use_calibrated)
        if m is None:
            continue
        rows.append(
            {
                "Dataset": DATASET_NAMES.get(eval_ds, eval_ds),
                "TP": fmt_int(m["tp"]),
                "TN": fmt_int(m["tn"]),
                "FP": fmt_int(m["fp"]),
                "FN": fmt_int(m["fn"]),
                "Acc": fmt_pct(m["accuracy"]),
                "Prec": fmt_pct(m["precision"]),
                "Recall": fmt_pct(m["recall"]),
                "F1": fmt_pct(m["f1"]),
            }
        )

    # LaTeX
    if include_counts:
        header = r"Dataset & TP & TN & FP & FN & Acc & Prec & Recall & F1 \\"
        cols = "l|rrrr|rrrr"
    else:
        header = r"Dataset & Acc & Prec & Recall & F1 \\"
        cols = "l|rrrr"

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{Probe performance ({name}, \\%)}}",
        f"  \\label{{tab:{name}}}",
        r"  \small",
        f"  \\begin{{tabular}}{{{cols}}}",
        r"    \toprule",
        f"    {header}",
        r"    \midrule",
    ]

    for r in rows:
        if include_counts:
            vals = [r["Dataset"], r["TP"], r["TN"], r["FP"], r["FN"], r["Acc"], r["Prec"], r["Recall"], r["F1"]]
        else:
            vals = [r["Dataset"], r["Acc"], r["Prec"], r["Recall"], r["F1"]]
        lines.append("    " + " & ".join(vals) + r" \\")

    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])

    (OUTPUT_DIR / f"{name}.tex").write_text("\n".join(lines))
    print(f"✓ {name}.tex")


def comparison_table(
    name: str,
    train_dataset: str,
    eval_datasets: list[str],
):
    """Export calibrated vs default comparison table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for eval_ds in eval_datasets:
        m_cal = _get_metrics(train_dataset, eval_ds, use_calibrated=True)
        m_def = _get_metrics(train_dataset, eval_ds, use_calibrated=False)
        if m_cal is None or m_def is None:
            continue
        rows.append(
            {
                "Dataset": DATASET_NAMES.get(eval_ds, eval_ds),
                "Recall_cal": fmt_pct(m_cal["recall"]),
                "Prec_cal": fmt_pct(m_cal["precision"]),
                "F1_cal": fmt_pct(m_cal["f1"]),
                "Recall_def": fmt_pct(m_def["recall"]),
                "Prec_def": fmt_pct(m_def["precision"]),
                "F1_def": fmt_pct(m_def["f1"]),
            }
        )

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        "  \\caption{Calibrated vs Default threshold (\\%)}",
        f"  \\label{{tab:{name}}}",
        r"  \small",
        r"  \begin{tabular}{l|rrr|rrr}",
        r"    \toprule",
        r"    & \multicolumn{3}{c|}{Calibrated} & \multicolumn{3}{c}{Default} \\",
        r"    Dataset & Recall & Prec & F1 & Recall & Prec & F1 \\",
        r"    \midrule",
    ]

    for r in rows:
        vals = [r["Dataset"], r["Recall_cal"], r["Prec_cal"], r["F1_cal"], r["Recall_def"], r["Prec_def"], r["F1_def"]]
        lines.append("    " + " & ".join(vals) + r" \\")

    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])

    (OUTPUT_DIR / f"{name}.tex").write_text("\n".join(lines))
    print(f"✓ {name}.tex")


# =============================================================================
# EXPORT DEFINITIONS
# =============================================================================

# Training datasets for heatmaps
TRAIN_DATASETS = [
    "instructed_pairs",
    "roleplaying-offpolicy_train",
    "apollo_instructed_pairs",
    "apollo_roleplaying",
]

# Evaluation datasets (no roleplaying-actor)
EVAL_DATASETS = [
    "roleplaying-plain",
    "roleplaying-offpolicy_train",
    "instructed_alien",
    "instructed_peasant",
    "instructed_sarcasm",
    "instructed_counterfactual",
    "instructed_wrong_answers",
    "instructed_pairs",
]

# DeceptionBench evaluation datasets
EVAL_DECEPTION_BENCH = [
    "deception_bench-L1_self",
    "deception_bench-L1_other",
    "deception_bench-L2_self_pressure",
    "deception_bench-L2_self_reward",
    "deception_bench-L2_other_pressure",
    "deception_bench-L2_other_reward",
]

# MASK evaluation datasets
EVAL_MASK = [
    "mask-continuations-labeled",
    "mask-disinformation-labeled",
    "mask-known_facts-labeled",
    "mask-provided_facts-None",
    "mask-statistics-labeled",
]

# Combined OOD evaluation (DeceptionBench + MASK)
EVAL_OOD = EVAL_DECEPTION_BENCH + EVAL_MASK

if __name__ == "__main__":
    print(f"Output: {OUTPUT_DIR}\n")

    # Standard heatmaps (threshold-dependent)
    heatmap("accuracy", TRAIN_DATASETS, EVAL_DATASETS, use_calibrated=True)
    heatmap("accuracy", TRAIN_DATASETS, EVAL_DATASETS, use_calibrated=False)
    heatmap("recall", TRAIN_DATASETS, EVAL_DATASETS, use_calibrated=True)
    heatmap("recall", TRAIN_DATASETS, EVAL_DATASETS, use_calibrated=False)
    heatmap("f1", TRAIN_DATASETS, EVAL_DATASETS, use_calibrated=True)
    heatmap("f1", TRAIN_DATASETS, EVAL_DATASETS, use_calibrated=False)

    # ROC AUC heatmaps (threshold-independent)
    roc_auc_heatmap(TRAIN_DATASETS, EVAL_DATASETS, "heatmap_roc_auc.pdf")
    roc_auc_heatmap(TRAIN_DATASETS, EVAL_OOD, "heatmap_roc_auc_ood.pdf")  # DeceptionBench + MASK combined

    # Tables
    table("metrics_calibrated", "instructed_pairs", EVAL_DATASETS, use_calibrated=True)
    table("metrics_default", "instructed_pairs", EVAL_DATASETS, use_calibrated=False)
    comparison_table("metrics_comparison", "instructed_pairs", EVAL_DATASETS)

    print(f"\n✓ Done! Files in {OUTPUT_DIR}")
