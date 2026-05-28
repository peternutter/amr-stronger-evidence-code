"""Metrics for multi-judge agreement analysis.

Provides functions to compute inter-rater agreement metrics for comparing
multiple judge models' evaluations of the same samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def check_data_completeness(labels_by_judge: dict[str, list[int]]) -> dict[str, float]:
    """Check completeness of judge data and report missing percentages.

    Args:
        labels_by_judge: Dict mapping judge name to list of labels per sample.

    Returns:
        Dict mapping judge name to percentage of missing data (0.0 to 1.0).
    """
    completeness = {}
    for judge, labels in labels_by_judge.items():
        n_total = len(labels)
        n_missing = sum(1 for label in labels if label is None or (isinstance(label, float) and np.isnan(label)))
        completeness[judge] = n_missing / n_total if n_total > 0 else 0.0
    return completeness


def compute_pairwise_kappa(labels_by_judge: dict[str, list[int]]) -> pd.DataFrame:
    """Compute Cohen's Kappa for all pairs of judges.

    Args:
        labels_by_judge: Dict mapping judge name to list of labels per sample.
            All judges must have the same number of samples.

    Returns:
        DataFrame with Kappa scores, indexed by judge pairs.
    """
    judges = list(labels_by_judge.keys())
    n_judges = len(judges)

    # Build matrix
    kappa_matrix = np.zeros((n_judges, n_judges))

    for i, judge_a in enumerate(judges):
        for j, judge_b in enumerate(judges):
            if i == j:
                kappa_matrix[i, j] = 1.0
            elif i < j:
                labels_a = labels_by_judge[judge_a]
                labels_b = labels_by_judge[judge_b]
                # Filter out None and NaN values
                valid_mask = [
                    (
                        a is not None
                        and b is not None
                        and not (isinstance(a, float) and np.isnan(a))
                        and not (isinstance(b, float) and np.isnan(b))
                    )
                    for a, b in zip(labels_a, labels_b, strict=False)
                ]
                valid_a = [a for a, v in zip(labels_a, valid_mask, strict=False) if v]
                valid_b = [b for b, v in zip(labels_b, valid_mask, strict=False) if v]

                if len(valid_a) > 1 and len(set(valid_a)) > 1 and len(set(valid_b)) > 1:
                    kappa = cohen_kappa_score(valid_a, valid_b)
                else:
                    kappa = np.nan  # Can't compute if all same label
                kappa_matrix[i, j] = kappa
                kappa_matrix[j, i] = kappa

    return pd.DataFrame(kappa_matrix, index=judges, columns=judges)


def compute_pairwise_agreement_rate(labels_by_judge: dict[str, list[int]]) -> pd.DataFrame:
    """Compute agreement rate for all pairs of judges.

    Args:
        labels_by_judge: Dict mapping judge name to list of labels per sample.

    Returns:
        DataFrame with agreement rates, indexed by judge pairs.
    """
    judges = list(labels_by_judge.keys())
    n_judges = len(judges)

    # Build matrix
    agreement_matrix = np.zeros((n_judges, n_judges))

    for i, judge_a in enumerate(judges):
        for j, judge_b in enumerate(judges):
            if i == j:
                agreement_matrix[i, j] = 1.0
            else:
                labels_a = labels_by_judge[judge_a]
                labels_b = labels_by_judge[judge_b]

                # Compute agreement between judge_a and judge_b
                agree_count = 0
                valid_count = 0

                for a, b in zip(labels_a, labels_b, strict=False):
                    # Skip if either is None or NaN
                    if (
                        a is None
                        or (isinstance(a, float) and np.isnan(a))
                        or b is None
                        or (isinstance(b, float) and np.isnan(b))
                    ):
                        continue
                    valid_count += 1
                    if a == b:
                        agree_count += 1

                agreement = agree_count / valid_count if valid_count > 0 else np.nan
                agreement_matrix[i, j] = agreement
                agreement_matrix[j, i] = agreement

    return pd.DataFrame(agreement_matrix, index=judges, columns=judges)


def compute_pairwise_fleiss_kappa(labels_by_judge: dict[str, list[int]]) -> pd.DataFrame:
    """Compute Fleiss' Kappa for all pairs of judges.

    For each pair of judges, computes Fleiss' Kappa treating it as a 2-rater problem.

    Args:
        labels_by_judge: Dict mapping judge name to list of labels per sample.

    Returns:
        DataFrame with Fleiss' Kappa scores, indexed by judge pairs.
    """
    judges = list(labels_by_judge.keys())
    n_judges = len(judges)

    # Build matrix
    fleiss_matrix = np.zeros((n_judges, n_judges))

    for i, judge_a in enumerate(judges):
        for j, judge_b in enumerate(judges):
            if i == j:
                fleiss_matrix[i, j] = 1.0
            else:
                labels_a = labels_by_judge[judge_a]
                labels_b = labels_by_judge[judge_b]

                # For two raters, Fleiss' Kappa equals Cohen's Kappa
                valid_mask = [
                    (
                        a is not None
                        and not (isinstance(a, float) and np.isnan(a))
                        and b is not None
                        and not (isinstance(b, float) and np.isnan(b))
                    )
                    for a, b in zip(labels_a, labels_b, strict=False)
                ]
                valid_a = [a for a, v in zip(labels_a, valid_mask, strict=False) if v]
                valid_b = [b for b, v in zip(labels_b, valid_mask, strict=False) if v]

                if len(valid_a) > 1 and len(set(valid_a)) > 1 and len(set(valid_b)) > 1:
                    kappa = cohen_kappa_score(valid_a, valid_b)
                else:
                    kappa = np.nan
                fleiss_matrix[i, j] = kappa
                fleiss_matrix[j, i] = kappa

    return pd.DataFrame(fleiss_matrix, index=judges, columns=judges)


def compute_agreement_rate(labels_by_judge: dict[str, list[int]]) -> float:
    """Compute percentage of samples where all judges agree.

    Args:
        labels_by_judge: Dict mapping judge name to list of labels per sample.

    Returns:
        Agreement rate as a float between 0 and 1.
    """
    if not labels_by_judge:
        return 0.0

    judges = list(labels_by_judge.keys())
    n_samples = len(next(iter(labels_by_judge.values())))

    if n_samples == 0:
        return 0.0

    agree_count = 0
    valid_count = 0

    for i in range(n_samples):
        sample_labels = [labels_by_judge[j][i] for j in judges]
        # Skip if any None or NaN
        if any(label is None or (isinstance(label, float) and np.isnan(label)) for label in sample_labels):
            continue
        valid_count += 1
        if len(set(sample_labels)) == 1:
            agree_count += 1

    return agree_count / valid_count if valid_count > 0 else 0.0


def compute_fleiss_kappa(labels_by_judge: dict[str, list[int]]) -> float:
    """Compute Fleiss' Kappa for multi-rater agreement.

    Fleiss' Kappa generalizes Cohen's Kappa to more than two raters.

    Args:
        labels_by_judge: Dict mapping judge name to list of labels per sample.

    Returns:
        Fleiss' Kappa score.
    """
    if not labels_by_judge:
        return 0.0

    judges = list(labels_by_judge.keys())
    n_raters = len(judges)
    n_samples = len(next(iter(labels_by_judge.values())))

    if n_samples == 0 or n_raters < 2:
        return 0.0

    # Collect all unique labels
    all_labels = set()
    for labels in labels_by_judge.values():
        all_labels.update(
            label for label in labels if label is not None and not (isinstance(label, float) and np.isnan(label))
        )
    categories = sorted(all_labels)
    n_categories = len(categories)

    if n_categories < 2:
        return 1.0  # All same label = perfect agreement

    # Build rating matrix: n_samples x n_categories
    # Each cell is count of raters who assigned that category
    rating_matrix = np.zeros((n_samples, n_categories))
    valid_samples = []

    for i in range(n_samples):
        sample_labels = [labels_by_judge[j][i] for j in judges]
        # Skip if any None or NaN
        if any(label is None or (isinstance(label, float) and np.isnan(label)) for label in sample_labels):
            continue
        valid_samples.append(i)
        for label in sample_labels:
            cat_idx = categories.index(label)
            rating_matrix[i, cat_idx] += 1

    if len(valid_samples) == 0:
        return 0.0

    rating_matrix = rating_matrix[valid_samples]
    n_valid = len(valid_samples)

    # Compute Fleiss' Kappa
    # p_j = proportion of all assignments to category j
    p = rating_matrix.sum(axis=0) / (n_valid * n_raters)

    # P_i = extent of agreement for sample i
    P = (np.sum(rating_matrix**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))

    # Mean P
    P_bar = P.mean()

    # P_e = expected agreement by chance
    P_e = np.sum(p**2)

    if P_e == 1.0:
        return 1.0  # Perfect agreement

    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


def get_disagreement_indices(labels_by_judge: dict[str, list[int]]) -> list[int]:
    """Return indices of samples where judges disagree.

    Args:
        labels_by_judge: Dict mapping judge name to list of labels per sample.

    Returns:
        List of sample indices where at least two judges gave different labels.
    """
    if not labels_by_judge:
        return []

    judges = list(labels_by_judge.keys())
    n_samples = len(next(iter(labels_by_judge.values())))

    disagreements = []
    for i in range(n_samples):
        sample_labels = [labels_by_judge[j][i] for j in judges]
        # Skip if any None or NaN
        valid_labels = [
            label for label in sample_labels if label is not None and not (isinstance(label, float) and np.isnan(label))
        ]
        if len(valid_labels) > 1 and len(set(valid_labels)) > 1:
            disagreements.append(i)

    return disagreements


def get_label_distribution(labels_by_judge: dict[str, list[int]]) -> pd.DataFrame:
    """Get label distribution per judge.

    Args:
        labels_by_judge: Dict mapping judge name to list of labels per sample.

    Returns:
        DataFrame with columns [judge, label, count, proportion].
    """
    rows = []
    for judge, labels in labels_by_judge.items():
        valid_labels = [label for label in labels if label is not None]
        total = len(valid_labels)
        for label in set(valid_labels):
            count = valid_labels.count(label)
            rows.append(
                {
                    "judge": judge,
                    "label": label,
                    "count": count,
                    "proportion": count / total if total > 0 else 0,
                }
            )
    return pd.DataFrame(rows)
