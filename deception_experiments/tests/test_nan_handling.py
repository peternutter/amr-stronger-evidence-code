"""Test NaN handling in judge metrics."""

import numpy as np
import pytest
from src.utils.judge_metrics import (
    check_data_completeness,
    compute_agreement_rate,
    compute_fleiss_kappa,
    compute_pairwise_kappa,
    get_disagreement_indices,
)


def test_check_data_completeness_with_nan():
    """Test that check_data_completeness correctly identifies NaN values."""
    labels_by_judge = {
        "judge_a": [1, 0, 1, np.nan, 1],
        "judge_b": [1, 0, 1, 1, np.nan],
        "judge_c": [1, 0, 1, 1, 1],
    }

    completeness = check_data_completeness(labels_by_judge)

    assert completeness["judge_a"] == 0.2  # 1 out of 5 is NaN
    assert completeness["judge_b"] == 0.2  # 1 out of 5 is NaN
    assert completeness["judge_c"] == 0.0  # No NaN


def test_check_data_completeness_with_none():
    """Test that check_data_completeness correctly identifies None values."""
    labels_by_judge = {
        "judge_a": [1, 0, None, 1, 1],
        "judge_b": [1, 0, 1, 1, None],
    }

    completeness = check_data_completeness(labels_by_judge)

    assert completeness["judge_a"] == 0.2  # 1 out of 5 is None
    assert completeness["judge_b"] == 0.2  # 1 out of 5 is None


def test_compute_pairwise_kappa_with_nan():
    """Test that compute_pairwise_kappa handles NaN values without crashing."""
    labels_by_judge = {
        "judge_a": [1, 0, 1, np.nan, 1, 0, 1, 0],
        "judge_b": [1, 0, 1, 1, np.nan, 0, 1, 0],
    }

    # Should not raise ValueError
    kappa_df = compute_pairwise_kappa(labels_by_judge)

    # Check that it returns a valid DataFrame
    assert kappa_df is not None
    assert kappa_df.shape == (2, 2)
    assert kappa_df.loc["judge_a", "judge_a"] == 1.0
    assert kappa_df.loc["judge_b", "judge_b"] == 1.0
    # Cross-kappa should be computed without NaN values
    assert not np.isnan(kappa_df.loc["judge_a", "judge_b"])


def test_compute_agreement_rate_with_nan():
    """Test that compute_agreement_rate handles NaN values correctly."""
    labels_by_judge = {
        "judge_a": [1, 1, 0, np.nan, 1],
        "judge_b": [1, 1, 0, 1, np.nan],
        "judge_c": [1, 1, 1, 1, 1],
    }

    # Should not raise ValueError
    rate = compute_agreement_rate(labels_by_judge)

    # Only samples 0 and 1 have all judges agreeing (both have all 1s)
    # Sample 2 has disagreement (0, 0, 1)
    # Samples 3 and 4 have NaN so are skipped
    # So 2 out of 3 valid samples agree
    assert rate == pytest.approx(2 / 3)


def test_compute_fleiss_kappa_with_nan():
    """Test that compute_fleiss_kappa handles NaN values correctly."""
    labels_by_judge = {
        "judge_a": [1, 0, 1, np.nan, 1],
        "judge_b": [1, 0, 1, 1, np.nan],
        "judge_c": [1, 0, 1, 1, 1],
    }

    # Should not raise ValueError
    kappa = compute_fleiss_kappa(labels_by_judge)

    # Should return a valid kappa score
    assert isinstance(kappa, float)
    assert not np.isnan(kappa)


def test_get_disagreement_indices_with_nan():
    """Test that get_disagreement_indices handles NaN values correctly."""
    labels_by_judge = {
        "judge_a": [1, 1, 0, np.nan, 1],
        "judge_b": [1, 0, 0, 1, np.nan],
        "judge_c": [1, 0, 1, 1, 1],
    }

    # Should not raise ValueError
    disagreements = get_disagreement_indices(labels_by_judge)

    # Sample 0: all 1s -> no disagreement
    # Sample 1: 1, 0, 0 -> disagreement
    # Sample 2: 0, 0, 1 -> disagreement
    # Sample 3: has NaN -> skipped
    # Sample 4: has NaN -> skipped
    assert disagreements == [1, 2]


def test_all_nan_data():
    """Test handling when all data is NaN."""
    labels_by_judge = {
        "judge_a": [np.nan, np.nan, np.nan],
        "judge_b": [np.nan, np.nan, np.nan],
    }

    completeness = check_data_completeness(labels_by_judge)
    assert all(c == 1.0 for c in completeness.values())

    # Should handle gracefully
    rate = compute_agreement_rate(labels_by_judge)
    assert rate == 0.0

    kappa_df = compute_pairwise_kappa(labels_by_judge)
    assert kappa_df is not None

    kappa = compute_fleiss_kappa(labels_by_judge)
    # When there are no valid categories, returns 1.0 (perfect agreement on nothing)
    assert kappa == 1.0

    disagreements = get_disagreement_indices(labels_by_judge)
    assert disagreements == []
