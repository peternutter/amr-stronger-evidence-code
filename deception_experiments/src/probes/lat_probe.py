import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler


class LATProbe(BaseEstimator, ClassifierMixin):
    """
    Linear Artificial Tomography (LAT) probe from Representation Engineering.

    Computes the direction by:
    1. Computing difference vectors between paired positive/negative samples
    2. Randomly flipping signs of half the difference vectors
    3. Taking the first principal component (PCA)
    4. Ensuring the direction points toward the positive class

    Args:
        threshold: Decision threshold for classification (default: 0.0)
        normalize: Whether to standardize features before computing direction
        random_state: Random seed for sign flipping
    """

    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def __init__(
        self,
        safe_name: str,
        threshold: float = 0.0,
        normalize: bool = True,
        random_state: int = 42,
    ):
        self.safe_name = safe_name
        self.threshold = threshold
        self.normalize = normalize
        self.random_state = random_state
        self.direction_ = None
        self.scaler_ = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the probe using LAT method.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (n_samples,)
        """
        # Store classes for sklearn compatibility
        self.classes_ = np.unique(y)

        if self.normalize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        # Separate classes (assuming paired data)
        pos_mask = y == 1
        neg_mask = y == 0
        X_pos = X[pos_mask]
        X_neg = X[neg_mask]

        # Compute difference vectors
        min_len = min(len(X_pos), len(X_neg))
        diffs = X_pos[:min_len] - X_neg[:min_len]

        # Randomly flip signs
        rng = np.random.default_rng(self.random_state)
        signs = np.where(rng.standard_normal(min_len) > 0, 1, -1)
        diffs = diffs * signs[:, np.newaxis]

        # Mean-center the differences
        diffs_centered = diffs - diffs.mean(axis=0)

        # if diffs are all the same, the direction is 0
        if np.allclose(diffs_centered, 0):
            self.direction_ = np.zeros(X.shape[1])
            return self

        # Compute first principal component
        # Using SVD: X = U @ S @ Vt, first PC is first column of V
        _, _, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
        self.direction_ = Vt[0]  # First row of Vt = first column of V

        # Ensure direction points toward positive class
        # by checking if positive samples have higher scores on average
        pos_scores = X_pos @ self.direction_
        neg_scores = X_neg @ self.direction_
        if pos_scores.mean() < neg_scores.mean():
            self.direction_ = -self.direction_

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision scores by projecting X onto direction.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Decision scores (n_samples,)
        """
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        scores = X @ self.direction_
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        scores = self.decision_function(X)
        return (scores > self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using sigmoid.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Class probabilities (n_samples, 2)
        """
        scores = self.decision_function(X)
        proba_pos = 1 / (1 + np.exp(-scores))
        proba_neg = 1 - proba_pos
        return np.column_stack([proba_neg, proba_pos])

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            "safe_name": self.safe_name,
            "threshold": self.threshold,
            "normalize": self.normalize,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
