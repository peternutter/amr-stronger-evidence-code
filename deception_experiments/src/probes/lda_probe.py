import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


class LDAProbe(BaseEstimator, ClassifierMixin):
    """
    Linear Discriminant Analysis (LDA) / Mean-Minus-Mean (MMS) probe.

    Computes the direction as the difference of means, optionally adjusted by the pooled
    covariance matrix (similar to Fisher's LDA).

    When use_covariance=False, this is equivalent to MMS (simple difference of means).
    When use_covariance=True, this is LDA with covariance adjustment.

    Args:
        threshold: Decision threshold for classification (default: 0.0)
        use_covariance: Whether to adjust by covariance matrix (default: True)
        regularization: Regularization parameter for covariance matrix (default: 0.01)
        normalize: Whether to standardize features before computing direction
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
        use_covariance: bool = True,
        regularization: float = 0.01,
        normalize: bool = False,
    ):
        self.safe_name = safe_name
        self.threshold = threshold
        self.use_covariance = use_covariance
        self.regularization = regularization
        self.normalize = normalize
        self.direction_ = None
        self.scaler_ = None
        self.classes_ = None

    @classmethod
    def create_tuned_estimator(cls, cv_folds: int = 5, n_jobs: int = -1, **probe_params):
        """
        Returns a GridSearchCV-wrapped LDA/MMS probe with automatic hyperparameter tuning.

        Tuning strategy:
        - regularization: Tests multiple regularization strengths (4 values)
        - All other parameters (use_covariance, threshold, normalize, etc.) are taken from probe_params

        Args:
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 uses all cores)
            **probe_params: All parameters from the probe config (passed through to estimator)

        Returns:
            GridSearchCV estimator that will find best hyperparameters during fit()
        """
        estimator = cls(**probe_params)

        # Only tune regularization
        param_grid = {
            "regularization": [0.001, 0.01, 0.1, 1.0],
        }

        return GridSearchCV(
            estimator, param_grid, cv=cv_folds, n_jobs=n_jobs, scoring="accuracy", refit=True, verbose=1
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the probe by computing (optionally covariance-adjusted) difference of means.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (n_samples,)
        """
        # Store classes for sklearn compatibility
        self.classes_ = np.unique(y)

        if self.normalize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        # Separate classes
        pos_mask = y == 1
        neg_mask = y == 0
        X_pos = X[pos_mask]
        X_neg = X[neg_mask]

        # Compute means
        pos_mean = X_pos.mean(axis=0)
        neg_mean = X_neg.mean(axis=0)
        mean_diff = pos_mean - neg_mean

        if not self.use_covariance:
            # Simple MMS: direction is just difference of means
            self.direction_ = mean_diff
        else:
            # LDA: adjust by pooled covariance matrix
            # Compute pooled covariance matrix
            pos_cov = np.cov(X_pos.T, bias=True)
            neg_cov = np.cov(X_neg.T, bias=True)
            pooled_cov = (pos_cov + neg_cov) / 2

            # Add regularization (ridge regression style)
            if self.regularization > 0:
                pooled_cov += self.regularization * np.eye(pooled_cov.shape[0])

            # Compute direction: cov^{-1} @ (mu_pos - mu_neg)
            try:
                cov_inv = np.linalg.inv(pooled_cov)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                cov_inv = np.linalg.pinv(pooled_cov)

            self.direction_ = cov_inv @ mean_diff

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
            "use_covariance": self.use_covariance,
            "regularization": self.regularization,
            "normalize": self.normalize,
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
