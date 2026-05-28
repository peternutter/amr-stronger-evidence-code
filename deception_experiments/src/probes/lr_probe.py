import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


class LogisticRegressionProbe(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression probe with configurable regularization and standardization.

    Wraps sklearn's LogisticRegression with explicit control over preprocessing.

    Args:
        C: Inverse of regularization strength (default: 1.0)
        penalty: Regularization type ('l1', 'l2', 'elasticnet', or 'none')
        normalize: Whether to standardize features
        class_weight: Class weights ('balanced' or None)
        solver: Optimization algorithm
        max_iter: Maximum iterations
        random_state: Random seed
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def __init__(
        self,
        safe_name: str,
        C: float = 1.0,
        penalty: str = "l2",
        normalize: bool = True,
        class_weight: str | None = "balanced",
        solver: str = "liblinear",
        max_iter: int = 10000,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        self.safe_name = safe_name
        self.C = C
        self.penalty = penalty
        self.normalize = normalize
        self.class_weight = class_weight
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.scaler_ = None
        self.model_ = None
        self.classes_ = None

    @classmethod
    def create_tuned_estimator(cls, cv_folds: int = 5, n_jobs: int = -1, **probe_params):
        """
        Returns a GridSearchCV-wrapped estimator to find the best C parameter.

        This approach ensures that the probe's own `fit` method, including its
        internal normalization, is called within the cross-validation loop.

        Args:
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 uses all cores)
            **probe_params: All parameters from the probe config

        Returns:
            GridSearchCV estimator that will find best C during fit()
        """
        estimator = cls(**probe_params)

        # C is the main hyperparameter to tune for logistic regression
        param_grid = {"C": [1.0, 10.0, 100.0]}

        return GridSearchCV(
            estimator, param_grid, cv=cv_folds, n_jobs=n_jobs, scoring="accuracy", refit=True, verbose=1
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the logistic regression probe.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (n_samples,)
        """
        # Store classes for sklearn compatibility
        self.classes_ = np.unique(y)

        if self.normalize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        self.model_ = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            class_weight=self.class_weight,
            solver=self.solver,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            fit_intercept=False,  # Consistent with detectors.py
        )
        self.model_.fit(X, y)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision scores.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Decision scores (n_samples,)
        """
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        return self.model_.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Class probabilities (n_samples, 2)
        """
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        return self.model_.predict_proba(X)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            "safe_name": self.safe_name,
            "C": self.C,
            "penalty": self.penalty,
            "normalize": self.normalize,
            "class_weight": self.class_weight,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
