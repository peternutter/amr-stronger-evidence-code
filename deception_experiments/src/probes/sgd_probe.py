import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


class SGDProbe(BaseEstimator, ClassifierMixin):
    """
    SGD Classifier probe with configurable regularization and standardization.

    Wraps sklearn's SGDClassifier with explicit control over preprocessing.

    Args:
        safe_name: Identifier for the probe (not passed to sklearn)
        loss: Loss function to use (default: 'log_loss' for logistic regression)
        penalty: Regularization type ('l1', 'l2', 'elasticnet', or 'none')
        alpha: Regularization strength (default: 0.0001)
        max_iter: Maximum iterations
        tol: Tolerance for stopping criterion
        normalize: Whether to standardize features
        n_jobs: Number of parallel jobs (SGD doesn't support this, kept for compatibility)
        random_state: Random seed
        early_stopping: Whether to use early stopping
        validation_fraction: Fraction of training data for validation in early stopping
        n_iter_no_change: Number of iterations with no change to wait before stopping
        class_weight: Class weights ('balanced' or None)
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def __init__(
        self,
        safe_name: str,
        loss: str = "log_loss",
        penalty: str = "l2",
        alpha: float = 0.0001,
        max_iter: int = 1000,
        tol: float = 1e-3,
        learning_rate: str = "optimal",
        eta0: float = 0.0,
        normalize: bool = True,
        n_jobs: int = 1,  # Not used by SGD but kept for config compatibility
        random_state: int = 42,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        class_weight: str | None = "balanced",
    ):
        self.safe_name = safe_name
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.normalize = normalize
        self.n_jobs = n_jobs  # Stored but not used
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.class_weight = class_weight
        self.scaler_ = None
        self.model_ = None
        self.classes_ = None

    @classmethod
    def create_tuned_estimator(cls, cv_folds: int = 5, n_jobs: int = -1, **probe_params):
        """
        Returns a GridSearchCV-wrapped SGD probe with automatic hyperparameter tuning.

        Tuning strategy:
        - alpha: Regularization strength (5 values on log scale)
        - penalty: L1, L2, or ElasticNet regularization
        - All other parameters are taken from probe_params

        Args:
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 uses all cores)
            **probe_params: All parameters from the probe config (passed through to estimator)

        Returns:
            GridSearchCV estimator that will find best hyperparameters during fit()
        """
        estimator = cls(**probe_params)

        param_grid = {
            "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
            "penalty": ["l2"],
        }

        return GridSearchCV(
            estimator, param_grid, cv=cv_folds, n_jobs=n_jobs, scoring="accuracy", refit=True, verbose=1
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the SGD classifier probe.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (n_samples,)
        """
        # Store classes for sklearn compatibility
        self.classes_ = np.unique(y)

        if self.normalize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        self.model_ = SGDClassifier(
            loss=self.loss,
            penalty=self.penalty,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            class_weight=self.class_weight,
            fit_intercept=False,  # Consistent with other probes
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
            "loss": self.loss,
            "penalty": self.penalty,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "learning_rate": self.learning_rate,
            "eta0": self.eta0,
            "normalize": self.normalize,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "early_stopping": self.early_stopping,
            "validation_fraction": self.validation_fraction,
            "n_iter_no_change": self.n_iter_no_change,
            "class_weight": self.class_weight,
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
