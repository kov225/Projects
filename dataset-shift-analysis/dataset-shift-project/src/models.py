"""
Model Registry and Training Module  Milestone 2

Provides two public surfaces:
  get_models()     Factory that returns a keyed dictionary of unfitted
                   estimators, including XGBoost when available.
  train_models()   Fits every model, isolates per-model failures, and
                   returns only the successfully trained estimators.

All models are instantiated with explicit random_state arguments so
that the fitted weights are deterministic given the same training data
and global seed (set via utils.set_global_seed).
"""

import warnings

from sklearn.dummy        import DummyClassifier
from sklearn.naive_bayes  import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm          import SVC
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)

from utils import get_logger

logger = get_logger(__name__)


def get_models(random_state: int = 42) -> dict:
    """
    Instantiates the full benchmark model suite.

    The suite deliberately spans diverse inductive biases:
      - Naive baseline (mode prediction) as the performance floor.
      - A generative probabilistic model (Gaussian Naive Bayes).
      - A linear discriminative model (Logistic Regression).
      - A kernel method (SVM with RBF kernel).
      - A single deep tree (Decision Tree) as an overfit-prone reference.
      - Two ensemble families: bagging (Random Forest) and boosting
        (Gradient Boosting, AdaBoost).
      - XGBoost when installed, providing a state-of-the-art boosting
        baseline.

    Args:
        random_state: Seed for all stochastic model components.

    Returns:
        Ordered dictionary mapping model name (str) to unfitted estimator.
    """
    models = {
        "Naive Baseline":    DummyClassifier(strategy="most_frequent"),
        "Naive Bayes":       GaussianNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            random_state=random_state
        ),
        "SVM (RBF)":         SVC(
            kernel="rbf", probability=True, C=1.0, gamma="scale",
            random_state=random_state
        ),
        "Decision Tree":     DecisionTreeClassifier(
            max_depth=10, random_state=random_state
        ),
        "Random Forest":     RandomForestClassifier(
            n_estimators=150, max_depth=None,
            random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05,
            max_depth=4, random_state=random_state
        ),
        "AdaBoost":          AdaBoostClassifier(
            n_estimators=100, algorithm="SAMME",
            random_state=random_state
        ),
    }

    # XGBoost is optional; a missing installation is silently skipped
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=150, learning_rate=0.05,
            max_depth=4, use_label_encoder=False,
            eval_metric="logloss", random_state=random_state,
            verbosity=0
        )
        logger.info("XGBoost detected and added to model suite.")
    except ImportError:
        logger.info("XGBoost not installed; skipping.")

    return models


def train_models(models: dict, X_train, y_train) -> dict:
    """
    Fits all models in the provided registry.

    Per-model exceptions are caught and logged so that a single training
    failure does not abort the entire experimental pipeline.  The SVM
    training runtime warning is surfaced proactively as it scales poorly
    with sample count.

    Args:
        models:  Dictionary of model name to unfitted estimator.
        X_train: Training feature matrix (n_samples, n_features).
        y_train: Training label vector (n_samples,).

    Returns:
        Dictionary containing only the successfully fitted estimators.
    """
    logger.info(
        "Starting model training. SVM fitting is O(n^2) to O(n^3) "
        "and may take several minutes on large datasets."
    )

    trained = {}
    for name, model in models.items():
        logger.info(f"Fitting {name} ...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            trained[name] = model
            logger.info(f"  {name} trained successfully.")
        except Exception as exc:
            logger.error(f"  {name} failed during training: {exc}")

    logger.info(f"Training complete. {len(trained)}/{len(models)} models ready.")
    return trained
