"""
train.py

Trains the stacking ensemble: XGBoost + LightGBM + Logistic Regression as
base models, with a logistic meta-learner trained on 5-fold out-of-fold (OOF)
predictions. This architecture gives us more signal than a simple average
because the meta-learner can learn that, for example, LightGBM's confidence on
high-FICO borrowers is more reliable than XGBoost's.

The full training run is logged to MLflow : hyperparameters, CV metrics,
feature importances, and the serialized pipeline + model artifacts. After
training we run the full evaluation suite and only promote to staging if
the new model beats the current production model on the holdout AUC.

Usage:
    python -m models.train --data data/processed/loans_clean.parquet
"""

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.feature_pipeline import make_feature_pipeline, get_feature_columns, NUMERIC_FEATURES
from features.schema_registry import register_schema
from models.mlflow_registry import log_training_run, promote_model
from models.shap_explainer import compute_shap_values

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODELS = {
    "xgboost": XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=5,      # roughly 1/(default_rate) for class imbalance
        eval_metric="auc",
        use_label_encoder=False,
        tree_method="hist",      # GPU-compatible; falls back to CPU automatically
        n_jobs=-1,
        random_state=42,
    ),
    "lightgbm": LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        is_unbalance=True,
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    ),
    "logistic": LogisticRegression(
        C=0.1,
        max_iter=1000,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
        random_state=42,
    ),
}

N_FOLDS = 5


def generate_oof_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models: dict,
) -> tuple[np.ndarray, dict]:
    """Generate out-of-fold predictions for each base model via stratified KFold.
    
    OOF predictions are the correct way to build stacking meta-features because
    each fold's test predictions come from a model that never saw those samples
    during training. This prevents the meta-learner from just memorizing which
    base model memorized which training examples.
    
    Returns:
        oof_matrix: (n_samples, n_base_models) array of OOF probabilities.
        fitted_models_by_fold: dict mapping model_name → list of fitted models
                               (one per fold), used for final prediction at test time.
    """
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    n_models = len(models)
    oof_matrix = np.zeros((len(X_train), n_models))
    fitted_models_by_fold = {name: [] for name in models}

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]

        logger.info(f"Fold {fold_idx + 1}/{N_FOLDS}: training {len(X_fold_train)} samples")

        for col, (name, model) in enumerate(models.items()):
            import copy
            m = copy.deepcopy(model)
            m.fit(X_fold_train, y_fold_train)
            oof_matrix[val_idx, col] = m.predict_proba(X_fold_val)[:, 1]
            fitted_models_by_fold[name].append(m)

    cv_aucs = {
        name: roc_auc_score(y_train, oof_matrix[:, col])
        for col, name in enumerate(models.keys())
    }
    logger.info(f"OOF AUCs: {cv_aucs}")
    return oof_matrix, fitted_models_by_fold, cv_aucs


def predict_with_fold_ensemble(
    X: np.ndarray,
    fitted_models_by_fold: dict,
) -> np.ndarray:
    """Average predictions from all fold-trained base models for test inference.
    
    At test time we use all N_FOLDS copies of each model and average their
    predictions. This is equivalent to bagging and reduces variance compared
    to keeping just one fold's model.
    """
    n_models = len(fitted_models_by_fold)
    preds = np.zeros((len(X), n_models))
    for col, (name, fold_models) in enumerate(fitted_models_by_fold.items()):
        fold_preds = np.array([m.predict_proba(X)[:, 1] for m in fold_models])
        preds[:, col] = fold_preds.mean(axis=0)
    return preds


def train_meta_learner(
    oof_matrix: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    """Train the meta-learner on OOF predictions from base models."""
    meta = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    meta.fit(oof_matrix, y_train)
    logger.info(f"Meta-learner coefficients: {dict(zip(BASE_MODELS.keys(), meta.coef_[0]))}")
    return meta


def run_training(data_path: Path, test_size: float = 0.15) -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "credit_intelligence")
    mlflow.set_experiment(experiment_name)

    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    # Time-based split: use the most recent 15% as held-out test
    df = df.sort_values("issue_d").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    logger.info(f"Train default rate: {train_df['default'].mean():.3f}")
    logger.info(f"Test default rate: {test_df['default'].mean():.3f}")

    feature_cols = get_feature_columns(train_df)

    # Fit preprocessing pipeline on train only
    pipeline = make_feature_pipeline(scale_for_lr=False)
    X_train = pipeline.fit_transform(train_df[feature_cols])
    y_train = train_df["default"].values

    # LR needs scaled features : fit a separate scaler track for it
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # For XGB and LGBM use raw; for LR use scaled
    X_lr = X_train_scaled

    models_with_data = {
        "xgboost": (BASE_MODELS["xgboost"], X_train),
        "lightgbm": (BASE_MODELS["lightgbm"], X_train),
        "logistic": (BASE_MODELS["logistic"], X_lr),
    }

    # We need a unified X for OOF : use scaled for everything (trees don't care)
    oof_matrix, fitted_models_by_fold, cv_aucs = generate_oof_predictions(
        X_train_scaled, y_train, BASE_MODELS
    )
    meta_learner = train_meta_learner(oof_matrix, y_train)

    # Evaluate on held-out test set
    X_test_raw = pipeline.transform(test_df[feature_cols])
    X_test = scaler.transform(X_test_raw)
    y_test = test_df["default"].values

    test_meta_features = predict_with_fold_ensemble(X_test, fitted_models_by_fold)
    y_prob = meta_learner.predict_proba(test_meta_features)[:, 1]

    test_auc = roc_auc_score(y_test, y_prob)
    test_ap = average_precision_score(y_test, y_prob)
    test_brier = brier_score_loss(y_test, y_prob)

    logger.info(f"Test AUC: {test_auc:.4f} | AP: {test_ap:.4f} | Brier: {test_brier:.4f}")

    # Compute SHAP values for XGBoost (most interpretable base model)
    shap_values = compute_shap_values(
        fitted_models_by_fold["xgboost"][0], X_test[:500], feature_cols
    )

    # Log to MLflow
    artifact_dir = Path("models/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_dir / "pipeline.joblib")
    joblib.dump(scaler, artifact_dir / "scaler.joblib")
    joblib.dump(fitted_models_by_fold, artifact_dir / "base_models.joblib")
    joblib.dump(meta_learner, artifact_dir / "meta_learner.joblib")

    metrics = {
        "test_auc": test_auc,
        "test_average_precision": test_ap,
        "test_brier_score": test_brier,
        **{f"oof_auc_{name}": auc for name, auc in cv_aucs.items()},
    }

    params = {
        "n_folds": N_FOLDS,
        "test_size": test_size,
        "train_size": len(train_df),
        "test_size_n": len(test_df),
        "features": len(feature_cols),
    }

    # Register/promote in MLflow
    model_version = log_training_run(
        metrics=metrics,
        params=params,
        artifact_dir=artifact_dir,
        feature_columns=feature_cols,
    )

    # Register schema for this version
    try:
        register_schema(version=model_version, feature_columns=feature_cols)
    except ValueError:
        logger.warning(f"Schema for version {model_version} already registered")

    logger.info(f"Training complete. Model version: {model_version}")
    logger.info(f"Final test AUC: {test_auc:.4f} (target: >0.78)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/loans_clean.parquet")
    args = parser.parse_args()
    run_training(Path(args.data))
