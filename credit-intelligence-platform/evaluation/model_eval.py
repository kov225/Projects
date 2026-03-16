"""
model_eval.py

Full evaluation suite for the stacking ensemble:
  - AUC, Average Precision, Brier Score
  - Calibration curve (reliability diagram)
  - Threshold optimization for F1 and cost-sensitive objective
  - SHAP faithfulness metric

Run this after training to validate that performance targets are met before
any manual promotion decision. The retraining trigger calls a subset of this
automatically during the automated validation check.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, precision_recall_curve, roc_curve,
)
from sklearn.preprocessing import StandardScaler

from features.feature_pipeline import make_feature_pipeline, get_feature_columns
from models.shap_explainer import compute_shap_faithfulness

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lender cost-sensitivity: a missed default costs 5x a false rejection
FN_COST = 5.0
FP_COST = 1.0


def cost_sensitive_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """Compute total cost for a given classification threshold.
    
    We negate cost so we can use argmax (higher is better) like we would with F1.
    A real lender would calibrate these weights to actual dollar loss rates.
    """
    preds = (y_pred >= threshold).astype(int)
    fn = ((y_true == 1) & (preds == 0)).sum()  # missed defaults
    fp = ((y_true == 0) & (preds == 1)).sum()  # false rejections
    return -(fn * FN_COST + fp * FP_COST)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str = "f1",
) -> tuple[float, float]:
    """Find decision threshold maximizing the given objective.
    
    Args:
        objective: 'f1' or 'cost_sensitive'
    
    Returns:
        (optimal_threshold, best_score)
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = []

    for t in thresholds:
        if objective == "f1":
            s = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        elif objective == "cost_sensitive":
            s = cost_sensitive_score(y_true, y_prob, t)
        scores.append(s)

    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def plot_calibration_curve(y_true, y_prob, out_path: Path) -> None:
    """Plot reliability diagram for the model."""
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(7, 5))
    plt.plot(mean_pred, frac_pos, "s-", label="Ensemble")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Calibration curve saved to {out_path}")


def run_ensemble_predict(
    X_test: np.ndarray,
    base_models: dict,
    meta_learner,
) -> np.ndarray:
    n_models = len(base_models)
    meta_feats = np.zeros((len(X_test), n_models))
    for col, (name, fold_models) in enumerate(base_models.items()):
        fold_preds = np.array([m.predict_proba(X_test)[:, 1] for m in fold_models])
        meta_feats[:, col] = fold_preds.mean(axis=0)
    return meta_learner.predict_proba(meta_feats)[:, 1]


def evaluate(data_path: Path, artifact_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path).sort_values("issue_d")
    split_idx = int(len(df) * 0.85)
    test_df = df.iloc[split_idx:]

    feature_cols = get_feature_columns(test_df)
    pipeline = joblib.load(artifact_dir / "pipeline.joblib")
    scaler = joblib.load(artifact_dir / "scaler.joblib")
    base_models = joblib.load(artifact_dir / "base_models.joblib")
    meta_learner = joblib.load(artifact_dir / "meta_learner.joblib")

    X_test_raw = pipeline.transform(test_df[feature_cols])
    X_test = scaler.transform(X_test_raw)
    y_test = test_df["default"].values

    y_prob = run_ensemble_predict(X_test, base_models, meta_learner)

    # Core metrics
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    f1_thresh, f1_best = find_optimal_threshold(y_test, y_prob, "f1")
    cost_thresh, _ = find_optimal_threshold(y_test, y_prob, "cost_sensitive")

    # SHAP faithfulness
    xgb_model = base_models["xgboost"][0]
    shap_faith = compute_shap_faithfulness(xgb_model, X_test[:500], feature_cols)

    # Plots
    plot_calibration_curve(y_test, y_prob, out_dir / "calibration_curve.png")

    results = {
        "test_auc": round(auc, 4),
        "test_average_precision": round(ap, 4),
        "test_brier_score": round(brier, 4),
        "optimal_f1_threshold": round(f1_thresh, 4),
        "optimal_f1_score": round(f1_best, 4),
        "optimal_cost_threshold": round(cost_thresh, 4),
        "shap_faithfulness_spearman_r": round(shap_faith, 4),
        "test_size": len(test_df),
        "default_rate": round(y_test.mean(), 4),
    }

    import json
    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation complete:\n{json.dumps(results, indent=2)}")
    print(f"\n{'='*50}")
    print(f"AUC:         {auc:.4f}  (target: >0.78)")
    print(f"SHAP Faith.: {shap_faith:.4f}  (target: >0.85)")
    print(f"Opt. F1 Thr: {f1_thresh:.4f}")
    print(f"Cost Thr:    {cost_thresh:.4f}")
    print(f"{'='*50}\n")

    return results


if __name__ == "__main__":
    evaluate(
        data_path=Path("data/processed/loans_clean.parquet"),
        artifact_dir=Path("models/artifacts"),
        out_dir=Path("evaluation/results"),
    )
