"""
Evaluation Module  Milestone 2

Provides a unified, statistically rigorous evaluation harness for all
experiments in this project.  Key capabilities added over Milestone 1:

  1. Full metric suite: Accuracy, Precision, Recall, F1, ROC-AUC,
     Calibration (Brier Score), and Confusion-Matrix-derived counts.
  2. Robustness scoring: an aggregate degradation index and relative
     drop computed against each model's clean baseline.
  3. Bootstrap confidence intervals (95%) for every metric.
  4. KS-test and PSI for distribution-level shift quantification.
  5. Stateless baseline caching via an explicit context object instead
     of mutable module-level globals.

Public API
----------
  EvaluationContext   Class that carries baseline state between calls.
  evaluate_models()   Main evaluation function; returns a tidy DataFrame.
  get_top_n_features() Random-Forest importance heuristic for shift targeting.
  compute_robustness_score()  Aggregate robustness index for a single model.
"""

import sys
import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, brier_score_loss,
    confusion_matrix,
)
from scipy.stats import ks_2samp, ttest_ind

sys.path.insert(0, os.path.dirname(__file__))
from utils import get_logger, compute_avg_psi

logger = get_logger(__name__)

BOOTSTRAP_ITERATIONS = 200
CI_LOWER = 2.5
CI_UPPER = 97.5


# ---------------------------------------------------------------------------
# Baseline State Container
# ---------------------------------------------------------------------------

class EvaluationContext:
    """
    Carries the baseline (clean) distribution and performance across the
    full experiment loop.

    Using an explicit context object instead of module-level globals makes
    the evaluation pipeline stateless from the caller's perspective: you can
    run multiple independent experiment sweeps by instantiating separate
    EvaluationContext objects without side effects.

    Attributes:
        baseline_X:       Cached clean test feature matrix.
        baseline_metrics: Per-model list of bootstrapped accuracy samples
                          collected during the baseline pass.
    """

    def __init__(self):
        self.baseline_X: np.ndarray | None = None
        self.baseline_metrics: dict = {}

    def is_ready(self) -> bool:
        return self.baseline_X is not None


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _predict(model, X: np.ndarray) -> tuple:
    """
    Produces class predictions and calibrated probability estimates for a
    single model regardless of its interface (predict_proba vs.
    decision_function vs. neither).

    Returns:
        Tuple (y_pred, y_prob) where y_prob is a 1-D array of positive-class
        probability estimates.
    """
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        y_prob = 1.0 / (1.0 + np.exp(-raw))   # sigmoid calibration
    else:
        y_prob = y_pred.astype(float)

    return y_pred, y_prob


def _point_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   y_prob: np.ndarray) -> dict:
    """
    Computes the full metric bundle for a single (y_true, y_pred, y_prob)
    triplet.  Returns a dictionary of scalar metric values.

    Handles edge cases (single-class samples, NaN probability estimates)
    gracefully by returning np.nan instead of raising.
    """
    multi = len(np.unique(y_true)) > 1
    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted",
                                     zero_division=0),
        "Recall":    recall_score(y_true, y_pred, average="weighted",
                                  zero_division=0),
        "F1_Score":  f1_score(y_true, y_pred, average="weighted",
                              zero_division=0),
    }

    metrics["ROC_AUC"] = roc_auc_score(y_true, y_prob) if multi else np.nan
    try:
        metrics["Brier_Score"] = brier_score_loss(y_true, y_prob)
    except ValueError:
        metrics["Brier_Score"] = np.nan

    return metrics


def _bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        y_prob: np.ndarray, n_iter: int = BOOTSTRAP_ITERATIONS
                        ) -> dict:
    """
    Constructs 95% bootstrap confidence intervals for all point metrics.

    Resampling is performed with replacement over the test-set rows.
    Single-class bootstrap draws (which are common at small sample sizes
    or extreme shift) are silently skipped rather than crashing.

    Returns:
        Dictionary containing lists of per-bootstrap metric values.
    """
    idx = np.arange(len(y_true))
    boot = {k: [] for k in ["Accuracy", "Precision", "Recall",
                             "F1_Score", "ROC_AUC", "Brier_Score"]}

    for _ in range(n_iter):
        s = np.random.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        m = _point_metrics(y_true[s], y_pred[s], y_prob[s])
        for k, v in m.items():
            boot[k].append(v)

    return boot


def _ci(values: list) -> tuple:
    """Returns (lower_95, upper_95) confidence interval from bootstrap samples."""
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return (np.nan, np.nan)
    return (np.nanpercentile(arr, CI_LOWER), np.nanpercentile(arr, CI_UPPER))


# ---------------------------------------------------------------------------
# Robustness Score
# ---------------------------------------------------------------------------

def compute_robustness_score(
    baseline_acc: float,
    shifted_acc: float,
    ks_statistic: float
) -> float:
    """
    Computes a composite Robustness Score for a single (model, shift) pair.

    The score integrates two signals:
      1. Relative performance retention (how much accuracy is preserved).
      2. Distribution shift magnitude (KS statistic) as a penalty.

    Formula:
        RS = retention_ratio * (1 - ks_statistic)

    Where retention_ratio = shifted_acc / baseline_acc (clamped to [0, 1]).
    A score near 1.0 means the model maintained performance despite large
    distributional divergence; near 0.0 means complete collapse.

    Args:
        baseline_acc: Clean-data accuracy.
        shifted_acc:  Accuracy under shift.
        ks_statistic: Average KS statistic between baseline and shifted X.

    Returns:
        Robustness score in [0, 1].
    """
    if baseline_acc <= 0:
        return 0.0
    retention = np.clip(shifted_acc / baseline_acc, 0.0, 1.0)
    return float(retention * (1.0 - np.clip(ks_statistic, 0.0, 1.0)))


def compute_relative_drop(baseline_acc: float, shifted_acc: float) -> float:
    """
    Returns the relative accuracy drop from baseline to shifted condition.

    Positive values indicate degradation. Negative values indicate
    (unlikely) improvement after shift.

    Formula: (baseline_acc - shifted_acc) / baseline_acc * 100
    """
    if baseline_acc <= 0:
        return 0.0
    return float((baseline_acc - shifted_acc) / baseline_acc * 100.0)


# ---------------------------------------------------------------------------
# Main Evaluation Function
# ---------------------------------------------------------------------------

def evaluate_models(
    trained_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    shift_type: str = "None",
    intensity: float = 0.0,
    ctx: EvaluationContext = None,
    continuous_indices: list = None,
) -> pd.DataFrame:
    """
    Evaluates all trained models against a (possibly shifted) test set and
    returns a tidy DataFrame with metrics, confidence intervals, distribution
    shift statistics, and robustness scores.

    The function is context-aware: when ctx is provided and the current pass
    is the baseline (intensity == 0.0 or shift_type == "Baseline"), it
    populates ctx.baseline_X and ctx.baseline_metrics for use by subsequent
    shifted evaluations.

    Args:
        trained_models:     Dict of name to fitted estimator.
        X_test:             Feature matrix for evaluation.
        y_test:             Ground-truth labels.
        shift_type:         Name of the applied shift (for result tagging).
        intensity:          Magnitude of the applied shift (for result tagging).
        ctx:                EvaluationContext carrying baseline state.
                            If None, a fresh context is created (baseline
                            caching will not persist across calls).
        continuous_indices: Feature columns used for PSI computation.
                            Defaults to all columns when None.

    Returns:
        pd.DataFrame with one row per model.
    """
    if ctx is None:
        ctx = EvaluationContext()

    is_baseline = (intensity == 0.0 or shift_type == "Baseline")
    if is_baseline and ctx.baseline_X is None:
        ctx.baseline_X = X_test.copy()
        logger.info("Baseline distribution cached.")

    # Distribution shift statistics
    ref = ctx.baseline_X if ctx.baseline_X is not None else X_test
    feat_idx = continuous_indices if continuous_indices else list(range(X_test.shape[1]))

    ks_stats = []
    for i in feat_idx:
        stat, _ = ks_2samp(ref[:, i], X_test[:, i])
        ks_stats.append(stat)
    avg_ks = float(np.mean(ks_stats)) if ks_stats else 0.0
    max_ks = float(np.max(ks_stats)) if ks_stats else 0.0
    avg_psi = compute_avg_psi(ref, X_test, feat_idx)

    rows = []
    for name, model in trained_models.items():
        y_pred, y_prob = _predict(model, X_test)
        point = _point_metrics(y_test, y_pred, y_prob)
        boot  = _bootstrap_metrics(y_test, y_pred, y_prob)

        # Baseline caching and hypothesis test
        if is_baseline:
            ctx.baseline_metrics[name] = boot["Accuracy"]
            p_val, significant = np.nan, False
        else:
            bl_acc = ctx.baseline_metrics.get(name, [])
            if len(bl_acc) >= 2 and len(boot["Accuracy"]) >= 2:
                _, p_val = ttest_ind(bl_acc, boot["Accuracy"], alternative="greater")
                significant = bool(p_val < 0.05)
            else:
                p_val, significant = np.nan, False

        # Robustness scores
        baseline_acc = np.mean(ctx.baseline_metrics.get(name, [point["Accuracy"]]))
        rob_score    = compute_robustness_score(baseline_acc, point["Accuracy"], avg_ks)
        rel_drop     = compute_relative_drop(baseline_acc, point["Accuracy"])

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))

        # Assemble row
        acc_lo, acc_hi = _ci(boot["Accuracy"])
        f1_lo,  f1_hi  = _ci(boot["F1_Score"])
        roc_lo, roc_hi = _ci(boot["ROC_AUC"])
        br_lo,  br_hi  = _ci(boot["Brier_Score"])
        pr_lo,  pr_hi  = _ci(boot["Precision"])
        rc_lo,  rc_hi  = _ci(boot["Recall"])

        rows.append({
            "Model":               name,
            "Shift_Type":          shift_type,
            "Intensity":           round(intensity, 4),
            "Accuracy":            point["Accuracy"],
            "Accuracy_Lower_CI":   acc_lo,
            "Accuracy_Upper_CI":   acc_hi,
            "Precision":           point["Precision"],
            "Precision_Lower_CI":  pr_lo,
            "Precision_Upper_CI":  pr_hi,
            "Recall":              point["Recall"],
            "Recall_Lower_CI":     rc_lo,
            "Recall_Upper_CI":     rc_hi,
            "F1_Score":            point["F1_Score"],
            "F1_Lower_CI":         f1_lo,
            "F1_Upper_CI":         f1_hi,
            "ROC_AUC":             point["ROC_AUC"],
            "ROC_Lower_CI":        roc_lo,
            "ROC_Upper_CI":        roc_hi,
            "Brier_Score":         point["Brier_Score"],
            "Brier_Lower_CI":      br_lo,
            "Brier_Upper_CI":      br_hi,
            "TP":                  int(tp),
            "TN":                  int(tn),
            "FP":                  int(fp),
            "FN":                  int(fn),
            "Avg_KS_Statistic":    avg_ks,
            "Max_KS_Statistic":    max_ks,
            "Avg_PSI":             avg_psi,
            "Robustness_Score":    rob_score,
            "Relative_Drop_Pct":   rel_drop,
            "P_Value":             p_val,
            "Significant_Shift":   significant,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature Importance Helper
# ---------------------------------------------------------------------------

def get_top_n_features(X_train: np.ndarray, y_train: np.ndarray,
                        n: int = 5) -> list:
    """
    Returns the indices of the N most informative features using a
    shallow Random Forest's Gini importance.

    This heuristic is used to direct concept-drift and feature-removal
    shift simulators toward the dimensions that carry the most signal,
    maximising the observable degradation effect.

    Args:
        X_train: Training feature matrix.
        y_train: Training label vector.
        n:       Number of top features to return.

    Returns:
        List of n integer column indices in descending importance order.
    """
    from sklearn.ensemble import RandomForestClassifier
    probe = RandomForestClassifier(n_estimators=50, max_depth=6,
                                   random_state=42, n_jobs=-1)
    probe.fit(X_train, y_train)
    return np.argsort(probe.feature_importances_)[::-1][:n].tolist()
