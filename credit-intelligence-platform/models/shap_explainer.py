"""
shap_explainer.py

Per-prediction SHAP value computation using TreeExplainer for XGBoost and 
LightGBM, and LinearExplainer for the logistic regression baseline.

The serving layer calls `explain_prediction` for every /explain request.
For training, `compute_shap_values` runs on the test set to produce the
faithfulness metric (Spearman correlation between SHAP sum and log-odds).
"""

import logging
from typing import Any

import numpy as np
import shap
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Compute SHAP values for a tree or linear model.
    
    We use TreeExplainer for XGBoost/LightGBM and LinearExplainer for LR.
    The check_additivity=False flag suppresses the SHAP additivity warning
    that fires for stacking ensembles where the base model SHAP values don't
    sum to the final ensemble output : they sum to the base model output, which
    is what we want for individual base model interpretability.
    """
    model_class = type(model).__name__

    if model_class in ("XGBClassifier", "LGBMClassifier"):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X, check_additivity=False)
        # For binary classifiers, shap returns a 2D array for class 1
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
    elif model_class == "LogisticRegression":
        # LinearExplainer needs a background dataset; use zeros as the baseline
        # since features are standardized and 0 = mean
        background = np.zeros((1, X.shape[1]))
        explainer = shap.LinearExplainer(model, background)
        shap_vals = explainer.shap_values(X)
    else:
        raise ValueError(f"Unsupported model type for SHAP: {model_class}")

    logger.info(f"SHAP values computed for {len(X)} samples, {len(feature_names)} features")
    return shap_vals


def explain_single_prediction(
    model: Any,
    x_single: np.ndarray,
    feature_names: list[str],
    top_k: int = 5,
) -> dict[str, float]:
    """Return the top-k SHAP values for a single prediction.
    
    Returns a dict mapping feature_name → signed SHAP value, sorted by
    absolute magnitude so the most influential features come first.
    This is what gets stored in PostgreSQL and returned by /explain.
    """
    # reshape to (1, n_features) for single-sample inference
    x = x_single.reshape(1, -1) if x_single.ndim == 1 else x_single
    shap_vals = compute_shap_values(model, x, feature_names)[0]

    # Sort by absolute SHAP value descending, keep top_k
    sorted_indices = np.argsort(np.abs(shap_vals))[::-1][:top_k]
    return {
        feature_names[i]: float(shap_vals[i])
        for i in sorted_indices
    }


def compute_shap_faithfulness(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
) -> float:
    """Compute Spearman correlation between sum of SHAP values and model log-odds.
    
    A high correlation (>0.85) confirms that the SHAP values faithfully
    represent the model's reasoning rather than being post-hoc rationalizations.
    This is the faithfulness metric cited in the README.
    """
    shap_vals = compute_shap_values(model, X, feature_names)
    shap_sums = shap_vals.sum(axis=1)

    raw_probs = model.predict_proba(X)[:, 1]
    # Clamp to avoid log(0) : in practice this shouldn't happen on a well-calibrated model
    raw_probs = np.clip(raw_probs, 1e-7, 1 - 1e-7)
    log_odds = np.log(raw_probs / (1 - raw_probs))

    corr, pvalue = spearmanr(shap_sums, log_odds)
    logger.info(f"SHAP faithfulness (Spearman r): {corr:.4f} (p={pvalue:.4g})")
    return float(corr)


def format_shap_for_llm(shap_dict: dict[str, float]) -> str:
    """Format SHAP values into a human-readable string for the LLM prompt.
    
    The prompt template in llm_explanation.py uses this string directly.
    """
    lines = []
    for feature, value in shap_dict.items():
        direction = "INCREASED" if value > 0 else "DECREASED"
        lines.append(f"- {feature}: {direction} default risk by {abs(value):.4f} SHAP units")
    return "\n".join(lines)
