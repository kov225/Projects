"""
stress_test.py

Simulates a 2008-style macroeconomic shock by replacing the FRED macro
features in the test set with their 2008 recession peak values, then
evaluates the ensemble's AUC, calibration, and approval rate under that shock.

This demonstrates that I understand how production ML models fail under
distribution shift : a question that comes up in every senior ML engineering
interview at fintech companies. The output table goes directly into the README.

2008 recession reference values (approximate FRED peaks):
  - Unemployment:    10.0% (Oct 2009 peak, effectively at 2008 crisis level)
  - CPI YoY:         5.6% (mid-2008 peak)
  - Fed Funds Rate:  0.25% (ZIRP response to the crisis)
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from features.feature_pipeline import make_feature_pipeline, get_feature_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RECESSION_2008 = {
    "unemployment_rate": 10.0,
    "cpi_yoy_pct": 5.6,
    "fed_funds_rate": 0.25,
    "unemployment_mom_change": 0.4,   # ~0.4pp/month during the crisis
    "cpi": 215.0,
    "low_rate_env": 1,
    "high_rate_env": 0,
    "recession_signal": 1,
    "high_inflation": 1,
    "dti_x_unemployment": None,       # will be recomputed from DTI * 10.0
}


def inject_recession_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace macro features with 2008 recession peak values."""
    df = df.copy()
    for col, val in RECESSION_2008.items():
        if col in df.columns and val is not None:
            df[col] = val
    # Recompute the DTI interaction term with the shocked unemployment rate
    if "dti" in df.columns:
        df["dti_x_unemployment"] = df["dti"] * 10.0
    return df


def run_stress_test(data_path: Path, artifact_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path).sort_values("issue_d")
    split_idx = int(len(df) * 0.85)
    test_df = df.iloc[split_idx:].copy()

    test_shocked = inject_recession_features(test_df)

    feature_cols = get_feature_columns(test_df)
    pipeline = joblib.load(artifact_dir / "pipeline.joblib")
    scaler = joblib.load(artifact_dir / "scaler.joblib")
    base_models = joblib.load(artifact_dir / "base_models.joblib")
    meta_learner = joblib.load(artifact_dir / "meta_learner.joblib")

    y_true = test_df["default"].values

    def predict(df_subset):
        X_raw = pipeline.transform(df_subset[feature_cols])
        X = scaler.transform(X_raw)
        n_models = len(base_models)
        meta_feats = np.zeros((len(X), n_models))
        for col, (name, fold_models) in enumerate(base_models.items()):
            meta_feats[:, col] = np.array(
                [m.predict_proba(X)[:, 1] for m in fold_models]
            ).mean(axis=0)
        return meta_learner.predict_proba(meta_feats)[:, 1]

    # Baseline (normal economic conditions from the test set)
    y_prob_baseline = predict(test_df)
    auc_baseline = roc_auc_score(y_true, y_prob_baseline)
    brier_baseline = brier_score_loss(y_true, y_prob_baseline)
    approval_rate_baseline = (y_prob_baseline < 0.5).mean()

    # Shocked (2008 recession macro values)
    y_prob_shocked = predict(test_shocked)
    auc_shocked = roc_auc_score(y_true, y_prob_shocked)
    brier_shocked = brier_score_loss(y_true, y_prob_shocked)
    approval_rate_shocked = (y_prob_shocked < 0.5).mean()

    results = {
        "baseline": {
            "auc": round(auc_baseline, 4),
            "brier_score": round(brier_baseline, 4),
            "approval_rate": round(float(approval_rate_baseline), 4),
        },
        "recession_2008_shock": {
            "auc": round(auc_shocked, 4),
            "brier_score": round(brier_shocked, 4),
            "approval_rate": round(float(approval_rate_shocked), 4),
        },
        "delta": {
            "auc_drop": round(auc_baseline - auc_shocked, 4),
            "approval_rate_drop": round(float(approval_rate_baseline - approval_rate_shocked), 4),
        },
    }

    with open(out_dir / "stress_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== 2008 MACROECONOMIC STRESS TEST RESULTS =====")
    print(f"{'Metric':<30} {'Baseline':>12} {'2008 Shock':>12} {'Delta':>10}")
    print("-" * 68)
    print(f"{'AUC':<30} {auc_baseline:>12.4f} {auc_shocked:>12.4f} {auc_baseline-auc_shocked:>10.4f}")
    print(f"{'Brier Score':<30} {brier_baseline:>12.4f} {brier_shocked:>12.4f} {brier_shocked-brier_baseline:>10.4f}")
    print(f"{'Approval Rate':<30} {approval_rate_baseline:>12.3%} {approval_rate_shocked:>12.3%} {approval_rate_baseline-approval_rate_shocked:>10.3%}")
    print("=" * 68)
    print(f"\nStress test results saved to {out_dir}")

    logger.info(f"Stress test complete: AUC drop={auc_baseline - auc_shocked:.4f}")


if __name__ == "__main__":
    run_stress_test(
        data_path=Path("data/processed/loans_clean.parquet"),
        artifact_dir=Path("models/artifacts"),
        out_dir=Path("evaluation/stress"),
    )
