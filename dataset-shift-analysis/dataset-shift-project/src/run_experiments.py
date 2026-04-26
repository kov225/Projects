"""
Experiment Orchestration Module  Milestone 2

This script is the single entry point for running the full benchmarking
suite.  It replaces the Milestone 1 run_experiments.py while deliberately
reusing the data_loader, models, evaluation, and shift_simulators modules.

Experiment design:
  Six shift families x eleven intensity levels (0.0 to 1.0 in steps of 0.1)
  x all models from the registry = a complete cross-product of conditions.

  Each shift family evaluates against the same trained baseline models so
  that all comparisons are apples-to-apples.

Output artifacts:
  results/experiment_results.csv   Full tidy result table.
  figures/*.png                    Auto-saved plots for all conditions.

Usage:
  python src/run_experiments.py
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Make the src directory importable when this script is run directly.
sys.path.insert(0, os.path.dirname(__file__))

from data_loader      import load_and_preprocess_data
from models           import get_models, train_models
from evaluation       import (EvaluationContext, evaluate_models,
                               get_top_n_features)
from shift_simulators import (
    apply_covariate_shift, apply_feature_scaling_drift,
    apply_label_shift, apply_concept_drift,
    apply_gaussian_noise,
    apply_mcar_missingness, apply_mar_missingness,
    apply_feature_removal,
)
from visualizations   import (
    plot_performance_curves, plot_model_heatmap,
    plot_robustness_ranking, plot_calibration_decay,
    plot_ks_psi_curves, plot_relative_drop_heatmap,
    save_figure,
)
from utils import get_logger, get_results_path, set_global_seed

logger = get_logger(__name__)

INTENSITIES = [round(i * 0.1, 1) for i in range(1, 11)]   # 0.1 ... 1.0
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Shift Registry
# ---------------------------------------------------------------------------

def build_shift_registry(continuous_indices, top_features, X_test, y_test):
    """
    Returns a list of (shift_name, [(intensity, X_shifted, y_shifted), ...])
    tuples that the main loop iterates over.

    Centralising shift construction here keeps the experiment loop clean and
    makes it trivial to add or remove shift families.
    """
    registry = []

    # A. Covariate Shift
    cov_rows = []
    for inten in INTENSITIES:
        X_s = apply_covariate_shift(X_test, continuous_indices, inten)
        cov_rows.append((inten, X_s, y_test))
    registry.append(("Covariate Shift", cov_rows))

    # A2. Feature Scaling Drift
    scl_rows = []
    for inten in INTENSITIES:
        X_s = apply_feature_scaling_drift(X_test, continuous_indices, inten)
        scl_rows.append((inten, X_s, y_test))
    registry.append(("Scaling Drift", scl_rows))

    # B. Label Shift
    lbl_rows = []
    for inten in INTENSITIES:
        X_s, y_s = apply_label_shift(X_test, y_test, inten)
        lbl_rows.append((inten, X_s, y_s))
    registry.append(("Label Shift", lbl_rows))

    # C. Concept Drift
    cdr_rows = []
    for inten in INTENSITIES:
        X_s, y_s = apply_concept_drift(X_test, y_test, top_features, inten)
        cdr_rows.append((inten, X_s, y_s))
    registry.append(("Concept Drift", cdr_rows))

    # D. Gaussian Noise
    gn_rows = []
    for inten in INTENSITIES:
        X_s = apply_gaussian_noise(X_test, continuous_indices, inten)
        gn_rows.append((inten, X_s, y_test))
    registry.append(("Gaussian Noise", gn_rows))

    # E. MCAR Missingness
    mcar_rows = []
    for inten in INTENSITIES:
        X_s = apply_mcar_missingness(X_test, inten)
        mcar_rows.append((inten, X_s, y_test))
    registry.append(("MCAR Missingness", mcar_rows))

    # E2. MAR Missingness  (condition on the first continuous feature)
    cond_feat = continuous_indices[0] if continuous_indices else 0
    mar_rows = []
    for inten in INTENSITIES:
        X_s = apply_mar_missingness(X_test, cond_feat,
                                    continuous_indices[1:6], inten)
        mar_rows.append((inten, X_s, y_test))
    registry.append(("MAR Missingness", mar_rows))

    # F. Feature Removal
    fr_rows = []
    for inten in INTENSITIES:
        X_s = apply_feature_removal(X_test, top_features, inten)
        fr_rows.append((inten, X_s, y_test))
    registry.append(("Feature Removal", fr_rows))

    return registry


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    set_global_seed(RANDOM_STATE)
    t0 = time.time()

    logger.info("=" * 65)
    logger.info("  Dataset Shift  Milestone 2 Experiment Suite")
    logger.info("=" * 65)

    # ---- Data ---------------------------------------------------------------
    logger.info("Loading and preprocessing data ...")
    (X_train, X_test, y_train, y_test,
     continuous_indices, _) = load_and_preprocess_data(random_state=RANDOM_STATE)
    logger.info(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # ---- Models -------------------------------------------------------------
    logger.info("Instantiating and training model suite ...")
    trained_models = train_models(get_models(RANDOM_STATE), X_train, y_train)

    # ---- Feature Importance -------------------------------------------------
    logger.info("Computing feature importance for shift targeting ...")
    top_features = get_top_n_features(X_train, y_train, n=5)
    logger.info(f"  Top feature indices: {top_features}")

    # ---- Shared evaluation context ------------------------------------------
    ctx = EvaluationContext()

    # ---- Baseline -----------------------------------------------------------
    logger.info("Evaluating baseline (no shift) ...")
    bl_df = evaluate_models(
        trained_models, X_test, y_test,
        shift_type="Baseline", intensity=0.0,
        ctx=ctx, continuous_indices=continuous_indices,
    )
    all_results = [bl_df]

    # ---- Shift registry -----------------------------------------------------
    registry = build_shift_registry(continuous_indices, top_features,
                                    X_test, y_test)

    for shift_name, intensity_rows in registry:
        logger.info(f"Running {shift_name} ({len(intensity_rows)} levels) ...")
        for intensity, X_s, y_s in intensity_rows:
            res = evaluate_models(
                trained_models, X_s, y_s,
                shift_type=shift_name, intensity=intensity,
                ctx=ctx, continuous_indices=continuous_indices,
            )
            all_results.append(res)
        logger.info(f"  {shift_name} complete.")

    # ---- Consolidate --------------------------------------------------------
    final_df = pd.concat(all_results, ignore_index=True)
    out_path  = get_results_path("experiment_results.csv")
    final_df.to_csv(out_path, index=False)
    logger.info(f"Results saved to {out_path}")
    logger.info(f"Total rows: {len(final_df)}")

    # ---- Auto-save figures --------------------------------------------------
    logger.info("Generating and saving figures ...")
    metrics_to_plot = ["Accuracy", "F1_Score", "ROC_AUC"]

    shift_names = [name for name, _ in registry]
    for shift_name in shift_names:
        safe = shift_name.replace(" ", "_").lower()
        for metric in metrics_to_plot:
            fig = plot_performance_curves(final_df, shift_name, metric)
            save_figure(fig, f"{safe}_{metric.lower()}_curves.png")
            plt.close(fig)

        fig = plot_model_heatmap(final_df, shift_name, "Accuracy")
        save_figure(fig, f"{safe}_accuracy_heatmap.png")
        plt.close(fig)

        fig = plot_robustness_ranking(final_df, shift_name)
        save_figure(fig, f"{safe}_robustness_ranking.png")
        plt.close(fig)

        fig = plot_calibration_decay(final_df, shift_name)
        save_figure(fig, f"{safe}_calibration_decay.png")
        plt.close(fig)

        fig = plot_ks_psi_curves(final_df, shift_name)
        save_figure(fig, f"{safe}_ks_psi.png")
        plt.close(fig)

    fig = plot_relative_drop_heatmap(final_df)
    save_figure(fig, "relative_drop_heatmap.png")
    plt.close(fig)

    elapsed = time.time() - t0
    logger.info(f"Experiment suite complete in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
