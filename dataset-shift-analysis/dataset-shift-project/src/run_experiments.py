"""
Experimental Pipeline Orchestration Module

This script serves as the primary entry point for executing the dataset shift 
experimental suite. It coordinates data loading, model baseline training, 
systematic shift simulation, and performance evaluation. Results are serialized 
to disk for downstream visualization and analysis.
"""

import os
import pandas as pd
import numpy as np
from data_loader import load_and_preprocess_data
from models import get_models, train_models
from shift_simulators import apply_covariate_shift, apply_prior_shift, apply_concept_adjacent_shift, apply_scaling_drift
from evaluation import evaluate_models, get_top_n_features, calculate_robustness_scores
from visualizations import auto_save_all_figures
import interpretability

def main():
    """
    Executes the comprehensive benchmarking pipeline across multiple shift types
    with statistical validation via multiple random seeds.
    """
    # Configuration
    N_SEEDS = 3
    intensities = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    # Path management
    BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    FIGURES_DIR = os.path.join(BASE_DIR, "figures")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    RESULTS_FILE = os.path.join(RESULTS_DIR, "experiment_results.csv")
    ROBUSTNESS_FILE = os.path.join(RESULTS_DIR, "robustness_leaderboard.csv")
    
    print("="*60)
    print(f"Dataset Shift Study: {N_SEEDS} Seeds Statistical Validation")
    print("="*60)
    
    all_runs_buffer = []
    
    for seed_idx in range(N_SEEDS):
        seed = 42 + seed_idx
        print(f"\n>>> Starting Run {seed_idx + 1}/{N_SEEDS} (Seed: {seed})")
        np.random.seed(seed)
        
        # Load and train
        X_train, X_test, y_train, y_test, continuous_indices, preprocessor = load_and_preprocess_data(random_state=seed)
        models_dict = get_models()
        trained_models = train_models(models_dict, X_train, y_train)
        
        # Identifying critical features
        top_n_indices = get_top_n_features(X_train, y_train, n=3)
        
        # 0. Baseline
        print("  Evaluating Baseline...")
        res = evaluate_models(trained_models, X_test, y_test, shift_type="Baseline", intensity=0.0)
        res['Seed'] = seed
        all_runs_buffer.append(res)
        
        # 1. Covariate Shift (Noise)
        print("  Simulating Covariate Shift (Noise)...")
        for intensity in intensities:
            cov_intensity = intensity * 2.0 
            X_test_shifted = apply_covariate_shift(X_test, continuous_indices, cov_intensity)
            res = evaluate_models(trained_models, X_test_shifted, y_test, shift_type="Covariate Shift", intensity=cov_intensity)
            res['Seed'] = seed
            all_runs_buffer.append(res)
            
        # 2. Scaling Drift (New)
        print("  Simulating Scaling Drift...")
        for intensity in intensities:
            X_test_shifted = apply_scaling_drift(X_test, continuous_indices, intensity)
            res = evaluate_models(trained_models, X_test_shifted, y_test, shift_type="Scaling Drift", intensity=intensity)
            res['Seed'] = seed
            all_runs_buffer.append(res)
            
        # 3. Prior Probability Shift
        print("  Simulating Prior Probability Shift...")
        for intensity in intensities:
            X_test_shifted, y_test_shifted = apply_prior_shift(X_test, y_test, intensity)
            res = evaluate_models(trained_models, X_test_shifted, y_test_shifted, shift_type="Prior Probability Shift", intensity=intensity)
            res['Seed'] = seed
            all_runs_buffer.append(res)
            
        # 4. Concept-Adjacent Shift
        print("  Simulating Concept-Adjacent Shift...")
        for intensity in intensities:
            X_test_shifted = apply_concept_adjacent_shift(X_test, top_n_indices, intensity)
            res = evaluate_models(trained_models, X_test_shifted, y_test, shift_type="Concept-Adjacent Shift", intensity=intensity)
            res['Seed'] = seed
            all_runs_buffer.append(res)

    # Consolidation
    print("\nConsolidating results across seeds...")
    raw_results_df = pd.concat(all_runs_buffer, ignore_index=True)
    
    # Aggregate stats: mean and std across seeds
    metrics = ["Accuracy", "F1_Score", "ROC_AUC", "Brier_Score", "KS_Statistic"]
    agg_dict = {m: ['mean', 'std'] for m in metrics}
    
    final_agg_results = raw_results_df.groupby(['Model', 'Shift_Type', 'Intensity']).agg(agg_dict).reset_index()
    # Flatten multi-index columns
    final_agg_results.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] 
        for col in final_agg_results.columns.values
    ]
    
    # Save results
    final_agg_results.to_csv(RESULTS_FILE, index=False)
    print(f"Aggregated results saved to: {RESULTS_FILE}")
    
    # Calculate and save Robustness Leaderboard
    print("Calculating Robustness Leaderboard...")
    robustness_df = calculate_robustness_scores(raw_results_df)
    robustness_df.to_csv(ROBUSTNESS_FILE, index=False)
    print(f"Robustness leaderboard saved to: {ROBUSTNESS_FILE}")
    
    # Auto-save figures
    print("Generating and saving figures...")
    auto_save_all_figures(final_agg_results, FIGURES_DIR)
    
    print("\n" + "="*60)
    print("Milestone 2 Experimental Suite Completed Successfully")
    print("="*60)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
