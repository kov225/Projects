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
from shift_simulators import apply_covariate_shift, apply_prior_shift, apply_concept_adjacent_shift
from evaluation import evaluate_models, get_top_n_features

def main():
    """
    Executes the comprehensive benchmarking pipeline across multiple shift types.

    The pipeline involves:
    1. Initializing data and fitting baseline models on clean training data.
    2. Iteratively applying three shift types (Covariate, Prior, Concept-Adjacent) 
       at increasing intensity levels.
    3. Quantifying model degradation relative to the zero-shift baseline.
    4. Persisting the consolidated result matrix to a CSV file.

    Parameters:
        None. All experimental hyperparameters are localized within the routine.

    Returns:
        None. Results are persisted to the filesystem.
    """
    # Seeding ensures predictability across stochastic model fitting and data sampling
    np.random.seed(42)
    
    # Path management for persistent artifact storage
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    RESULTS_FILE = os.path.join(RESULTS_DIR, "experiment_results.csv")
    
    print("="*60)
    print("Beginning Dataset Shift Longitudinal Study")
    print("="*60)
    
    # Baseline data acquisition and cleaning
    X_train, X_test, y_train, y_test, continuous_indices, _ = load_and_preprocess_data()
    print(f"Dataset initialized. Train samples={X_train.shape[0]}, Test samples={X_test.shape[0]}")
    
    # Model instantiation and initial training on uncorrupted data
    models_dict = get_models()
    trained_models = train_models(models_dict, X_train, y_train)
    
    all_results_buffer = []
    
    # Baseline evaluation serves as the reference point for all subsequent degradation analysis
    print("\nCalculating Baseline Performance (Shift Intensity = 0.0)...")
    baseline_res = evaluate_models(trained_models, X_test, y_test, shift_type="Baseline", intensity=0.0)
    all_results_buffer.append(baseline_res)
    
    # Shift sweep parameters
    intensities = [0.1, 0.25, 0.5, 0.75, 1.0] 
    
    # Identifying critical features to target for the Concept-Adjacent shift simulation
    top_n_indices = get_top_n_features(X_train, y_train, n=3)
    
    # 1. Covariate Shift Simulation (Injection of Gaussian noise into continuous dimensions)
    print("\nSimulating Covariate Shift...")
    for intensity in intensities:
        # Intensity is mapped to standard deviation units for the Gaussian kernel
        cov_intensity = intensity * 2.0 
        X_test_shifted = apply_covariate_shift(X_test, continuous_indices, cov_intensity)
        res = evaluate_models(trained_models, X_test_shifted, y_test, shift_type="Covariate Shift", intensity=cov_intensity)
        all_results_buffer.append(res)
        print(f"  Processed intensity: {cov_intensity:.2f}")
        
    # 2. Prior Probability Shift Simulation (Controlled resampling of class distribution)
    print("\nSimulating Prior Probability Shift...")
    for intensity in intensities:
        X_test_shifted, y_test_shifted = apply_prior_shift(X_test, y_test, intensity)
        res = evaluate_models(trained_models, X_test_shifted, y_test_shifted, shift_type="Prior Probability Shift", intensity=intensity)
        all_results_buffer.append(res)
        print(f"  Processed intensity: {intensity:.2f}")
        
    # 3. Concept-Adjacent Shift Simulation (Correlation disruption on informative features)
    print("\nSimulating Concept-Adjacent Shift...")
    for intensity in intensities:
        X_test_shifted = apply_concept_adjacent_shift(X_test, top_n_indices, intensity)
        res = evaluate_models(trained_models, X_test_shifted, y_test, shift_type="Concept-Adjacent Shift", intensity=intensity)
        all_results_buffer.append(res)
        print(f"  Processed intensity: {intensity:.2f}")
        
    # Result consolidation and file serialization
    final_results_df = pd.concat(all_results_buffer, ignore_index=True)
    final_results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\nExperimental suite completed. Longitudinal results archived at: '{RESULTS_FILE}'")

if __name__ == "__main__":
    main()
