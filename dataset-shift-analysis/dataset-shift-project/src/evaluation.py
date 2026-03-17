"""
Model Evaluation and Feature Importance Module

This module provides tools for quantifying model performance across multiple 
statistical dimensions. It includes a unified evaluation framework for 
calculating classification metrics and a utility for identifying informative 
features to guide concept-adjacent shift simulations.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
import sys
import os

# We insert the local directory to the front of the path to ensure that our 
# local 'statistics.py' is prioritized over the Python standard library.
sys.path.insert(0, os.path.dirname(__file__))
import statistics

# We use module-level caches to store the uncorrupted baseline data and its performance.
# This allows us to compare shifted distributions against the original.
_BASELINE_X_CACHE = None
_BASELINE_METRICS_CACHE = {} # Maps model names to their baseline bootstrapped accuracies

def _calculate_metrics(y_true, y_pred, y_prob, has_multiple_classes):
    """
    Internal helper to calculate a bundle of metrics for a single sample.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    if has_multiple_classes:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            roc_auc = np.nan
    else:
        roc_auc = np.nan
        
    try:
        brier = brier_score_loss(y_true, y_prob)
    except ValueError:
        brier = np.nan
        
    return acc, f1, roc_auc, brier

def evaluate_models(trained_models, X_test, y_test, shift_type="None", intensity=0.0):
    """
    Evaluates a collection of trained models with statistical bootstrapping.

    This function quantifying model performance while providing confidence 
    intervals for all metrics. It also calculates the Kolmogorov-Smirnov 
    statistic to measure the physical divergence of the input features relative 
    to the baseline distribution.

    Args:
        trained_models (dict): Mapping of model names to fitted instances.
        X_test (np.ndarray): The feature matrix for evaluation.
        y_test (np.ndarray): The ground truth labels.
        shift_type (str): The name of the dataset shift applied.
        intensity (float): The magnitude of the shift applied.

    Returns:
        pd.DataFrame: A DataFrame containing metrics, confidence intervals, 
                      and distribution shift statistics.
    """
    global _BASELINE_X_CACHE, _BASELINE_METRICS_CACHE
    results = []
    
    # We cache the very first clean test set encountered to serve as the reference 
    # point for all future comparisons.
    is_baseline = (intensity == 0.0 or shift_type == "Baseline")
    if _BASELINE_X_CACHE is None and is_baseline:
        _BASELINE_X_CACHE = X_test.copy()
    
    # Calculate distribution shift using the statistics module
    ks_results = statistics.calculate_feature_drift(
        _BASELINE_X_CACHE, X_test, list(range(X_test.shape[1]))
    )
    
    has_multiple_classes = len(np.unique(y_test)) > 1
    n_iterations = 100
    
    for name, model in trained_models.items():
        # Baseline point estimates
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-decision)) 
        else:
            y_prob = y_pred
            
        point_acc, point_f1, point_roc, point_brier = _calculate_metrics(
            y_test, y_pred, y_prob, has_multiple_classes
        )
        
        # Bootstrapping loop for uncertainty quantification
        boot_acc, boot_f1, boot_roc, boot_brier = [], [], [], []
        indices = np.arange(len(y_test))
        
        for _ in range(n_iterations):
            # Resampling with replacement simulates multiple draws from the same population
            resample_idx = np.random.choice(indices, size=len(indices), replace=True)
            y_true_b = y_test[resample_idx]
            y_pred_b = y_pred[resample_idx]
            y_prob_b = y_prob[resample_idx]
            
            # Metric check for single-class resamples to prevent calculation errors
            b_multi = len(np.unique(y_true_b)) > 1
            a, f, r, b = _calculate_metrics(y_true_b, y_pred_b, y_prob_b, b_multi)
            
            boot_acc.append(a)
            boot_f1.append(f)
            boot_roc.append(r)
            boot_brier.append(b)
            
        # Store baseline accuracies for future hypothesis testing
        if is_baseline:
            _BASELINE_METRICS_CACHE[name] = boot_acc
            p_val = np.nan
            significant = False
        else:
            # Shifted vs Baseline Hypothesis Testing
            baseline_dist = _BASELINE_METRICS_CACHE.get(name, [])
            if baseline_dist:
                hr = statistics.perform_hypothesis_test(baseline_dist, boot_acc)
                p_val = hr["p_value"]
                significant = hr["significant"]
            else:
                p_val = np.nan
                significant = False

        results.append({
            "Model": name,
            "Shift_Type": shift_type,
            "Intensity": intensity,
            "Accuracy": point_acc,
            "Accuracy_Lower_CI": np.nanpercentile(boot_acc, 2.5),
            "Accuracy_Upper_CI": np.nanpercentile(boot_acc, 97.5),
            "F1_Score": point_f1,
            "F1_Lower_CI": np.nanpercentile(boot_f1, 2.5),
            "F1_Upper_CI": np.nanpercentile(boot_f1, 97.5),
            "ROC_AUC": point_roc,
            "ROC_Lower_CI": np.nanpercentile(boot_roc, 2.5),
            "ROC_Upper_CI": np.nanpercentile(boot_roc, 97.5),
            "Brier_Score": point_brier,
            "Brier_Lower_CI": np.nanpercentile(boot_brier, 2.5),
            "Brier_Upper_CI": np.nanpercentile(boot_brier, 97.5),
            "KS_Statistic": ks_results["avg_ks"],
            "P_Value": p_val,
            "Significant_Shift": significant
        })
        
    return pd.DataFrame(results)

def get_top_n_features(X_train, y_train, n=5):
    """
    Identifies the top N most informative features using a Random Forest heuristic.

    By training a shallow ensemble on the training data, we can extract Gini 
    importance scores. These indices are used to selectively corrupt the 
    most critical dimensions during 'Concept-Adjacent' shift simulations.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.
        n (int): The number of top feature indices to retrieve.

    Returns:
        list: A list of integer indices corresponding to the most importantes features.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Shallow depth keeps computation fast while still surfacing strong linear/non-linear signals
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    
    # Argsort provides indices that sort the array; we slice the tail for highest values
    top_n_indices = np.argsort(importances)[::-1][:n].tolist()
    
    return top_n_indices
