"""
Statistical Analysis and Distribution Shift Measurement Module

This module provides tools for quantifying the statistical divergence between 
data distributions. It specifically implements the Kolmogorov-Smirnov (KS) test 
to measure feature-level drift during dataset shift simulations, providing a 
mathematical basis for intensity measurements.
"""

import numpy as np
from scipy.stats import ks_2samp

def calculate_feature_drift(X_baseline, X_shifted, feature_indices):
    """
    Quantifies the distribution shift for specific features using the KS test.

    The Kolmogorov-Smirnov test is computed for each specified feature between 
     the baseline and the shifted datasets. This measures the maximum distance 
    between the empirical cumulative distribution functions, providing a value 
    between 0 (identical) and 1 (completely divergent).

    Args:
        X_baseline (np.ndarray): The original, uncorrupted feature matrix.
        X_shifted (np.ndarray): The feature matrix after shift simulation.
        feature_indices (list): Indices of features to analyze (typically continuous).

    Returns:
        dict: A dictionary containing the average KS statistic and individual 
              feature p-values for granular analysis.
    """
    if X_baseline is None or X_shifted is None:
        return {"avg_ks": 0.0, "max_ks": 0.0}
        
    ks_stats = []
    
    for idx in feature_indices:
        # We compare the same feature across the two samples
        stat, p_val = ks_2samp(X_baseline[:, idx], X_shifted[:, idx])
        ks_stats.append(stat)
        
    avg_ks = np.mean(ks_stats) if ks_stats else 0.0
    max_ks = np.max(ks_stats) if ks_stats else 0.0
    
    return {
        "avg_ks": avg_ks,
        "max_ks": max_ks
    }

def demonstrate_clt(data, n_samples=100, sample_size=30):
    """
    Demonstrates the Central Limit Theorem using model performance samples.
    
    CLT states that the distribution of sample means will be approximately normal, 
    regardless of the underlying distribution, as sample size increases.
    """
    sample_means = []
    for _ in range(n_samples):
        sample = np.random.choice(data, size=sample_size, replace=True)
        sample_means.append(np.mean(sample))
    
    return {
        "mean_of_means": np.mean(sample_means),
        "std_of_means": np.std(sample_means),
        "sample_means": sample_means
    }

def demonstrate_lln(data, max_samples=1000):
    """
    Demonstrates the Law of Large Numbers.
    
    LLN states that as the number of trials increases, the actual ratio of outcomes 
    will converge on the theoretical, or expected, ratio of outcomes.
    """
    running_means = []
    for i in range(1, max_samples + 1):
        running_means.append(np.mean(data[:i]))
    
    return running_means

def perform_hypothesis_test(baseline_metrics, shifted_metrics):
    """
    Performs a t-test to determine if the performance drop is statistically significant.
    
    H0: There is no difference in mean performance.
    Ha: The mean performance of the shifted data is lower than the baseline.
    """
    from scipy.stats import ttest_ind
    
    stat, p_val = ttest_ind(baseline_metrics, shifted_metrics, alternative='greater')
    
    return {
        "t_statistic": stat,
        "p_value": p_val,
        "significant": p_val < 0.05
    }
