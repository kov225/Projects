"""
Dataset Shift Simulation Suite: Formal Distributional Perturbations

This module implements synthetic induction of three primary dataset shift categories:
1. Covariate Shift: P(X) changes while P(Y|X) remains invariant.
2. Prior Probability Shift: P(Y) changes while P(X|Y) remains invariant.
3. Concept Shift: P(Y|X) changes (simulated via feature-target disruption).

These are used to evaluate model robustness by measuring performance degradation
as the test distribution diverges from the training manifold.
"""

import logging
import numpy as np
import pandas as pd
from copy import deepcopy

logger = logging.getLogger(__name__)

def apply_covariate_shift(X, continuous_indices, intensity=0.0, noise_type="gaussian"):
    """
    Simulates Covariate Shift by altering the marginal distribution P(X).
    
    Args:
        X (np.ndarray): Feature matrix.
        continuous_indices (list): Indices of features to perturb.
        intensity (float): Std dev of noise (Gaussian) or scale (Laplacian).
        noise_type (str): 'gaussian' for standard drift, 'laplacian' for heavy-tailed shift.
    """
    if intensity == 0.0:
        return X
        
    X_shifted = deepcopy(X)
    num_samples = X_shifted.shape[0]
    num_continuous = len(continuous_indices)
    
    if noise_type == "gaussian":
        noise = np.random.normal(loc=0.0, scale=intensity, size=(num_samples, num_continuous))
    else:
        # Laplacian noise simulates environments with more frequent extreme outliers
        noise = np.random.laplace(loc=0.0, scale=intensity, size=(num_samples, num_continuous))
    
    for i, col_idx in enumerate(continuous_indices):
        # Multiplicative drift (systematic) + Additive noise (stochastic)
        X_shifted[:, col_idx] = X_shifted[:, col_idx] * (1.0 + (intensity * 0.1)) + noise[:, i]
        
    logger.debug(f"P(X) perturbed via {noise_type} noise | Intensity: {intensity}")
    return X_shifted

def apply_prior_shift(X, y, intensity=0.0):
    """
    Simulates Prior Probability Shift: P(Y) diverges while P(X|Y) is fixed.
    
    This is implemented via class-conditional undersampling of the minority class.
    """
    if intensity == 0.0:
        return X, y
        
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y 
        
    majority_class = classes[np.argmax(counts)]
    minority_class = classes[np.argmin(counts)]
    
    maj_indices = np.where(y == majority_class)[0]
    min_indices = np.where(y == minority_class)[0]
    
    # Efron's Resampling: Ensure at least 5% of minority remains for metric stability
    drop_fraction = min(0.95, intensity) 
    num_keep_min = max(1, int(len(min_indices) * (1.0 - drop_fraction)))
    
    keep_min_indices = np.random.choice(min_indices, size=num_keep_min, replace=False)
    new_indices = np.concatenate([maj_indices, keep_min_indices])
    np.random.shuffle(new_indices)
    
    logger.debug(f"P(Y) shifted | Minority retention: {1.0 - drop_fraction:.2%}")
    return X[new_indices], y[new_indices]

def apply_concept_adjacent_shift(X, top_n_indices, intensity=0.0):
    """
    Simulates Concept Shift via feature-specific concept disruption.
    
    Replaces a fraction of high-importance features with uniform noise, 
    breaking the learned P(Y|X) relationship.
    """
    if intensity == 0.0:
        return X
        
    X_shifted = deepcopy(X)
    num_samples = X_shifted.shape[0]
    num_corrupt = int(num_samples * min(1.0, intensity))
    
    for col_idx in top_n_indices:
        col_min, col_max = np.min(X[:, col_idx]), np.max(X[:, col_idx])
        corrupt_indices = np.random.choice(num_samples, size=num_corrupt, replace=False)
        
        # Injecting 'out-of-distribution' uniform samples
        X_shifted[corrupt_indices, col_idx] = np.random.uniform(col_min, col_max, size=num_corrupt)
        
    logger.debug(f"P(Y|X) disrupted for {len(top_n_indices)} primary features")
    return X_shifted

def apply_scaling_drift(X, continuous_indices, intensity=0.0):
    """
    Simulates Feature Scaling Drift by multiplying continuous features 
    by a drift factor.

    This is a variant of Covariate Shift where the scale of features 
    changes systematically, rather than just adding noise.

    Args:
        X (np.ndarray): Feature matrix.
        continuous_indices (list): Indices of features to be scaled.
        intensity (float): The magnitude of the scaling drift.

    Returns:
        np.ndarray: The scaled feature matrix.
    """
    if intensity == 0.0:
        return X
        
    X_shifted = deepcopy(X)
    
    # Systematic multiplicative drift
    # At intensity 1.0, features are scaled by 1.5x
    scale_factor = 1.0 + (intensity * 0.5)
    
    for col_idx in continuous_indices:
        X_shifted[:, col_idx] = X_shifted[:, col_idx] * scale_factor
        
    return X_shifted


def apply_feature_permutation_drift(X, col_indices, intensity=0.0):
    """
    Simulates Feature Permutation Drift by randomly shuffling a subset of 
    feature values across samples.

    This preserves the individual marginal distribution but destroys the 
    joint distribution and dependencies between features for a certain 
    proportion of the dataset.

    Args:
        X (np.ndarray): Feature matrix.
        col_indices (list): Indices of features to be permuted.
        intensity (float): The probability/proportion of samples to permute.

    Returns:
        np.ndarray: The permuted feature matrix.
    """
    if intensity == 0.0:
        return X
        
    X_shifted = deepcopy(X)
    X_shifted = np.array(X_shifted)
    num_samples = X_shifted.shape[0]
    
    # We identify the number of rows to shuffle based on intensity
    num_permute = int(num_samples * min(1.0, intensity))
    
    for col_idx in col_indices:
        permute_indices = np.random.choice(num_samples, size=num_permute, replace=False)
        shuffled_values = X_shifted[permute_indices, col_idx]
        np.random.shuffle(shuffled_values)
        X_shifted[permute_indices, col_idx] = shuffled_values
        
    return X_shifted


