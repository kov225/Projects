"""
Dataset Shift Simulation Module

This module provides algorithms for synthetically inducing three categories of 
dataset shift: Covariate Shift, Prior Probability Shift, and Concept-Adjacent 
Shift. These simulations are used to quantify the robustness of machine 
learning models under varying degrees of environmental distribution change.
"""

import numpy as np
import pandas as pd
from copy import deepcopy

def apply_covariate_shift(X, continuous_indices, intensity=0.0):
    """
    Simulates Covariate Shift by injecting Gaussian noise and a linear drift 
    into continuous features.

    The marginal distribution P(X) is modified by scaling each feature and 
    adding noise sampled from a zero-centered normal distribution with a 
    standard deviation equal to the provided intensity.

    Args:
        X (np.ndarray): The feature matrix to be transformed.
        continuous_indices (list): Indices of columns targeted for noise injection.
        intensity (float): The magnitude of the noise/drift (standard deviation).

    Returns:
        np.ndarray: The modified feature matrix.
    """
    if intensity == 0.0:
        return X
        
    # Deepcopy prevents mutations from propagating to the baseline test set
    X_shifted = deepcopy(X)
    
    num_samples = X_shifted.shape[0]
    num_continuous = len(continuous_indices)
    
    # Noise generation follows a normal Gaussian kernel
    noise = np.random.normal(loc=0.0, scale=intensity, size=(num_samples, num_continuous))
    
    # We apply both a multiplicative drift and additive noise for a more realistic shift
    for i, col_idx in enumerate(continuous_indices):
        X_shifted[:, col_idx] = X_shifted[:, col_idx] * (1.0 + (intensity * 0.1)) + noise[:, i]
        
    return X_shifted

def apply_prior_shift(X, y, intensity=0.0):
    """
    Simulates Prior Probability Shift by altering the class distribution P(Y).

    The function biases the test set by selectively dropping minority class 
    samples, thereby increasing the prevalence of the majority class. This 
    simulates environments where the relative frequency of labels has diverged 
    from the training distribution.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.
        intensity (float): The proportion of minority samples to potentially drop.

    Returns:
        tuple: A tuple (X_shifted, y_shifted) containing the resampled data.
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
    
    # We cap the drop fraction at 0.95 to ensure at least some minority samples remain 
    # for metric calculations (e.g., F1, ROC-AUC require at least one positive sample).
    drop_fraction = min(0.95, intensity) 
    num_keep_min = int(len(min_indices) * (1.0 - drop_fraction))
    
    if num_keep_min == 0:
        num_keep_min = 1 
        
    # Random selection without replacement maintains the intra-class distribution
    keep_min_indices = np.random.choice(min_indices, size=num_keep_min, replace=False)
    
    # Consolidated indices are shuffled to remove ordering bias
    new_indices = np.concatenate([maj_indices, keep_min_indices])
    np.random.shuffle(new_indices)
    
    return X[new_indices], y[new_indices]

def apply_concept_adjacent_shift(X, top_n_indices, intensity=0.0):
    """
    Simulates Concept-Adjacent Shift by corrupting key feature columns.

    This routine targets the most informative features and replaces a subset 
    of their values with random noise sampled uniformly from the feature's 
    original range. This disrupts the learned concept (P(Y|X)) for the 
    specified intensity of samples.

    Args:
        X (np.ndarray): Feature matrix.
        top_n_indices (list): Indices of features to be corrupted.
        intensity (float): The probability/proportion of samples to corrupt.

    Returns:
        np.ndarray: The corrupted feature matrix.
    """
    if intensity == 0.0:
        return X
        
    X_shifted = deepcopy(X)
    num_samples = X_shifted.shape[0]
    
    # The number of corrupted rows is proportional to the shift intensity
    num_corrupt = int(num_samples * min(1.0, intensity))
    
    for col_idx in top_n_indices:
        # We calculate the feature range to ensure noise stays within 'realistic' bounds
        col_min = np.min(X[:, col_idx])
        col_max = np.max(X[:, col_idx])
        
        # Consistent row indices are not required; randomization happens per feature
        corrupt_indices = np.random.choice(num_samples, size=num_corrupt, replace=False)
        
        # Injection of uniform noise destroys the conditional probability relationship
        random_noise = np.random.uniform(low=col_min, high=col_max, size=num_corrupt)
        X_shifted[corrupt_indices, col_idx] = random_noise
        
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


