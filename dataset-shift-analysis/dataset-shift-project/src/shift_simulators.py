"""
Dataset Shift Simulation Module  Milestone 2

Six mechanistically distinct shift families are implemented here.
Every public function follows a consistent contract:

    Input   -> X (np.ndarray), optional y (np.ndarray), intensity (float in [0, 1])
    Output  -> (X_out, y_out) or just X_out depending on whether labels change

Intensity = 0.0 always returns the unmodified inputs so that the baseline
evaluation pass can reuse the same code path without branching at call sites.

Shift taxonomy implemented:
    A. Covariate Shift         P(X) changes, P(Y|X) unchanged
    B. Label Shift             P(Y) changes, P(X|Y) unchanged
    C. Concept Drift           P(Y|X) changes, P(X) unchanged
    D. Noise Corruption        Additive Gaussian + categorical scrambling
    E. Missingness Shift       MCAR and MAR masking patterns
    F. Feature Removal         Zeroing of informative columns (simulates sensor failure)
"""

import numpy as np
from copy import deepcopy


# ---------------------------------------------------------------------------
# A. Covariate Shift
# ---------------------------------------------------------------------------

def apply_covariate_shift(
    X: np.ndarray,
    continuous_indices: list,
    intensity: float = 0.0
) -> np.ndarray:
    """
    Simulates Covariate Shift by injecting scaled Gaussian noise and a
    multiplicative drift into continuous features.

    The marginal P(X) is modified while the conditional P(Y|X) is preserved.
    Noise standard deviation scales linearly with intensity so that
    intensity = 1.0 corresponds to noise of magnitude 2.0 sigma (relative to
    the standardized feature space where features have unit variance).

    Args:
        X:                  Feature matrix (n_samples, n_features).
        continuous_indices: Column indices of continuous features to perturb.
        intensity:          Shift magnitude in [0, 1]. 0 returns X unchanged.

    Returns:
        Perturbed feature matrix of the same shape as X.
    """
    if intensity == 0.0:
        return X

    X_out = deepcopy(X)
    noise_scale = intensity * 2.0
    n_samples = X_out.shape[0]
    n_cont = len(continuous_indices)

    noise = np.random.normal(loc=0.0, scale=noise_scale,
                             size=(n_samples, n_cont))
    drift_factor = 1.0 + (intensity * 0.15)

    for k, col in enumerate(continuous_indices):
        X_out[:, col] = X_out[:, col] * drift_factor + noise[:, k]

    return X_out


def apply_feature_scaling_drift(
    X: np.ndarray,
    continuous_indices: list,
    intensity: float = 0.0
) -> np.ndarray:
    """
    Simulates distributional shift via heterogeneous feature rescaling.

    Each continuous feature is scaled by a factor drawn from a log-normal
    distribution whose spread grows with intensity.  This models scenarios
    where sensor calibration drifts or measurement units change silently
    between deployment environments.

    Args:
        X:                  Feature matrix (n_samples, n_features).
        continuous_indices: Indices of continuous features to rescale.
        intensity:          Shift magnitude in [0, 1].

    Returns:
        Rescaled feature matrix of the same shape as X.
    """
    if intensity == 0.0:
        return X

    X_out = deepcopy(X)
    sigma_log = intensity * 0.8

    for col in continuous_indices:
        scale_factor = np.random.lognormal(mean=0.0, sigma=sigma_log)
        X_out[:, col] = X_out[:, col] * scale_factor

    return X_out


# ---------------------------------------------------------------------------
# B. Label Shift (Prior Probability Shift)
# ---------------------------------------------------------------------------

def apply_label_shift(
    X: np.ndarray,
    y: np.ndarray,
    intensity: float = 0.0
) -> tuple:
    """
    Simulates Label Shift by selectively undersampling the minority class.

    The marginal P(Y) is deliberately altered while the class-conditional
    P(X|Y) distributions remain intact.  As intensity rises the minority
    class is progressively thinned, driving the effective class ratio toward
    the majority-only limit.

    Args:
        X:         Feature matrix (n_samples, n_features).
        y:         Label vector (n_samples,).
        intensity: Fraction of minority samples to drop, in [0, 1].
                   Capped at 0.95 to ensure at least one minority sample
                   survives for metric calculations.

    Returns:
        Tuple (X_shifted, y_shifted) with resampled rows.
    """
    if intensity == 0.0:
        return X, y

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y

    majority = classes[np.argmax(counts)]
    minority = classes[np.argmin(counts)]

    maj_idx = np.where(y == majority)[0]
    min_idx = np.where(y == minority)[0]

    drop_frac = min(0.95, intensity)
    n_keep    = max(1, int(len(min_idx) * (1.0 - drop_frac)))

    kept_min = np.random.choice(min_idx, size=n_keep, replace=False)
    all_idx  = np.concatenate([maj_idx, kept_min])
    np.random.shuffle(all_idx)

    return X[all_idx], y[all_idx]


# ---------------------------------------------------------------------------
# C. Concept Drift
# ---------------------------------------------------------------------------

def apply_concept_drift(
    X: np.ndarray,
    y: np.ndarray,
    top_feature_indices: list,
    intensity: float = 0.0
) -> tuple:
    """
    Simulates Concept Drift by corrupting key features and simultaneously
    flipping a proportion of labels.

    Two mechanisms compound to disrupt P(Y|X):
      1. Top informative features are overwritten with uniform random values
         drawn from each feature's observed range (corrupting the signal).
      2. A fraction of labels equal to intensity/2 are randomly flipped to
         inject incorrect ground truth (modeling annotation drift).

    Args:
        X:                   Feature matrix (n_samples, n_features).
        y:                   Label vector (n_samples,).
        top_feature_indices: Columns targeted for corruption.
        intensity:           Drift magnitude in [0, 1].

    Returns:
        Tuple (X_drifted, y_drifted).
    """
    if intensity == 0.0:
        return X, y

    X_out = deepcopy(X)
    y_out = y.copy()
    n_samples = X_out.shape[0]

    n_corrupt_rows = int(n_samples * min(1.0, intensity))
    for col in top_feature_indices:
        col_min = X[:, col].min()
        col_max = X[:, col].max()
        corrupt_rows = np.random.choice(n_samples, size=n_corrupt_rows, replace=False)
        X_out[corrupt_rows, col] = np.random.uniform(col_min, col_max, size=n_corrupt_rows)

    # Label flip fraction is half the feature corruption rate to avoid
    # rendering labels completely meaningless at moderate intensities.
    n_flip = int(n_samples * min(0.5, intensity * 0.5))
    if n_flip > 0:
        flip_rows = np.random.choice(n_samples, size=n_flip, replace=False)
        y_out[flip_rows] = 1 - y_out[flip_rows]

    return X_out, y_out


# ---------------------------------------------------------------------------
# D. Noise Corruption
# ---------------------------------------------------------------------------

def apply_gaussian_noise(
    X: np.ndarray,
    continuous_indices: list,
    intensity: float = 0.0
) -> np.ndarray:
    """
    Adds independent Gaussian noise to every continuous feature.

    Unlike the Covariate Shift simulator which also applies multiplicative
    drift, this function applies purely additive i.i.d. Gaussian noise.
    This is the simplest robustness stress test and serves as a diagnostic
    baseline for how much raw noise a model can tolerate.

    Args:
        X:                  Feature matrix (n_samples, n_features).
        continuous_indices: Columns to add noise to.
        intensity:          Noise standard deviation (scales with intensity).

    Returns:
        Noisy feature matrix of the same shape as X.
    """
    if intensity == 0.0:
        return X

    X_out = deepcopy(X)
    noise = np.random.normal(0.0, intensity, size=(X_out.shape[0], len(continuous_indices)))
    for k, col in enumerate(continuous_indices):
        X_out[:, col] += noise[:, k]

    return X_out


# ---------------------------------------------------------------------------
# E. Missingness Shift
# ---------------------------------------------------------------------------

def apply_mcar_missingness(
    X: np.ndarray,
    intensity: float = 0.0,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Missing Completely At Random (MCAR): each cell is independently masked
    with probability equal to intensity.

    Masked cells are replaced with fill_value (default = 0.0, which maps
    to the mean in a standardized feature space).  This models random
    instrumentation failures or survey non-response with no systematic bias.

    Args:
        X:          Feature matrix (n_samples, n_features).
        intensity:  Masking probability per cell, in [0, 1].
        fill_value: Replacement value for masked entries.

    Returns:
        Feature matrix with MCAR missingness applied.
    """
    if intensity == 0.0:
        return X

    X_out = deepcopy(X)
    mask = np.random.rand(*X_out.shape) < intensity
    X_out[mask] = fill_value

    return X_out


def apply_mar_missingness(
    X: np.ndarray,
    condition_feature_idx: int,
    target_feature_indices: list,
    intensity: float = 0.0,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Missing At Random (MAR): target features are masked only for samples
    where the conditioning feature exceeds its median.

    This models a structured missing-data mechanism where missingness
    depends on observed (but not target) data, for example a sensor that
    stops reporting when another reading is above a threshold.

    Args:
        X:                      Feature matrix (n_samples, n_features).
        condition_feature_idx:  Column whose value triggers missingness.
        target_feature_indices: Columns to be masked in triggered rows.
        intensity:              Fraction of triggered rows to actually mask.
        fill_value:             Replacement value for masked entries.

    Returns:
        Feature matrix with MAR missingness applied.
    """
    if intensity == 0.0:
        return X

    X_out = deepcopy(X)
    threshold = np.median(X[:, condition_feature_idx])
    eligible = np.where(X[:, condition_feature_idx] > threshold)[0]

    n_mask = int(len(eligible) * min(1.0, intensity))
    rows_to_mask = np.random.choice(eligible, size=n_mask, replace=False)

    for col in target_feature_indices:
        X_out[rows_to_mask, col] = fill_value

    return X_out


# ---------------------------------------------------------------------------
# F. Feature Removal
# ---------------------------------------------------------------------------

def apply_feature_removal(
    X: np.ndarray,
    feature_indices: list,
    intensity: float = 0.0
) -> np.ndarray:
    """
    Simulates feature removal by zeroing out the top informative columns.

    The number of features zeroed is proportional to intensity.  At
    intensity = 1.0 all provided feature_indices are zeroed, modelling a
    complete sensor or data-pipeline failure for those channels.

    Args:
        X:               Feature matrix (n_samples, n_features).
        feature_indices: Candidate columns for removal (ranked by importance).
        intensity:       Fraction of provided features to remove, in [0, 1].

    Returns:
        Feature matrix with selected columns zeroed.
    """
    if intensity == 0.0 or not feature_indices:
        return X

    X_out = deepcopy(X)
    n_remove = max(1, int(len(feature_indices) * intensity))
    cols_to_remove = feature_indices[:n_remove]

    for col in cols_to_remove:
        X_out[:, col] = 0.0

    return X_out
