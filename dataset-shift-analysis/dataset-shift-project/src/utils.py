"""
Utility Module

Centralized helpers for reproducibility seeding, directory management,
Population Stability Index (PSI), and structured console logging.
All experiment modules import from here rather than duplicating logic.
"""

import os
import sys
import logging
import numpy as np


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = "shift_study") -> logging.Logger:
    """
    Returns a consistently formatted logger for the given module name.

    The logger writes to stdout at INFO level. All modules in this project
    should obtain their logger through this factory to maintain a uniform
    output style.

    Args:
        name: Logger name, typically set to __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """
    Seeds NumPy's global random state for cross-module reproducibility.

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Path Management
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_results_path(filename: str = "experiment_results.csv") -> str:
    """Returns the absolute path to the results CSV file."""
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, filename)


def get_figures_path(filename: str) -> str:
    """Returns the absolute path for a figure saved under the figures/ directory."""
    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return os.path.join(figures_dir, filename)


def get_outputs_path(filename: str) -> str:
    """Returns the absolute path for an artifact under the outputs/ directory."""
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    return os.path.join(outputs_dir, filename)


# ---------------------------------------------------------------------------
# Population Stability Index (PSI)
# ---------------------------------------------------------------------------

def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes the Population Stability Index between two univariate distributions.

    PSI measures how much a feature distribution has shifted between two
    populations (e.g. training vs. deployment). The standard interpretation:
      PSI < 0.10 : No significant shift
      PSI < 0.20 : Moderate shift requiring monitoring
      PSI >= 0.20: Significant shift requiring action

    The implementation uses equal-width bins derived from the combined range
    of both arrays to avoid empty-bucket edge cases.

    Args:
        expected: Reference (baseline) 1-D array of feature values.
        actual:   Target (shifted) 1-D array of feature values.
        n_bins:   Number of histogram bins. Defaults to 10.

    Returns:
        Scalar PSI value.
    """
    combined = np.concatenate([expected, actual])
    bin_edges = np.linspace(combined.min(), combined.max(), n_bins + 1)

    # Proportions per bin for each distribution
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _   = np.histogram(actual,   bins=bin_edges)

    # Convert to proportions; add epsilon to guard against zero-division and log(0)
    eps = 1e-8
    expected_pct = expected_counts / (len(expected) + eps) + eps
    actual_pct   = actual_counts   / (len(actual)   + eps) + eps

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


def compute_avg_psi(X_baseline: np.ndarray, X_shifted: np.ndarray,
                    feature_indices: list, n_bins: int = 10) -> float:
    """
    Computes the mean PSI across a specified set of feature columns.

    Args:
        X_baseline:      Reference feature matrix (n_samples, n_features).
        X_shifted:       Shifted feature matrix (n_samples, n_features).
        feature_indices: List of column indices to include in the calculation.
        n_bins:          Number of histogram bins per feature.

    Returns:
        Average PSI value across the selected features.
    """
    if X_baseline is None or X_shifted is None:
        return 0.0
    psi_scores = [
        compute_psi(X_baseline[:, i], X_shifted[:, i], n_bins)
        for i in feature_indices
    ]
    return float(np.mean(psi_scores)) if psi_scores else 0.0
