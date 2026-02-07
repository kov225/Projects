# drift_detection.py
# This script measures drift using PSI (Population Stability Index)
# on the model's prediction scores before and after we simulate a shift
# in the input data.

import os

import joblib
import numpy as np
import pandas as pd

from train import load_kaggle_german_credit


MODEL_PATH = os.path.join("models", "credit_model.pkl")


def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between two distributions of scores.

    expected: baseline scores (e.g. predictions on training-like data)
    actual:   new scores (e.g. predictions on new incoming data)

    Higher PSI means more drift. In many practical guides:
    - PSI < 0.1  : low drift
    - 0.1–0.25   : moderate drift
    - > 0.25     : significant drift
    """
    # To keep things simple, I bin scores into equal-width buckets between 0 and 1.
    bins = np.linspace(0, 1, buckets + 1)

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_dist = expected_counts / expected_counts.sum()
    actual_dist = actual_counts / actual_counts.sum()

    psi = 0.0
    for e, a in zip(expected_dist, actual_dist):
        # Avoid division by zero or log of zero
        if e == 0 or a == 0:
            continue
        psi += (e - a) * np.log(e / a)

    return float(psi)


def main():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run train.py first."
        )

    print("Loading trained pipeline...")
    pipeline = joblib.load(MODEL_PATH)

    print("Loading baseline data...")
    df = load_kaggle_german_credit()
    X = df.drop(columns=["label"])

    # Baseline predictions (this is like our 'normal' world)
    print("Computing baseline predictions...")
    baseline_scores = pipeline.predict_proba(X)[:, 1]

    # Now I simulate a simple type of drift:
    # I increase Credit_amount and Duration a bit to represent a portfolio
    # with larger and longer loans.
    print("Creating a drifted version of the data...")
    drifted_X = X.copy()
    if "Credit_amount" in drifted_X.columns:
        drifted_X["Credit_amount"] = drifted_X["Credit_amount"] * 1.3
    if "Duration" in drifted_X.columns:
        drifted_X["Duration"] = drifted_X["Duration"] * 1.2

    print("Computing predictions on drifted data...")
    drifted_scores = pipeline.predict_proba(drifted_X)[:, 1]

    psi_score = calculate_psi(baseline_scores, drifted_scores)

    print("\nPSI drift result:")
    print(f"PSI score: {psi_score:.3f}")
    if psi_score > 0.25:
        print("Interpretation: Significant drift detected.")
    elif psi_score > 0.10:
        print("Interpretation: Moderate drift detected.")
    else:
        print("Interpretation: Low drift detected.")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
