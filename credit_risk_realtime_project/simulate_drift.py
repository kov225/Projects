# simulate_drift.py
# This script focuses more on the feature side: it prints how some basic
# statistics change when we simulate a population shift, and it compares
# the average predicted risk before and after.

import os

import joblib
import numpy as np
import pandas as pd

from train import load_kaggle_german_credit


MODEL_PATH = os.path.join("models", "credit_model.pkl")


def main():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run train.py first."
        )

    print("Loading trained pipeline...")
    pipeline = joblib.load(MODEL_PATH)

    print("Loading original data...")
    df = load_kaggle_german_credit()
    X = df.drop(columns=["label"])

    # Compute some simple stats before drift
    print("\nOriginal data statistics:")
    for col in ["Age", "Credit_amount", "Duration"]:
        if col in X.columns:
            print(f"  {col} mean: {X[col].mean():.2f}")

    # Baseline predicted probabilities
    baseline_scores = pipeline.predict_proba(X)[:, 1]
    print(f"\nBaseline average default probability: {baseline_scores.mean():.3f}")

    # Simulate a different portfolio: slightly younger customers but larger loans
    drifted_X = X.copy()
    if "Age" in drifted_X.columns:
        drifted_X["Age"] = np.maximum(drifted_X["Age"] - 5, 18)
    if "Credit_amount" in drifted_X.columns:
        drifted_X["Credit_amount"] = drifted_X["Credit_amount"] * 1.4
    if "Duration" in drifted_X.columns:
        drifted_X["Duration"] = drifted_X["Duration"] * 1.1

    print("\nDrifted data statistics:")
    for col in ["Age", "Credit_amount", "Duration"]:
        if col in drifted_X.columns:
            print(f"  {col} mean: {drifted_X[col].mean():.2f}")

    drifted_scores = pipeline.predict_proba(drifted_X)[:, 1]
    print(f"\nDrifted average default probability: {drifted_scores.mean():.3f}")

    print("\nThis script gives a rough intuition for how changes in the incoming "
          "population can shift the overall risk level the model sees.\n")


if __name__ == "__main__":
    main()
