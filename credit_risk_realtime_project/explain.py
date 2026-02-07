# explain.py
# This script generates SHAP explanations for the trained credit model.
# The goal is to show which features drive the predictions,
# which is important for credit risk and model governance.

import os

import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from train import load_kaggle_german_credit


MODEL_PATH = os.path.join("models", "credit_model.pkl")


def main():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run train.py first."
        )

    print("Loading trained pipeline...")
    pipeline = joblib.load(MODEL_PATH)

    print("Loading data for explanation...")
    df = load_kaggle_german_credit()
    X = df.drop(columns=["label"])

    # Take a small sample to keep SHAP reasonably fast
    sample = X.sample(n=300, random_state=42)

    # Split out preprocessor and model from the pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    print("Transforming data with the same preprocessor used during training...")
    X_transformed = preprocessor.transform(sample)

    # Build SHAP explainer for the XGBoost model
    print("Building SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    print("Computing SHAP values...")
    shap_values = explainer.shap_values(X_transformed)

    # SHAP summary plot: which features matter overall
    print("Saving SHAP summary plot to shap_summary.png")
    plt.figure()
    shap.summary_plot(shap_values, X_transformed, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close()

    print("\nSHAP summary saved as shap_summary.png in the current folder.")
    print("You can open this image to see which features drive default risk.\n")


if __name__ == "__main__":
    main()
