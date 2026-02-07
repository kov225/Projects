# train.py
# This script trains a credit risk model using the German Credit dataset from Kaggle.
# I am using an XGBoost model wrapped in a scikit-learn Pipeline so that
# preprocessing and the model stay together when I save and load it.

import os

import joblib
import numpy as np
import pandas as pd
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from xgboost import XGBClassifier


def load_kaggle_german_credit() -> pd.DataFrame:
    """
    Load the German Credit dataset from Kaggle using kagglehub.

    In the uciml/german-credit dataset there is a file called german_credit_data.csv
    which has columns like:

    Age, Sex, Job, Housing, Saving accounts, Checking account,
    Credit amount, Duration, Purpose, Risk

    Risk is usually 'good' or 'bad'.
    I will map it to 0 and 1 so that the model can learn on numeric labels.
    """
    print("Downloading German Credit dataset from Kaggle (this is cached after first run)...")
    path = kagglehub.dataset_download("uciml/german-credit")
    csv_path = os.path.join(path, "german_credit_data.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find german_credit_data.csv under {path}. "
            "Open that folder locally and check the exact CSV name, "
            "then update csv_path in train.py if needed."
        )

    df = pd.read_csv(csv_path)

    if "Risk" not in df.columns:
        raise ValueError(
            "Expected a 'Risk' column in german_credit_data.csv. "
            "Open the CSV and confirm the label column name."
        )

    # Map label: 'good' -> 0, 'bad' -> 1 (1 means higher risk)
    df["label"] = df["Risk"].map({"good": 0, "bad": 1})
    if df["label"].isnull().any():
        raise ValueError(
            "Risk column had unexpected values. Expected only 'good' and 'bad'."
        )

    # We keep all original feature columns (Age, Sex, Job, etc.) and drop the original Risk.
    df = df.drop(columns=["Risk"])

    return df


def build_pipeline(df: pd.DataFrame):
    """
    Build a preprocessing and model pipeline.

    Idea:
    - Categorical features get one-hot encoded
    - Numeric features are passed through
    - The XGBoost model sits at the end

    I am keeping the logic inside a scikit-learn Pipeline so that when I save the model,
    I can reload it later and call .predict() or .predict_proba() directly on a raw DataFrame.
    """
    target_col = "label"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Any column with dtype 'object' is treated as categorical here.
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    # XGBoost tends to work well on tabular credit data.
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return X, y, clf


def main():
    print("\nStep 1: Loading Kaggle German Credit dataset...")
    df = load_kaggle_german_credit()
    print(f"Dataset loaded with shape: {df.shape}")

    print("\nStep 2: Building pipeline (preprocessing + XGBoost model)...")
    X, y, pipeline = build_pipeline(df)

    print("\nStep 3: Splitting train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")

    print("\nStep 4: Training model...")
    pipeline.fit(X_train, y_train)

    print("\nStep 5: Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\nAccuracy: {acc:.3f}")
    print(f"ROC AUC:  {auc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nStep 6: Saving trained pipeline...")
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "credit_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")

    print("\nTraining complete. You can now load this pipeline in api.py, drift scripts, or explain.py.\n")


if __name__ == "__main__":
    main()
