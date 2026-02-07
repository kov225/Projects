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

    Assumes there is a file named german_credit_data.csv in the dataset folder.
    That file normally has columns such as:

        Age, Sex, Job, Housing, Saving accounts, Checking account,
        Credit amount, Duration, Purpose, Risk

    Risk is usually 'good' or 'bad'.
    We will map it to 0 and 1.
    """
    print("Downloading German Credit dataset from Kaggle...")
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
            "Risk column had unexpected values. Expected 'good' and 'bad'."
        )

    df = df.drop(columns=["Risk"])

    return df


def build_pipeline(df: pd.DataFrame):
    """
    Build a preprocessing and model pipeline.

    Categorical features are one hot encoded.
    Numeric features are left as is.
    """
    target_col = "label"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Categorical = object dtype
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

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
    print("Loading Kaggle German Credit dataset...")
    df = load_kaggle_german_credit()
    print(f"Dataset loaded with shape: {df.shape}")

    X, y, pipeline = build_pipeline(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC: {auc:.3f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "credit_model.pkl")

    print(f"Saving trained model to {model_path}...")
    joblib.dump(pipeline, model_path)

    print("Training complete.")


if __name__ == "__main__":
    main()

