# api.py
# This file exposes the trained credit risk model as a simple FastAPI service.
# The idea is to show how the model can be called in a real application.

import os
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from train import load_kaggle_german_credit


# Use the same path we used in train.py
MODEL_PATH = os.path.join("models", "credit_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model file not found at {MODEL_PATH}. "
        "Run train.py first so that the trained pipeline is saved."
    )

print(f"Loading trained pipeline from {MODEL_PATH}...")
pipeline = joblib.load(MODEL_PATH)
print("Model loaded successfully.")


# I load the dataset once here so that I can recover the original feature columns.
# This way I can make sure the input format in the API matches what the model expects.
_raw_df = load_kaggle_german_credit()
FEATURE_COLUMNS = _raw_df.drop(columns=["label"]).columns.tolist()


class CreditRequest(BaseModel):
    # These fields match the german_credit_data.csv columns (except Risk).
    # Some of them are optional in the API to keep it simple; if they are missing
    # I will fill with default values.
    Age: int
    Sex: str
    Job: int
    Housing: str
    Saving_accounts: Optional[str] = None
    Checking_account: Optional[str] = None
    Credit_amount: int
    Duration: int
    Purpose: str


class CreditResponse(BaseModel):
    default_probability: float
    predicted_label: int
    risk_level: str


app = FastAPI(
    title="Credit Risk Scoring API",
    description="Simple FastAPI wrapper around the trained XGBoost credit model.",
)


@app.get("/")
def root():
    return {
        "message": "Credit risk scoring API is up. "
                   "Send a POST request to /predict with customer features."
    }


@app.post("/predict", response_model=CreditResponse)
def predict(request: CreditRequest):
    # Convert the incoming Pydantic object into a single-row DataFrame
    data = pd.DataFrame([request.dict()])

    # Make sure we have all feature columns expected by the pipeline.
    # If any column is missing (for example account info), we fill it with a default.
    for col in FEATURE_COLUMNS:
        if col not in data.columns:
            data[col] = None

    # Reorder columns to match the training DataFrame
    data = data[FEATURE_COLUMNS]

    # Call the pipeline directly; it will handle preprocessing + model.
    proba_default = pipeline.predict_proba(data)[0, 1]
    label = int(pipeline.predict(data)[0])

    # Turning label into a simple human-readable string
    risk_level = "high" if label == 1 else "low"

    return CreditResponse(
        default_probability=float(proba_default),
        predicted_label=label,
        risk_level=risk_level,
    )
