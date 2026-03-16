"""
feature_pipeline.py

Assembles the full sklearn pipeline from the custom transformers and
standard sklearn steps. The pipeline runs on a pandas DataFrame and outputs
a numpy array ready for tree-based models.

We use ColumnTransformer for parallel numeric/categorical tracks and then
chain our custom transformers in the order that makes logical sense: macro
engineering first, then debt burden (needs installment + annual_inc), then
stability (needs emp_length), then FICO bucketing, then standard scaling
for the logistic regression baseline.

The pipeline is serialized to disk alongside the model in every MLflow run
so that the serving layer can call `pipeline.transform(X)` without knowing
anything about the raw schema.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.transformers import (
    DebtBurdenRatioTransformer,
    EmploymentStabilityTransformer,
    FICOBucketTransformer,
    MacroFeatureEngineer,
)

# Features that go into the numeric impute-and-scale track.
# These all exist after the custom transforms run.
NUMERIC_FEATURES = [
    "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc",
    "dti", "delinq_2yrs", "fico_score", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "mort_acc",
    "pub_rec_bankruptcies", "grade_enc", "sub_grade_enc", "term_months",
    "unemployment_rate", "cpi", "fed_funds_rate",
    "unemployment_mom_change", "cpi_yoy_pct",
    # derived by custom transformers (added in fit/transform order)
    "debt_burden_ratio", "dti_x_unemployment",
    "fico_bucket", "employment_stability",
    "low_rate_env", "high_rate_env", "recession_signal", "high_inflation",
    "emp_length",
]


def make_feature_pipeline(scale_for_lr: bool = False) -> Pipeline:
    """Build the preprocessing pipeline.
    
    Args:
        scale_for_lr: If True, append StandardScaler at the end. The tree-based
                      models don't need it, but the logistic regression baseline does.
    
    Returns:
        Unfitted sklearn Pipeline.
    """
    steps = [
        ("macro_engineer", MacroFeatureEngineer()),
        ("debt_burden", DebtBurdenRatioTransformer()),
        ("employment_stability", EmploymentStabilityTransformer()),
        ("fico_bucket", FICOBucketTransformer()),
    ]

    if scale_for_lr:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the feature columns that actually exist in the dataframe.
    
    The full NUMERIC_FEATURES list is the superset; we filter to what's
    present so that the same pipeline code works on subsets of the full
    feature matrix during testing.
    """
    return [c for c in NUMERIC_FEATURES if c in df.columns]


def fit_and_save_pipeline(
    df: pd.DataFrame,
    pipeline: Pipeline,
    save_path: Path,
) -> np.ndarray:
    """Fit the pipeline on df and save it for serving."""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    X_transformed = pipeline.fit_transform(X)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    return X_transformed


def load_pipeline(path: Path) -> Pipeline:
    return joblib.load(path)


def transform_single_record(record: dict, pipeline: Pipeline) -> np.ndarray:
    """Transform a single loan application dict through the fitted pipeline.
    
    Used by the serving layer to convert raw API payload → feature vector.
    """
    df = pd.DataFrame([record])
    feature_cols = get_feature_columns(df)
    # Fill any completely missing columns with 0 rather than erroring
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    return pipeline.transform(df[NUMERIC_FEATURES])
