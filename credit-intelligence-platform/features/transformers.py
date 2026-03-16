"""
transformers.py

Custom sklearn-compatible transformers for the credit risk feature pipeline.
Each transformer inherits from BaseEstimator and TransformerMixin so it
slots cleanly into a Pipeline and plays well with cross_val_score and
GridSearchCV. The fit/transform split ensures that any statistics computed
during fit (like FICO bin edges) are derived only from training data and
then applied identically at inference time.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DebtBurdenRatioTransformer(BaseEstimator, TransformerMixin):
    """Constructs the debt burden ratio: installment / (annual_inc / 12).
    
    This is more informative than DTI alone because it captures the actual
    monthly cash flow impact relative to income rather than the total debt
    load. We clip it at [0, 5] to handle the rare case where income is tiny.

    We also add an interaction term: dti * unemployment_rate, which captures
    how much worse high leverage becomes in a weak labor market. This feature
    type comes directly from the type of macro-aware feature engineering we
    did at CASHe for the lending segmentation pipeline.
    """

    def __init__(self):
        self.median_income_ = None

    def fit(self, X: pd.DataFrame, y=None) -> "DebtBurdenRatioTransformer":
        self.median_income_ = X["annual_inc"].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        monthly_income = X["annual_inc"].fillna(self.median_income_) / 12
        monthly_income = monthly_income.replace(0, 1)  # avoid division by zero

        X["debt_burden_ratio"] = (X["installment"] / monthly_income).clip(0, 5)

        # Macro interaction: how bad is your leverage in *this* economic environment?
        if "unemployment_rate" in X.columns and "dti" in X.columns:
            X["dti_x_unemployment"] = (
                X["dti"].fillna(X["dti"].median()) * X["unemployment_rate"].fillna(5.0)
            )

        return X

    def get_feature_names_out(self, input_features=None):
        return ["debt_burden_ratio", "dti_x_unemployment"]


class FICOBucketTransformer(BaseEstimator, TransformerMixin):
    """Converts continuous FICO score into ordinal risk buckets.
    
    The credit industry already buckets FICO this way operationally.
    Using buckets rather than the raw score makes the feature more robust
    to small measurement noise and aligns the model's representation with
    how underwriters actually think about credit quality.
    """

    BINS = [0, 579, 619, 659, 699, 739, 779, 850]
    LABELS = [0, 1, 2, 3, 4, 5, 6]  # subprime → exceptional

    def fit(self, X: pd.DataFrame, y=None) -> "FICOBucketTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        fico_col = "fico_score" if "fico_score" in X.columns else "fico_range_low"
        if fico_col in X.columns:
            X["fico_bucket"] = pd.cut(
                X[fico_col],
                bins=self.BINS,
                labels=self.LABELS,
                include_lowest=True,
            ).astype(float)
        return X

    def get_feature_names_out(self, input_features=None):
        return ["fico_bucket"]


class EmploymentStabilityTransformer(BaseEstimator, TransformerMixin):
    """Computes an employment stability score from emp_length and verification_status.
    
    Longer employment tenure is a strong signal of income stability. We
    scale it to [0, 1] and add a bonus for income-verified borrowers
    because verification substantially reduces the risk of stated-income fraud.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "EmploymentStabilityTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # emp_length is in years (0-10+), cap at 10 and normalize
        emp_norm = X.get("emp_length", pd.Series(5.0, index=X.index)).fillna(0).clip(0, 10) / 10

        # Verification bonus: "Verified" or "Source Verified" add 0.15 to the score
        verify_bonus = 0.0
        for col in X.columns:
            if "verification_status" in col and "_Verified" in col:
                verify_bonus = X[col].astype(float) * 0.15
                break

        X["employment_stability"] = (emp_norm + verify_bonus).clip(0, 1)
        return X

    def get_feature_names_out(self, input_features=None):
        return ["employment_stability"]


class MacroFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives additional features from the macro time series columns.
    
    The raw unemployment/CPI/fed_funds values are useful but adding
    volatility signals and rate regime indicators gives the model more
    discriminative power during economic transitions. These features were
    inspired by the kind of macro-risk features I built in the CASHe
    lending market segmentation pipeline.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "MacroFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if "fed_funds_rate" in X.columns:
            # Low-rate environment flag: Fed funds < 1% → near-zero interest rate policy
            X["low_rate_env"] = (X["fed_funds_rate"] < 1.0).astype(int)
            # High interest rate environment: > 4% puts pressure on variable-rate borrowers
            X["high_rate_env"] = (X["fed_funds_rate"] > 4.0).astype(int)

        if "unemployment_rate" in X.columns:
            X["recession_signal"] = (X["unemployment_rate"] > 7.0).astype(int)

        if "cpi_yoy_pct" in X.columns:
            X["high_inflation"] = (X["cpi_yoy_pct"] > 5.0).astype(int)

        return X

    def get_feature_names_out(self, input_features=None):
        return ["low_rate_env", "high_rate_env", "recession_signal", "high_inflation"]
