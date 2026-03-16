"""
test_feature_pipeline.py

Unit tests for the custom sklearn transformers. These are the kind of tests
that a disciplined ML engineer writes to catch silent bugs in preprocessing
: the class of bug that doesn't raise an exception but quietly corrupts
your feature matrix and shows up six weeks later as a mysterious AUC drop.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from features.transformers import (
    DebtBurdenRatioTransformer,
    EmploymentStabilityTransformer,
    FICOBucketTransformer,
    MacroFeatureEngineer,
)
from features.feature_pipeline import make_feature_pipeline


@pytest.fixture
def sample_df():
    """Minimal realistic loan application dataframe."""
    return pd.DataFrame({
        "loan_amnt": [10000, 25000, 5000],
        "installment": [320.0, 800.0, 165.0],
        "annual_inc": [60000.0, 120000.0, 35000.0],
        "dti": [15.0, 28.0, 8.0],
        "fico_score": [720.0, 650.0, 580.0],
        "fico_range_low": [720.0, 650.0, 580.0],
        "fico_range_high": [724.0, 654.0, 584.0],
        "emp_length": [5.0, 10.0, 1.0],
        "unemployment_rate": [4.5, 4.5, 4.5],
        "cpi": [300.0, 300.0, 300.0],
        "fed_funds_rate": [2.0, 2.0, 2.0],
        "cpi_yoy_pct": [2.5, 2.5, 2.5],
        "unemployment_mom_change": [0.1, 0.1, 0.1],
        "grade_enc": [1, 3, 5],
    })


class TestDebtBurdenRatioTransformer:
    def test_output_column_exists(self, sample_df):
        t = DebtBurdenRatioTransformer()
        result = t.fit_transform(sample_df)
        assert "debt_burden_ratio" in result.columns

    def test_ratio_is_clipped_at_5(self, sample_df):
        # Create an extreme case: tiny income, high installment
        df = sample_df.copy()
        df.loc[0, "annual_inc"] = 100.0
        df.loc[0, "installment"] = 9999.0
        t = DebtBurdenRatioTransformer()
        result = t.fit_transform(df)
        assert result.loc[0, "debt_burden_ratio"] <= 5.0

    def test_interaction_term_computed(self, sample_df):
        t = DebtBurdenRatioTransformer()
        result = t.fit_transform(sample_df)
        assert "dti_x_unemployment" in result.columns
        # interaction = dti * unemployment_rate
        expected = sample_df["dti"].iloc[0] * sample_df["unemployment_rate"].iloc[0]
        assert abs(result["dti_x_unemployment"].iloc[0] - expected) < 1e-6

    def test_fit_stores_median_income(self, sample_df):
        t = DebtBurdenRatioTransformer()
        t.fit(sample_df)
        assert t.median_income_ == pytest.approx(sample_df["annual_inc"].median())


class TestFICOBucketTransformer:
    def test_output_column_exists(self, sample_df):
        t = FICOBucketTransformer()
        result = t.fit_transform(sample_df)
        assert "fico_bucket" in result.columns

    def test_buckets_are_ordinal_integers(self, sample_df):
        t = FICOBucketTransformer()
        result = t.fit_transform(sample_df)
        buckets = result["fico_bucket"].dropna().values
        # All bucket values should be between 0 and 6
        assert np.all(buckets >= 0)
        assert np.all(buckets <= 6)

    def test_high_fico_gets_high_bucket(self, sample_df):
        """720 should be a higher bucket than 580."""
        t = FICOBucketTransformer()
        result = t.fit_transform(sample_df)
        # Row 0 has fico 720, row 2 has fico 580
        assert result["fico_bucket"].iloc[0] > result["fico_bucket"].iloc[2]


class TestEmploymentStabilityTransformer:
    def test_output_in_unit_interval(self, sample_df):
        t = EmploymentStabilityTransformer()
        result = t.fit_transform(sample_df)
        assert "employment_stability" in result.columns
        assert result["employment_stability"].between(0, 1).all()

    def test_longer_employment_higher_stability(self, sample_df):
        """10 years employment should produce higher stability than 1 year."""
        t = EmploymentStabilityTransformer()
        result = t.fit_transform(sample_df)
        # Row 1 has emp_length=10, row 2 has emp_length=1
        assert result["employment_stability"].iloc[1] > result["employment_stability"].iloc[2]

    def test_handles_missing_emp_length(self):
        df = pd.DataFrame({"installment": [300.0], "annual_inc": [50000.0]})
        t = EmploymentStabilityTransformer()
        result = t.fit_transform(df)
        assert "employment_stability" in result.columns
        assert result["employment_stability"].iloc[0] == pytest.approx(0.0)


class TestMacroFeatureEngineer:
    def test_low_rate_flag(self, sample_df):
        df = sample_df.copy()
        df["fed_funds_rate"] = 0.25  # ZIRP
        t = MacroFeatureEngineer()
        result = t.fit_transform(df)
        assert result["low_rate_env"].iloc[0] == 1
        assert result["high_rate_env"].iloc[0] == 0

    def test_recession_signal(self, sample_df):
        df = sample_df.copy()
        df["unemployment_rate"] = 9.5
        t = MacroFeatureEngineer()
        result = t.fit_transform(df)
        assert result["recession_signal"].iloc[0] == 1

    def test_normal_conditions_no_flags(self, sample_df):
        t = MacroFeatureEngineer()
        result = t.fit_transform(sample_df)
        assert result["recession_signal"].iloc[0] == 0


class TestPipelineIntegration:
    def test_pipeline_transforms_without_error(self, sample_df):
        pipeline = make_feature_pipeline(scale_for_lr=False)
        # Should not raise
        result = pipeline.fit_transform(sample_df)
        assert result is not None

    def test_pipeline_with_scaler(self, sample_df):
        pipeline = make_feature_pipeline(scale_for_lr=True)
        result = pipeline.fit_transform(sample_df)
        assert result is not None

    def test_pipeline_is_serializable(self, sample_df, tmp_path):
        import joblib
        pipeline = make_feature_pipeline()
        pipeline.fit(sample_df)
        path = tmp_path / "pipeline.joblib"
        joblib.dump(pipeline, path)
        loaded = joblib.load(path)
        result = loaded.transform(sample_df)
        assert result is not None
