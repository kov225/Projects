from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from experiments.aa_test import AATestRunner


def test_aa_validation():
    """Verify that the experimentation system controls Type I error.

    We run a series of A/A tests and confirm that the empirical false
    positive rate is close to the expected five percent. This is the
    most important validation for any experimentation platform.
    """
    rng = np.random.default_rng(42)
    n_rows = 10000
    df = pd.DataFrame({
        "treatment": [0] * n_rows, # All control
        "outcome": rng.normal(size=n_rows)
    })
    
    runner = AATestRunner()
    # We use a smaller number for unit testing speed (100 instead of 1000)
    res = runner.run_simulation(df, "outcome", iterations=100)
    
    # 5 percent alpha should yield ~5 positives per 100
    # Allow reasonable statistical noise for a test
    assert 0.01 <= res["false_positive_rate"] <= 0.10
