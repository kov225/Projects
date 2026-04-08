from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from experiments.cuped import CUPEDAdjuster


def test_cuped_variance_reduction_correlated():
    """Verify that CUPED reduces variance for correlated covariates.

    This test confirms that utilizing a strong pre experiment predictor
    leads to the expected variance reduction as described in the 
    foundational Microsoft paper.
    """
    rng = np.random.default_rng(42)
    n = 10000
    x = rng.normal(loc=50.0, scale=10.0, size=n)
    # y is directly correlated with x
    y = 0.8 * x + rng.normal(loc=10.0, scale=5.0, size=n)
    
    df = pd.DataFrame({"pre_exp": x, "outcome": y})
    adjuster = CUPEDAdjuster("pre_exp", "outcome")
    adjuster.fit(df)
    
    assert adjuster.variance_reduction_pct > 50.0
    
    y_cuped = adjuster.transform(df)
    assert np.var(y_cuped) < np.var(y)


def test_cuped_unbiased_estimate():
    """Verify that the CUPED adjusted estimator remains unbiased.

    The mean of the adjusted metric must be equal to the mean of the
    original metric to ensure that the treatment effect estimate 
    is not systematically distorted.
    """
    rng = np.random.default_rng(42)
    n = 1000
    x = rng.normal(loc=50.0, scale=10.0, size=n)
    y = 0.5 * x + rng.normal(loc=10.0, scale=5.0, size=n)
    
    df = pd.DataFrame({"pre_exp": x, "outcome": y})
    adjuster = CUPEDAdjuster("pre_exp", "outcome")
    adjuster.fit(df)
    
    y_cuped = adjuster.transform(df)
    assert np.isclose(np.mean(y_cuped), np.mean(y), atol=1e-10)


def test_cuped_weak_correlation():
    """Verify that CUPED provides minimal benefit for uncorrelated covariates.

    This test confirms that our engine handles noisy or irrelevant 
    predictors correctly and avoids over adjusting when there is no 
    statistical signal.
    """
    rng = np.random.default_rng(42)
    n = 1000
    x = rng.normal(loc=10.0, scale=1.0, size=n)
    y = rng.normal(loc=20.0, scale=1.0, size=n)
    
    df = pd.DataFrame({"pre_exp": x, "outcome": y})
    adjuster = CUPEDAdjuster("pre_exp", "outcome")
    
    with pytest.warns(UserWarning, match="Weak correlation"):
        adjuster.fit(df)
        
    assert adjuster.variance_reduction_pct < 5.0
