from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from metrics.composite import CompositeEngagementMetric
from metrics.sensitivity import MetricSensitivity


def test_composite_weights_sum_to_one():
    """Verify that PCA derived weights are correctly normalized.

    The weights for the individual engagement metrics must sum to unity
    to ensure that the final composite score is correctly scaled 
    for comparison across different experiments.
    """
    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame({
        "m1": rng.normal(size=n),
        "m2": rng.normal(size=n),
        "m3": rng.normal(size=n)
    })
    
    metric = CompositeEngagementMetric(["m1", "m2", "m3"])
    metric.fit(df)
    
    weights = metric.get_weights_summary()
    assert np.isclose(sum(weights.values()), 1.0, atol=1e-10)


def test_sensitivity_monotonicity():
    """Verify that metric detection power increases with sample size.

    Statistical theory guarantees that larger samples resolve smaller
    effects. This test confirms that our MetricSensitivity module
    correctly reflects this fundamental property of experimentation.
    """
    rng = np.random.default_rng(42)
    values = rng.normal(loc=10.0, scale=2.0, size=100000)
    sensitivity = MetricSensitivity()
    
    mde_small = sensitivity.compute_mde(values, n=1000)
    mde_medium = sensitivity.compute_mde(values, n=5000)
    mde_large = sensitivity.compute_mde(values, n=20000)
    
    # Larger n must produce smaller MDE
    assert mde_small > mde_medium > mde_large
