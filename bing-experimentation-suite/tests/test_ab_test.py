from __future__ import annotations

import numpy as np
import pytest
from experiments.ab_test import TwoSampleTTest


def test_t_test_significant_positive():
    """Verify that a clear treatment lift returns the SHIP recommendation.

    This test uses a synthetic dataset with a large enough difference
    to ensure that the p value is well below our threshold.
    """
    rng = np.random.default_rng(42)
    ctrl = rng.normal(loc=10.0, scale=1.0, size=1000)
    trtm = rng.normal(loc=11.0, scale=1.0, size=1000)
    
    res = TwoSampleTTest().run(ctrl, trtm)
    
    assert res.is_significant == True
    assert res.recommendation == "SHIP"
    assert res.absolute_difference > 0


def test_t_test_significant_negative():
    """Verify that a clear treatment drop returns the NO SHIP recommendation.

    This scenario identifies cases where the experimental group
    performs significantly worse than the baseline.
    """
    rng = np.random.default_rng(42)
    ctrl = rng.normal(loc=10.0, scale=1.0, size=1000)
    trtm = rng.normal(loc=9.0, scale=1.0, size=1000)
    
    res = TwoSampleTTest().run(ctrl, trtm)
    
    assert res.is_significant == True
    assert res.recommendation == "NO SHIP"
    assert res.absolute_difference < 0


def test_t_test_inconclusive():
    """Verify that a small or noisy difference returns INCONCLUSIVE.

    This test ensures that our statistical engine correctly identifies
    when it does not have enough evidence to support a product change.
    """
    rng = np.random.default_rng(42)
    ctrl = rng.normal(loc=10.0, scale=1.0, size=1000)
    trtm = rng.normal(loc=10.02, scale=1.0, size=1000)
    
    res = TwoSampleTTest().run(ctrl, trtm)
    
    assert res.is_significant == False
    assert res.recommendation == "INCONCLUSIVE"
