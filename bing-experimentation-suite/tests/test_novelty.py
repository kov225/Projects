from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from experiments.novelty import NoveltyEffectDetector


def test_novelty_detection_positive():
    """Verify that the engine detects a simulated novelty decay.

    We inject a decaying treatment effect into the dataset and confirm
    that the NoveltyEffectDetector correctly identifies the pattern
    and provides a reasonable estimate of the true long term effect.
    """
    rng = np.random.default_rng(42)
    weeks = np.repeat(np.arange(1, 9), 1000)
    treatment = rng.binomial(1, 0.5, size=len(weeks))
    
    # Baseline 10.0
    # Novelty 1.0 (week 1), decaying to 0.1 (week 8)
    true_lift = 0.1
    novelty_lift = 1.0 * np.exp(-0.5 * (weeks - 1))
    
    outcomes = 10.0 + treatment * (true_lift + novelty_lift) + rng.normal(scale=0.1, size=len(weeks))
    
    df = pd.DataFrame({
        "week_number": weeks,
        "treatment": treatment,
        "outcome": outcomes
    })
    
    detector = NoveltyEffectDetector()
    res = detector.detect(df, "outcome")
    
    assert res.novelty_detected == True
    assert res.novelty_magnitude > 0.5
    assert np.isclose(res.estimated_true_effect, true_lift, atol=0.05)


def test_no_novelty_detection():
    """Verify that a stable treatment effect does not trigger an alarm.

    In experiments where the lift is constant across all weeks, the
    detector should correctly conclude that no novelty effect is 
    present, ensuring a low false positive rate for anomaly detection.
    """
    rng = np.random.default_rng(42)
    weeks = np.repeat(np.arange(1, 9), 1000)
    treatment = rng.binomial(1, 0.5, size=len(weeks))
    
    # Constant lift
    true_lift = 0.5
    outcomes = 10.0 + treatment * true_lift + rng.normal(scale=0.1, size=len(weeks))
    
    df = pd.DataFrame({
        "week_number": weeks,
        "treatment": treatment,
        "outcome": outcomes
    })
    
    detector = NoveltyEffectDetector()
    res = detector.detect(df, "outcome")
    
    assert res.novelty_detected == False
    assert abs(res.novelty_magnitude) < 0.1
