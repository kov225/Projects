import numpy as np
from scipy import stats
from experiments.ab_test import TwoSampleTTest
from experiments.novelty import NoveltyEffectDetector
import pandas as pd

def debug_ab_test():
    print("Testing T-Test...")
    rng = np.random.default_rng(42)
    ctrl = rng.normal(loc=10.0, scale=1.0, size=1000)
    trtm = rng.normal(loc=11.0, scale=1.0, size=1000)
    
    res = TwoSampleTTest().run(ctrl, trtm)
    print(f"Mean Control: {res.mean_control}")
    print(f"Mean Treatment: {res.mean_treatment}")
    print(f"P-Value: {res.p_value}")
    print(f"Is Significant: {res.is_significant}")

def debug_novelty():
    print("\nTesting Novelty Detection...")
    rng = np.random.default_rng(42)
    weeks = np.repeat(np.arange(1, 9), 1000)
    treatment = rng.binomial(1, 0.5, size=len(weeks))
    
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
    print(f"Novelty Detected: {res.novelty_detected}")
    print(f"Estimated True Effect: {res.estimated_true_effect}")
    print(f"Novelty Magnitude: {res.novelty_magnitude}")
    print(f"Decay Rate: {res.decay_rate}")
    print(f"Fit P-Value: {res.fit_p_value}")

if __name__ == "__main__":
    debug_ab_test()
    debug_novelty()
