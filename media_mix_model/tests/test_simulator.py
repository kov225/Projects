import pytest
import numpy as np
from media_mix_model.simulator import MMMSimulator
from media_mix_model.optimizer import MMMOptimizer

def test_geometric_adstock_bounds():
    sim = MMMSimulator()
    x = np.array([100, 0, 0, 0])
    alpha = 0.5
    adstocked = sim.geometric_adstock(x, alpha)
    
    # Expected: 100, 50, 25, 12.5
    assert adstocked[0] == 100
    assert adstocked[1] == 50
    assert adstocked[2] == 25
    assert adstocked[3] == 12.5
    assert np.all(adstocked >= 0)

def test_hill_saturation_asymptote():
    sim = MMMSimulator()
    # At very high x, Hill function should approach 1.0
    val = sim.hill_saturation(1_000_000, 1000, 3)
    assert val > 0.99
    
    # At x=0, should be 0
    assert sim.hill_saturation(0, 1000, 3) == 0

def test_simulator_runs_and_saves():
    sim = MMMSimulator(n_weeks=10)
    df = sim.run()
    assert len(df) == 10
    assert "conversions" in df.columns
    assert "linear_tv_spend" in df.columns
