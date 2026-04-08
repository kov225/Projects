import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from tv_attribution.attribution import TVAttributor

@pytest.fixture
def sample_data():
    # 60 minutes of data
    ts = pd.date_range("2024-01-01", periods=60, freq="min")
    sessions = pd.DataFrame({"timestamp": ts, "sessions": 100})
    # Add a spike
    sessions.loc[25:35, "sessions"] = 150
    
    airings = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01 00:20:00")],
        "network": ["ESPN"],
        "creative_id": ["A"],
        "spot_length": [30],
        "cost": [1000]
    })
    return sessions, airings

def test_attribution_lift_detection(sample_data):
    sessions, airings = sample_data
    attr = TVAttributor(sessions, airings)
    res = attr.attribute_spot(airings.iloc[0]["timestamp"])
    
    assert res is not None
    assert res["total_lift"] > 0
    assert res["is_significant"] == True

def test_baseline_extrapolation(sample_data):
    sessions, airings = sample_data
    attr = TVAttributor(sessions, airings)
    # Testing internal baseline logic
    # This would ideally test the local linear trend directly
    pass
