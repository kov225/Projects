from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_telemetry(
    num_rows: int = 500_000,
    num_weeks: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic search engine telemetry dataset with novelty effects.

    This function simulates user interactions across two experiment arms. It
    includes a pre experiment engagement covariate and a decaying novelty
    effect that mimics real world online experimentation scenarios at scale.
    """
    rng = np.random.default_rng(seed)

    # Basic user distribution and assignment
    user_ids = [f"u_{i:06d}" for i in range(100_000)]
    selected_users = rng.choice(user_ids, size=num_rows)
    treatment = rng.binomial(1, 0.5, size=num_rows)

    segments = ["power_user", "casual_user", "new_user"]
    segment_probs = [0.2, 0.5, 0.3]
    user_segments = rng.choice(segments, size=num_rows, p=segment_probs)

    # Pre experiment engagement covariate for CUPED
    pre_exp = rng.gamma(shape=2.0, scale=20.0, size=num_rows)

    # Week logic and sessions
    start_date = datetime(2025, 1, 1)
    week_numbers = rng.integers(1, num_weeks + 1, size=num_rows)
    session_dates = [
        start_date + timedelta(days=int(w * 7) + int(rng.integers(0, 7)))
        for w in week_numbers
    ]

    # Treatment effect simulation with novelty decay
    # True long term lift is 2 percent for CTR and 3 percent for dwell time.
    # Novelty adds extra lift in early weeks that decays exponentially.
    true_ctr_lift = 0.02
    true_dwell_lift = 0.03
    novelty_amplitude = 0.05
    decay_rate = 0.5

    novelty_multiplier = novelty_amplitude * np.exp(-decay_rate * (week_numbers - 1))
    effective_ctr_lift = true_ctr_lift + novelty_multiplier
    effective_dwell_lift = true_dwell_lift + novelty_multiplier

    # Baseline metrics (CTR mean ~0.15, dwell mean ~45s)
    base_ctr = 0.15
    ctr_probs = base_ctr * (1 + treatment * effective_ctr_lift)
    clicked = rng.binomial(1, ctr_probs)

    # Dwell time follows a log normal distribution
    # We use pre experiment engagement to correlate the outcome
    # mu is around 3.8 for ~45s mean
    mu_base = 3.8 + 0.01 * pre_exp
    mu_treatment = mu_base + treatment * np.log(1 + effective_dwell_lift)
    dwell_times = rng.lognormal(mean=mu_treatment, sigma=0.5)

    # Other binary metrics
    reformulated = rng.binomial(1, 0.1 * (1 - 0.05 * treatment))
    abandoned = rng.binomial(1, 0.2 * (1 - 0.03 * treatment))

    df = pd.DataFrame({
        "query_id": [f"q_{i:07d}" for i in range(num_rows)],
        "user_id": selected_users,
        "treatment": treatment,
        "session_date": session_dates,
        "clicked": clicked,
        "dwell_time_seconds": dwell_times,
        "reformulated": reformulated,
        "abandoned": abandoned,
        "pre_experiment_engagement": pre_exp,
        "week_number": week_numbers,
        "user_segment": user_segments,
    })

    return df


if __name__ == "__main__":
    print("Generating 500,000 rows of telemetry data...")
    data = generate_synthetic_telemetry()
    output_path = "data/raw/synthetic_telemetry.parquet"
    import os
    os.makedirs("data/raw", exist_ok=True)
    data.to_parquet(output_path, index=False)
    print(f"Dataset saved to {output_path}")
