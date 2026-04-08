from __future__ import annotations

import numpy as np
import pandas as pd
from .ab_test import TwoSampleTTest


class AATestRunner:
    """A diagnostic tool for validating experimentation systems.

    This class runs hundreds of A/A tests where both groups receive
    the identical control experience. It computes the empirical false
    positive rate across different statistical methods to ensure that
    each method controls the Type I error correctly.
    """

    def run_simulation(
        self, df: pd.DataFrame, outcome_col: str, iterations: int = 1000, alpha: float = 0.05
    ) -> dict[str, float]:
        """Perform one thousand A/A tests and report the false positive rate.

        We randomly split the control group into two new groups and run
        a standard T test on each iteration. For a well calibrated method,
        the percentage of tests that return a significant result should be
        approximately equal to the chosen alpha level.
        """
        # Filter only control users to ensure a true A/A scenario
        control_data = df[df["treatment"] == 0][outcome_col].values
        n = len(control_data)
        
        sig_count = 0
        p_values = []

        for _ in range(iterations):
            # Randomly shuffle and split
            indices = np.random.permutation(n)
            split = n // 2
            aa_control = control_data[indices[:split]]
            aa_treatment = control_data[indices[split:]]
            
            res = TwoSampleTTest().run(aa_control, aa_treatment, alpha=alpha)
            if res.is_significant:
                sig_count += 1
            p_values.append(res.p_value)

        false_positive_rate = sig_count / iterations
        
        return {
            "iterations": iterations,
            "alpha": alpha,
            "false_positive_rate": false_positive_rate,
            "is_calibrated": 0.03 <= false_positive_rate <= 0.07,
            "mean_p_value": np.mean(p_values)
        }
