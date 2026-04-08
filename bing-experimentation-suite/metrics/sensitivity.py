from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class MetricSensitivity:
    """A tool for evaluating the detection power of a custom metric.

    This class computes the minimum detectable effect at various sample
    sizes and power levels. It also uses bootstrap simulations to measure
    how frequently the metric correctly identifies an injected treatment
    signal under realistic experiment conditions.
    """

    def compute_mde(
        self, values: np.ndarray, n: int, alpha: float = 0.05, power: float = 0.80
    ) -> float:
        """Calculate the absolute MDE for a given sample size.

        The formula uses the standard deviation of the metric and the
        corresponding Z scores for the significance level and target
        power. This defines the smallest change that the experiment is
        mathematically capable of resolving.
        """
        std = np.std(values, ddof=1)
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        # Absolute MDE for a two sample test with equal sample sizes
        mde = (z_alpha + z_beta) * np.sqrt(2 * (std**2) / n)
        return float(mde)

    def empirical_sensitivity(
        self, values: np.ndarray, n: int, lift: float = 0.02, iterations: int = 500
    ) -> float:
        """Measure detection frequency through bootstrap simulation.

        We take a sample of the baseline metric and inject a known
        percentage lift. By running hundreds of simulated t tests, we
        measure the proportion of iterations where the metric captures
        the signal, providing an empirical estimate of its power.
        """
        success_count = 0
        mu = np.mean(values)
        
        for _ in range(iterations):
            ctrl = np.random.choice(values, size=n, replace=True)
            trtm = np.random.choice(values, size=n, replace=True) * (1 + lift)
            
            # Simple t test
            _, p_val = stats.ttest_ind(trtm, ctrl, equal_var=False)
            if p_val < 0.05:
                success_count += 1
                
        return success_count / iterations

    def variance_over_time(self, df: pd.DataFrame, outcome_col: str) -> pd.Series:
        """Compute the historical variance of the metric by week.

        This tracks whether the metric is becoming more or less noisy
        over time, which is critical for maintaining stable experiment
        sensitivity in the presence of seasonal shifts or user changes.
        """
        return df.groupby("week_number")[outcome_col].var()
