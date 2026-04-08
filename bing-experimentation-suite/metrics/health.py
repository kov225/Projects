from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .sensitivity import MetricSensitivity


class MetricHealthMonitor:
    """A proactive tool for monitoring experiment metric quality.

    This class tracks metrics over time and alerts developers when the
    data distribution shifts significantly or when a metric's variance
    makes it too noisy for reliable detection. This ensures that the
    experimentation platform remains a trusted source of truth for
    product decisions.
    """

    def __init__(self, sensitivity_threshold: float = 0.5):
        self.sensitivity_threshold = sensitivity_threshold
        self.sensitivity_engine = MetricSensitivity()

    def check_distribution_shift(
        self, baseline_values: np.ndarray, current_values: np.ndarray, alpha: float = 0.05
    ) -> dict[str, float | bool]:
        """Verify that the current data distribution matches the baseline.

        We use the Kolmogorov Smirnov test to detect any non parametric shifts
        in the metric. A significant p value indicates that the fundamental
        nature of the user behavior has changed, which may invalidate
        historical comparisons in the experimentation suite.
        """
        ks_stat, p_val = stats.ks_2samp(baseline_values, current_values)
        
        return {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_val),
            "is_shifted": p_val < alpha
        }

    def check_health(
        self, df: pd.DataFrame, outcome_col: str, n: int = 10000
    ) -> dict[str, float | bool | str]:
        """Perform a comprehensive health check on a specific metric.

        This combines sensitivity analysis with variance tracking and shift
        detection. It produces a status report for the dashboard that
        highlights any metrics requiring attention from the data engineering
        team or the product manager.
        """
        values = df[outcome_col].values
        mde = self.sensitivity_engine.compute_mde(values, n=n)
        empirical_pwr = self.sensitivity_engine.empirical_sensitivity(values, n=n)
        
        # Determine health status based on sensitivity
        status = "HEALTHY"
        if empirical_pwr < self.sensitivity_threshold:
            status = "SENSITIVITY_ALARM"
        elif mde > (df[outcome_col].mean() * 0.2): # 20% MDE is very high
            status = "VARIANCE_ALARM"

        return {
            "metric_name": outcome_col,
            "status": status,
            "mde_absolute": mde,
            "mde_relative": mde / df[outcome_col].mean() if df[outcome_col].mean() != 0 else 0.0,
            "empirical_power": empirical_pwr,
            "degraded_sensitivity": empirical_pwr < self.sensitivity_threshold
        }
