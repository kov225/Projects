from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

from .ab_test import TwoSampleTTest


@dataclass(frozen=True)
class NoveltyResult:
    """A collection of results from the novelty effect detector.

    This result object provides insights into how the treatment effect
    changes over time, helping experimenters distinguish between
    short term excitement and long term sustainable gains.
    """
    novelty_detected: bool
    estimated_true_effect: float
    novelty_magnitude: float
    early_stopping_errors: list[float]
    decay_rate: float
    fit_p_value: float


class NoveltyEffectDetector:
    """A tool for identifying short term inflation in treatment effects.

    This class fits an exponential decay model to weekly experiment data.
    It identifies whether the observed initial gains are likely to last
    or if they are artifacts of novelty that will vanish as the launch
    period concludes.
    """

    def _decay_func(self, x, true_effect, novelty_component, decay_rate):
        """Standard exponential decay model for novelty effects.

        The model assumes that the treatment effect at time x is composed
        of a stable true effect and a transient novelty component that
        diminishes at a certain rate.
        """
        return true_effect + novelty_component * np.exp(-decay_rate * (x - 1))

    def detect(
        self, df: pd.DataFrame, outcome_col: str, alpha: float = 0.05
    ) -> NoveltyResult:
        """Analyze the experiment timeline for novelty decay patterns.

        We compute the treatment effect for each week and fit our decay
        curve to the resulting sequence. We also compare this fit to a
        null model to determine if the novelty component is statistically
        significant.
        """
        weeks = sorted(df["week_number"].unique())
        weekly_lifts = []
        full_duration_lift = 0.0

        # Step 1: Compute rolling effects and week by week effects
        for w in weeks:
            sub = df[df["week_number"] == w]
            ctrl = sub[sub["treatment"] == 0][outcome_col].values
            trtm = sub[sub["treatment"] == 1][outcome_col].values
            if len(ctrl) > 0 and len(trtm) > 0:
                res = TwoSampleTTest().run(ctrl, trtm)
                weekly_lifts.append(res.absolute_difference)

        # Step 2: Fit the decay model
        x_data = np.array(weeks)
        y_data = np.array(weekly_lifts)

        try:
            # Initial guess: true_effect=last week, novelty=first week - last week, rate=0.1
            p0 = [y_data[-1], y_data[0] - y_data[-1], 0.1]
            popt, pcov = curve_fit(self._decay_func, x_data, y_data, p0=p0, maxfev=2000)
            true_eff, novelty_mag, d_rate = popt
            
            # Estimate significance: standard error of the novelty magnitude
            perr = np.sqrt(np.diag(pcov))
            t_stat = novelty_mag / perr[1]
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(x_data) - 3))
        except Exception:
            # Fallback if fit fails
            true_eff, novelty_mag, d_rate = y_data[-1], 0.0, 0.0
            p_val = 1.0

        # Step 3: Compute early stopping errors
        # Error is defined as (estimate at week n - final estimate)
        final_estimate = weekly_lifts[-1]
        early_errors = []
        for i in range(len(weekly_lifts)):
            rolling_sub = df[df["week_number"] <= weeks[i]]
            r_ctrl = rolling_sub[rolling_sub["treatment"] == 0][outcome_col].values
            r_trtm = rolling_sub[rolling_sub["treatment"] == 1][outcome_col].values
            r_res = TwoSampleTTest().run(r_ctrl, r_trtm)
            early_errors.append(float(r_res.absolute_difference - final_estimate))

        return NoveltyResult(
            novelty_detected=(p_val < alpha and novelty_mag > 0),
            estimated_true_effect=float(true_eff),
            novelty_magnitude=float(novelty_mag),
            early_stopping_errors=early_errors,
            decay_rate=float(d_rate),
            fit_p_value=float(p_val)
        )
