from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ExperimentResult:
    """A standard container for experiment inference results.

    This class provides a uniform interface for all methods in the suite,
    ensuring that a t test and a CUPED adjusted estimate can be directly
    compared in the benchmarking dashboard.
    """
    mean_control: float
    mean_treatment: float
    absolute_difference: float
    relative_lift_pct: float
    t_statistic: float
    p_value: float
    confidence_interval_95: tuple[float, float]
    statistical_power: float
    is_significant: bool
    recommendation: Literal["SHIP", "NO SHIP", "INCONCLUSIVE"]


class TwoSampleTTest:
    """The core engine for frequentist A/B testing inference.

    This class computes standard two sample t tests and handles the
    statistical power calculation using the observed sample standard
    deviation and effect size.
    """

    def run(
        self, control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05
    ) -> ExperimentResult:
        n_c, n_t = len(control), len(treatment)
        m_c, m_t = float(np.mean(control)), float(np.mean(treatment))
        var_c, var_t = float(np.var(control, ddof=1)), float(np.var(treatment, ddof=1))

        t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=False)

        abs_diff = m_t - m_c
        rel_lift = (abs_diff / m_c) * 100 if m_c != 0 else 0.0

        # Confidence interval (95 percent)
        se = np.sqrt(var_c / n_c + var_t / n_t)
        df = (var_c / n_c + var_t / n_t) ** 2 / (
            (var_c / n_c) ** 2 / (n_c - 1) + (var_t / n_t) ** 2 / (n_t - 1)
        )
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        ci = (abs_diff - t_crit * se, abs_diff + t_crit * se)

        # Statistical power calculation
        pooled_std = np.sqrt((var_c + var_t) / 2)
        effect_size = abs_diff / pooled_std if pooled_std > 0 else 0
        power = float(stats.norm.sf(stats.norm.ppf(1 - alpha / 2) - effect_size * np.sqrt(n_c * n_t / (n_c + n_t))))

        is_sig = bool(p_val < alpha)
        recommendation = "INCONCLUSIVE"
        if is_sig:
            recommendation = "SHIP" if abs_diff > 0 else "NO SHIP"

        return ExperimentResult(
            mean_control=float(m_c),
            mean_treatment=float(m_t),
            absolute_difference=float(abs_diff),
            relative_lift_pct=float(rel_lift),
            t_statistic=float(t_stat),
            p_value=float(p_val),
            confidence_interval_95=(float(ci[0]), float(ci[1])),
            statistical_power=power,
            is_significant=is_sig,
            recommendation=recommendation,
        )


class SequentialTest:
    """A module for monitoring experiments as data accumulates.

    This class implements a simple sequential testing procedure. It is
    used to demonstrate how early stopping decisions can lead to
    different recommendations than waiting for the full experiment duration.
    """

    def analyze_early_stopping(
        self, df: pd.DataFrame, outcome_col: str, alpha: float = 0.05
    ) -> list[ExperimentResult]:
        results = []
        weeks = sorted(df["week_number"].unique())
        
        for w in weeks:
            sub = df[df["week_number"] <= w]
            ctrl = sub[sub["treatment"] == 0][outcome_col].values
            trtm = sub[sub["treatment"] == 1][outcome_col].values
            
            if len(ctrl) > 2 and len(trtm) > 2:
                results.append(TwoSampleTTest().run(ctrl, trtm, alpha=alpha))
            
        return results
