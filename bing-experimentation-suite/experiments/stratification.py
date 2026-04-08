from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .ab_test import ExperimentResult, TwoSampleTTest


class PostStratificationEstimator:
    """A tool for reducing variance by adjusting for unbalanced segments.

    This class computes a weighted average of treatment effects across
    different user strata. It corrects for any chance imbalance in group
    assignment and ensures that the final estimate reflects the official
    population structure rather than the sample distribution.
    """

    def run(
        self, df: pd.DataFrame, outcome_col: str, stratum_col: str, alpha: float = 0.05
    ) -> ExperimentResult:
        """Estimate the treatment effect using a post stratified weighted average.

        We first compute the mean and variance within each stratum for
        both experimental arms. Then, we aggregate these into a single
        population level estimate using the weights of each stratum in
        the total sample.
        """
        strata = df[stratum_col].unique()
        pop_weights = df[stratum_col].value_counts(normalize=True).to_dict()

        m_c_tot, m_t_tot = 0.0, 0.0
        var_c_tot, var_t_tot = 0.0, 0.0
        n_c, n_t = len(df[df["treatment"] == 0]), len(df[df["treatment"] == 1])

        for s in strata:
            weight = pop_weights[s]
            sub_c = df[(df["treatment"] == 0) & (df[stratum_col] == s)][outcome_col].values
            sub_t = df[(df["treatment"] == 1) & (df[stratum_col] == s)][outcome_col].values

            if len(sub_c) < 2 or len(sub_t) < 2:
                continue

            m_c_tot += weight * np.mean(sub_c)
            m_t_tot += weight * np.mean(sub_t)
            var_c_tot += (weight**2) * np.var(sub_c, ddof=1) / len(sub_c)
            var_t_tot += (weight**2) * np.var(sub_t, ddof=1) / len(sub_t)

        abs_diff = m_t_tot - m_c_tot
        # We leverage the standard T test runner for consistent result format
        res = TwoSampleTTest().run(
            df[df["treatment"] == 0][outcome_col].values,
            df[df["treatment"] == 1][outcome_col].values,
            alpha=alpha
        )

        # Update the result with stratified values
        # The standard error of the stratified estimator is sqrt of sum of weighted variances
        se_strat = np.sqrt(var_c_tot + var_t_tot)
        t_stat_strat = abs_diff / se_strat if se_strat > 0 else 0.0
        p_val_strat = 2 * (1 - stats.norm.cdf(abs(t_stat_strat)))

        return ExperimentResult(
            mean_control=float(m_c_tot),
            mean_treatment=float(m_t_tot),
            absolute_difference=float(abs_diff),
            relative_lift_pct=(abs_diff / m_c_tot) * 100 if m_c_tot != 0 else 0.0,
            t_statistic=float(t_stat_strat),
            p_value=float(p_val_strat),
            confidence_interval_95=(abs_diff - 1.96 * se_strat, abs_diff + 1.96 * se_strat),
            statistical_power=res.statistical_power,
            is_significant=p_val_strat < alpha,
            recommendation="SHIP" if p_val_strat < alpha and abs_diff > 0 else ("NO SHIP" if p_val_strat < alpha else "INCONCLUSIVE"),
        )


class RegressionAdjustedEstimator:
    """An advanced estimator that controls for multiple covariates.

    We use an Ordinary Least Squares model to regress the outcome on the
    treatment indicator and any available pre experiment features. This
    technique generalizes CUPED and post stratification into a single
    statistical framework for precise inference in complex experiments.
    """

    def run(
        self, df: pd.DataFrame, outcome_col: str, covariate_cols: list[str], alpha: float = 0.05
    ) -> ExperimentResult:
        """Estimate the treatment effect through a linear regression coefficient.

        The model includes a treatment indicator and a set of covariates.
        The coefficient of the indicator is the adjusted treatment effect,
        efficiently utilizing the signal from pre experiment data to narrow
        the error margin.
        """
        x = df[["treatment"] + covariate_cols].copy()
        # Add dummies for non numerical columns like user segment
        x = pd.get_dummies(x, drop_first=True)
        x = sm.add_constant(x)
        y = df[outcome_col].values

        model = sm.OLS(y, x).fit()
        coeff = model.params["treatment"]
        se = model.bse["treatment"]
        p_val = model.pvalues["treatment"]
        ci = model.conf_int().loc["treatment"].values

        # Baseline means for result consistency
        m_c = df[df["treatment"] == 0][outcome_col].mean()
        m_t = df[df["treatment"] == 1][outcome_col].mean()

        return ExperimentResult(
            mean_control=float(m_c),
            mean_treatment=float(m_t),
            absolute_difference=float(coeff),
            relative_lift_pct=(coeff / m_c) * 100 if m_c != 0 else 0.0,
            t_statistic=float(model.tvalues["treatment"]),
            p_value=float(p_val),
            confidence_interval_95=(float(ci[0]), float(ci[1])),
            statistical_power=0.0, # Placeholder for simplicity
            is_significant=p_val < alpha,
            recommendation="SHIP" if p_val < alpha and coeff > 0 else ("NO SHIP" if p_val < alpha else "INCONCLUSIVE"),
        )
