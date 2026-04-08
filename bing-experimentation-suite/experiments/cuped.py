from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class CUPEDAdjuster:
    """A frequentist variance reduction tool for online experiments.

    This class implements the Controlled Experiments Using Pre Experiment
    Data methodology as described by Microsoft researchers in 2013. The
    core idea is that utilizing a predictive covariate can significantly
    reduce the standard error of the treatment effect estimate, leading
    to faster decision making.
    """

    def __init__(self, covariate_col: str, outcome_col: str):
        self.covariate_col = covariate_col
        self.outcome_col = outcome_col
        self.theta = 0.0
        self.mean_x = 0.0
        self.variance_reduction_pct = 0.0

    def fit(self, df: pd.DataFrame) -> CUPEDAdjuster:
        """Estimate the optimal theta scaling factor for the covariate.

        We compute theta using the global covariance and variance of the
        pre experiment metric to ensure that the adjusted estimator
        remains unbiased across both experimental arms.
        """
        x = df[self.covariate_col].values
        y = df[self.outcome_col].values

        cov_yx = np.cov(y, x)[0, 1]
        var_x = np.var(x, ddof=1)

        if var_x > 0:
            self.theta = cov_yx / var_x
        else:
            self.theta = 0.0

        self.mean_x = np.mean(x)

        # Compute theoretical variance reduction: rho squared
        rho = np.corrcoef(y, x)[0, 1]
        if abs(rho) < 0.1:
            warnings.warn(
                f"Weak correlation ({rho:.4f}) between {self.covariate_col} "
                f"and {self.outcome_col}. CUPED may provide minimal benefit."
            )
            
        self.variance_reduction_pct = (rho ** 2) * 100
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Apply the linear transformation to the outcome variable.

        The adjusted value is the original outcome minus the product
        of theta and the centered covariate. This maintains the same
        expected value as the original metric while narrowing the
        confidence interval.
        """
        x = df[self.covariate_col].values
        y = df[self.outcome_col].values
        
        y_cuped = y - self.theta * (x - self.mean_x)
        return pd.Series(y_cuped, index=df.index, name=f"{self.outcome_col}_cuped")
