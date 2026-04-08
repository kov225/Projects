from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class MetricCorrelationAnalysis:
    """A collection of tools for designing independent metric sets.

    This class computes the full correlation matrix between engagement
    metrics. It applies a Bonferroni correction to ensure that any
    identified relationships are statistically significant and not
    the result of multiple comparisons across many metric pairs.
    """

    def __init__(self, metric_cols: list[str]):
        self.metric_cols = metric_cols

    def analyze_correlations(self, df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
        """Compute significant correlations with Bonferroni correction.

        We calculate Pearson correlations for all pairs and compute their
        respective p values. The significance threshold is adjusted
        by the number of tests to avoid discovering spurious relationships
        between noisy engagement signals.
        """
        corr_matrix = df[self.metric_cols].corr()
        n_tests = len(self.metric_cols) * (len(self.metric_cols) - 1) / 2
        adj_alpha = alpha / n_tests
        
        results = []
        for i in range(len(self.metric_cols)):
            for j in range(i + 1, len(self.metric_cols)):
                m1, m2 = self.metric_cols[i], self.metric_cols[j]
                r, p = stats.pearsonr(df[m1], df[m2])
                
                results.append({
                    "Metric 1": m1,
                    "Metric 2": m2,
                    "Correlation": r,
                    "p Value": p,
                    "Is Significant": p < adj_alpha,
                    "Redundant": abs(r) > 0.8
                })
        
        return pd.DataFrame(results)

    def find_redundant_metrics(self, df: pd.DataFrame, threshold: float = 0.8) -> list[tuple[str, str]]:
        """Identify pairs of metrics that are highly correlated.

        Metrics that move in lockstep do not add independent information
        to a composite score. Reducing redundancy helps in building a
        cleaner and more interpretable experimentation dashboard for
        product teams.
        """
        corr = df[self.metric_cols].corr()
        redundant = []
        for i in range(len(self.metric_cols)):
            for j in range(i + 1, len(self.metric_cols)):
                if abs(corr.iloc[i, j]) > threshold:
                    redundant.append((self.metric_cols[i], self.metric_cols[j]))
        return redundant
