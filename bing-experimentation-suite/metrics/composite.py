from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class CompositeEngagementMetric:
    """A scalar value that combines multiple engagement signals.

    This class learns a robust weighing scheme for different metrics by
    applying principal component analysis to their correlation matrix.
    This principled approach avoids the arbitrariness of hand tuned
    weights and identifies the direction of maximum common variance in
    user engagement behaviors.
    """

    def __init__(self, metric_cols: list[str]):
        self.metric_cols = metric_cols
        self.pca = PCA(n_components=1)
        self.scaler = StandardScaler()
        self.weights = {}

    def fit(self, df: pd.DataFrame) -> CompositeEngagementMetric:
        """Identify the principal vector of engagement from user data.

        We first standardize each metric to unit variance and zero mean
        so that differences in scale do not bias the weights. Then, we
        extract the first principal component and use its absolute
        loadings as our normalized weights.
        """
        x = df[self.metric_cols].values
        x_scaled = self.scaler.fit_transform(x)
        self.pca.fit(x_scaled)

        # Normalise absolute loadings to sum to one
        loadings = np.abs(self.pca.components_[0])
        total_loading = np.sum(loadings)
        
        for name, loading in zip(self.metric_cols, loadings):
            self.weights[name] = float(loading / total_loading)

        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Compute the weighted composite score for each row in the dataset.

        The final score is a sum of the individual metrics, each scaled
        by its learned weight. This produces a single engagement number
        that can be used as the primary outcome for experimentation.
        """
        composite_score = np.zeros(len(df))
        for col, weight in self.weights.items():
            # Standardize column for fair combination
            col_data = df[col].values
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std > 0:
                standardized_col = (col_data - col_mean) / col_std
                composite_score += weight * standardized_col
        
        return pd.Series(composite_score, index=df.index, name="composite_engagement_score")

    def get_contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the contribution of each metric to the total score.

        This helps developers understand which specific user behaviors
        are driving changes in the overall composite metric.
        """
        contributions = {}
        for col, weight in self.weights.items():
            contributions[col] = df[col] * weight
        return pd.DataFrame(contributions)
        
    def get_weights_summary(self) -> dict[str, float]:
        """Return the learned weights for each component metric.

        This summary is used in the dashboard to explain how the
        composite engagement score is structured.
        """
        return self.weights
