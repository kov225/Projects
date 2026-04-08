"""
Robustness Scoring Module

This module calculates a unified robustness score for machine learning 
algorithms based on their performance stability under varying dataset 
shift intensities. It uses the rate of decay in ROC AUC to rank models 
from most resilient to most sensitive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RobustnessScorer:
    """A collection of tools for ranking model stability.

    This class computes a single robustness score for each experiment. 
    The core idea is to measure the area under the performance decay 
    curve (AUC of AUC), where a higher value indicates that a model 
    maintains its accuracy for longer as the environment degrades.
    """

    def calculate_robustness(
        self, intensities: list[float], performance_metrics: list[float]
    ) -> float:
        """Compute the robustness score using numerical integration.

        We take a sequence of shift intensities and the corresponding 
        model performance scores. The robustness is defined as the 
        integral of the performance curve normalized by the baseline 
        performance at zero intensity.
        """
        if not intensities or not performance_metrics:
            return 0.0
            
        base_perf = performance_metrics[0]
        if base_perf == 0:
            return 0.0
            
        # Normalise performance by baseline
        norm_perf = [p / base_perf for p in performance_metrics]
        
        # Area under the curve using the trapezoidal rule
        score = np.trapz(norm_perf, x=intensities)
        
        # Final score is normalized such that a perfectly robust 
        # model (no decay) with max intensity 1.0 gets a 100.
        max_intensity = max(intensities)
        return float((score / max_intensity) * 100)

    def rank_models(self, experiment_results: pd.DataFrame) -> pd.DataFrame:
        """Create a ranking of models based on their calculated robustness.

        This summarizes the longitudinal study by identifying which 
        architectures are best suited for production deployment in 
        shifting environments.
        """
        rankings = []
        model_names = experiment_results["model_name"].unique()
        
        for name in model_names:
            subset = experiment_results[experiment_results["model_name"] == name].sort_values("intensity")
            score = self.calculate_robustness(
                subset["intensity"].tolist(), subset["auc_roc"].tolist()
            )
            rankings.append({"Model": name, "Robustness Score": score})
            
        return pd.DataFrame(rankings).sort_values("Robustness Score", ascending=False)
