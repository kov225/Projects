"""
Interpretability Layer for Dataset Shift Analysis

This module provides tools to understand WHY models respond differently to shift.
It includes feature importance analysis and confidence distribution plotting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

def get_feature_importance(model, X, y, feature_names=None):
    """
    Calculates feature importance for a given model and dataset.
    Uses model-native importance if available, otherwise falls back to 
    permutation importance.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        # Fallback to permutation importance for models like NB or SVM RBF
        r = permutation_importance(model, X, y, n_repeats=5, random_state=42)
        importances = r.importances_mean
        
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]
        
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)

def plot_confidence_distribution(model, X_baseline, X_shifted, model_name="Model"):
    """
    Plots the distribution of prediction confidence (max probability) 
    comparing baseline vs shifted data.
    """
    if not hasattr(model, "predict_proba"):
        # Fallback for models without predict_proba using decision function if available
        if hasattr(model, "decision_function"):
            d_base = model.decision_function(X_baseline)
            d_shift = model.decision_function(X_shifted)
            # Sigmoid scaling
            p_base = 1 / (1 + np.exp(-d_base))
            p_shift = 1 / (1 + np.exp(-d_shift))
            # Binary confidence: distance from 0.5
            conf_base = np.abs(p_base - 0.5) * 2
            conf_shift = np.abs(p_shift - 0.5) * 2
        else:
            return None
    else:
        probs_base = model.predict_proba(X_baseline)
        probs_shift = model.predict_proba(X_shifted)
        conf_base = np.max(probs_base, axis=1)
        conf_shift = np.max(probs_shift, axis=1)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(conf_base, label="Baseline (Clean)", shade=True, ax=ax)
    sns.kdeplot(conf_shift, label="Shifted (Aggregated)", shade=True, ax=ax)
    
    ax.set_title(f"Confidence Distribution Shift: {model_name}")
    ax.set_xlabel("Confidence (Max Probability)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    
    return fig

def plot_importance_shift(importance_base, importance_shifted, model_name="Model"):
    """
    Compares top feature importances before and after shift.
    """
    df = pd.DataFrame({
        'Feature': importance_base.index,
        'Baseline': importance_base.values,
        'Shifted': importance_shifted.reindex(importance_base.index).values
    }).melt(id_vars='Feature', var_name='Scenario', value_name='Importance')
    
    # Take top 10 for readability
    top_features = importance_base.index[:10]
    df = df[df['Feature'].isin(top_features)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x='Importance', y='Feature', hue='Scenario', ax=ax)
    ax.set_title(f"Feature Importance Shift: {model_name}")
    plt.tight_layout()
    
    return fig
