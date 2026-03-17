"""
Data Visualization and Result Interpretation Module

This module provides high-level plotting utilities for visualizing experimental 
benchmarks. It leverages Matplotlib and Seaborn to generate degradation curves 
and comparative heatmaps, facilitating a granular analysis of model robustness 
under various dataset shift regimes.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_performance_curves(results_df, shift_type, metric):
    """
    Generates longitudinal performance degradation curves for all models.

    This function filters results for a specific shift regime and plots the 
    specified performance metric as a function of shift intensity. It provides 
    a visual baseline for identifying the cliff-points where models start to 
    diverge or fail.

    Args:
        results_df (pd.DataFrame): Consolidated experimental results.
        shift_type (str): The specific shift regime to visualize.
        metric (str): The primary performance metric for the y-axis.

    Returns:
        matplotlib.figure.Figure: The generated visualization object.
    """
    df_filtered = results_df[results_df["Shift_Type"] == shift_type].copy()
    
    # Standardized aspect ratio for consistency across reporting platforms
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Seaborn lineplot handles model grouping and color palette management
    sns.lineplot(
        data=df_filtered, 
        x="Intensity", 
        y=metric, 
        hue="Model", 
        marker="o",
        ax=ax,
        linewidth=2.5,
        markersize=8
    )
    
    ax.set_title(f"{metric} Decay Profile: {shift_type}", fontsize=14, pad=15)
    ax.set_xlabel("Shift Intensity (Magnitude)", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Legend orientation is anchored outside the axis to prevent occlusion of data points
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    return fig

def plot_model_comparison_heatmap(results_df, shift_type, metric):
    """
    Constructs a cross-sectional heatmap of model performance across intensities.

    By pivoting the result matrix, this function provides a comparative overview 
    of all models simultaneously. It is particularly effective for identifying 
    global trends and anomalous model behaviors at extreme shift intensities.

    Args:
        results_df (pd.DataFrame): Consolidated experimental results.
        shift_type (str): The shift regime to analyze.
        metric (str): The performance metric to populate the heatmap cells.

    Returns:
        matplotlib.figure.Figure: The generated visualization object.
    """
    df_filtered = results_df[results_df["Shift_Type"] == shift_type].copy()
    
    # Data transformation: Matrix formation (Model x Intensity)
    pivot_df = df_filtered.pivot(index="Model", columns="Intensity", values=metric)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sequential colormaps are selected based on the metric's optimization direction
    if metric == "Brier_Score":
        # Red spectrum for loss-based metrics (lower is better)
        cmap = "YlOrRd"
    else:
        # Green-Blue spectrum for accuracy-based metrics (higher is better)
        cmap = "YlGnBu"
        
    sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt=".3f", 
        cmap=cmap, 
        ax=ax,
        cbar_kws={'label': metric},
        linewidths=0.5
    )
    
    ax.set_title(f"Model Comparative Matrix: {metric} ({shift_type})", fontsize=14, pad=15)
    plt.tight_layout()
    
    return fig
