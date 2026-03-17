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
import os

import os

def plot_performance_curves(results_df, shift_type, metric, save_path=None):
    """
    Generates longitudinal performance degradation curves for all models.
    Supports shaded error bands if 'mean' and 'std' columns are present.
    """
    df_filtered = results_df[results_df["Shift_Type"] == shift_type].copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check if we have aggregated results (mean/std)
    if f"{metric}_mean" in df_filtered.columns:
        y_col = f"{metric}_mean"
        error_col = f"{metric}_std"
        
        for model in df_filtered["Model"].unique():
            m_df = df_filtered[df_filtered["Model"] == model].sort_values("Intensity")
            ax.plot(m_df["Intensity"], m_df[y_col], marker='o', label=model, linewidth=2)
            ax.fill_between(
                m_df["Intensity"], 
                m_df[y_col] - m_df[error_col], 
                m_df[y_col] + m_df[error_col], 
                alpha=0.2
            )
    else:
        # Standard lineplot for single-seed results
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
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig

def plot_model_comparison_heatmap(results_df, shift_type, metric, save_path=None):
    """
    Constructs a cross-sectional heatmap of model performance across intensities.
    """
    df_filtered = results_df[results_df["Shift_Type"] == shift_type].copy()
    
    # Use mean if available
    val_col = f"{metric}_mean" if f"{metric}_mean" in df_filtered.columns else metric
    
    pivot_df = df_filtered.pivot(index="Model", columns="Intensity", values=val_col)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cmap = "YlOrRd" if metric == "Brier_Score" else "YlGnBu"
        
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
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig

def auto_save_all_figures(results_df, figures_dir):
    """
    Iterates through all shift types and metrics to save standard plots.
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    shift_types = [s for s in results_df["Shift_Type"].unique() if s != "Baseline"]
    metrics = ["Accuracy", "F1_Score", "ROC_AUC", "Brier_Score"]
    
    for shift in shift_types:
        shift_name = shift.lower().replace(" ", "_")
        for metric in metrics:
            # Curve plot
            plot_performance_curves(
                results_df, shift, metric, 
                save_path=os.path.join(figures_dir, f"curve_{shift_name}_{metric.lower()}.png")
            )
            plt.close()
            
            # Heatmap
            plot_model_comparison_heatmap(
                results_df, shift, metric,
                save_path=os.path.join(figures_dir, f"heatmap_{shift_name}_{metric.lower()}.png")
            )
            plt.close()

