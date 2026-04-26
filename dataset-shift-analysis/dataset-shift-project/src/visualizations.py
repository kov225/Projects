"""
Visualization Module  Milestone 2

Produces all publication-quality figures used by the Streamlit dashboard
and the main experiment runner.  Every function returns a
matplotlib.figure.Figure so that callers can either display it interactively
(Streamlit st.pyplot) or save it to disk.

Figure catalogue:
  plot_performance_curves()        Accuracy/F1/ROC vs. intensity per model.
  plot_model_heatmap()             Cross-model metric matrix at all intensities.
  plot_robustness_ranking()        Horizontal bar chart of robustness scores.
  plot_calibration_decay()         Brier score as a function of shift intensity.
  plot_ks_psi_curves()             Distribution shift metrics vs. intensity.
  plot_confidence_band()           Single-model metric with bootstrap CI band.
  plot_relative_drop_heatmap()     Percent accuracy drop across all conditions.
  save_figure()                    Utility: persist any Figure to disk.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from utils import get_figures_path

# Shared aesthetic configuration applied globally at module import time
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor":  "#0e1117",
    "axes.facecolor":    "#0e1117",
    "axes.edgecolor":    "#3a3f4b",
    "axes.labelcolor":   "#d0d0d0",
    "text.color":        "#d0d0d0",
    "xtick.color":       "#d0d0d0",
    "ytick.color":       "#d0d0d0",
    "grid.color":        "#2a2d35",
    "legend.facecolor":  "#1c1f26",
    "legend.edgecolor":  "#3a3f4b",
    "figure.dpi":        120,
})

_PALETTE = sns.color_palette("tab10", 12)
_BAD_METRIC  = {"Brier_Score"}   # lower is better
_GOOD_METRIC = {"Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC",
                 "Robustness_Score"}


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _prepare_with_baseline(df: pd.DataFrame, shift_type: str) -> pd.DataFrame:
    """
    Prepends the baseline rows (relabeled to the active shift type) so that
    all curves start at intensity = 0, giving a continuous trajectory.
    """
    bl = df[df["Shift_Type"] == "Baseline"].copy()
    bl["Shift_Type"] = shift_type
    active = df[df["Shift_Type"] == shift_type].copy()

    if active.empty:
        return bl
    if 0.0 not in active["Intensity"].values:
        return pd.concat([bl, active], ignore_index=True)
    return active


def _apply_dark_spine(ax):
    """Applies consistent dark-mode spine styling."""
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3f4b")


# ---------------------------------------------------------------------------
# 1. Performance Curves (per model, per shift)
# ---------------------------------------------------------------------------

def plot_performance_curves(
    results_df: pd.DataFrame,
    shift_type: str,
    metric: str = "Accuracy",
    show_ci: bool = True,
) -> plt.Figure:
    """
    Plots metric vs. shift intensity for all models as line curves.

    When show_ci is True and the matching CI columns exist, a shaded
    confidence band is drawn around each model's curve.

    Args:
        results_df: Consolidated experiment result DataFrame.
        shift_type: Name of the shift regime to visualize.
        metric:     Column name of the metric to plot on the y-axis.
        show_ci:    Whether to draw 95% bootstrap CI bands.

    Returns:
        matplotlib Figure.
    """
    if metric not in results_df.columns:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.text(0.5, 0.5, f"{metric} column not found.\nPlease run the M2 pipeline.",
                ha="center", va="center", transform=ax.transAxes, color="#e0e0e0")
        _apply_dark_spine(ax)
        return fig

    df = _prepare_with_baseline(results_df, shift_type)
    df = df.sort_values("Intensity")
    models = df["Model"].unique()

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, model in enumerate(models):
        mdf = df[df["Model"] == model].sort_values("Intensity")
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(mdf["Intensity"], mdf[metric], marker="o", linewidth=2.2,
                markersize=6, label=model, color=color)

        lo_col = f"{metric}_Lower_CI"
        hi_col = f"{metric}_Upper_CI"
        if show_ci and lo_col in mdf.columns and hi_col in mdf.columns:
            ax.fill_between(mdf["Intensity"], mdf[lo_col], mdf[hi_col],
                            alpha=0.12, color=color)

    direction = "lower is better" if metric in _BAD_METRIC else "higher is better"
    ax.set_title(f"{metric} vs. Shift Intensity  |  {shift_type}\n({direction})",
                 fontsize=13, pad=14, color="#e0e0e0")
    ax.set_xlabel("Shift Intensity", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.legend(loc="best", fontsize=8, framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.35)
    _apply_dark_spine(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Model Heatmap
# ---------------------------------------------------------------------------

def plot_model_heatmap(
    results_df: pd.DataFrame,
    shift_type: str,
    metric: str = "Accuracy",
) -> plt.Figure:
    """
    Draws a cross-sectional heatmap (Model x Intensity) for the specified
    metric under a given shift regime.

    Args:
        results_df: Consolidated experiment result DataFrame.
        shift_type: Shift regime to visualize.
        metric:     Metric to populate the heatmap cells.

    Returns:
        matplotlib Figure.
    """
    if metric not in results_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, f"{metric} column not found.\nPlease run the M2 pipeline.",
                ha="center", va="center", transform=ax.transAxes, color="#e0e0e0")
        _apply_dark_spine(ax)
        return fig

    df = _prepare_with_baseline(results_df, shift_type)
    pivot = df.pivot_table(index="Model", columns="Intensity",
                           values=metric, aggfunc="mean")
    pivot.columns = [f"{c:.2f}" for c in pivot.columns]

    cmap = "RdYlGn_r" if metric in _BAD_METRIC else "RdYlGn"
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.4),
                                    max(5, len(pivot) * 0.85)))

    sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, ax=ax,
                linewidths=0.4, linecolor="#1e2128",
                cbar_kws={"label": metric, "shrink": 0.8},
                annot_kws={"size": 9})

    ax.set_title(f"Model x Intensity Heatmap  |  {metric}  |  {shift_type}",
                 fontsize=13, pad=14, color="#e0e0e0")
    ax.set_xlabel("Shift Intensity", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Robustness Ranking Bar Chart
# ---------------------------------------------------------------------------

def plot_robustness_ranking(
    results_df: pd.DataFrame,
    shift_type: str,
) -> plt.Figure:
    """
    Horizontal bar chart ranking all models by their mean Robustness Score
    across all non-baseline intensities for the selected shift type.

    Args:
        results_df: Consolidated experiment result DataFrame.
        shift_type: Shift regime to rank.

    Returns:
        matplotlib Figure.
    """
    if "Robustness_Score" not in results_df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Robustness_Score column not found.\nPlease run the M2 pipeline.",
                ha="center", va="center", transform=ax.transAxes, color="#e0e0e0")
        _apply_dark_spine(ax)
        return fig

    df = results_df[
        (results_df["Shift_Type"] == shift_type) &
        (results_df["Intensity"] > 0.0)
    ].copy()

    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No shifted data available.",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    ranked = (
        df.groupby("Model")["Robustness_Score"]
          .mean()
          .sort_values(ascending=True)
          .reset_index()
    )

    colors = sns.color_palette("RdYlGn", len(ranked))
    fig, ax = plt.subplots(figsize=(9, max(4, len(ranked) * 0.65)))

    bars = ax.barh(ranked["Model"], ranked["Robustness_Score"],
                   color=colors, edgecolor="#1e2128", linewidth=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=9, color="#e0e0e0")

    ax.set_xlim(0, 1.05)
    ax.set_title(f"Mean Robustness Score Ranking  |  {shift_type}",
                 fontsize=13, pad=14, color="#e0e0e0")
    ax.set_xlabel("Robustness Score  (higher = more robust)", fontsize=11)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    _apply_dark_spine(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Calibration Decay (Brier Score)
# ---------------------------------------------------------------------------

def plot_calibration_decay(
    results_df: pd.DataFrame,
    shift_type: str,
) -> plt.Figure:
    """
    Plots Brier Score (a proper scoring rule measuring calibration quality)
    as a function of shift intensity for all models.

    Lower Brier Scores are better; a rising curve signals probability
    calibration degradation under distribution shift.

    Args:
        results_df: Consolidated experiment result DataFrame.
        shift_type: Shift regime to visualize.

    Returns:
        matplotlib Figure.
    """
    return plot_performance_curves(results_df, shift_type,
                                   metric="Brier_Score", show_ci=True)


# ---------------------------------------------------------------------------
# 5. Distribution Shift Diagnostics (KS + PSI)
# ---------------------------------------------------------------------------

def plot_ks_psi_curves(
    results_df: pd.DataFrame,
    shift_type: str,
) -> plt.Figure:
    """
    Dual-panel plot showing how the KS statistic and Population Stability
    Index (PSI) evolve as shift intensity increases.

    This chart decouples "how much did the data change" from "how much did
    performance drop", allowing an analyst to judge whether a model is
    sensitive even to small distribution changes.

    Args:
        results_df: Consolidated experiment result DataFrame.
        shift_type: Shift regime to visualize.

    Returns:
        matplotlib Figure with two vertically stacked subplots.
    """
    if "Avg_KS_Statistic" not in results_df.columns or "Avg_PSI" not in results_df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "KS/PSI columns not found.\nPlease run the M2 pipeline.",
                ha="center", va="center", transform=ax.transAxes, color="#e0e0e0")
        _apply_dark_spine(ax)
        return fig

    df = _prepare_with_baseline(results_df, shift_type)
    df = df.sort_values("Intensity")
    models = df["Model"].unique()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for i, model in enumerate(models):
        mdf = df[df["Model"] == model].sort_values("Intensity")
        c = _PALETTE[i % len(_PALETTE)]
        ax1.plot(mdf["Intensity"], mdf["Avg_KS_Statistic"], marker="s",
                 linewidth=1.8, markersize=5, label=model, color=c)
        ax2.plot(mdf["Intensity"], mdf["Avg_PSI"], marker="^",
                 linewidth=1.8, markersize=5, label=model, color=c)

    ax1.axhline(0.1, linestyle=":", color="#facc15", linewidth=1.2,
                label="KS = 0.10 threshold")
    ax1.set_ylabel("Avg. KS Statistic", fontsize=11)
    ax1.set_title(f"Distribution Shift Diagnostics  |  {shift_type}",
                  fontsize=13, pad=12, color="#e0e0e0")
    ax1.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.6)
    ax1.grid(True, linestyle="--", alpha=0.3)
    _apply_dark_spine(ax1)

    ax2.axhline(0.10, linestyle=":", color="#facc15", linewidth=1.2,
                label="PSI = 0.10 (monitoring)")
    ax2.axhline(0.20, linestyle=":", color="#f87171", linewidth=1.2,
                label="PSI = 0.20 (action)")
    ax2.set_ylabel("Avg. PSI", fontsize=11)
    ax2.set_xlabel("Shift Intensity", fontsize=11)
    ax2.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.6)
    ax2.grid(True, linestyle="--", alpha=0.3)
    _apply_dark_spine(ax2)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Confidence Band (Single Model)
# ---------------------------------------------------------------------------

def plot_confidence_band(
    results_df: pd.DataFrame,
    shift_type: str,
    model_name: str,
    metric: str = "Accuracy",
) -> plt.Figure:
    """
    Zoomed single-model view with a shaded 95% bootstrap confidence band.

    Args:
        results_df: Consolidated experiment result DataFrame.
        shift_type: Shift regime to visualize.
        model_name: Model to highlight.
        metric:     Performance metric to plot.

    Returns:
        matplotlib Figure.
    """
    if metric not in results_df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, f"{metric} column not found.\nPlease run the M2 pipeline.",
                ha="center", va="center", transform=ax.transAxes, color="#e0e0e0")
        _apply_dark_spine(ax)
        return fig

    df = _prepare_with_baseline(results_df, shift_type)
    mdf = df[df["Model"] == model_name].sort_values("Intensity")

    fig, ax = plt.subplots(figsize=(9, 5))
    lo_col, hi_col = f"{metric}_Lower_CI", f"{metric}_Upper_CI"

    ax.plot(mdf["Intensity"], mdf[metric], marker="o", linewidth=2.4,
            color="#60a5fa", label=f"{model_name}  ({metric})")

    if lo_col in mdf.columns and hi_col in mdf.columns:
        ax.fill_between(mdf["Intensity"], mdf[lo_col], mdf[hi_col],
                        alpha=0.22, color="#60a5fa", label="95% CI")

    ax.set_title(f"{model_name}  |  {metric}  |  {shift_type}",
                 fontsize=13, pad=14, color="#e0e0e0")
    ax.set_xlabel("Shift Intensity", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.35)
    _apply_dark_spine(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Relative Drop Heatmap
# ---------------------------------------------------------------------------

def plot_relative_drop_heatmap(results_df: pd.DataFrame) -> plt.Figure:
    """
    Heatmap of mean percentage accuracy drop (Relative_Drop_Pct) for every
    (Model, Shift_Type) combination, averaged over all intensities > 0.

    This gives a single-page overview of which model-shift combinations are
    most damaging.

    Args:
        results_df: Consolidated experiment result DataFrame.

    Returns:
        matplotlib Figure.
    """
    df = results_df[results_df["Intensity"] > 0.0].copy()

    if df.empty or "Relative_Drop_Pct" not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Relative_Drop_Pct column not found.",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    pivot = df.pivot_table(index="Model", columns="Shift_Type",
                           values="Relative_Drop_Pct", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2),
                                    max(5, len(pivot) * 0.75)))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                linewidths=0.4, linecolor="#1e2128",
                cbar_kws={"label": "Mean Accuracy Drop (%)", "shrink": 0.7},
                annot_kws={"size": 9})

    ax.set_title("Mean Relative Accuracy Drop (%) by Model and Shift Type",
                 fontsize=13, pad=14, color="#e0e0e0")
    ax.set_xlabel("Shift Type", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save Helper
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, filename: str) -> str:
    """
    Saves a Figure to the project's figures/ directory.

    Args:
        fig:      matplotlib Figure to save.
        filename: Output filename (e.g. "covariate_accuracy.png").

    Returns:
        Absolute path to the saved file.
    """
    path = get_figures_path(filename)
    fig.savefig(path, bbox_inches="tight", dpi=150,
                facecolor=fig.get_facecolor())
    return path
