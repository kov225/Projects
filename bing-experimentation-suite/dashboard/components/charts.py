from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st


def plot_forest_results(df: pd.DataFrame) -> alt.Chart:
    """Create a forest plot of experiment results.

    This chart displays the treatment effect estimates and confidence
    intervals for all statistical methods. It allows for a visual
    comparison of how each technique reduces uncertainty.
    """
    base = alt.Chart(df).encode(
        y=alt.Y("Method:N", title="Statistical Method"),
        x=alt.X("Estimated Effect:Q", title="Treatment Lift"),
    )

    points = base.mark_point(filled=True, size=100)
    
    # Confidence intervals
    error_bars = base.mark_errorbar().encode(
        x=alt.X("ci_low:Q", title="Treatment Lift"),
        x2="ci_high:Q"
    )

    return (points + error_bars).properties(
        width=600,
        height=300,
        title="Treatment Effect Comparison with 95% Confidence Intervals"
    )


def plot_novelty_decay(weeks: list[int], effects: list[float], fit: list[float]) -> alt.Chart:
    """Visualize the decay of the treatment effect over time.

    The chart shows the actual weekly lift alongside the fitted exponential
    decay curve. It highlights whether the initial gains are stable or if
    they are diminishing due to novelty.
    """
    df = pd.DataFrame({
        "Week": weeks,
        "Actual Effect": effects,
        "Predicted Decay": fit
    })
    
    df_melted = df.melt("Week", var_name="Metric", value_name="Value")
    
    return alt.Chart(df_melted).mark_line(point=True).encode(
        x=alt.X("Week:O", title="Week of Experiment"),
        y=alt.Y("Value:Q", title="Treatment Lift"),
        color=alt.Color("Metric:N", title="Series Type")
    ).properties(
        width=600,
        height=300,
        title="Novelty Effect Decay Analysis"
    )


def plot_correlation_heatmap(df: pd.DataFrame) -> alt.Chart:
    """Display a heatmap of the metric correlation matrix.

    This visualization identifies which engagement signals are moving
    together. High correlations suggest redundancy, while low correlations
    indicate independent signals for the composite metric.
    """
    return alt.Chart(df.reset_index().melt(id_vars="index")).mark_rect().encode(
        x=alt.X("index:N", title=None),
        y=alt.Y("variable:N", title=None),
        color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis"), title="Correlation"),
        tooltip=["index", "variable", "value"]
    ).properties(
        width=500,
        height=500,
        title="Metric Correlation Matrix"
    )
