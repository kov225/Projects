from __future__ import annotations

import pandas as pd
import streamlit as st
import numpy as np

from metrics.health import MetricHealthMonitor
from metrics.correlation import MetricCorrelationAnalysis
from metrics.sensitivity import MetricSensitivity
from data.generate import generate_synthetic_telemetry
from components.charts import plot_correlation_heatmap
from components.tables import display_health_summary


def show_page():
    """Display the metric health monitoring page.

    This page provides a deep dive into the reliability of our engagement
    signals. It identifies redundant metrics and alerts the team 
    if any metric becomes too noisy for detection.
    """
    st.header("Metric Health Diagnostics")
    st.markdown("""
    Monitor the stability and sensitivity of your engagement metrics. 
    We analyze correlations and compute power curves to ensure 
    our experimentation platform remains reliable.
    """)

    df = generate_synthetic_telemetry()
    metrics = ["clicked", "dwell_time_seconds", "reformulated", "abandoned"]

    # Step 1: Health Summary
    st.subheader("Current Metric Health")
    monitor = MetricHealthMonitor()
    health_results = []
    for m in metrics:
        health_results.append(monitor.check_health(df, m))
    display_health_summary(health_results)

    # Step 2: Correlation Analysis
    st.subheader("Metric Correlation Analysis")
    corr_analyzer = MetricCorrelationAnalysis(metrics)
    corr_df = df[metrics].corr()
    st.altair_chart(plot_correlation_heatmap(corr_df), use_container_width=True)
    
    st.markdown("""
    Higher correlations between metrics suggest they are capturing 
    similar user engagement signals. This is a key input for our 
    PCA based composite metric weights.
    """)

    # Step 3: Sensitivity Curves
    st.subheader("Sensitivity & Power Curves")
    sensitivity = MetricSensitivity()
    sample_sizes = [5000, 10000, 20000, 50000]
    
    metric_to_plot = st.selectbox("Select Metric for Power Curve", metrics)
    values = df[metric_to_plot].values
    
    mde_list = []
    for n in sample_sizes:
        mde_list.append({
            "Sample Size": n,
            "Absolute MDE": sensitivity.compute_mde(values, n=n),
            "Relative MDE (%)": (sensitivity.compute_mde(values, n=n) / values.mean()) * 100
        })
    
    st.dataframe(pd.DataFrame(mde_list).style.format({"Absolute MDE": "{:.4f}", "Relative MDE (%)": "{:.2f}%"}))
    st.markdown("""
    The Minimum Detectable Effect decreases as the sample size 
    increases, reflecting higher statistical precision.
    """)


if __name__ == "__main__":
    show_page()
