from __future__ import annotations

import pandas as pd
import streamlit as st
import numpy as np

from experiments.variance_benchmark import VarianceBenchmark
from experiments.novelty import NoveltyEffectDetector
from data.generate import generate_synthetic_telemetry
from components.charts import plot_forest_results, plot_novelty_decay
from components.tables import display_results_table


def show_page():
    """Display the experiment results analysis page.

    This page allows users to run multi variant analyses on their dataset.
    It highlights the power of variance reduction and identifies any
    novelty decay patterns to ensure robust product launch decisions.
    """
    st.header("Experiment Results Analysis")
    st.markdown("""
    Upload your experiment data to perform a deep statistical analysis. 
    We compare four different methods and check for novelty effects 
    to provide the most reliable recommendation.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file with experiment data", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("No file uploaded. Displaying results from the default synthetic dataset.")
        df = generate_synthetic_telemetry()

    outcome_col = st.selectbox("Select Outcome Metric", ["clicked", "dwell_time_seconds"])
    covariate_col = "pre_experiment_engagement"
    stratum_col = "user_segment"

    # Step 1: Run Variance Benchmark
    st.subheader("Statistical Method Comparison")
    benchmark = VarianceBenchmark()
    results_df = benchmark.run_benchmark(df, outcome_col, covariate_col, stratum_col)
    
    # Format for chart (ci_low and ci_high for forest plot)
    # Estimated effect +/- CI Width / 2
    results_df["ci_low"] = results_df["Estimated Effect"] - (results_df["CI Width"] / 2)
    results_df["ci_high"] = results_df["Estimated Effect"] + (results_df["CI Width"] / 2)
    
    col1, col2 = st.columns(2)
    with col1:
        display_results_table(results_df.drop(columns=["ci_low", "ci_high"]))
    with col2:
        st.altair_chart(plot_forest_results(results_df), use_container_width=True)

    # Step 2: Novelty Analysis
    st.subheader("Novelty Effect Detection")
    detector = NoveltyEffectDetector()
    novelty_res = detector.detect(df, outcome_col)
    
    st.write(f"Novelty Detected: **{novelty_res.novelty_detected}**")
    st.write(f"Estimated True Long-Term Effect: **{novelty_res.estimated_true_effect:.4f}**")
    
    weeks = sorted(df["week_number"].unique())
    weekly_lifts = []
    for w in weeks:
        sub = df[df["week_number"] == w]
        weekly_lifts.append(sub[sub["treatment"] == 1][outcome_col].mean() - sub[sub["treatment"] == 0][outcome_col].mean())
    
    fit_values = [
        novelty_res.estimated_true_effect + novelty_res.novelty_magnitude * np.exp(-novelty_res.decay_rate * (w - 1))
        for w in weeks
    ]
    
    st.altair_chart(plot_novelty_decay(weeks, weekly_lifts, fit_values), use_container_width=True)
    
    # Recommendation
    st.subheader("Final Product Recommendation")
    best_method = "CUPED" if results_df.iloc[1]["Var Reduction Pct"] > 0 else "Standard T Test"
    final_rec = results_df[results_df["Method"] == best_method]["Recommendation"].values[0]
    
    if novelty_res.novelty_detected:
        st.warning(f"Note: Significant novelty effect detected. The true lift is estimated at {novelty_res.estimated_true_effect:.4f}.")
    
    st.success(f"Recommendation based on {best_method}: **{final_rec}**")


if __name__ == "__main__":
    show_page()
