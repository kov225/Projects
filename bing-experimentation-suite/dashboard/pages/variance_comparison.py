from __future__ import annotations

import pandas as pd
import streamlit as st
import numpy as np

from experiments.aa_test.py import AATestRunner # Wait, it is in experiments/aa_test.py
from experiments.aa_test import AATestRunner
from experiments.variance_benchmark import VarianceBenchmark
from data.generate import generate_synthetic_telemetry


def show_page():
    """Display the variance reduction and A/A validation page.

    This page performs intensive simulations to validate the 
    statistical properties of each method. It confirms that the 
    Type I error is correctly controlled and measures the empirical 
    benefit of variance reduction.
    """
    st.header("Variance Reduction & Validation")
    st.markdown("""
    Compare the statistical properties of different experimentation 
    methods. We validate the false positive rate through simulated 
    A/A tests and measure how variance reduction impacts precision.
    """)

    df = generate_synthetic_telemetry()
    outcome_col = "clicked"
    covariate_col = "pre_experiment_engagement"
    stratum_col = "user_segment"

    # Step 1: A/A Simulation
    st.subheader("A/A Test Validation (Type I Error Control)")
    aa_runner = AATestRunner()
    st.write("Running 500 simulated A/A tests...")
    aa_res = aa_runner.run_simulation(df, outcome_col, iterations=500)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Empirical False Positive Rate", f"{aa_res['false_positive_rate'] * 100:.2f}%")
    with col2:
        st.metric("Target Alpha", "5.00%")
    with col3:
        status_color = "green" if aa_res["is_calibrated"] else "red"
        st.write(f"Status: **{('CALIBRATED' if aa_res['is_calibrated'] else 'MISCALIBRATED')}**")
    
    st.markdown("""
    A false positive rate close to five percent confirms that our 
    system is well calibrated. Higher values suggest a risk of 
    overstating significance in product experiments.
    """)

    # Step 2: Variance Reduction Summary
    st.subheader("Method Comparison: Variance & Precision")
    benchmark = VarianceBenchmark()
    bench_df = benchmark.run_benchmark(df, outcome_col, covariate_col, stratum_col)
    
    st.table(bench_df[["Method", "Standard Error", "CI Width", "Var Reduction Pct"]])
    
    st.markdown("""
    Variance reduction techniques like CUPED and Regression Adjustment 
    allow for more precise estimates with narrower confidence intervals. 
    This enables us to ship experiments faster with the same statistical confidence.
    """)


if __name__ == "__main__":
    show_page()
