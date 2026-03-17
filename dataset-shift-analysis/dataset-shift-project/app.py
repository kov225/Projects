"""
Dataset Shift Explorer Application

This module implements a Streamlit-based interactive dashboard for visualizing 
the results of the dataset shift experiments. It provides a faceted interface 
for exploring model performance across different shift regimes (Covariate, 
Prior, Concept-Adjacent) and performance metrics (Accuracy, F1, ROC-AUC, 
Brier Score).
"""

import streamlit as st
import pandas as pd
import os
import sys

# Link to local source directory for specialized visualization routines
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
try:
    from visualizations import plot_performance_curves, plot_model_comparison_heatmap
except ImportError:
    # Fail-safe for environment misconfigurations during deployment
    pass 

# Configuration of global dashboard metadata and visual themes
st.set_page_config(page_title="Dataset Shift Explorer", layout="wide", page_icon="📈")

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "experiment_results.csv")

@st.cache_data
def load_results():
    """
    Ingests experimental results from the local filesystem with caching enabled.

    Args:
        None (Path is resolved relative to module location).

    Returns:
        pd.DataFrame or None: The ingested result matrix if detected, else None.
    """
    if not os.path.exists(RESULTS_PATH):
        return None
    return pd.read_csv(RESULTS_PATH)

df = load_results()

st.title("Dataset Shift: Longitudinal Robustness Study")
st.markdown("""
This dashboard facilitates a granular analysis of how seven classical machine 
learning architectures respond to systematic distribution changes. Use the 
sidebar filters to explore degradation profiles across various metrics and 
shift regimes.
""")

if df is None:
    st.error("Consolidated results matrix not detected in specified path.")
    st.info("Execution of the experimental pipeline ('python src/run_experiments.py') is required to populate the interface.")
    st.stop()
    
# Dynamically extract available categorical levels for filtering
models = sorted(df["Model"].unique().tolist())
shift_types = sorted([s for s in df["Shift_Type"].unique() if s != "Baseline"])
metrics = ["Accuracy", "F1_Score", "ROC_AUC", "Brier_Score"]

# Sidebar: Parameter Control Interface
st.sidebar.header("Global Configuration")
selected_shift = st.sidebar.selectbox("Shift Regime Selection", shift_types)
selected_metric = st.sidebar.selectbox("Primary Evaluation Metric", metrics)

st.sidebar.markdown("---")
st.sidebar.subheader("Granular Analysis")
selected_model = st.sidebar.selectbox("Target Model for Detailed View", models)

st.sidebar.markdown("---")
st.sidebar.info("Operational Note: The baseline measurement (Intensity = 0.0) is "
                "normalized across all shift trials to ensure comparative validity.")

# To provide a continuous visualization, the baseline (0.0) is prepended to 
# the active shift trial data.
df_baseline = df[df["Shift_Type"] == "Baseline"].copy()
df_baseline["Shift_Type"] = selected_shift

df_current_shift = df[df["Shift_Type"] == selected_shift].copy()
if 0.0 not in df_current_shift["Intensity"].values:
    df_plot = pd.concat([df_baseline, df_current_shift], ignore_index=True)
else:
    df_plot = df_current_shift

# Main Dashboard Layout: Split View
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Performance Decay Profile: {selected_metric} ({selected_shift})")
    st.markdown("Longitudinal view of model correction rate vs. distribution divergence.")
    try:
        fig_curve = plot_performance_curves(df_plot, selected_shift, selected_metric)
        st.pyplot(fig_curve)
    except Exception as e:
        st.error(f"Visualization pipeline failure: {e}")

with col2:
    st.subheader(f"Detailed Analysis: {selected_model}")
    # Segmented view for high-fidelty inspection of a single architecture
    model_df = df_plot[df_plot["Model"] == selected_model].sort_values("Intensity")
    
    st.dataframe(
        model_df[["Intensity", selected_metric]].style.format({selected_metric: "{:.3f}"}),
        hide_index=True,
        use_container_width=True
    )
    
st.markdown("---")
st.subheader("Comparative Matrix Heatmap")
st.markdown(f"Cross-sectional intensity analysis across the entire model suite.")

try:
    fig_heatmap = plot_model_comparison_heatmap(df_plot, selected_shift, selected_metric)
    st.pyplot(fig_heatmap)
except Exception as e:
    st.error(f"Matrix generation failure: {e}")
