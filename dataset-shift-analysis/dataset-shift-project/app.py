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
ROBUSTNESS_PATH = os.path.join(os.path.dirname(__file__), "results", "robustness_leaderboard.csv")

@st.cache_data
def load_results():
    if not os.path.exists(RESULTS_PATH):
        return None, None
    df = pd.read_csv(RESULTS_PATH)
    
    robustness_df = None
    if os.path.exists(ROBUSTNESS_PATH):
        robustness_df = pd.read_csv(ROBUSTNESS_PATH)
        
    return df, robustness_df

df, robustness_df = load_results()

st.title("Dataset Shift: Longitudinal Robustness Study (Milestone 2)")
st.markdown("""
This dashboard facilitates a granular analysis of how classical machine learning 
architectures respond to systematic distribution changes, now with 
**statistical validation across multiple seeds**.
""")

if df is None:
    st.error("Consolidated results matrix not detected.")
    st.info("Run 'python src/run_experiments.py' first.")
    st.stop()
    
# Extract available levels
models = sorted(df["Model"].unique().tolist())
shift_types = sorted([s for s in df["Shift_Type"].unique() if s != "Baseline"])
metrics = ["Accuracy", "F1_Score", "ROC_AUC", "Brier_Score"]

# Sidebar
st.sidebar.header("Global Selection")
selected_shift = st.sidebar.selectbox("Shift Regime", shift_types)
selected_metric = st.sidebar.selectbox("Evaluation Metric", metrics)

st.sidebar.markdown("---")
st.sidebar.subheader("Detailed Inspection")
selected_model = st.sidebar.selectbox("Target Model", models)

# Create Tabs
tab1, tab2, tab3 = st.tabs(["📉 Shift Analysis", "🏆 Robustness Leaderboard", "🔍 Model Insights"])

with tab1:
    # Baseline prep
    df_baseline = df[df["Shift_Type"] == "Baseline"].copy()
    df_baseline["Shift_Type"] = selected_shift
    df_current_shift = df[df["Shift_Type"] == selected_shift].copy()
    
    if 0.0 not in df_current_shift["Intensity"].values:
        df_plot = pd.concat([df_baseline, df_current_shift], ignore_index=True)
    else:
        df_plot = df_current_shift

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Performance Decay: {selected_metric}")
        try:
            fig_curve = plot_performance_curves(df_plot, selected_shift, selected_metric)
            st.pyplot(fig_curve)
        except Exception as e:
            st.error(f"Curve Error: {e}")

    with col2:
        st.subheader(f"Details: {selected_model}")
        # Map metric to mean for display if aggregated
        display_col = f"{selected_metric}_mean" if f"{selected_metric}_mean" in df_plot.columns else selected_metric
        model_df = df_plot[df_plot["Model"] == selected_model].sort_values("Intensity")
        
        cols_to_show = ["Intensity", display_col]
        if f"{selected_metric}_std" in df_plot.columns:
            cols_to_show.append(f"{selected_metric}_std")
            
        st.dataframe(
            model_df[cols_to_show].style.format("{:.3f}"),
            hide_index=True,
            use_container_width=True
        )
        
    st.markdown("---")
    st.subheader("Comparative Heatmap")
    try:
        fig_heatmap = plot_model_comparison_heatmap(df_plot, selected_shift, selected_metric)
        st.pyplot(fig_heatmap)
    except Exception as e:
        st.error(f"Heatmap Error: {e}")

with tab2:
    st.subheader("Global Robustness Leaderboard")
    st.markdown("Robustness is calculated via the **Area Under the Degradation Curve (AUDC)**, "
                "normalized relative to the baseline performance. (1.0 = Perfect Robustness)")
    
    if robustness_df is not None:
        # Filter for selected metric and shift
        r_filtered = robustness_df[(robustness_df["Metric"] == selected_metric) & 
                                  (robustness_df["Shift_Type"] == selected_shift)]
        
        if not r_filtered.empty:
            r_sorted = r_filtered.sort_values("Robustness_Score", ascending=False)
            st.dataframe(
                r_sorted.style.background_gradient(cmap="RdYlGn", subset=["Robustness_Score"]).format({"Robustness_Score": "{:.3f}"}),
                use_container_width=True,
                hide_index=True
            )
            
            # Bar chart
            st.bar_chart(r_sorted.set_index("Model")["Robustness_Score"])
        else:
            st.warning("No robustness data for the selected combination.")
    else:
        st.info("Robustness leaderboard data placeholder.")

with tab3:
    st.subheader("Interpretability Layer")
    st.markdown("Select a model and shift to view how feature importance and confidence change.")
    
    st.info("This tab will display pre-computed interpretability assets in the final version.")
    
    # Placeholder for interpretability plots
    figures_dir = os.path.join(os.path.dirname(__file__), "figures")
    # For now, just show a message or list available figures
    if os.path.exists(figures_dir):
        st.markdown("**Available Automated Figures:**")
        st.write(os.listdir(figures_dir)[:10]) # Show first 10
    else:
        st.info("No figures found. Run the experiment pipeline to generate them.")
