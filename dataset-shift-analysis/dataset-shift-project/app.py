"""
Dataset Shift Explorer  Streamlit Dashboard  Milestone 2

A research-grade interactive dashboard for exploring model robustness
to eight types of dataset shift.  The interface is organised into four
analysis tabs:

  1. Performance Curves    Metric vs. intensity trajectories per model.
  2. Comparative Heatmap   Cross-sectional model x intensity matrix.
  3. Robustness Analysis   Robustness scores, relative drop, calibration.
  4. Distribution Shift    KS statistic and PSI diagnostic charts.

All charts use a dark theme and respond to sidebar controls without
page reloads thanks to Streamlit's reactive data-flow model.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from visualizations import (
    plot_performance_curves, plot_model_heatmap,
    plot_robustness_ranking, plot_calibration_decay,
    plot_ks_psi_curves, plot_confidence_band,
    plot_relative_drop_heatmap,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dataset Shift Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS  dark card look, brand colours
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b22; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1c2533, #1a2035);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"]  { color: #94a3b8 !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"]  { color: #e2e8f0 !important; font-size: 1.55rem; }
    [data-testid="stMetricDelta"]  { font-size: 0.80rem; }

    /* Tab strip */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #161b22;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #94a3b8;
        font-weight: 500;
        padding: 6px 18px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: #fff !important;
    }

    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #60a5fa;
        margin-bottom: 4px;
        border-left: 3px solid #2563eb;
        padding-left: 10px;
    }

    /* Info callout */
    .info-box {
        background: #1c2533;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 12px 16px;
        color: #94a3b8;
        font-size: 0.85rem;
        line-height: 1.6;
    }

    /* Divider */
    hr { border-color: #2d3748 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results",
                             "experiment_results.csv")


@st.cache_data(show_spinner=False)
def load_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    return pd.read_csv(RESULTS_PATH)


df_raw = load_results()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='color:#e2e8f0;font-size:2rem;font-weight:700;margin-bottom:4px'>"
    "📊 Dataset Shift Explorer</h1>"
    "<p style='color:#64748b;font-size:0.95rem;margin-top:0'>"
    "Assessing Model Robustness to Environmental Distribution Change  "
    "Milestone 2  DSCI 441"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

if df_raw is None:
    st.error("No results file found at `results/experiment_results.csv`.")
    st.info(
        "Run the experiment pipeline first:\n\n"
        "```bash\npython main.py\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='color:#60a5fa;font-weight:700;font-size:1.05rem;"
        "margin-bottom:12px'>Control Panel</div>",
        unsafe_allow_html=True,
    )

    shift_types_all = sorted([s for s in df_raw["Shift_Type"].unique()
                               if s != "Baseline"])
    selected_shift  = st.selectbox("Shift Type", shift_types_all,
                                   key="shift_type_sel")

    all_metrics = ["Accuracy", "Precision", "Recall",
                   "F1_Score", "ROC_AUC", "Brier_Score"]
    available_metrics = [m for m in all_metrics if m in df_raw.columns]
    selected_metric   = st.selectbox("Primary Metric", available_metrics,
                                     key="metric_sel")

    all_models     = sorted(df_raw["Model"].unique().tolist())
    selected_model = st.selectbox("Model (detail view)", all_models,
                                  key="model_sel")

    st.markdown("---")

    intensity_range = st.slider(
        "Intensity Filter Range",
        min_value=0.0, max_value=1.0,
        value=(0.0, 1.0), step=0.1,
        key="intensity_slider",
    )

    st.markdown("---")
    st.markdown(
        "<div class='info-box'>"
        "<b>Intensity = 0.0</b> is the clean baseline.<br>"
        "Higher values represent stronger distribution shift.<br><br>"
        "<b>Robustness Score</b> near 1.0 means the model is insensitive "
        "to the applied shift.<br><br>"
        "<b>PSI thresholds</b><br>"
        "< 0.10  No action<br>"
        "0.10 to 0.20  Monitor<br>"
        "> 0.20  Investigate"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Filtered data helpers
# ---------------------------------------------------------------------------
lo_int, hi_int = intensity_range


def filtered_df(shift=selected_shift):
    bl = df_raw[df_raw["Shift_Type"] == "Baseline"].copy()
    bl["Shift_Type"] = shift
    act = df_raw[df_raw["Shift_Type"] == shift].copy()
    combined = pd.concat([bl, act], ignore_index=True) if 0.0 not in act["Intensity"].values else act
    return combined[
        combined["Intensity"].between(lo_int, hi_int)
    ].copy()


# ---------------------------------------------------------------------------
# Summary metrics (top of page)
# ---------------------------------------------------------------------------
df_shifted  = df_raw[
    (df_raw["Shift_Type"] == selected_shift) &
    (df_raw["Intensity"] > 0.0)
].copy()
df_baseline = df_raw[df_raw["Shift_Type"] == "Baseline"].copy()

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

with col_m1:
    n_models = df_raw["Model"].nunique()
    st.metric("Models Evaluated", n_models)

with col_m2:
    n_shifts = len(shift_types_all)
    st.metric("Shift Types", n_shifts)

with col_m3:
    n_inten = df_raw["Intensity"].nunique() - 1
    st.metric("Intensity Levels", n_inten)

with col_m4:
    if not df_baseline.empty and selected_metric in df_baseline.columns:
        bl_mean = df_baseline[selected_metric].mean()
        st.metric("Baseline (mean)", f"{bl_mean:.3f}")
    else:
        st.metric("Baseline", "N/A")

with col_m5:
    if not df_shifted.empty and "Robustness_Score" in df_shifted.columns:
        best_rob = df_shifted.groupby("Model")["Robustness_Score"].mean().idxmax()
        st.metric("Most Robust Model", best_rob)
    else:
        st.metric("Most Robust", "N/A")

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Performance Curves",
    "🗂️  Comparative Heatmap",
    "🏆  Robustness Analysis",
    "🔬  Distribution Shift",
])

# ---- Tab 1: Performance Curves --------------------------------------------
with tab1:
    st.markdown(
        f"<div class='section-header'>Performance Decay: "
        f"{selected_metric} under {selected_shift}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Shaded bands represent 95% bootstrap confidence intervals. "
        "A steeper downward slope signals greater sensitivity to this shift type."
    )
    fdf = filtered_df()
    if fdf.empty:
        st.warning("No data for the selected filters.")
    else:
        fig_curves = plot_performance_curves(fdf, selected_shift,
                                             selected_metric, show_ci=True)
        st.pyplot(fig_curves, use_container_width=True)

    st.markdown("---")
    st.markdown(
        f"<div class='section-header'>Detailed View: {selected_model}</div>",
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns([1, 2])
    with col_a:
        mdf_detail = fdf[fdf["Model"] == selected_model].sort_values("Intensity")
        cols_show  = ["Intensity", selected_metric,
                      f"{selected_metric}_Lower_CI", f"{selected_metric}_Upper_CI",
                      "Robustness_Score", "Relative_Drop_Pct",
                      "Avg_KS_Statistic", "Significant_Shift"]
        cols_show  = [c for c in cols_show if c in mdf_detail.columns]
        fmt = {c: "{:.4f}" for c in cols_show if mdf_detail[c].dtype == float
               if c in mdf_detail.columns}
        st.dataframe(
            mdf_detail[cols_show].style.format(fmt),
            hide_index=True, use_container_width=True,
        )
    with col_b:
        if not fdf.empty:
            fig_band = plot_confidence_band(fdf, selected_shift,
                                           selected_model, selected_metric)
            st.pyplot(fig_band, use_container_width=True)


# ---- Tab 2: Comparative Heatmap -------------------------------------------
with tab2:
    st.markdown(
        f"<div class='section-header'>Model x Intensity Heatmap: "
        f"{selected_metric} under {selected_shift}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Each cell shows the mean metric value for the (model, intensity) "
        "combination.  Green tones indicate high performance; red indicates degradation."
    )
    fdf2 = filtered_df()
    if fdf2.empty:
        st.warning("No data for the selected filters.")
    else:
        fig_heat = plot_model_heatmap(fdf2, selected_shift, selected_metric)
        st.pyplot(fig_heat, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div class='section-header'>Global Relative Drop Heatmap</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Mean percentage accuracy drop averaged over all intensities for "
        "every (model, shift-type) pair.  Darker red cells represent the "
        "most destructive combinations."
    )
    if "Relative_Drop_Pct" in df_raw.columns:
        fig_drop = plot_relative_drop_heatmap(df_raw)
        st.pyplot(fig_drop, use_container_width=True)
    else:
        st.info("Relative_Drop_Pct column not present. Re-run experiments.")


# ---- Tab 3: Robustness Analysis -------------------------------------------
with tab3:
    st.markdown(
        f"<div class='section-header'>Robustness Score Ranking: "
        f"{selected_shift}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Mean Robustness Score across all non-zero intensities. "
        "Score = accuracy-retention-ratio x (1 - KS statistic). "
        "A score near 1.0 means performance held up despite large distribution shift."
    )
    fdf3 = filtered_df()
    if not fdf3.empty:
        fig_rob = plot_robustness_ranking(fdf3, selected_shift)
        st.pyplot(fig_rob, use_container_width=True)

    st.markdown("---")
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown(
            f"<div class='section-header'>Calibration Decay (Brier Score): "
            f"{selected_shift}</div>",
            unsafe_allow_html=True,
        )
        st.caption("Lower Brier Score = better calibration.  "
                   "Rising curves indicate probability estimates are becoming unreliable.")
        if not fdf3.empty:
            fig_cal = plot_calibration_decay(fdf3, selected_shift)
            st.pyplot(fig_cal, use_container_width=True)

    with col_d:
        st.markdown(
            "<div class='section-header'>Robustness Score vs. Shift Type"
            " (all shifts, selected model)</div>",
            unsafe_allow_html=True,
        )
        if "Robustness_Score" in df_raw.columns:
            rob_by_type = (
                df_raw[
                    (df_raw["Model"] == selected_model) &
                    (df_raw["Intensity"] > 0.0)
                ]
                .groupby("Shift_Type")["Robustness_Score"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={"Shift_Type": "Shift Type",
                                  "Robustness_Score": "Mean Robustness"})
            )
            st.dataframe(
                rob_by_type.style.format({"Mean Robustness": "{:.4f}"}),
                hide_index=True, use_container_width=True,
            )


# ---- Tab 4: Distribution Shift Diagnostics --------------------------------
with tab4:
    st.markdown(
        f"<div class='section-header'>KS Statistic and PSI: "
        f"{selected_shift}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "KS Statistic measures the maximum distance between empirical CDFs. "
        "PSI quantifies overall distributional shift. "
        "Both should be read alongside performance metrics to establish causality."
    )
    fdf4 = filtered_df()
    if fdf4.empty:
        st.warning("No data for the selected filters.")
    else:
        fig_ks = plot_ks_psi_curves(fdf4, selected_shift)
        st.pyplot(fig_ks, use_container_width=True)

    st.markdown("---")
    st.markdown(
        f"<div class='section-header'>Statistical Significance Table: "
        f"{selected_shift}  |  {selected_model}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "P-values from a one-sided Welch t-test comparing bootstrapped "
        "accuracy distributions (shifted vs. baseline). "
        "Significant = True when p < 0.05."
    )
    sig_df = fdf4[fdf4["Model"] == selected_model].copy()
    if not sig_df.empty and "P_Value" in sig_df.columns:
        sig_cols = ["Intensity", "Accuracy", "Avg_KS_Statistic",
                    "Avg_PSI", "P_Value", "Significant_Shift"]
        sig_cols = [c for c in sig_cols if c in sig_df.columns]
        fmt_sig  = {c: "{:.4f}" for c in sig_cols
                    if sig_df[c].dtype == float and c in sig_df.columns}

        def _color_sig(val):
            if val is True:
                return "color: #f87171; font-weight: 600"
            return "color: #4ade80"

        styled = sig_df[sig_cols].style.format(fmt_sig)
        if "Significant_Shift" in sig_cols:
            styled = styled.applymap(_color_sig, subset=["Significant_Shift"])
        st.dataframe(styled, hide_index=True, use_container_width=True)
    else:
        st.info("Statistical significance data not available.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.80rem;padding-bottom:12px'>"
    "Dataset Shift Explorer  Milestone 2  DSCI 441 Machine Learning  "
    "UCI Adult Income Dataset  8 Shift Types  9 Models"
    "</div>",
    unsafe_allow_html=True,
)
