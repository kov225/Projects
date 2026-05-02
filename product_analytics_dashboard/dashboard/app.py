"""Streamlit dashboard for the Product Analytics project.

Run from the project root:

    streamlit run dashboard/app.py

The dashboard reads from data/processed/ which is produced by notebook 01.
If those parquets do not exist yet, the homepage prints instructions to
generate them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import experiment_analysis as exa
from src import metrics as m
from src import segmentation as seg

PROCESSED = ROOT / "data" / "processed"
PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#937860", "#DA8BC3"]


st.set_page_config(
    page_title="Product Analytics Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_events() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "events_extended.parquet")
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


@st.cache_data(show_spinner=False)
def load_users() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "users_extended.parquet")
    df["first_seen_date"] = pd.to_datetime(df["first_seen_date"])
    return df


@st.cache_data(show_spinner=False)
def load_experiments() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exps = pd.read_parquet(PROCESSED / "experiments.parquet")
    assigns = pd.read_parquet(PROCESSED / "experiment_assignments.parquet")
    results = pd.read_parquet(PROCESSED / "experiment_results.parquet")
    summary = pd.read_parquet(PROCESSED / "experiment_summary.parquet")
    return exps, assigns, results, summary


@st.cache_data(show_spinner=False)
def cached_segmentation(events: pd.DataFrame, k: int = 5) -> dict:
    features = seg.build_user_features(events)
    result = seg.fit_segments(features, k=k, seed=11)
    names = seg.label_segments(result.profile)
    labelled = result.features.copy()
    labelled["segment"] = labelled["segment_id"].map(names)
    return {
        "features": labelled,
        "profile": result.profile.rename(index=names),
        "names": names,
        "silhouette": result.silhouette,
    }


def required_files_exist() -> bool:
    needed = [
        "events_extended.parquet",
        "users_extended.parquet",
        "experiments.parquet",
        "experiment_results.parquet",
        "experiment_summary.parquet",
    ]
    return all((PROCESSED / n).exists() for n in needed)


def empty_state() -> None:
    st.title("Product Analytics Dashboard")
    st.warning(
        "Processed data not found. Run notebook 01 (data acquisition and "
        "extension) and notebook 04 (experiments) before launching the dashboard."
    )
    st.code(
        "jupyter nbconvert --to notebook --execute notebooks/01_data_acquisition.ipynb\n"
        "jupyter nbconvert --to notebook --execute notebooks/04_ab_testing_analysis.ipynb",
        language="bash",
    )


def metric_card(label: str, value: str, delta: str | None = None) -> None:
    st.metric(label=label, value=value, delta=delta)


def apply_filters(
    events: pd.DataFrame,
    users: pd.DataFrame,
    date_range: tuple[pd.Timestamp, pd.Timestamp],
    devices: list[str],
    sources: list[str],
    countries: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    e = events
    e = e[(e["event_date"] >= date_range[0]) & (e["event_date"] <= date_range[1])]
    if devices:
        e = e[e["device_category"].isin(devices)]
    if sources:
        e = e[e["traffic_source"].isin(sources)]
    if countries:
        e = e[e["country"].isin(countries)]
    u = users[users["user_pseudo_id"].isin(e["user_pseudo_id"].unique())]
    return e, u


def overview_tab(events: pd.DataFrame, users: pd.DataFrame) -> None:
    st.subheader("Activity at a glance")
    wau = m.weekly_active_users(events)
    dau = m.daily_active_users(events)
    mau = m.monthly_active_users(events)

    latest_dau = int(dau["dau"].iloc[-30:].mean()) if len(dau) else 0
    latest_wau = int(wau["wau"].iloc[-4:].mean()) if len(wau) else 0
    latest_mau = int(mau["mau"].iloc[-1]) if len(mau) else 0
    sticky = round(latest_dau / latest_mau, 3) if latest_mau else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("DAU (30d avg)", f"{latest_dau:,}")
    with c2:
        metric_card("WAU (last 4w avg)", f"{latest_wau:,}")
    with c3:
        metric_card("MAU (latest)", f"{latest_mau:,}")
    with c4:
        metric_card("DAU/MAU stickiness", f"{sticky:.1%}")

    st.markdown("##### Weekly active users")
    fig = px.line(wau, x="week_start", y="wau", color_discrete_sequence=[PALETTE[0]])
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=None, yaxis_title="WAU",
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Conversion funnel")
    funnel = m.conversion_funnel(events)
    fig = go.Figure(
        go.Funnel(
            y=funnel["step"],
            x=funnel["users"],
            textinfo="value+percent initial",
            marker={"color": PALETTE[1]},
        )
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Source breakdown of the underlying data")
    source_counts = events["source"].value_counts().rename("events").to_frame()
    source_counts["share"] = source_counts["events"] / source_counts["events"].sum()
    st.dataframe(
        source_counts.style.format({"events": "{:,}", "share": "{:.1%}"}),
        use_container_width=True,
    )


def cohort_tab(events: pd.DataFrame, users: pd.DataFrame) -> None:
    st.subheader("Cohort retention")
    matrix = m.cohort_retention_matrix(events, users)
    if matrix.empty:
        st.info("No cohort matrix could be built for the current filter selection.")
        return

    max_weeks = st.slider("Weeks since first visit", 4, 24, 12)
    display = matrix.iloc[:, : max_weeks + 1] * 100
    display.index = [str(d.date()) for d in display.index]
    fig = px.imshow(
        display,
        color_continuous_scale="BuPu",
        aspect="auto",
        labels=dict(x="Weeks since first visit", y="Cohort week", color="Retention (%)"),
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Retention curves at selected horizons")
    horizon_choice = st.multiselect(
        "Cohorts to highlight",
        options=list(display.index),
        default=list(display.index[:: max(1, len(display.index) // 5)])[:5],
    )
    if horizon_choice:
        sub = display.loc[horizon_choice].T
        fig = px.line(sub, x=sub.index, y=sub.columns, color_discrete_sequence=PALETTE)
        fig.update_layout(
            height=380, xaxis_title="Weeks since first visit", yaxis_title="Retention (%)",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


def experiments_tab(
    experiments: pd.DataFrame,
    results: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    st.subheader("Experiment results")
    feature_filter = st.multiselect(
        "Feature area",
        options=sorted(summary["feature_area"].unique()),
        default=[],
    )
    rec_filter = st.multiselect(
        "Recommendation",
        options=sorted(summary["recommendation"].unique()),
        default=[],
    )
    view = summary.copy()
    if feature_filter:
        view = view[view["feature_area"].isin(feature_filter)]
    if rec_filter:
        view = view[view["recommendation"].isin(rec_filter)]

    st.dataframe(
        view[
            [
                "experiment_name", "feature_area", "experiment_type",
                "winner_variant", "control_rate", "winner_rate",
                "lift_rel", "p_value", "ci_low", "ci_high", "power", "recommendation",
            ]
        ].style.format({
            "control_rate": "{:.2%}", "winner_rate": "{:.2%}",
            "lift_rel": "{:+.1%}", "p_value": "{:.3f}",
            "ci_low": "{:+.3f}", "ci_high": "{:+.3f}", "power": "{:.2f}",
        }),
        use_container_width=True,
        height=420,
    )

    if view.empty:
        st.info("No experiments match the current filters.")
        return

    st.markdown("##### Deep dive on a selected experiment")
    chosen_name = st.selectbox("Experiment", options=view["experiment_name"].tolist())
    chosen_id = experiments[experiments["experiment_name"] == chosen_name]["experiment_id"].iloc[0]
    arms = exa.analyse_experiment(results, chosen_id)

    fig = px.bar(
        arms,
        x="variant",
        y="conv_rate",
        color="variant",
        color_discrete_sequence=PALETTE,
        text=arms["conv_rate"].map(lambda v: f"{v:.1%}"),
    )
    fig.update_layout(
        height=360, yaxis_tickformat=".1%", showlegend=False,
        xaxis_title=None, yaxis_title="Conversion rate",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        arms.style.format({
            "conv_rate": "{:.2%}", "lift_abs": "{:+.3f}", "lift_rel": "{:+.1%}",
            "p_value": "{:.4f}", "ci_low": "{:+.3f}", "ci_high": "{:+.3f}",
            "cohen_h": "{:.3f}", "power": "{:.2f}",
        }),
        use_container_width=True,
    )

    if experiments.set_index("experiment_id").loc[chosen_id, "experiment_type"] == "multivariate":
        st.markdown("##### Pairwise comparisons")
        pairwise = exa.pairwise_multivariate(results, chosen_id)
        st.dataframe(
            pairwise.style.format({
                "rate_a": "{:.2%}", "rate_b": "{:.2%}", "lift_abs": "{:+.3f}",
                "p_value": "{:.4f}", "ci_low": "{:+.3f}", "ci_high": "{:+.3f}", "cohen_h": "{:.3f}",
            }),
            use_container_width=True,
        )


def segments_tab(events: pd.DataFrame, users: pd.DataFrame) -> None:
    st.subheader("Behavioural segments")
    if len(events) < 2_000:
        st.info("Filter selection is too small to fit a stable segmentation. Widen the filters.")
        return

    bundle = cached_segmentation(events, k=5)
    profile = bundle["profile"]
    sizes = bundle["features"]["segment"].value_counts().rename("users").to_frame()

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.metric("Silhouette score", f"{bundle['silhouette']:.3f}")
        st.dataframe(sizes.style.format("{:,}"), use_container_width=True)
        fig = px.pie(
            sizes,
            values="users",
            names=sizes.index,
            color_discrete_sequence=PALETTE,
            hole=0.45,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("##### Segment metrics")
        joined = bundle["features"].merge(
            users[["user_pseudo_id", "traffic_source", "device_category"]], on="user_pseudo_id"
        )
        meta = joined.groupby("segment").agg(
            users=("user_pseudo_id", "nunique"),
            avg_events=("total_events", "mean"),
            avg_revenue=("total_revenue", "mean"),
            purchase_rate=("purchases_count", lambda s: float((s > 0).mean())),
            top_source=("traffic_source", lambda s: s.value_counts().idxmax()),
            top_device=("device_category", lambda s: s.value_counts().idxmax()),
        ).sort_values("avg_revenue", ascending=False)
        st.dataframe(
            meta.style.format({
                "users": "{:,.0f}", "avg_events": "{:,.1f}",
                "avg_revenue": "${:,.2f}", "purchase_rate": "{:.1%}",
            }),
            use_container_width=True,
        )

        st.markdown("##### Behavioural shape (scaled 0 to 1)")
        scaled = (profile - profile.min()) / (profile.max() - profile.min()).replace(0, np.nan)
        scaled = scaled.fillna(0)
        long = scaled.reset_index().melt(id_vars="segment_id", var_name="feature", value_name="value")
        fig = px.bar(
            long, x="feature", y="value", color="segment_id",
            barmode="group", color_discrete_sequence=PALETTE,
        )
        fig.update_layout(
            height=420, yaxis_title="Scaled value", xaxis_title=None,
            margin=dict(l=10, r=10, t=10, b=60), legend_title="Segment",
        )
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    if not required_files_exist():
        empty_state()
        return

    events = load_events()
    users = load_users()
    experiments, _assigns, results, summary = load_experiments()

    st.title("Product Analytics Dashboard")
    st.caption(
        "Real Google Analytics 4 BigQuery data (Google Merchandise Store, "
        "Nov 2020 to Jan 2021), extended to 156 weeks via documented "
        "statistical resampling."
    )

    with st.sidebar:
        st.header("Filters")
        date_min = events["event_date"].min().date()
        date_max = events["event_date"].max().date()
        date_range = st.date_input(
            "Date range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_dt = pd.Timestamp(date_range[0])
            end_dt = pd.Timestamp(date_range[1])
        else:
            start_dt = pd.Timestamp(date_min)
            end_dt = pd.Timestamp(date_max)

        device_options = sorted(events["device_category"].dropna().unique().tolist())
        source_options = sorted(events["traffic_source"].dropna().unique().tolist())
        country_options = sorted(events["country"].dropna().unique().tolist())

        devices = st.multiselect("Device category", device_options)
        sources = st.multiselect("Traffic source", source_options)
        countries = st.multiselect("Country", country_options[:25])

        st.divider()
        st.markdown(
            "**Data source.** Primary data is the public BigQuery dataset "
            "`bigquery-public-data.ga4_obfuscated_sample_ecommerce`. The 92 "
            "real days are extended to 156 weeks via statistical resampling "
            "that preserves the original behavioural distributions. Every "
            "row carries a source flag."
        )

    filtered_events, filtered_users = apply_filters(
        events, users, (start_dt, end_dt), devices, sources, countries
    )

    if len(filtered_events) == 0:
        st.warning("No data matches the current filters.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Cohort analysis", "Experiments", "User segments"])
    with tab1:
        overview_tab(filtered_events, filtered_users)
    with tab2:
        cohort_tab(filtered_events, filtered_users)
    with tab3:
        experiments_tab(experiments, results, summary)
    with tab4:
        segments_tab(filtered_events, filtered_users)


if __name__ == "__main__":
    main()
