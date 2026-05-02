"""Top level ETL helpers used by the notebooks and the Streamlit app."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import data_acquisition as acq
from . import data_extension as ext
from . import data_quality as dq
from . import experiment_analysis as exa

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_and_validate(
    use_bigquery: bool = True,
    project_id: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Acquire the raw GA4 data and run the standard quality checks."""
    if use_bigquery:
        events = acq.fetch_events(project_id=project_id, use_cache=True)
        items = acq.fetch_items(project_id=project_id, use_cache=True)
    else:
        events = acq.load_cached_events()
        items = acq.load_cached_items()

    LOGGER.info("acquired %d events and %d item rows", len(events), len(items))
    return {"events": events, "items": items}


def build_extended_dataset(
    events_real: pd.DataFrame,
    write: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the documented extension and optionally persist to parquet."""
    events_all, users_all = ext.extend_dataset(events_real)
    if write:
        ext.write_extended(events_all, users_all)
    return {"events": events_all, "users": users_all}


def build_user_features(events: pd.DataFrame, items: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate the raw event log into the per-user behavioural feature table."""
    from .segmentation import build_user_features as _build

    return _build(events, items)


def build_weekly_metrics(events: pd.DataFrame) -> pd.DataFrame:
    """Weekly active users, sessions, and revenue. Useful for the dashboard."""
    df = events.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["week_start"] = df["event_date"].dt.to_period("W-MON").dt.start_time
    weekly = df.groupby("week_start").agg(
        wau=("user_pseudo_id", "nunique"),
        sessions=("session_id", "nunique"),
        events=("event_timestamp", "count"),
        revenue=("purchase_revenue", lambda s: float(s.fillna(0).sum())),
        purchases=("event_name", lambda s: int((s == "purchase").sum())),
    ).reset_index()
    weekly["revenue_per_active_user"] = weekly["revenue"] / weekly["wau"].replace(0, pd.NA)
    return weekly


def build_experiment_summary(
    extended_events: pd.DataFrame,
    users: pd.DataFrame,
    write: bool = True,
) -> dict[str, pd.DataFrame]:
    """Simulate experiments on top of the extended user base and summarise."""
    experiments, assignments, results = exa.simulate_experiments(users)
    summary = exa.summarise_experiments(results, experiments)
    if write:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        experiments.to_parquet(PROCESSED_DIR / "experiments.parquet", index=False)
        assignments.to_parquet(PROCESSED_DIR / "experiment_assignments.parquet", index=False)
        results.to_parquet(PROCESSED_DIR / "experiment_results.parquet", index=False)
        summary.to_parquet(PROCESSED_DIR / "experiment_summary.parquet", index=False)
    return {
        "experiments": experiments,
        "assignments": assignments,
        "results": results,
        "summary": summary,
    }


def run_full_pipeline(
    use_bigquery: bool = True,
    project_id: str | None = None,
) -> dict[str, pd.DataFrame]:
    """One call to take the project from BigQuery (or cache) to processed parquet."""
    raw = load_and_validate(use_bigquery=use_bigquery, project_id=project_id)
    extended = build_extended_dataset(raw["events"])
    quality_report = dq.run_all_checks(extended["users"], extended["events"])
    LOGGER.info("data quality report:\n%s", quality_report.to_string(index=False))
    experiments_bundle = build_experiment_summary(extended["events"], extended["users"])
    return {
        "raw_events": raw["events"],
        "raw_items": raw["items"],
        "events": extended["events"],
        "users": extended["users"],
        "quality_report": quality_report,
        **experiments_bundle,
    }
