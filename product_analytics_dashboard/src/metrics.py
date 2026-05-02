"""Product metric calculations: DAU/WAU/MAU, retention, funnels, time-to-value."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def daily_active_users(events: pd.DataFrame) -> pd.DataFrame:
    """Distinct users per calendar day."""
    df = events.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    return (
        df.groupby(df["event_date"].dt.date)["user_pseudo_id"].nunique().rename("dau").reset_index()
    )


def weekly_active_users(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["week_start"] = df["event_date"].dt.to_period("W-MON").dt.start_time
    return (
        df.groupby("week_start")["user_pseudo_id"].nunique().rename("wau").reset_index()
    )


def monthly_active_users(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["month_start"] = df["event_date"].dt.to_period("M").dt.start_time
    return (
        df.groupby("month_start")["user_pseudo_id"].nunique().rename("mau").reset_index()
    )


def stickiness(events: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """Rolling DAU / MAU stickiness ratio. Returns one row per calendar day.

    Uses an explode-expand trick to avoid the naive O(D*N) scan: each unique
    (user, active_date) pair is expanded into the next `window_days` days,
    so the user counts as 'monthly active' on each of those days.
    """
    df = events.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    user_dates = df[["user_pseudo_id", "event_date"]].drop_duplicates()

    daily = user_dates.groupby("event_date")["user_pseudo_id"].nunique().rename("dau")

    offsets = pd.to_timedelta(np.arange(window_days), unit="D")
    expanded = (
        user_dates.assign(_offset=[offsets] * len(user_dates))
        .explode("_offset")
    )
    expanded["effective_date"] = expanded["event_date"] + expanded["_offset"]
    expanded = expanded.drop_duplicates(["user_pseudo_id", "effective_date"])

    monthly = (
        expanded.groupby("effective_date")["user_pseudo_id"]
        .nunique()
        .rename("mau")
    )

    full_range = pd.date_range(daily.index.min(), monthly.index.max(), freq="D")
    out = pd.DataFrame({"date": full_range})
    out = out.merge(daily.rename_axis("date").reset_index(), on="date", how="left")
    out = out.merge(monthly.rename_axis("date").reset_index(), on="date", how="left")
    out["dau"] = out["dau"].fillna(0).astype(int)
    out["mau"] = out["mau"].fillna(0).astype(int)
    out["stickiness"] = out["dau"] / out["mau"].replace(0, np.nan)
    return out


def cohort_retention_matrix(
    events: pd.DataFrame,
    users: pd.DataFrame,
) -> pd.DataFrame:
    """Cohort-week vs weeks-since-first-visit retention matrix.

    Cohorts are defined by the user's first_seen_date floored to the
    Monday-aligned week.
    """
    ev = events.copy()
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    ev["event_week"] = ev["event_date"].dt.to_period("W-MON").dt.start_time

    u = users.copy()
    u["first_seen_date"] = pd.to_datetime(u["first_seen_date"])
    u["cohort_week"] = u["first_seen_date"].dt.to_period("W-MON").dt.start_time

    merged = ev.merge(u[["user_pseudo_id", "cohort_week"]], on="user_pseudo_id", how="inner")
    merged["weeks_since"] = (
        (merged["event_week"] - merged["cohort_week"]) / np.timedelta64(1, "W")
    ).astype(int)
    merged = merged[merged["weeks_since"] >= 0]

    cohort_size = u.groupby("cohort_week")["user_pseudo_id"].nunique().rename("cohort_size")
    active = (
        merged.groupby(["cohort_week", "weeks_since"])["user_pseudo_id"].nunique().reset_index(name="active")
    )
    active = active.merge(cohort_size, on="cohort_week", how="left")
    active["retention"] = active["active"] / active["cohort_size"]

    matrix = active.pivot(index="cohort_week", columns="weeks_since", values="retention")
    return matrix


def conversion_funnel(
    events: pd.DataFrame,
    steps: Sequence[str] = ("first_visit", "view_item", "add_to_cart", "begin_checkout", "purchase"),
) -> pd.DataFrame:
    """Funnel of unique users hitting each step at least once."""
    df = events.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    rows: list[dict] = []
    seen_users: set | None = None
    for step in steps:
        if step == "first_visit":
            users = set(df["user_pseudo_id"].unique())
        else:
            users = set(df.loc[df["event_name"] == step, "user_pseudo_id"].unique())
        if seen_users is None:
            seen_users = users
        else:
            seen_users = seen_users & users
        rows.append({"step": step, "users": len(seen_users)})
    out = pd.DataFrame(rows)
    out["conv_rate"] = out["users"] / out["users"].iloc[0]
    return out


def time_to_first_purchase(events: pd.DataFrame, users: pd.DataFrame) -> pd.Series:
    """Days from first_seen_date to first purchase, per converting user."""
    purchases = events[events["event_name"] == "purchase"][["user_pseudo_id", "event_date"]]
    purchases = purchases.copy()
    purchases["event_date"] = pd.to_datetime(purchases["event_date"])
    first_purchase = purchases.groupby("user_pseudo_id")["event_date"].min()

    u = users[["user_pseudo_id", "first_seen_date"]].copy()
    u["first_seen_date"] = pd.to_datetime(u["first_seen_date"])

    merged = u.merge(first_purchase.rename("first_purchase"), on="user_pseudo_id", how="inner")
    days = (merged["first_purchase"] - merged["first_seen_date"]).dt.days.clip(lower=0)
    return days.rename("days_to_first_purchase")


def feature_adoption_by_week(events: pd.DataFrame) -> pd.DataFrame:
    """Weekly share of active users that touched each event type at least once."""
    df = events.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["week_start"] = df["event_date"].dt.to_period("W-MON").dt.start_time

    weekly_users = df.groupby("week_start")["user_pseudo_id"].nunique().rename("active")
    feature = (
        df.groupby(["week_start", "event_name"])["user_pseudo_id"]
        .nunique()
        .reset_index(name="adopters")
    )
    feature = feature.merge(weekly_users, on="week_start", how="left")
    feature["adoption_rate"] = feature["adopters"] / feature["active"]
    return feature


def revenue_per_user(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    df["purchase_revenue"] = df["purchase_revenue"].fillna(0)
    return (
        df.groupby("user_pseudo_id", as_index=False)["purchase_revenue"]
        .sum()
        .rename(columns={"purchase_revenue": "revenue"})
    )
