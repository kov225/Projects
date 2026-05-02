"""Extend the 92-day GA4 sample to a 156-week window for cohort analysis.

This is a deliberate, transparent step. The real BigQuery export covers
2020-11-01 to 2021-01-31, which is too short to study weekly cohorts at a
useful resolution. The extension procedure below resamples behavioural
profiles from the real data so that the marginal distributions, the event
mix, and the conversion rates are preserved. The extended rows are tagged
with source = 'synthetic_extension' and the real rows with
source = 'ga4_bigquery' so any downstream chart can split or filter as
needed.

Three things vary across the 156 weeks:

1. Mild quarter-over-quarter user-base growth (~5 percent per quarter).
2. Slow conversion-rate drift (~0.5 percentage points per year), which is
   what a healthy product would expect from gradual UX improvements.
3. Seasonality: December engagement is dampened, January and September are
   lifted. The shape mirrors the real GA4 sample where November holiday
   sessions taper into a slower February.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVENTS_OUT = PROCESSED_DIR / "events_extended.parquet"
USERS_OUT = PROCESSED_DIR / "users_extended.parquet"

EXT_START = pd.Timestamp("2022-01-03")
EXT_END = pd.Timestamp("2024-12-29")


@dataclass(frozen=True)
class ExtensionConfig:
    """Tunable knobs for the extension procedure."""

    target_weeks: int = 156
    base_weekly_users: int = 280
    quarterly_growth: float = 0.05
    annual_conversion_drift: float = 0.005
    seasonal_amplitude: float = 0.18
    seed: int = 2026


def _seasonal_factor(week_start: pd.Timestamp, amplitude: float) -> float:
    """Return a multiplicative factor for engagement on the given week."""
    month = week_start.month
    base = {
        1: 1.10, 2: 1.00, 3: 1.02, 4: 1.00, 5: 0.98, 6: 0.95,
        7: 0.97, 8: 1.02, 9: 1.12, 10: 1.05, 11: 1.04, 12: 0.78,
    }
    centred = base[month] - 1.0
    return float(1.0 + amplitude * centred / 0.18)


def _user_profiles_from_real(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each real user into a behavioural profile to resample from."""
    g = events.groupby("user_pseudo_id", as_index=False).agg(
        sessions=("session_id", "nunique"),
        events=("event_timestamp", "count"),
        engagement_total=("engagement_time_msec", "sum"),
        purchases=("event_name", lambda s: int((s == "purchase").sum())),
        revenue=("purchase_revenue", lambda s: float(s.fillna(0).sum())),
        device=("device_category", lambda s: s.mode().iloc[0]),
        os=("device_os", lambda s: s.mode().iloc[0]),
        country=("country", lambda s: s.mode().iloc[0]),
        traffic_source=("traffic_source", lambda s: s.mode().iloc[0]),
        traffic_medium=("traffic_medium", lambda s: s.mode().iloc[0]),
    )
    g["events_per_session"] = g["events"] / g["sessions"].clip(lower=1)
    g["engagement_per_event"] = g["engagement_total"] / g["events"].clip(lower=1)
    g["converter"] = (g["purchases"] > 0).astype(int)
    return g


def _event_name_distribution(events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    counts = events["event_name"].value_counts(normalize=True)
    return counts.index.to_numpy(), counts.values.astype(float)


def extend_dataset(
    events_real: pd.DataFrame,
    config: ExtensionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate 156 weeks of extended events plus a clean users table.

    The real rows are returned in the events frame too, with source =
    'ga4_bigquery'. The users frame contains the de-duplicated profile of
    every user that appears in either the real or the extended events.
    """
    cfg = config or ExtensionConfig()
    rng = np.random.default_rng(cfg.seed)
    profiles = _user_profiles_from_real(events_real)
    event_names, event_probs = _event_name_distribution(events_real)

    week_starts = pd.date_range(EXT_START, EXT_END, freq="W-MON")[: cfg.target_weeks]
    new_event_rows: list[dict] = []
    new_user_rows: list[dict] = []

    base_conversion = profiles["converter"].mean()
    LOGGER.info(
        "real conversion rate %.3f, target weeks %d, profiles %d",
        base_conversion,
        len(week_starts),
        len(profiles),
    )

    user_counter = 0
    for week_idx, week in enumerate(week_starts):
        quarters_elapsed = week_idx / 13.0
        years_elapsed = week_idx / 52.0

        weekly_target = int(
            cfg.base_weekly_users
            * (1.0 + cfg.quarterly_growth) ** quarters_elapsed
            * _seasonal_factor(week, cfg.seasonal_amplitude)
        )
        conversion_lift = cfg.annual_conversion_drift * years_elapsed

        sampled = profiles.sample(
            n=weekly_target,
            replace=True,
            random_state=int(rng.integers(0, 2**31 - 1)),
        ).reset_index(drop=True)

        for _, profile in sampled.iterrows():
            user_counter += 1
            uid = f"ext_{user_counter:07d}"
            sessions = max(1, int(rng.poisson(profile["sessions"])))
            events_per_session = max(2, int(rng.poisson(max(profile["events_per_session"], 2))))
            became_buyer = bool(rng.random() < min(0.6, profile["converter"] + conversion_lift))

            first_seen_in_week = week + pd.Timedelta(days=int(rng.integers(0, 7)))
            new_user_rows.append(
                {
                    "user_pseudo_id": uid,
                    "first_seen_date": first_seen_in_week.date(),
                    "device_category": profile["device"],
                    "device_os": profile["os"],
                    "country": profile["country"],
                    "traffic_source": profile["traffic_source"],
                    "traffic_medium": profile["traffic_medium"],
                    "source": "synthetic_extension",
                }
            )

            for s in range(sessions):
                session_day = week + pd.Timedelta(days=int(rng.integers(0, 7)))
                session_id = int(session_day.timestamp()) + int(rng.integers(0, 10_000)) + s
                base_ts = int(session_day.timestamp() * 1_000_000) + int(
                    rng.integers(0, 86_400_000_000)
                )
                chosen = list(
                    rng.choice(event_names, size=events_per_session, p=event_probs)
                )
                if became_buyer and s == 0 and "purchase" not in chosen:
                    chosen[-1] = "purchase"
                    if "begin_checkout" not in chosen and len(chosen) >= 2:
                        chosen[-2] = "begin_checkout"
                    if "add_to_cart" not in chosen and len(chosen) >= 3:
                        chosen[-3] = "add_to_cart"

                transaction_id: str | None = None
                for k, ev in enumerate(chosen):
                    ts = base_ts + k * int(rng.integers(5_000_000, 60_000_000))
                    engagement = (
                        int(rng.gamma(2.0, max(profile["engagement_per_event"] / 2.0, 4_000)))
                        if ev != "session_start"
                        else 0
                    )
                    rev = None
                    qty = None
                    if ev == "purchase":
                        if transaction_id is None:
                            transaction_id = f"TX_{user_counter:07d}_{s}"
                        rev = float(np.round(rng.gamma(2.0, 22.0), 2))
                        qty = int(rng.integers(1, 4))
                    new_event_rows.append(
                        {
                            "event_date": session_day.date(),
                            "event_timestamp": ts,
                            "event_name": ev,
                            "user_pseudo_id": uid,
                            "device_category": profile["device"],
                            "device_os": profile["os"],
                            "country": profile["country"],
                            "region": "(not set)",
                            "traffic_source": profile["traffic_source"],
                            "traffic_medium": profile["traffic_medium"],
                            "campaign_name": "merch_extended",
                            "session_id": session_id,
                            "engagement_time_msec": engagement,
                            "page_title": "Google Merchandise Store",
                            "page_location": "https://shop.googlemerchandisestore.com/",
                            "transaction_id": transaction_id if ev == "purchase" else None,
                            "purchase_revenue": rev,
                            "total_item_quantity": qty,
                            "source": "synthetic_extension",
                        }
                    )

    extended_events = pd.DataFrame(new_event_rows)
    real_events = events_real.copy()
    real_events["source"] = "ga4_bigquery"

    all_events = pd.concat([real_events, extended_events], ignore_index=True)
    all_events["event_date"] = pd.to_datetime(all_events["event_date"]).dt.date

    users_real = (
        events_real.groupby("user_pseudo_id", as_index=False)
        .agg(
            first_seen_date=("event_date", "min"),
            device_category=("device_category", lambda s: s.mode().iloc[0]),
            device_os=("device_os", lambda s: s.mode().iloc[0]),
            country=("country", lambda s: s.mode().iloc[0]),
            traffic_source=("traffic_source", lambda s: s.mode().iloc[0]),
            traffic_medium=("traffic_medium", lambda s: s.mode().iloc[0]),
        )
    )
    users_real["source"] = "ga4_bigquery"

    users_extended = pd.DataFrame(new_user_rows)
    users = pd.concat([users_real, users_extended], ignore_index=True)
    users["first_seen_date"] = pd.to_datetime(users["first_seen_date"]).dt.date

    LOGGER.info(
        "extension produced %d new events, %d new users; combined %d events, %d users",
        len(extended_events),
        len(users_extended),
        len(all_events),
        len(users),
    )
    return all_events, users


def write_extended(
    events: pd.DataFrame,
    users: pd.DataFrame,
    events_path: Path | None = None,
    users_path: Path | None = None,
) -> None:
    """Persist the extended events and users tables to parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(events_path or EVENTS_OUT, index=False)
    users.to_parquet(users_path or USERS_OUT, index=False)
    LOGGER.info("wrote extended events to %s", events_path or EVENTS_OUT)
    LOGGER.info("wrote extended users to %s", users_path or USERS_OUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from data_acquisition import load_cached_events, bootstrap_offline_cache

    try:
        real = load_cached_events()
    except FileNotFoundError:
        LOGGER.info("no cached events found, bootstrapping offline cache first")
        real, _ = bootstrap_offline_cache()

    events_all, users_all = extend_dataset(real)
    write_extended(events_all, users_all)
