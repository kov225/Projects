"""Common fixtures and path wiring for pytest."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def tiny_users() -> pd.DataFrame:
    """Three users with sensible metadata."""
    return pd.DataFrame(
        {
            "user_pseudo_id": ["u1", "u2", "u3"],
            "first_seen_date": [date(2024, 1, 8), date(2024, 1, 15), date(2024, 2, 1)],
            "device_category": ["mobile", "desktop", "mobile"],
            "device_os": ["iOS", "Windows", "Android"],
            "country": ["United States", "Canada", "India"],
            "traffic_source": ["google", "(direct)", "newsletter"],
            "traffic_medium": ["organic", "(none)", "email"],
            "source": ["ga4_bigquery", "ga4_bigquery", "synthetic_extension"],
        }
    )


@pytest.fixture()
def tiny_events(tiny_users: pd.DataFrame) -> pd.DataFrame:
    """Six events spread across the three users so cohort logic has shape."""
    rows = [
        {"user_pseudo_id": "u1", "event_date": date(2024, 1, 8), "event_timestamp": 1, "event_name": "session_start", "session_id": 100, "engagement_time_msec": 0, "purchase_revenue": None, "transaction_id": None, "total_item_quantity": None},
        {"user_pseudo_id": "u1", "event_date": date(2024, 1, 8), "event_timestamp": 2, "event_name": "view_item", "session_id": 100, "engagement_time_msec": 8000, "purchase_revenue": None, "transaction_id": None, "total_item_quantity": None},
        {"user_pseudo_id": "u1", "event_date": date(2024, 1, 15), "event_timestamp": 3, "event_name": "purchase", "session_id": 101, "engagement_time_msec": 12000, "purchase_revenue": 49.99, "transaction_id": "T1", "total_item_quantity": 2},
        {"user_pseudo_id": "u2", "event_date": date(2024, 1, 15), "event_timestamp": 4, "event_name": "view_item", "session_id": 200, "engagement_time_msec": 5000, "purchase_revenue": None, "transaction_id": None, "total_item_quantity": None},
        {"user_pseudo_id": "u2", "event_date": date(2024, 1, 22), "event_timestamp": 5, "event_name": "add_to_cart", "session_id": 201, "engagement_time_msec": 7000, "purchase_revenue": None, "transaction_id": None, "total_item_quantity": None},
        {"user_pseudo_id": "u3", "event_date": date(2024, 2, 1), "event_timestamp": 6, "event_name": "session_start", "session_id": 300, "engagement_time_msec": 0, "purchase_revenue": None, "transaction_id": None, "total_item_quantity": None},
    ]
    df = pd.DataFrame(rows)
    df["device_category"] = df["user_pseudo_id"].map(tiny_users.set_index("user_pseudo_id")["device_category"])
    df["device_os"] = df["user_pseudo_id"].map(tiny_users.set_index("user_pseudo_id")["device_os"])
    df["country"] = df["user_pseudo_id"].map(tiny_users.set_index("user_pseudo_id")["country"])
    df["region"] = "(not set)"
    df["traffic_source"] = df["user_pseudo_id"].map(tiny_users.set_index("user_pseudo_id")["traffic_source"])
    df["traffic_medium"] = df["user_pseudo_id"].map(tiny_users.set_index("user_pseudo_id")["traffic_medium"])
    df["campaign_name"] = "test"
    df["page_title"] = "Test"
    df["page_location"] = "https://example.com"
    df["source"] = df["user_pseudo_id"].map(tiny_users.set_index("user_pseudo_id")["source"])
    return df


@pytest.fixture()
def tiny_experiments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"experiment_id": "EXP_001", "experiment_name": "test_one", "feature_area": "checkout",
             "start_date": date(2024, 1, 1), "end_date": date(2024, 2, 1),
             "experiment_type": "a_b", "variant_count": 2,
             "hypothesis": "test", "designed_outcome": "clear_winner"},
        ]
    )


@pytest.fixture()
def tiny_assignments() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    user_ids = [f"u{i}" for i in range(60)]
    variants = rng.choice(["control", "variant_a"], size=60)
    return pd.DataFrame(
        {
            "user_pseudo_id": user_ids,
            "experiment_id": "EXP_001",
            "variant": variants,
            "assignment_date": [date(2024, 1, 5)] * 60,
        }
    )


@pytest.fixture()
def tiny_results(tiny_assignments: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rates = {"control": 0.10, "variant_a": 0.30}
    rows: list[dict] = []
    for _, r in tiny_assignments.iterrows():
        converted = bool(rng.random() < rates[r["variant"]])
        rows.append(
            {
                "user_pseudo_id": r["user_pseudo_id"],
                "experiment_id": r["experiment_id"],
                "variant": r["variant"],
                "converted": converted,
                "revenue_impact": 25.0 if converted else 0.0,
                "engagement_score": 60.0 if converted else 35.0,
            }
        )
    return pd.DataFrame(rows)
