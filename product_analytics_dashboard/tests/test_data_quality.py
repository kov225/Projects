"""Tests for the data_quality module."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src import data_quality as dq


def test_referential_integrity_passes_when_users_match(tiny_events, tiny_users):
    passed, msg = dq.check_referential_integrity(tiny_events, tiny_users)
    assert passed, msg
    assert "OK" in msg


def test_referential_integrity_fails_for_orphan_event():
    events = pd.DataFrame({"user_pseudo_id": ["a", "b"]})
    users = pd.DataFrame({"user_pseudo_id": ["a"]})
    passed, msg = dq.check_referential_integrity(events, users)
    assert not passed
    assert "1" in msg


def test_referential_integrity_handles_missing_columns():
    passed, msg = dq.check_referential_integrity(
        pd.DataFrame({"x": [1]}), pd.DataFrame({"y": [1]})
    )
    assert not passed
    assert "missing" in msg


def test_check_date_ranges_passes_inside_window(tiny_events):
    passed, msg = dq.check_date_ranges(
        tiny_events, "event_date", min_date=date(2023, 1, 1), max_date=date(2024, 12, 31)
    )
    assert passed, msg


def test_check_date_ranges_fails_when_outside_window(tiny_events):
    passed, _ = dq.check_date_ranges(
        tiny_events, "event_date", min_date=date(2025, 1, 1)
    )
    assert not passed


def test_check_date_ranges_flags_unparseable_values():
    df = pd.DataFrame({"event_date": ["2024-01-01", "not-a-date"]})
    passed, msg = dq.check_date_ranges(df)
    assert not passed
    assert "unparseable" in msg


def test_check_duplicates_finds_dupes():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    passed, msg = dq.check_duplicates(df, ["a", "b"])
    assert not passed
    assert "duplicate" in msg


def test_check_duplicates_passes_when_clean(tiny_users):
    passed, _ = dq.check_duplicates(tiny_users, ["user_pseudo_id"])
    assert passed


def test_check_null_rates_respects_threshold():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [None, None, 3, 4, 5]})
    passed, msg = dq.check_null_rates(df, threshold=0.10)
    assert not passed
    assert "above" in msg


def test_check_null_rates_can_ignore_columns():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [None, None, 3, 4, 5]})
    passed, _ = dq.check_null_rates(df, threshold=0.10, ignore_columns=["b"])
    assert passed


def test_validate_experiment_assignments_happy_path(tiny_experiments, tiny_assignments):
    tiny_experiments_with_count = tiny_experiments.copy()
    tiny_experiments_with_count["variant_count"] = 2
    passed, msg = dq.validate_experiment_assignments(tiny_assignments, tiny_experiments_with_count)
    assert passed, msg


def test_validate_experiment_assignments_flags_duplicate_user(tiny_experiments, tiny_assignments):
    bad = pd.concat([tiny_assignments, tiny_assignments.head(1)], ignore_index=True)
    passed, msg = dq.validate_experiment_assignments(bad, tiny_experiments)
    assert not passed
    assert "more than once" in msg


def test_run_all_checks_returns_dataframe_with_required_columns(
    tiny_events, tiny_users
):
    report = dq.run_all_checks(tiny_users, tiny_events)
    assert {"check", "passed", "message"}.issubset(report.columns)
    assert len(report) >= 5


def test_check_null_rates_handles_empty_dataframe():
    passed, _ = dq.check_null_rates(pd.DataFrame({"a": []}))
    assert passed
