"""Data quality checks used by the acquisition and extension pipelines.

Each public function returns a (passed, message) tuple so the caller can log
the message and decide whether to abort. Keeping the contract uniform makes
it straightforward to chain checks in a notebook or in pytest.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd


CheckResult = tuple[bool, str]


def check_referential_integrity(
    events: pd.DataFrame,
    users: pd.DataFrame,
    event_user_col: str = "user_pseudo_id",
    users_user_col: str = "user_pseudo_id",
) -> CheckResult:
    """Every user_id in events must also appear in the users table."""
    if event_user_col not in events.columns or users_user_col not in users.columns:
        return False, "missing user id columns on one of the frames"

    event_users = set(events[event_user_col].dropna().unique())
    known_users = set(users[users_user_col].dropna().unique())
    orphans = event_users - known_users

    if orphans:
        sample = list(orphans)[:3]
        return False, f"{len(orphans)} event user ids not in users table (sample: {sample})"
    return True, f"referential integrity OK across {len(event_users)} users"


def check_date_ranges(
    df: pd.DataFrame,
    date_col: str = "event_date",
    min_date: date | None = None,
    max_date: date | None = None,
) -> CheckResult:
    """Ensure dates are parseable and fall inside an expected window."""
    if date_col not in df.columns:
        return False, f"column {date_col} not found"

    parsed = pd.to_datetime(df[date_col], errors="coerce")
    n_bad = int(parsed.isna().sum())
    if n_bad:
        return False, f"{n_bad} rows have unparseable {date_col}"

    if min_date is not None and parsed.dt.date.min() < min_date:
        return False, f"dates earlier than {min_date} found"
    if max_date is not None and parsed.dt.date.max() > max_date:
        return False, f"dates later than {max_date} found"

    return True, f"{date_col} ranges from {parsed.dt.date.min()} to {parsed.dt.date.max()}"


def check_duplicates(df: pd.DataFrame, key_columns: Iterable[str]) -> CheckResult:
    """Flag any duplicate combinations of the given key columns."""
    keys = list(key_columns)
    missing = [k for k in keys if k not in df.columns]
    if missing:
        return False, f"key columns missing: {missing}"

    dup_count = int(df.duplicated(subset=keys).sum())
    if dup_count:
        return False, f"{dup_count} duplicate rows on {keys}"
    return True, f"no duplicates on {keys}"


def check_null_rates(
    df: pd.DataFrame,
    threshold: float = 0.05,
    ignore_columns: Iterable[str] | None = None,
) -> CheckResult:
    """Fail if any column has a null rate above the threshold."""
    ignore = set(ignore_columns or [])
    rates = df.drop(columns=list(ignore & set(df.columns)), errors="ignore").isna().mean()
    offenders = rates[rates > threshold]
    if len(offenders):
        worst = offenders.sort_values(ascending=False).head(3).to_dict()
        return False, f"columns above {threshold:.0%} null rate: {worst}"
    return True, f"all column null rates below {threshold:.0%}"


def validate_experiment_assignments(
    assignments: pd.DataFrame,
    experiments: pd.DataFrame,
) -> CheckResult:
    """Every assignment must reference a known experiment with a valid variant."""
    required_a = {"user_pseudo_id", "experiment_id", "variant", "assignment_date"}
    required_e = {"experiment_id", "variant_count"}
    if not required_a.issubset(assignments.columns):
        return False, f"assignments missing columns: {required_a - set(assignments.columns)}"
    if not required_e.issubset(experiments.columns):
        return False, f"experiments missing columns: {required_e - set(experiments.columns)}"

    known = experiments.set_index("experiment_id")["variant_count"].to_dict()
    unknown = set(assignments["experiment_id"]) - set(known)
    if unknown:
        return False, f"{len(unknown)} assignments reference unknown experiments"

    valid_variants = {"control", "variant_a", "variant_b", "variant_c"}
    bad = set(assignments["variant"].unique()) - valid_variants
    if bad:
        return False, f"unrecognised variants: {bad}"

    one_per_user = assignments.duplicated(subset=["user_pseudo_id", "experiment_id"]).sum()
    if one_per_user:
        return False, f"{int(one_per_user)} users assigned more than once to the same experiment"

    return True, f"validated {len(assignments):,} assignments across {len(known)} experiments"


def run_all_checks(
    users: pd.DataFrame,
    events: pd.DataFrame,
    experiments: pd.DataFrame | None = None,
    assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the standard battery of checks and return a tidy report."""
    rows: list[dict] = []

    def record(name: str, result: CheckResult) -> None:
        rows.append({"check": name, "passed": result[0], "message": result[1]})

    record("referential_integrity_events_users", check_referential_integrity(events, users))
    record("event_date_range", check_date_ranges(events, "event_date"))
    record("user_signup_date_range", check_date_ranges(users, "first_seen_date"))
    record(
        "events_no_duplicates",
        check_duplicates(events, ["user_pseudo_id", "event_timestamp", "event_name"]),
    )
    record("users_no_duplicates", check_duplicates(users, ["user_pseudo_id"]))
    record(
        "events_null_rates",
        check_null_rates(
            events,
            threshold=0.30,
            ignore_columns=["transaction_id", "purchase_revenue", "total_item_quantity"],
        ),
    )
    record("users_null_rates", check_null_rates(users, threshold=0.10))

    if experiments is not None and assignments is not None:
        record("experiment_assignments", validate_experiment_assignments(assignments, experiments))

    return pd.DataFrame(rows)
