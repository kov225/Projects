from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class TelemetryRecord:
    """A record representing a single search engine user interaction.

    Each field maps directly to the synthetic telemetry produced in the
    data generation pipeline. This schema ensures consistency across
    the experimentation and metric calculation modules.
    """
    query_id: str
    user_id: str
    treatment: int  # 0 for control, 1 for treatment
    session_date: datetime
    clicked: int  # 0 or 1
    dwell_time_seconds: float
    reformulated: int  # 0 or 1
    abandoned: int  # 0 or 1
    pre_experiment_engagement: float
    week_number: int
    user_segment: Literal["power_user", "casual_user", "new_user"]
