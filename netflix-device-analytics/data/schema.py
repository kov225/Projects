from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


DeviceType = Literal[
    "smart_tv",
    "mobile",
    "tablet",
    "gaming_console",
    "laptop",
    "desktop",
    "streaming_stick",
    "set_top_box",
    "smart_monitor",
    "projector",
]


Region = Literal[
    "north_america_us_east",
    "north_america_us_west",
    "north_america_canada",
    "north_america_mexico",
    "europe_uk_ireland",
    "europe_west",
    "europe_nordics",
    "europe_south",
    "asia_pacific_japan",
    "asia_pacific_korea",
    "asia_pacific_singapore",
    "asia_pacific_india",
    "asia_pacific_australia_nz",
    "south_america_brazil",
    "south_america_argentina",
    "south_america_chile",
    "oceania_australia",
    "oceania_new_zealand",
    "middle_east_uae",
    "africa_south_africa",
]


ErrorCategory = Literal[
    "playback_failure",
    "buffering",
    "authentication",
    "drm_error",
    "network_timeout",
    "codec_error",
    "bitrate_adaptation",
    "silent",
]


@dataclass(frozen=True)
class DeviceEvent:
    event_id: str
    device_id: str
    device_type: DeviceType
    firmware_version: str
    region: Region
    error_category: ErrorCategory
    request_latency_ms: float
    bytes_transferred: int
    session_duration_seconds: float
    timestamp: datetime
    week_number: int
    is_anomaly: bool


def firmware_version(major: int, minor: int, patch: int) -> str:
    return f"v{major}.{minor}.{patch}"

