from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Literal, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from data.schema import DeviceEvent


ScenarioName = Literal["firmware_rollout", "regional_cdn_failure", "silent_metric_drift"]


@dataclass(frozen=True)
class GeneratorConfig:
    n_events: int = 10_000_000
    n_weeks: int = 90
    seed: int = 11
    start_timestamp: str = "2025-01-01T00:00:00Z"
    output_dir: str = "data/output"
    parquet_dirname: str = "device_events_parquet"
    manifest_filename: str = "validation_manifest.jsonl"
    partition_cols: tuple[str, str] = ("week_number", "device_type")
    batch_size: int = 250_000


DEVICE_TYPES = [
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

REGIONS = [
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

ERROR_CATEGORIES = [
    "playback_failure",
    "buffering",
    "authentication",
    "drm_error",
    "network_timeout",
    "codec_error",
    "bitrate_adaptation",
    "silent",
]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_start(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def _lognormal_from_median(rng: np.random.Generator, median: float, sigma: float, size: int) -> np.ndarray:
    mu = math.log(median)
    return rng.lognormal(mean=mu, sigma=sigma, size=size)


def _firmware_versions() -> list[str]:
    versions: list[str] = []
    for major in range(1, 7):
        for minor in range(0, 5):
            versions.append(f"v{major}.{minor}.0")
    return versions


def _sample_firmware(rng: np.random.Generator, device_type: np.ndarray) -> np.ndarray:
    """
    Models adoption such that newer major versions have higher probability, but laggards exist.
    """
    versions = _firmware_versions()
    majors = np.array([int(v.split(".")[0][1:]) for v in versions])
    base = np.exp(0.65 * (majors - majors.max()))
    base = base / base.sum()

    # Laggard devices have higher mass on older majors.
    lag_factor = np.select(
        [device_type == "smart_tv", device_type == "set_top_box", device_type == "projector"],
        [0.55, 0.45, 0.35],
        default=0.0,
    )
    p = np.tile(base, (len(device_type), 1))
    older = majors <= (majors.max() - 2)
    p[:, older] = p[:, older] * (1.0 + lag_factor[:, None] * 1.5)
    p = p / p.sum(axis=1, keepdims=True)

    idx = np.array([rng.choice(len(versions), p=row) for row in p], dtype=int)
    return np.array([versions[i] for i in idx], dtype=object)


def _region_latency_multiplier(region: np.ndarray) -> np.ndarray:
    return np.select(
        [
            np.char.startswith(region.astype(str), "north_america"),
            np.char.startswith(region.astype(str), "europe"),
            np.char.startswith(region.astype(str), "asia_pacific"),
            np.char.startswith(region.astype(str), "south_america"),
            np.char.startswith(region.astype(str), "oceania"),
        ],
        [0.95, 1.05, 1.25, 1.15, 1.12],
        default=1.10,
    ).astype(float)


def _base_error_probs(device_type: np.ndarray) -> pd.DataFrame:
    """
    Returns per-row probabilities over the seven non-silent categories, with silent being derived.
    """
    # Start from global baseline for non-silent errors (total error probability).
    total_err = np.select(
        [device_type == "mobile", device_type == "smart_tv", device_type == "gaming_console"],
        [0.055, 0.070, 0.060],
        default=0.045,
    ).astype(float)

    # Device-specific emphasis.
    drm_boost = (device_type == "smart_tv").astype(float) * 1.8
    net_boost = (device_type == "mobile").astype(float) * 1.9
    codec_boost = (device_type == "gaming_console").astype(float) * 1.7

    weights = pd.DataFrame(
        {
            "playback_failure": np.full(len(device_type), 1.0),
            "buffering": np.full(len(device_type), 1.2),
            "authentication": np.full(len(device_type), 0.7),
            "drm_error": np.full(len(device_type), 0.8) * (1.0 + drm_boost),
            "network_timeout": np.full(len(device_type), 0.9) * (1.0 + net_boost),
            "codec_error": np.full(len(device_type), 0.6) * (1.0 + codec_boost),
            "bitrate_adaptation": np.full(len(device_type), 0.9),
        }
    )
    wsum = weights.sum(axis=1).to_numpy(dtype=float)
    probs = weights.div(wsum, axis=0).mul(total_err, axis=0)
    return probs


def _inject_scenarios(
    df: pd.DataFrame,
    rng: np.random.Generator,
    manifest: list[dict],
) -> pd.DataFrame:
    """
    Injects the three required scenarios and writes a manifest describing expected slices.
    """
    df = df.copy()

    # Scenario 1: firmware rollout anomaly.
    s1_weeks = (df["week_number"] >= 30) & (df["week_number"] <= 38)
    s1_devices = df["device_type"].isin(["smart_tv", "streaming_stick"])
    s1_firmware = df["firmware_version"] == "v4.2.0"
    s1_mask = s1_weeks & s1_devices & s1_firmware
    if s1_mask.any():
        # 4x spike in playback_failure error rate via re-labeling from silent.
        idx = df.index[s1_mask & (df["error_category"] == "silent")]
        take = idx[rng.random(len(idx)) < 0.12]  # calibrated to yield visible spike
        df.loc[take, "error_category"] = "playback_failure"
        df.loc[take, "is_anomaly"] = True
    manifest.append(
        {
            "scenario": "firmware_rollout",
            "start_week": 30,
            "end_week": 38,
            "device_types": ["smart_tv", "streaming_stick"],
            "regions": "all",
            "firmware_version": "v4.2.0",
            "expected_effect": "4x spike in playback_failure errors",
        }
    )

    # Scenario 2: regional CDN failure.
    s2_weeks = (df["week_number"] >= 55) & (df["week_number"] <= 62)
    s2_regions = df["region"].astype(str).str.startswith("asia_pacific")
    s2_mask = s2_weeks & s2_regions
    if s2_mask.any():
        df.loc[s2_mask, "request_latency_ms"] *= 3.0
        # 5x increase in network_timeout by re-labeling a fraction of silent.
        idx = df.index[s2_mask & (df["error_category"] == "silent")]
        take = idx[rng.random(len(idx)) < 0.10]
        df.loc[take, "error_category"] = "network_timeout"
        df.loc[take, "is_anomaly"] = True
    manifest.append(
        {
            "scenario": "regional_cdn_failure",
            "start_week": 55,
            "end_week": 62,
            "device_types": "all",
            "regions": "asia_pacific_*",
            "firmware_version": "all",
            "expected_effect": "3x request latency and 5x network_timeout errors",
        }
    )

    # Scenario 3: silent metric drift (data collection failure).
    s3_start_week = 70
    s3_weeks = df["week_number"] >= s3_start_week
    s3_device = df["device_type"] == "smart_monitor"
    s3_region = df["region"] == "europe_west"
    s3_mask = s3_weeks & s3_device & s3_region
    if s3_mask.any():
        # Gradually drop buffering to near zero over 15 weeks by converting buffering -> silent.
        week = df.loc[s3_mask, "week_number"].to_numpy(dtype=float)
        t = np.clip((week - s3_start_week) / 15.0, 0.0, 1.0)
        drop_prob = 0.95 * t
        buf_idx = df.index[s3_mask & (df["error_category"] == "buffering")]
        # Map each buffering row to its drop probability via week_number lookup.
        drop_p = drop_prob[(df.loc[buf_idx, "week_number"].to_numpy(dtype=float) - s3_start_week).clip(0, 15).astype(int)]
        keep = rng.random(len(buf_idx)) < drop_p
        df.loc[buf_idx[keep], "error_category"] = "silent"
        df.loc[buf_idx[keep], "is_anomaly"] = True
    manifest.append(
        {
            "scenario": "silent_metric_drift",
            "start_week": 70,
            "end_week": 90,
            "device_types": ["smart_monitor"],
            "regions": ["europe_west"],
            "firmware_version": "all",
            "expected_effect": "buffering errors drop to near zero over 15 weeks",
            "detection_owner": "data_quality",
        }
    )
    return df


def generate_events(cfg: GeneratorConfig) -> tuple[Path, Path]:
    rng = np.random.default_rng(cfg.seed)

    out_root = Path(cfg.output_dir)
    parquet_root = out_root / cfg.parquet_dirname
    _ensure_dir(parquet_root)

    manifest_path = out_root / cfg.manifest_filename
    if manifest_path.exists():
        manifest_path.unlink()

    start_ts = _parse_start(cfg.start_timestamp)

    all_manifest: list[dict] = []
    written = 0
    while written < cfg.n_events:
        n = min(cfg.batch_size, cfg.n_events - written)

        week_number = rng.integers(1, cfg.n_weeks + 1, size=n, endpoint=False).astype(int)
        # Uniform timestamps within the week; hours granularity supports hourly rollups later.
        ts = np.array(
            [start_ts + timedelta(days=int((w - 1) * 1), hours=int(h)) for w, h in zip(week_number, rng.integers(0, 24, size=n))],
            dtype=object,
        )

        device_type = rng.choice(DEVICE_TYPES, size=n, p=[0.18, 0.24, 0.08, 0.06, 0.08, 0.07, 0.10, 0.12, 0.05, 0.02])
        region = rng.choice(REGIONS, size=n)
        firmware = _sample_firmware(rng, device_type)

        latency = _lognormal_from_median(rng, median=120.0, sigma=0.60, size=n) * _region_latency_multiplier(region)
        latency = np.clip(latency, 5.0, 2000.0).astype(float)

        session = _lognormal_from_median(rng, median=25.0 * 60.0, sigma=0.55, size=n)
        session = np.clip(session, 30.0, 6.0 * 3600.0).astype(float)

        bytes_transferred = (session * rng.normal(1.8e6, 3.5e5, size=n)).clip(1e5, 5e9).astype(np.int64)

        probs = _base_error_probs(device_type)
        # Sample one category by drawing non-silent first, otherwise silent.
        u = rng.random(n)
        total_err = probs.sum(axis=1).to_numpy(dtype=float)
        is_err = u < total_err
        error = np.full(n, "silent", dtype=object)
        if is_err.any():
            p_rows = probs.loc[is_err].to_numpy(dtype=float)
            p_rows = p_rows / p_rows.sum(axis=1, keepdims=True)
            cats = probs.columns.to_list()
            choices = np.array([rng.choice(len(cats), p=row) for row in p_rows], dtype=int)
            error[is_err] = np.array([cats[i] for i in choices], dtype=object)

        df = pd.DataFrame(
            {
                "event_id": [str(uuid4()) for _ in range(n)],
                "device_id": [f"d{int(x):09d}" for x in rng.integers(0, 2_500_000, size=n)],
                "device_type": device_type.astype(str),
                "firmware_version": firmware.astype(str),
                "region": region.astype(str),
                "error_category": error.astype(str),
                "request_latency_ms": latency,
                "bytes_transferred": bytes_transferred,
                "session_duration_seconds": session,
                "timestamp": pd.to_datetime(ts, utc=True),
                "week_number": week_number,
                "is_anomaly": np.zeros(n, dtype=bool),
            }
        )

        df = _inject_scenarios(df, rng=rng, manifest=all_manifest)

        # Partitioned write.
        df.to_parquet(parquet_root, index=False, partition_cols=list(cfg.partition_cols), engine="pyarrow")

        written += n

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in all_manifest:
            f.write(json.dumps(row) + "\n")

    return parquet_root, manifest_path


def _cfg_from_env() -> GeneratorConfig:
    n = int(os.getenv("N_EVENTS", "10000000"))
    batch = int(os.getenv("BATCH_SIZE", "250000"))
    out = os.getenv("OUTPUT_DIR", "data/output")
    return GeneratorConfig(n_events=n, batch_size=batch, output_dir=out)


if __name__ == "__main__":
    cfg = _cfg_from_env()
    parquet_root, manifest_path = generate_events(cfg)
    print(str(parquet_root))
    print(str(manifest_path))

