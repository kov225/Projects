from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset


@dataclass(frozen=True)
class MSMARCOConfig:
    split: str = "train"
    cache_dir_env: str = "MSMARCO_CACHE_DIR"
    max_rows: int = 200_000
    seed: int = 13


def download_msmarco_passage(cfg: MSMARCOConfig = MSMARCOConfig()) -> pd.DataFrame:
    """
    Downloads the MSMARCO passage ranking dataset from Hugging Face and returns a
    compact table that can be merged onto telemetry by query bucket or query_id.

    The goal is not to replicate a full ranking pipeline but to bring real query
    complexity signals into the synthetic world, which makes segment analyses and
    stratification exercises more realistic.
    """
    cache_dir = os.getenv(cfg.cache_dir_env, None)
    ds = load_dataset("microsoft/ms_marco", "passage", split=cfg.split, cache_dir=cache_dir)

    df = ds.to_pandas()
    keep = []
    for col in ["query", "query_id"]:
        if col in df.columns:
            keep.append(col)
    for col in ["is_selected", "label", "relevance", "passage_label"]:
        if col in df.columns:
            keep.append(col)
    if "passage" in df.columns:
        keep.append("passage")

    df = df[keep].copy()
    if "query_id" not in df.columns and "query" in df.columns:
        df["query_id"] = df["query"].astype(str).map(lambda s: f"ms_{abs(hash(s)) % 10**9:09d}")

    df = df.sample(n=min(cfg.max_rows, len(df)), random_state=cfg.seed).reset_index(drop=True)
    df["query_length"] = df["query"].astype(str).str.len()
    df["query_num_tokens"] = df["query"].astype(str).str.split().map(len)
    return df


def write_msmarco_features(out_path: Optional[str] = None) -> Path:
    df = download_msmarco_passage()
    out_dir = Path(__file__).resolve().parent / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(out_path) if out_path else (out_dir / "msmarco_query_features.parquet")
    df.to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    print(str(write_msmarco_features()))

