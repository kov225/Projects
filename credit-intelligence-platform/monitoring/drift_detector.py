import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from scipy import stats
from scipy.stats import wasserstein_distance

# Researcher-grade monitoring configuration
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PSI_THRESHOLD = float(os.environ.get("PSI_THRESHOLD", "0.2"))
KS_PVALUE_THRESHOLD = float(os.environ.get("KS_PVALUE_THRESHOLD", "0.05"))
WINDOW_SIZE = int(os.environ.get("DRIFT_WINDOW_SIZE", "1000"))


def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "credit_intelligence"),
        user=os.environ.get("POSTGRES_USER", "credit_app"),
        password=os.environ.get("POSTGRES_PASSWORD", "changeme"),
    )


def ensure_drift_tables() -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS drift_reports (
                    id              SERIAL PRIMARY KEY,
                    checked_at      TIMESTAMPTZ DEFAULT NOW(),
                    window_size     INT,
                    drifted_features JSONB,
                    psi_scores      JSONB,
                    ks_pvalues      JSONB,
                    wasserstein_dist JSONB,
                    any_drifted     BOOLEAN
                );
                CREATE TABLE IF NOT EXISTS reference_features (
                    feature_name TEXT PRIMARY KEY,
                    distribution JSONB
                );
            """)
            conn.commit()
    finally:
        conn.close()


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Population Stability Index (PSI).
    
    PSI = sum((P_cur - P_ref) * ln(P_cur / P_ref))
    
    A PSI > 0.2 indicates significant distributional shift requiring attention.
    """
    eps = 1e-6
    # Robust binning using reference percentiles
    bin_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / (ref_counts.sum() + eps)
    cur_pct = cur_counts / (cur_counts.sum() + eps)

    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def load_reference_features(feature_cols: list[str]) -> pd.DataFrame | None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT feature_name, distribution FROM reference_features")
            rows = cur.fetchall()
        if not rows:
            return None
        data = {}
        for feature_name, dist_json in rows:
            if feature_name in feature_cols:
                data[feature_name] = json.loads(dist_json) if isinstance(dist_json, str) else dist_json
        return pd.DataFrame(data)
    finally:
        conn.close()


def save_reference_features(ref_df: pd.DataFrame) -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for col in ref_df.select_dtypes(include=[np.number]).columns:
                values = ref_df[col].dropna().tolist()
                cur.execute("""
                    INSERT INTO reference_features (feature_name, distribution)
                    VALUES (%s, %s)
                    ON CONFLICT (feature_name) DO NOTHING
                """, (col, json.dumps(values)))
        conn.commit()
        logger.info(f"Reference distribution anchored for {len(ref_df.columns)} features.")
    finally:
        conn.close()


def load_recent_predictions(window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT features FROM predictions
                ORDER BY created_at DESC
                LIMIT %s
            """, (window_size,))
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        records = [json.loads(row[0]) if isinstance(row[0], str) else row[0] for row in rows]
        return pd.DataFrame(records)
    finally:
        conn.close()


def run_drift_detection(feature_cols: list[str] | None = None) -> dict[str, Any]:
    """
    Executes covariate drift detection suite.
    
    This run computes:
    1. Kolmogorov-Smirnov (KS) test for distribution overlap.
    2. Population Stability Index (PSI) for structural shifts.
    3. Wasserstein Distance for absolute drift magnitude.
    """
    ensure_drift_tables()
    current_df = load_recent_predictions(window_size=WINDOW_SIZE)

    if current_df.empty or len(current_df) < 100:
        logger.warning(f"Insufficient telemetry ({len(current_df)}) for robust drift detection.")
        return {"skipped": True, "reason": "insufficient_data"}

    ref_df = load_reference_features(feature_cols or list(current_df.columns))
    if ref_df is None:
        save_reference_features(current_df)
        return {"skipped": True, "reason": "reference_set_initialized"}

    numeric_cols = [c for c in current_df.select_dtypes(include=[np.number]).columns
                    if c in ref_df.columns]

    psi_scores, ks_pvalues, w_distances = {}, {}, {}
    drifted = []

    for col in numeric_cols:
        ref_vals = np.array(ref_df[col].dropna())
        cur_vals = current_df[col].dropna().values

        if len(ref_vals) < 50 or len(cur_vals) < 50:
            continue

        # 1. PSI (Stability)
        psi = compute_psi(ref_vals, cur_vals)
        # 2. KS Test (Significance)
        _, ks_p = stats.ks_2samp(ref_vals, cur_vals)
        # 3. Wasserstein (Magnitude)
        w_dist = wasserstein_distance(ref_vals, cur_vals)

        psi_scores[col] = round(psi, 4)
        ks_pvalues[col] = round(ks_p, 6)
        w_distances[col] = round(float(w_dist), 6)

        if psi > PSI_THRESHOLD or ks_p < KS_PVALUE_THRESHOLD:
            drifted.append(col)
            logger.warning(f"Drift Detected: {col} | PSI: {psi:.3f} | KS p: {ks_p:.4f} | W-Dist: {w_dist:.4f}")

    any_drifted = len(drifted) > 0
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO drift_reports 
                    (window_size, drifted_features, psi_scores, ks_pvalues, wasserstein_dist, any_drifted)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                len(current_df),
                json.dumps(drifted),
                json.dumps(psi_scores),
                json.dumps(ks_pvalues),
                json.dumps(w_distances),
                any_drifted,
            ))
            conn.commit()
    finally:
        conn.close()

    logger.info(f"Drift scan finalized. Features flagged: {len(drifted)}")
    return {
        "drifted_features": drifted,
        "psi_scores": psi_scores,
        "ks_pvalues": ks_pvalues,
        "wasserstein_dist": w_distances,
        "any_drifted": any_drifted,
        "window_size": len(current_df),
    }
