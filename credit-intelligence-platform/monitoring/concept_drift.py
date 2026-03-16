"""
concept_drift.py

Tracks model AUC on a rolling labeled holdout set. Covariate drift can be
caught by PSI/KS, but concept drift : where the relationship between features
and labels changes : won't show up in input distributions at all. The only
way to detect it is to actually evaluate the model on recently labeled data.

We maintain a holdout set of 500 records that gets a fraction of actual
outcomes filled in as they resolve (charge-off typically happens 60-180 days
after origination in LendingClub data). For simulation purposes the drift
scenarios in kafka_producer.py inject the concept drift signal immediately.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import psycopg2
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

load_dotenv()
logger = logging.getLogger(__name__)

AUC_DROP_THRESHOLD = float(os.environ.get("AUC_DROP_THRESHOLD", "0.03"))
HOLDOUT_SIZE = int(os.environ.get("HOLDOUT_SIZE", "500"))


def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "credit_intelligence"),
        user=os.environ.get("POSTGRES_USER", "credit_app"),
        password=os.environ.get("POSTGRES_PASSWORD", "changeme"),
    )


def ensure_concept_drift_tables() -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS concept_drift_reports (
                    id              SERIAL PRIMARY KEY,
                    checked_at      TIMESTAMPTZ DEFAULT NOW(),
                    holdout_size    INT,
                    current_auc     FLOAT,
                    baseline_auc    FLOAT,
                    auc_drop        FLOAT,
                    drift_detected  BOOLEAN,
                    model_version   TEXT
                );
                CREATE TABLE IF NOT EXISTS baseline_auc (
                    model_version TEXT PRIMARY KEY,
                    auc_value     FLOAT,
                    set_at        TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            conn.commit()
    finally:
        conn.close()


def set_baseline_auc(model_version: str, auc: float) -> None:
    """Record the baseline AUC for a newly promoted model.
    
    Called by the retraining trigger after a successful promotion. The baseline
    is the AUC on the held-out test set at promotion time, anchored to the
    moment the model went live.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO baseline_auc (model_version, auc_value)
                VALUES (%s, %s)
                ON CONFLICT (model_version) DO UPDATE SET auc_value = EXCLUDED.auc_value
            """, (model_version, auc))
            conn.commit()
        logger.info(f"Baseline AUC for {model_version}: {auc:.4f}")
    finally:
        conn.close()


def get_baseline_auc(model_version: str) -> float | None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT auc_value FROM baseline_auc WHERE model_version = %s",
                (model_version,)
            )
            row = cur.fetchone()
        return float(row[0]) if row else None
    finally:
        conn.close()


def get_labeled_holdout(model_version: str, holdout_size: int = HOLDOUT_SIZE):
    """Fetch recent predictions that have actual outcomes filled in.
    
    In the simulation, we treat the `default` field passed through Kafka as
    the ground truth label. In a real system this would come from your loan
    servicing platform months after origination.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT default_prob, actual_default
                FROM predictions
                WHERE model_version = %s
                  AND actual_default IS NOT NULL
                ORDER BY created_at DESC
                LIMIT %s
            """, (model_version, holdout_size))
            rows = cur.fetchall()
        return rows
    finally:
        conn.close()


def run_concept_drift_check(model_version: str) -> dict[str, Any]:
    """Evaluate rolling AUC on labeled holdout and compare to baseline.
    
    If the rolling AUC drops more than AUC_DROP_THRESHOLD below the baseline,
    we flag concept drift. The retraining trigger polls this function.
    """
    ensure_concept_drift_tables()
    rows = get_labeled_holdout(model_version)

    if len(rows) < 50:
        logger.warning(f"Only {len(rows)} labeled holdout records : skipping concept drift check")
        return {"skipped": True, "reason": "insufficient_labeled_data"}

    probs = np.array([r[0] for r in rows])
    labels = np.array([r[1] for r in rows])

    if len(np.unique(labels)) < 2:
        return {"skipped": True, "reason": "only_one_class_in_holdout"}

    current_auc = float(roc_auc_score(labels, probs))
    baseline_auc = get_baseline_auc(model_version)

    if baseline_auc is None:
        set_baseline_auc(model_version, current_auc)
        return {"skipped": True, "reason": "baseline_initialized", "baseline_auc": current_auc}

    drop = baseline_auc - current_auc
    drift_detected = drop > AUC_DROP_THRESHOLD

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO concept_drift_reports
                    (holdout_size, current_auc, baseline_auc, auc_drop, drift_detected, model_version)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (len(rows), current_auc, baseline_auc, drop, drift_detected, model_version))
            conn.commit()
    finally:
        conn.close()

    if drift_detected:
        logger.warning(
            f"CONCEPT DRIFT: AUC dropped {drop:.4f} (current={current_auc:.4f}, "
            f"baseline={baseline_auc:.4f}, threshold={AUC_DROP_THRESHOLD})"
        )
    else:
        logger.info(f"Concept drift check OK: AUC={current_auc:.4f}, drop={drop:.4f}")

    return {
        "current_auc": current_auc,
        "baseline_auc": baseline_auc,
        "auc_drop": drop,
        "drift_detected": drift_detected,
        "holdout_size": len(rows),
        "model_version": model_version,
    }
