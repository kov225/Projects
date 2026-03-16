"""
ab_router.py

Handles champion/challenger traffic splitting for A/B model testing.
The split percentage is read from CHALLENGER_TRAFFIC_PCT in the environment.
Each scoring request is logged to PostgreSQL with the model version that
served it, so we can compare champion vs. challenger performance offline.
"""

import logging
import os
import random
import uuid
from datetime import datetime, timezone
from typing import Any

import psycopg2
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "credit_intelligence"),
        user=os.environ.get("POSTGRES_USER", "credit_app"),
        password=os.environ.get("POSTGRES_PASSWORD", "changeme"),
    )


def ensure_predictions_table() -> None:
    """Create the predictions table if it doesn't exist yet.
    
    We store the full feature vector as JSONB so we can reconstruct any
    prediction's features for drift analysis or audit purposes without
    having to keep a separate copy. SHAP values go in a separate column.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    request_id      TEXT PRIMARY KEY,
                    applicant_id    TEXT NOT NULL,
                    model_version   TEXT NOT NULL,
                    model_role      TEXT NOT NULL,    -- 'champion' or 'challenger'
                    risk_score      FLOAT NOT NULL,
                    default_prob    FLOAT NOT NULL,
                    decision        TEXT NOT NULL,    -- 'approve' or 'deny'
                    decision_threshold FLOAT NOT NULL,
                    features        JSONB,
                    shap_values     JSONB,
                    actual_default  INT,             -- filled in later for evaluation
                    created_at      TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_predictions_applicant 
                    ON predictions(applicant_id);
                CREATE INDEX IF NOT EXISTS idx_predictions_created_at 
                    ON predictions(created_at);
                CREATE INDEX IF NOT EXISTS idx_predictions_model_version 
                    ON predictions(model_version);
            """)
            conn.commit()
    finally:
        conn.close()


def log_prediction(
    request_id: str,
    applicant_id: str,
    model_version: str,
    model_role: str,
    risk_score: float,
    default_prob: float,
    decision: str,
    decision_threshold: float,
    features: dict[str, Any],
    shap_values: dict[str, float] | None = None,
) -> None:
    """Persist a prediction record to PostgreSQL."""
    import json
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predictions (
                    request_id, applicant_id, model_version, model_role,
                    risk_score, default_prob, decision, decision_threshold,
                    features, shap_values
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (request_id) DO NOTHING
            """, (
                request_id, applicant_id, model_version, model_role,
                risk_score, default_prob, decision, decision_threshold,
                json.dumps(features, default=float),
                json.dumps(shap_values or {}, default=float),
            ))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to log prediction {request_id}: {e}")
    finally:
        conn.close()


class ABRouter:
    """Routes scoring requests to champion or challenger model based on traffic split.
    
    The router is stateless : each request independently draws a random number,
    which ensures the split is correct in expectation without needing sticky
    sessions or sequential assignment. At low traffic volumes there will be
    sampling variance in the actual split.
    """

    def __init__(self, challenger_pct: float | None = None) -> None:
        self.challenger_pct = challenger_pct or float(
            os.environ.get("CHALLENGER_TRAFFIC_PCT", "0.2")
        )
        logger.info(f"A/B router initialized: {self.challenger_pct:.0%} challenger traffic")

    def route(self) -> str:
        """Return 'champion' or 'challenger' based on the configured split."""
        return "challenger" if random.random() < self.challenger_pct else "champion"

    def make_request_id(self) -> str:
        return str(uuid.uuid4())

    def get_model_version_for_role(
        self, role: str, champion_version: str, challenger_version: str | None
    ) -> str:
        if role == "challenger" and challenger_version:
            return challenger_version
        return champion_version
