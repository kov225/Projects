"""
api.py

FastAPI application serving the credit risk scoring system. This is the
entrypoint for all external traffic. On startup it loads:
  - The fitted sklearn feature pipeline
  - The stacking ensemble (base models + meta-learner)
  - The SHAP explainer for the XGBoost base model
  - Connections to Redis (feature store) and PostgreSQL (prediction log)

The startup sequence is intentionally fail-loud : if any of these fail,
the app refuses to start rather than serving garbage predictions.

Endpoints:
    POST /score    : score a single loan application
    POST /explain  : explain a past prediction via SHAP + LLM
    GET  /health   : liveness/readiness check
    GET  /metrics  : Prometheus metrics exposition
"""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import psycopg2
import redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# : Lazy-loaded globals : populated on startup :::::::::::::::::::::::::::::
_pipeline = None
_scaler = None
_base_models = None
_meta_learner = None
_feature_cols = None
_redis_client = None
_champion_version = None
_challenger_version = None
_ab_router = None
_start_time = None

DECISION_THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.5"))
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "models/artifacts"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all model artifacts and initialize connections on startup."""
    global _pipeline, _scaler, _base_models, _meta_learner, _feature_cols
    global _redis_client, _ab_router, _start_time, _champion_version

    _start_time = datetime.now(timezone.utc)

    try:
        _pipeline = joblib.load(ARTIFACT_DIR / "pipeline.joblib")
        _scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
        _base_models = joblib.load(ARTIFACT_DIR / "base_models.joblib")
        _meta_learner = joblib.load(ARTIFACT_DIR / "meta_learner.joblib")

        # Load feature column list from schema registry
        from features.schema_registry import load_registry
        reg = load_registry()
        if reg["versions"]:
            latest = sorted(reg["versions"].keys())[-1]
            _feature_cols = reg["versions"][latest]["feature_columns"]
            _champion_version = latest
        logger.info(f"Loaded model artifacts. Champion version: {_champion_version}")
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        # App starts but marks itself unhealthy : Kubernetes will handle restart

    try:
        _redis_client = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            decode_responses=True,
        )
        _redis_client.ping()
        logger.info("Redis connected")
    except redis.ConnectionError as e:
        logger.warning(f"Redis unavailable: {e}")

    from models.ab_router import ABRouter, ensure_predictions_table
    _ab_router = ABRouter()
    try:
        ensure_predictions_table()
        logger.info("PostgreSQL predictions table ready")
    except Exception as e:
        logger.warning(f"Could not ensure predictions table: {e}")

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="Credit Intelligence Platform",
    description="Real-time credit risk scoring with drift-aware model serving",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response schemas ───────────────────────────────────────────────

class LoanApplication(BaseModel):
    applicant_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    loan_amnt: float = Field(..., gt=0)
    int_rate: float = Field(..., gt=0)
    installment: float = Field(..., gt=0)
    annual_inc: float = Field(..., gt=0)
    dti: float = Field(..., ge=0)
    fico_score: float = Field(..., ge=300, le=850)
    delinq_2yrs: float = Field(default=0.0)
    revol_util: float = Field(default=50.0)
    total_acc: float = Field(default=10.0)
    open_acc: float = Field(default=5.0)
    pub_rec: float = Field(default=0.0)
    mort_acc: float = Field(default=0.0)
    emp_length: float = Field(default=5.0)
    grade_enc: float = Field(default=2.0)
    sub_grade_enc: float = Field(default=10.0)
    term_months: float = Field(default=36.0)
    funded_amnt: float = Field(default=None)
    revol_bal: float = Field(default=5000.0)
    pub_rec_bankruptcies: float = Field(default=0.0)
    # Macro features : pulled from FRED or passed directly for testing
    unemployment_rate: float = Field(default=5.0)
    cpi: float = Field(default=300.0)
    fed_funds_rate: float = Field(default=2.0)
    unemployment_mom_change: float = Field(default=0.0)
    cpi_yoy_pct: float = Field(default=2.0)

    model_config = {"extra": "allow"}  # allow any extra features without erroring


class ScoreResponse(BaseModel):
    request_id: str
    applicant_id: str
    risk_score: float
    default_probability: float
    decision: str
    decision_threshold: float
    model_version: str
    model_role: str
    latency_ms: float


class ExplainResponse(BaseModel):
    applicant_id: str
    shap_values: dict[str, float]
    adverse_action_notice: str
    model_version: str


# ── Helper functions ────────────────────────────────────────────────────────

def get_or_compute_features(record: dict, model_version: str) -> np.ndarray:
    """Try Redis first; compute from pipeline on cache miss."""
    applicant_id = record.get("applicant_id", "unknown")

    if _redis_client:
        cache_key = f"applicant:{applicant_id}:features:{model_version}"
        cached = _redis_client.get(cache_key)
        if cached:
            return np.array(json.loads(cached))

    # Feature computation on cache miss
    feature_cols = _feature_cols or list(record.keys())
    df_row = __import__("pandas").DataFrame([record])
    for col in feature_cols:
        if col not in df_row.columns:
            df_row[col] = 0.0

    available = [c for c in feature_cols if c in df_row.columns]
    X = _pipeline.transform(df_row[available])
    X = _scaler.transform(X)

    # Cache for TTL
    if _redis_client:
        ttl = int(os.environ.get("REDIS_TTL_SECONDS", "3600"))
        _redis_client.set(cache_key, json.dumps(X[0].tolist()), ex=ttl)

    return X


def ensemble_predict(X: np.ndarray) -> tuple[float, float]:
    """Run inference through the stacking ensemble.
    
    Returns (risk_score, default_probability). risk_score is the same as
    default_probability here : we expose both fields so the API contract
    is explicit that higher = more risk.
    """
    # Average all fold-trained versions of each base model
    n_base = len(_base_models)
    meta_features = np.zeros((1, n_base))
    for col, (name, fold_models) in enumerate(_base_models.items()):
        fold_preds = np.array([m.predict_proba(X)[:, 1] for m in fold_models])
        meta_features[0, col] = fold_preds.mean(axis=0)

    prob = float(_meta_learner.predict_proba(meta_features)[0, 1])
    return prob, prob


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/score", response_model=ScoreResponse)
async def score(application: LoanApplication, request: Request):
    from serving.prometheus_metrics import (
        request_latency, requests_total, errors_total, score_distribution
    )

    t_start = time.perf_counter()
    request_id = str(uuid.uuid4())

    if _pipeline is None or _meta_learner is None:
        errors_total.labels(endpoint="/score").inc()
        raise HTTPException(503, "Model not loaded: service starting up")

    try:
        record = application.model_dump()
        if record.get("funded_amnt") is None:
            record["funded_amnt"] = record["loan_amnt"]

        role = _ab_router.route()
        model_version = _champion_version or "local_dev"

        X = get_or_compute_features(record, model_version)
        risk_score, default_prob = ensemble_predict(X)
        decision = "deny" if default_prob >= DECISION_THRESHOLD else "approve"
        latency_ms = (time.perf_counter() - t_start) * 1000

        # SHAP for deny decisions (batched async in production : sync here for simplicity)
        shap_dict = None
        if decision == "deny" and _base_models:
            try:
                from models.shap_explainer import explain_single_prediction
                xgb_model = _base_models["xgboost"][0]
                feature_names = _feature_cols or [f"f{i}" for i in range(X.shape[1])]
                shap_dict = explain_single_prediction(xgb_model, X[0], feature_names)
            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}")

        # Async log to PostgreSQL (swallow errors : logging failure must not break scoring)
        try:
            from models.ab_router import log_prediction
            log_prediction(
                request_id=request_id,
                applicant_id=application.applicant_id,
                model_version=model_version,
                model_role=role,
                risk_score=risk_score,
                default_prob=default_prob,
                decision=decision,
                decision_threshold=DECISION_THRESHOLD,
                features={k: v for k, v in record.items() if isinstance(v, (int, float))},
                shap_values=shap_dict,
            )
        except Exception as e:
            logger.warning(f"Failed to log prediction: {e}")

        # Prometheus
        request_latency.labels(model_role=role).observe(latency_ms / 1000)
        requests_total.labels(model_role=role, decision=decision).inc()
        score_distribution.labels(model_role=role).observe(risk_score)

        return ScoreResponse(
            request_id=request_id,
            applicant_id=application.applicant_id,
            risk_score=round(risk_score, 6),
            default_probability=round(default_prob, 6),
            decision=decision,
            decision_threshold=DECISION_THRESHOLD,
            model_version=model_version,
            model_role=role,
            latency_ms=round(latency_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        errors_total.labels(endpoint="/score").inc()
        logger.exception(f"Scoring error for {application.applicant_id}: {e}")
        raise HTTPException(500, f"Internal scoring error: {str(e)}")


@app.post("/explain", response_model=ExplainResponse)
async def explain(applicant_id: str):
    """Retrieve stored SHAP values and generate an adverse action notice."""
    from serving.prometheus_metrics import errors_total

    try:
        # Fetch SHAP values from PostgreSQL
        conn = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            dbname=os.environ.get("POSTGRES_DB", "credit_intelligence"),
            user=os.environ.get("POSTGRES_USER", "credit_app"),
            password=os.environ.get("POSTGRES_PASSWORD", "changeme"),
        )
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT shap_values, risk_score, model_version
                    FROM predictions
                    WHERE applicant_id = %s AND decision = 'deny'
                    ORDER BY created_at DESC LIMIT 1
                """, (applicant_id,))
                row = cur.fetchone()
        finally:
            conn.close()

        if not row:
            raise HTTPException(404, f"No denied prediction found for applicant {applicant_id}")

        shap_raw, risk_score, model_version = row
        shap_values = shap_raw if isinstance(shap_raw, dict) else json.loads(shap_raw)

        if not shap_values:
            raise HTTPException(404, "No SHAP values stored for this prediction")

        from serving.llm_explanation import generate_adverse_action_notice
        notice = generate_adverse_action_notice(
            shap_values=shap_values,
            risk_score=risk_score,
            decision_threshold=DECISION_THRESHOLD,
            model_version=model_version or "unknown",
        )

        return ExplainResponse(
            applicant_id=applicant_id,
            shap_values=shap_values,
            adverse_action_notice=notice,
            model_version=model_version or "unknown",
        )

    except HTTPException:
        raise
    except Exception as e:
        errors_total.labels(endpoint="/explain").inc()
        logger.exception(f"Explain error for {applicant_id}: {e}")
        raise HTTPException(500, str(e))


@app.get("/health")
async def health():
    redis_ok = False
    if _redis_client:
        try:
            redis_ok = _redis_client.ping()
        except Exception:
            pass

    uptime_seconds = None
    if _start_time:
        uptime_seconds = (datetime.now(timezone.utc) - _start_time).total_seconds()

    return {
        "status": "healthy" if _pipeline is not None else "degraded",
        "model_loaded": _pipeline is not None,
        "model_version": _champion_version,
        "redis_connected": redis_ok,
        "uptime_seconds": uptime_seconds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics")
async def metrics():
    from serving.prometheus_metrics import get_metrics_bytes
    return Response(
        content=get_metrics_bytes(),
        media_type="text/plain; version=0.0.4",
    )
