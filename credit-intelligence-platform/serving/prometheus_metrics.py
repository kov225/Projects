"""
prometheus_metrics.py

Prometheus counters, histograms, and gauges exposed through the FastAPI /metrics
endpoint. FastAPI calls into these from the middleware and route handlers.

We expose:
- Request latency as a histogram (lets Grafana compute p50/p95/p99)
- Total requests and error count as counters
- Champion vs. challenger decision rate as a gauge (updated per request)
- Drift score per feature (updated by the drift detector)
- Rolling AUC (updated by concept drift monitor)
"""

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest

# Use a fresh registry rather than the default global one so we have full
# control over what gets exported and can unit test in isolation
REGISTRY = CollectorRegistry(auto_describe=True)

request_latency = Histogram(
    "credit_scorer_request_latency_seconds",
    "End-to-end request latency for /score",
    labelnames=["model_role"],   # champion or challenger
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 1.0],
    registry=REGISTRY,
)

requests_total = Counter(
    "credit_scorer_requests_total",
    "Total scoring requests",
    labelnames=["model_role", "decision"],
    registry=REGISTRY,
)

errors_total = Counter(
    "credit_scorer_errors_total",
    "Total request errors",
    labelnames=["endpoint"],
    registry=REGISTRY,
)

drift_score = Gauge(
    "credit_scorer_drift_psi",
    "Most recent PSI drift score per feature",
    labelnames=["feature"],
    registry=REGISTRY,
)

rolling_auc = Gauge(
    "credit_scorer_rolling_auc",
    "Rolling AUC from labeled holdout evaluation",
    labelnames=["model_version"],
    registry=REGISTRY,
)

decision_rate = Gauge(
    "credit_scorer_approval_rate",
    "Rolling approval rate (last 1000 requests)",
    labelnames=["model_role"],
    registry=REGISTRY,
)

score_distribution = Histogram(
    "credit_scorer_risk_score",
    "Distribution of risk scores",
    labelnames=["model_role"],
    buckets=[i / 20 for i in range(21)],  # 0.0 to 1.0 in 0.05 steps
    registry=REGISTRY,
)


def get_metrics_bytes() -> bytes:
    """Return the Prometheus text exposition format for the /metrics endpoint."""
    return generate_latest(REGISTRY)


def update_drift_scores(psi_scores: dict[str, float]) -> None:
    """Update gauge values after each drift detection run."""
    for feature, psi in psi_scores.items():
        drift_score.labels(feature=feature).set(psi)


def update_rolling_auc(model_version: str, auc: float) -> None:
    rolling_auc.labels(model_version=model_version).set(auc)
