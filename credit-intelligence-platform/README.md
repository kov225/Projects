# Credit Intelligence Platform

**Stack:** Python 3.11, FastAPI, XGBoost, LightGBM, scikit-learn, SHAP, MLflow,
Apache Kafka, Redis, PostgreSQL, Evidently, Prometheus, Grafana, Docker
Compose, Mistral-7B (Ollama) or GPT-4o-mini, fredapi, APScheduler, Locust.

A containerized credit risk scoring stack that ingests historical LendingClub
loans, enriches them with FRED macro features under a publication lag correct
point in time join, trains a stacking ensemble, and serves predictions through
a monitored FastAPI service with drift triggered retraining and SHAP based
adverse action explanations.

The project is built around three ideas I wanted to learn end to end:

1. **Temporally honest features.** The FRED macro join uses a one month
   publication lag rather than a level join on year-month. A loan originated
   in February only had access to January's unemployment release. Skipping
   that detail is a common form of look ahead leakage that inflates training
   AUC.
2. **Drift as an operational signal, not a chart.** Drift detection (PSI plus
   KS on inputs, rolling AUC on labeled holdout) is wired to an APScheduler
   trigger that retrains and promotes models through MLflow when thresholds
   are breached.
3. **Adverse action explanations as a real product surface.** SHAP top
   reasons are translated by an LLM (GPT-4o-mini in the cloud path or local
   Mistral-7B via Ollama) into ECOA/FCRA style language. The LLM choice is
   configurable so the same notice can be generated offline.

---

## Observed numbers

These are the numbers I see on my hardware after a clean run. Treat them as
ballpark, not a benchmark.

| Metric | Observed | Target |
|--------|----------|--------|
| Ensemble AUC (held out test) | ~0.79 | > 0.78 |
| API p99 latency, Locust @ 50 users | ~95 ms | < 120 ms |
| Drift detection recall, scenario suite | ~0.9 | > 0.9 |
| Retrain to promote pipeline (E2E) | ~6.5 min | < 8 min |
| SHAP faithfulness (Spearman r vs. ablation) | ~0.87 | > 0.85 |
| Training loans | 2.2 M | n/a |
| Macro features (FRED) | 5 series | n/a |
| Compose services | 9 | n/a |

---

## Architecture

```
+---------------------------------------------------------------+
|                        DATA LAYER                             |
|  LendingClub CSV -> historical_loader.py -> loans_clean.pq    |
|  FRED API -> fred_macro_pipeline.py -> point in time join     |
|  Kafka Producer -> loan stream -> Kafka -> Kafka Consumer     |
+---------------------------+-----------------------------------+
                            |
+---------------------------v-----------------------------------+
|                      FEATURE LAYER                            |
|  Custom sklearn transformers (DebtBurden, FICO, MacroInteract)|
|  Redis Feature Store (TTL plus version namespaced cache)      |
|  Schema Registry (versioned JSON feature/model sync)          |
+---------------------------+-----------------------------------+
                            |
+---------------------------v-----------------------------------+
|                       MODEL LAYER                             |
|  XGBoost + LightGBM + LR -> 5 fold OOF -> Logistic meta       |
|  MLflow Registry (Staging -> Production -> Archived)          |
|  A/B Router: Champion / Challenger traffic split              |
|  SHAP TreeExplainer + top 5 values stored in PostgreSQL       |
+---------------------------+-----------------------------------+
                            |
+---------------------------v-----------------------------------+
|                      SERVING LAYER                            |
|  POST /score  : score + A/B route + log + SHAP                |
|  POST /explain: SHAP waterfall + LLM adverse action notice    |
|  GET  /health : liveness check                                |
|  GET  /metrics: Prometheus exposition                         |
+---------------------------+-----------------------------------+
                            |
+---------------------------v-----------------------------------+
|                    MONITORING LAYER                           |
|  Evidently PSI + KS drift detection (sliding window 1000)     |
|  Rolling AUC concept drift (holdout 500)                      |
|  APScheduler retraining trigger (every 6h if drift detected)  |
|  Prometheus + Grafana dashboard (p50/p95/p99, drift, AUC)     |
+---------------------------------------------------------------+
```

---

## Project structure

```
credit-intelligence-platform/
  data/
    historical_loader.py     LendingClub CSV to clean Parquet
    fred_macro_pipeline.py   FRED API pull + point in time join
    kafka_producer.py        Loan stream simulator (3 drift scenarios)
    kafka_consumer.py        Kafka -> Redis -> /score integration
  features/
    transformers.py          DebtBurden, FICOBucket, MacroEngineer, EmpStability
    feature_pipeline.py      Full sklearn Pipeline + save/load helpers
    feature_store.py         Redis cache with TTL and version namespacing
    schema_registry.py       Versioned feature/model sync (fails loud on mismatch)
  models/
    train.py                 Stacking ensemble: XGB+LGB+LR + 5 fold OOF meta-learner
    mlflow_registry.py       Experiment logging + AUC gated model promotion
    ab_router.py             Champion/challenger A/B router + PostgreSQL logging
    shap_explainer.py        SHAP per prediction + faithfulness metric
  serving/
    api.py                   FastAPI: /score /explain /health /metrics
    llm_explanation.py       GPT-4o-mini / Ollama Mistral-7B adverse action notice
    prometheus_metrics.py    Counters, histograms, gauges
  monitoring/
    drift_detector.py        PSI + KS drift detection -> PostgreSQL
    concept_drift.py         Rolling AUC on labeled holdout
    retrain_trigger.py       APScheduler: drift -> retrain -> promote
    grafana_dashboard.json   Pre built dashboard (import directly)
  evaluation/
    model_eval.py            AUC, AP, Brier, calibration, F1 opt + cost sensitive threshold
    fairness_audit.py        Demographic parity, equalized odds, markdown report
    stress_test.py           2008 recession macro shock simulation
  prometheus/
    prometheus.yml           Scrape config
  locust/
    locustfile.py            p99 latency load test @ 50 users
  tests/
    test_feature_pipeline.py 12 unit tests for transformers + pipeline
  docker-compose.yml         9 services orchestrated
  Dockerfile                 API + scheduler container
  requirements.txt           Pinned dependencies
  .env.example               All required config variables
  README.md
```

---

## Drift simulation scenarios

Three explicit drift scenarios are implemented in `kafka_producer.py` and
validated against the monitoring layer.

**Scenario 1, economic shock.** Unemployment spikes from 4 percent to 10
percent, CPI rises 15 percent. PSI on `unemployment_rate` and
`dti_x_unemployment` exceeds 0.2 within the first detection window.
Retraining is triggered automatically.

**Scenario 2, population shift.** `annual_inc` and `loan_amnt` ramp down 40
percent over the simulated 60 day stream to simulate a younger, lower income
borrower demographic. KS test p-values drop below 0.05 across multiple income
related features. Both PSI and KS alarms fire.

**Scenario 3, concept drift.** 15 percent of non default labels are silently
flipped to default with no change to any input feature. PSI stays silent
(correct: the inputs haven't moved). AUC on the rolling labeled holdout
drops by more than the configured 0.03 threshold, triggering an alert from
the concept drift monitor. This is the hardest case for drift detection and
is handled by a separate monitoring path.

---

## Reproducibility

### Prerequisites
- Docker Desktop with Compose v2
- Python 3.11 (for running scripts outside Docker)
- LendingClub dataset CSV from Kaggle (`loan.csv`, ~1.6GB)

### Quick start

```bash
# 1. Clone and configure
git clone https://github.com/koushikvennalakanti/credit-intelligence-platform
cd credit-intelligence-platform
cp .env.example .env
# Fill in FRED_API_KEY, OPENAI_API_KEY (or set EXPLANATION_BACKEND=ollama)

# 2. Data preparation (run outside Docker)
pip install -r requirements.txt
python -m data.historical_loader data/raw/loan.csv
python -m data.fred_macro_pipeline
# This joins macro features onto the loans and saves data/processed/loans_clean.parquet

# 3. Train the ensemble
python -m models.train --data data/processed/loans_clean.parquet

# 4. Run evaluations
python -m evaluation.model_eval
python -m evaluation.fairness_audit
python -m evaluation.stress_test

# 5. Start all services
docker compose up -d

# 6. Verify
curl http://localhost:8000/health
```

### Score a loan application

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 15000,
    "int_rate": 14.5,
    "installment": 520.0,
    "annual_inc": 72000,
    "dti": 18.5,
    "fico_score": 680,
    "emp_length": 4.0
  }'
```

### Get an adverse action notice

```bash
curl -X POST "http://localhost:8000/explain?applicant_id=<applicant_id>"
```

### Run load test

```bash
locust -f locust/locustfile.py --host http://localhost:8000 \
       --users 50 --spawn-rate 10 --run-time 60s --headless
```

### Simulate drift

```bash
# Economic shock
python -m data.kafka_producer economic_shock

# Gradual population shift
python -m data.kafka_producer population_shift

# Concept drift (hardest case)
python -m data.kafka_producer concept_drift
```

### Run unit tests

```bash
pytest tests/ -v
```

---

## Service endpoints

| Service | URL | Purpose |
|---------|-----|---------|
| FastAPI | http://localhost:8000 | Credit scoring API |
| MLflow | http://localhost:5000 | Experiment tracking and model registry |
| Grafana | http://localhost:3000 | Live monitoring dashboard (admin/admin) |
| Prometheus | http://localhost:9090 | Metrics storage |
| Kafka | localhost:9092 | Loan application stream |
| PostgreSQL | localhost:5432 | Predictions and drift reports |
| Redis | localhost:6379 | Feature cache |

---

## Why these technology choices

The macro features use a one month publication lag rather than a level join
on year-month because BLS and Fed releases are delayed. A February
origination only had access to January's unemployment numbers; ignoring that
lag is a common form of temporal leakage that inflates training AUC by 2 to
4 points and degrades immediately in production. I started caring about this
detail while building geo clustering and lending segmentation pipelines at
CASHe.

LightGBM and XGBoost are included together because they have meaningfully
different inductive biases. XGBoost's regularization tends to be more
conservative on tail risk, while LightGBM's leaf wise growth often fits
complex interactions better. The meta learner weights them contextually
rather than picking one. Logistic regression sits underneath both as the
explainable baseline.

The `EXPLANATION_BACKEND` flag is intentional. Mistral-7B is the same model I
quantized in the inference project, and using it here means the optimization
work has a downstream application that benefits from low latency local
inference, rather than two unrelated demos.

SHAP uses `TreeExplainer` rather than `KernelExplainer` because the former
runs in `O(T*L*D)` time (depth times leaves) instead of `O(M^2*N)`, which
makes per request SHAP feasible in the serving hot path without breaking the
latency target.

---

## Fairness audit

The fairness audit uses U.S. state as a geographic demographic proxy
following established fair lending research methodology. The script computes
demographic parity difference and equalized odds difference across
geographic clusters and attempts per group threshold adjustment as a
mitigation. A written compliance report is generated at
`evaluation/fairness/fairness_report.md`. The state level proxy is
deliberately a first step. A complete fair lending review would require
census tract demographics and HMDA data.

---

## Honest caveats

- **Ollama is not in `docker-compose.yml`.** When `EXPLANATION_BACKEND=ollama`,
  the API expects an Ollama instance reachable at the configured URL. The
  GPT-4o-mini path is fully self contained and is what the default config
  uses.
- **Test coverage is uneven.** The feature pipeline has unit tests; the API,
  drift, and LLM modules are exercised by the load test and end to end
  scenarios but not by isolated unit tests yet.
- **The "2.2M loans" figure is dataset size, not throughput.** Real time
  scoring is benchmarked separately via Locust at 50 concurrent users.

## Next steps

- Replace the geographic proxy in the fairness audit with zip code level
  census tract demographics from the Census Bureau API.
- Add unit coverage to the drift detector and the LLM explanation module.
- Deploy to AWS ECS with ECR plus RDS PostgreSQL for a cloud native run.
- Extend the A/B router with Bayesian stopping rules.
