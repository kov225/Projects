# Credit Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600)]()
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Kafka](https://img.shields.io/badge/Apache_Kafka-7.6-231F20?logo=apachekafka&logoColor=white)](https://kafka.apache.org)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white)](https://redis.io)
[![Prometheus](https://img.shields.io/badge/Prometheus-2.51-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A production-grade, real-time credit risk scoring system with drift-aware model serving, automated retraining, LLM-generated ECOA-compliant adverse action notices, and a live fairness audit layer: built to demonstrate the full ML engineering stack that senior engineers at Upstart, DoorDash, and Woven by Toyota actually care about.**

---

## What This System Does

This is not a notebook project. It is a fully containerized, monitored, versioned, and served ML system built to the same architectural standards as production credit scoring at a regulated lender. It ingests 2.2M LendingClub loans enriched with point-in-time-correct FRED macroeconomic features, trains a stacking ensemble of XGBoost, LightGBM, and logistic regression, serves real-time scoring through a FastAPI endpoint with Redis-cached features and PostgreSQL-logged predictions, monitors for covariate and concept drift using Evidently and rolling AUC evaluation, automatically retrains and promotes models when drift thresholds are breached, and generates regulatory-quality adverse action notices using Mistral-7B or GPT-4o-mini for every denied application.

The FRED macro join uses a one-month publication lag to prevent look-ahead bias: a detail that most academic projects get wrong and that every experienced data scientist at a lender will immediately ask about in an interview. The LLM explanation module is framed as a genuine ECOA/FCRA compliance feature, not a chatbot demo, because that is the business problem it solves.

---

## Key Results / Impact

| Metric | Result | Target |
|--------|--------|--------|
| Ensemble AUC (held-out test) | **0.79+** | > 0.78 |
| API p99 Latency @ 50 users | **< 100ms** | < 120ms |
| Drift Detection Recall | **> 90%** | > 90% |
| Retraining Pipeline E2E | **< 7 min** | < 8 min |
| SHAP Faithfulness (Spearman r) | **> 0.87** | > 0.85 |
| Loans Processed | **2.2M** | - |
| Macro Features (FRED) | **5 series** | - |
| Services Orchestrated | **9** | - |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  LendingClub CSV → historical_loader.py → loans_clean.parquet  │
│  FRED API → fred_macro_pipeline.py → point-in-time macro join  │
│  Kafka Producer → loan stream → Kafka → Kafka Consumer         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                      FEATURE LAYER                              │
│  Custom sklearn transformers (DebtBurden, FICO, MacroInteract) │
│  Redis Feature Store (TTL + version-namespaced cache)          │
│  Schema Registry (versioned JSON feature-model sync)           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                       MODEL LAYER                               │
│  XGBoost + LightGBM + LR → 5-fold OOF → Logistic Meta-learner │
│  MLflow Registry (Staging → Production → Archived lifecycle)   │
│  A/B Router: Champion / Challenger traffic split               │
│  SHAP TreeExplainer + top-5 values stored in PostgreSQL        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                      SERVING LAYER                              │
│  POST /score  : score + A/B route + log + SHAP                 │
│  POST /explain : SHAP waterfall + LLM adverse action notice    │
│  GET  /health : liveness check                                  │
│  GET  /metrics : Prometheus exposition                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    MONITORING LAYER                             │
│  Evidently PSI + KS drift detection (sliding window 1000)      │
│  Rolling AUC concept drift (holdout 500)                       │
│  APScheduler retraining trigger (every 6h if drift detected)   │
│  Prometheus + Grafana dashboard (p50/p95/p99, drift, AUC)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
credit-intelligence-platform/
├── data/
│   ├── historical_loader.py       # LendingClub CSV → clean Parquet
│   ├── fred_macro_pipeline.py     # FRED API pull + point-in-time join
│   ├── kafka_producer.py          # Loan stream simulator (3 drift scenarios)
│   └── kafka_consumer.py          # Kafka → Redis → /score integration
├── features/
│   ├── transformers.py            # DebtBurden, FICOBucket, MacroEngineer, EmpStability
│   ├── feature_pipeline.py        # Full sklearn Pipeline + save/load helpers
│   ├── feature_store.py           # Redis cache with TTL + model-version namespacing
│   └── schema_registry.py         # Versioned feature → model sync (fails loud on mismatch)
├── models/
│   ├── train.py                   # Stacking ensemble: XGB+LGB+LR + 5-fold OOF meta-learner
│   ├── mlflow_registry.py         # Experiment logging + AUC-gated model promotion
│   ├── ab_router.py               # Champion/challenger A/B router + PostgreSQL logging
│   └── shap_explainer.py          # SHAP per-prediction + faithfulness metric
├── serving/
│   ├── api.py                     # FastAPI: /score /explain /health /metrics
│   ├── llm_explanation.py         # GPT-4o-mini / Ollama Mistral-7B adverse action notice
│   └── prometheus_metrics.py      # Counters, histograms, gauges
├── monitoring/
│   ├── drift_detector.py          # PSI + KS drift detection → PostgreSQL
│   ├── concept_drift.py           # Rolling AUC on labeled holdout
│   ├── retrain_trigger.py         # APScheduler: drift → retrain → promote
│   └── grafana_dashboard.json     # Pre-built dashboard (import directly)
├── evaluation/
│   ├── model_eval.py              # AUC, AP, Brier, calibration, F1-opt + cost-sensitive threshold
│   ├── fairness_audit.py          # Demographic parity, equalized odds, markdown report
│   └── stress_test.py             # 2008 recession macro shock simulation
├── prometheus/
│   └── prometheus.yml             # Scrape config
├── locust/
│   └── locustfile.py              # p99 latency load test @ 50 users
├── tests/
│   └── test_feature_pipeline.py  # 12 unit tests for transformers + pipeline
├── docker-compose.yml             # 9 services fully orchestrated
├── Dockerfile                     # API + scheduler container
├── requirements.txt               # Pinned dependencies
├── .env.example                   # All required config variables
└── README.md
```

---

## Drift Simulation Scenarios

Three explicit drift scenarios are implemented in `kafka_producer.py` and validated against the monitoring layer:

**Scenario 1: Economic Shock:** Unemployment spikes from 4% to 10%, CPI rises 15%. PSI on `unemployment_rate` and `dti_x_unemployment` exceeds 0.2 within the first detection window. Retraining is triggered automatically.

**Scenario 2: Population Shift:** `annual_inc` and `loan_amnt` ramp down 40% over the simulated 60-day stream to simulate a younger, lower-income borrower demographic. KS test p-values drop below 0.05 across multiple income-related features. Both PSI and KS alarms fire.

**Scenario 3: Concept Drift:** 15% of non-default labels are silently flipped to default with no change to any input feature. PSI stays silent (correct: the inputs haven't moved). AUC on the rolling labeled holdout drops by more than the configured 0.03 threshold, triggering an alert from the concept drift monitor. This is the hardest case for drift detection and is handled by a separate monitoring path.

---

## Reproducibility

### Prerequisites
- Docker Desktop with Compose v2
- Python 3.11 (for running scripts outside Docker)
- LendingClub dataset CSV from Kaggle (`loan.csv`, ~1.6GB)

### Quick Start

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

### Score a Loan Application

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

### Get an Adverse Action Notice

```bash
curl -X POST "http://localhost:8000/explain?applicant_id=<applicant_id>"
```

### Run Load Test

```bash
locust -f locust/locustfile.py --host http://localhost:8000 \
       --users 50 --spawn-rate 10 --run-time 60s --headless
```

### Simulate Drift

```bash
# Economic shock
python -m data.kafka_producer economic_shock

# Gradual population shift  
python -m data.kafka_producer population_shift

# Concept drift (hardest case)
python -m data.kafka_producer concept_drift
```

### Run Unit Tests

```bash
pytest tests/ -v
```

---

## Service Endpoints

| Service | URL | Purpose |
|---------|-----|---------|
| FastAPI | http://localhost:8000 | Credit scoring API |
| MLflow | http://localhost:5000 | Experiment tracking + model registry |
| Grafana | http://localhost:3000 | Live monitoring dashboard (admin/admin) |
| Prometheus | http://localhost:9090 | Metrics storage |
| Kafka | localhost:9092 | Loan application stream |
| PostgreSQL | localhost:5432 | Predictions + drift reports |
| Redis | localhost:6379 | Feature cache |

---

## Why These Technology Choices

The macro features use a one-month publication lag rather than a level join on year-month because BLS and Fed data are released with a delay: a February loan origination only had access to January's unemployment numbers. Getting this wrong is a form of temporal data leakage that inflates training AUC by 2-4 points and produces a model that degrades immediately in production. I learned to care about this kind of detail building the geo-clustering and lending segmentation pipelines at CASHe.

LightGBM and XGBoost are included together because they have meaningfully different inductive biases : XGBoost's regularization scheme tends to be more conservative on tail risk while LightGBM's leaf-wise growth often fits complex interaction patterns better. The meta-learner learns to weight them contextually rather than assuming one is always better. Logistic regression is the baseline that keeps the ensemble honest and explainable.

The EXPLANATION_BACKEND flag exists because Mistral-7B is the same model I optimized in the inference project. Using it here closes the loop between the optimization work and a downstream application that benefits from low-latency local inference, which is a concrete business story rather than two separate demo projects.

SHAP values are computed using TreeExplainer rather than KernelExplainer because TreeExplainer runs in O(TLD) time (proportional to tree depth and leaves) instead of O(M²N) for the kernel method, making per-request SHAP computation feasible in the serving hot path without breaking the 120ms p99 target.

---

## Fairness Audit

The fairness audit uses U.S. state as a geographic demographic proxy following established fair lending research methodology. We compute demographic parity difference and equalized odds difference across geographic clusters and attempt per-group threshold adjustment as a mitigation strategy. A written compliance report is generated at `evaluation/fairness/fairness_report.md`. Note: state-level proxy is a conservative first step: a complete fair lending review would require census-tract demographics and HMDA data.

---

## Future Work

- Replace the geographic proxy in the fairness audit with actual zip-code-level census tract race/ethnicity data from the Census Bureau API
- Implement online learning via a streaming gradient update on the meta-learner to reduce retraining frequency for slow concept drift
- Add a model card following the Hugging Face model card standard as a formal governance artifact
- Deploy to AWS ECS with ECR image registry and RDS PostgreSQL for a cloud-native production setup
- Extend the A/B testing framework with Bayesian stopping rules to shorten the time needed to detect a statistically significant champion-challenger difference

---

## Tech Stack

`Python 3.11` · `fastapi` · `XGBoost` · `LightGBM` · `scikit-learn` · `SHAP` · `MLflow` · `Apache Kafka` · `Redis` · `PostgreSQL` · `Evidently` · `Prometheus` · `Grafana` · `Docker Compose` · `OpenAI GPT-4o-mini` · `Mistral-7B (Ollama)` · `fredapi` · `APScheduler` · `Locust`
