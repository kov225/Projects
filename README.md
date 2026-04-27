# 📊 Koushik Vennalakanti | Applied Data Science & ML Engineering Portfolio

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://python.org)
[![R](https://img.shields.io/badge/R-276DC3?logo=r&logoColor=white)](https://r-project.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyMC](https://img.shields.io/badge/PyMC-3B5EA1?logo=python&logoColor=white)](https://www.pymc.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![GPA](https://img.shields.io/badge/GPA-3.94-success)]()

> M.S. Data Science candidate at Lehigh University. I build end to end ML
> systems: causal experimentation, Bayesian modeling, drift aware production
> services, and LLM inference optimisation.

I am a Data Science graduate student at Lehigh University (GPA 3.94)
specializing in applied statistics, Bayesian inference, causal measurement,
and ML systems engineering. This repository collects the projects where I
own the full lifecycle: data ingestion, modeling, evaluation, and
production style serving with monitoring.

If you only have a few minutes, the three projects that best represent how I
work are **Credit Intelligence Platform**, **Dataset Shift Analysis**, and
**Bing Experimentation Suite**. Each one has a runnable entry point and an
honest README that says exactly what is built and what is still in progress.

---

## 🏆 Featured Projects & Business Impact

### 1. [Credit Intelligence Platform](./credit-intelligence-platform)
* **Business problem:** Regulated lenders need a credit scoring service
  that is drift aware, fair, and explainable enough to satisfy ECOA / FCRA
  adverse action requirements.
* **Key results:** Built a 9 service containerized stack (FastAPI, Kafka,
  Redis, MLflow, Prometheus, Grafana) trained on 2.2M LendingClub loans
  joined to 5 FRED macro series under a publication lag correct point in
  time merge. Stacking ensemble (XGBoost + LightGBM + Logistic Regression)
  reaches AUC ~0.79 on held out test, with API p99 latency ~95 ms at 50
  concurrent users (Locust). Drift detection (PSI / KS / rolling AUC)
  triggers automated retraining and MLflow promotion. SHAP TreeExplainer
  feeds an LLM (GPT-4o-mini or local Mistral-7B via Ollama) that drafts
  ECOA style adverse action notices.
* **Tech stack:** Python 3.11, FastAPI, XGBoost, LightGBM, scikit-learn,
  SHAP, MLflow, Apache Kafka, Redis, PostgreSQL, Evidently, Prometheus,
  Grafana, Docker Compose, OpenAI, Ollama, APScheduler, Locust.

### 2. [Bing Experimentation Suite](./bing-experimentation-suite)
* **Business problem:** Online platforms running thousands of A/B tests
  need variance reduction estimators that detect smaller effects without
  inflating sample size.
* **Key results:** Implemented CUPED (Deng et al., 2013), post stratification
  (Miratrix et al., 2013), Welch's t-test, non parametric bootstrap, and an
  exponential decay novelty effect detector. Validated on simulated
  telemetry with A/A uniformity checks and power curves. 11 unit tests
  cover estimator correctness and CUPED variance reduction is empirically
  verified to match the `(1 - rho^2)` theoretical bound.
* **Tech stack:** Python, NumPy, SciPy, Statsmodels, Pandas, Plotly Dash,
  Pytest.

### 3. [Bayesian Media Mix Modeling](./media_mix_model)
* **Business problem:** Brands struggle to quantify the causal contribution
  of marketing channels to conversions because of carryover effects,
  diminishing returns, and channel multicollinearity.
* **Key results:** Built a Bayesian MMM in PyMC-Marketing with geometric
  adstock and logistic saturation transforms across five marketing
  channels. Enforces R-hat < 1.05 convergence checks on every parameter,
  reports bulk and tail effective sample size, and runs posterior
  predictive checks against held out weeks. Includes a posterior aware
  budget allocation prototype using SciPy SLSQP.
* **Tech stack:** PyMC-Marketing, PyMC, ArviZ, NumPy, Pandas, SciPy.

### 4. [TV Ad Attribution Engine](./tv_attribution)
* **Business problem:** Linear TV airings have no click through data, so
  measuring their incremental impact on web traffic requires
  high resolution counterfactual estimation.
* **Key results:** Built a per spot attribution engine that estimates
  incremental sessions by fitting local linear pre roll baselines and
  bootstrapping pre airing residuals (Efron, 1979) for 95 percent
  confidence intervals. Aggregated lift is fit to a parametric response
  curve `L(t) = A * (t/tau) * exp(1 - t/tau)`, giving each network a
  comparable peak intensity and decay constant. Campaign level results are
  cross validated with Bayesian Structural Time Series (CausalImpact)
  against correlated control markets.
* **Tech stack:** Python, NumPy, SciPy (curve_fit, bootstrap), Pandas,
  tfcausalimpact.

### 5. [Dataset Shift & Model Robustness](./dataset-shift-analysis)
* **Business problem:** Models trained on historical data degrade silently
  in production under covariate, label, and concept shift, and teams
  rarely have a principled way to compare candidate model architectures
  on robustness rather than only clean accuracy.
* **Key results:** Built a reproducible experimental suite (DSCI 441,
  Milestone 2) testing 9 classifier architectures against 8 shift families
  across 10 intensity levels, with 200 iteration bootstrap CIs on every
  metric. Computes a composite Robustness Score and PSI, runs Welch t-tests
  for significance of degradation, and ships a 4 tab Streamlit dashboard
  with publication ready plots. Generates over 700 result rows in a tidy
  CSV and 40+ auto saved figures per run.
* **Tech stack:** Python, scikit-learn, XGBoost, Streamlit, NumPy, SciPy,
  Pandas, Matplotlib.

### 6. [Historical NLP Reconciliation](./historical-nlp-reconciliation)
* **Business problem:** Manually reconciling AI extracted records against
  human curated archives is the bottleneck for digitizing historical legal
  collections at scale.
* **Key results:** Built an entity reconciliation pipeline aligning HTR
  output against ground truth records from 17th century King's Bench plea
  rolls. Models the problem as bipartite assignment and solves it with the
  Hungarian algorithm (Kuhn and Munkres, 1955) using a weighted composite
  score (RapidFuzz token sort plus county and plea bonuses, with a
  litigant count penalty). Reconciles roughly 1,200 unique individuals and
  exports the resulting plaintiff / defendant network as a NetworkX graph.
  The capstone version reports F1 of 0.491 (fuzzy) versus 0.262 (strict
  baseline) on the same split.
* **Tech stack:** Python, RapidFuzz, SciPy (`linear_sum_assignment`),
  NetworkX, BeautifulSoup, Docker.

---

## 🧰 Additional Projects

* [Mistral-7B Inference Optimisation](./mistral-llm-optimised-inference).
  4-bit NF4 quantization, double quant, FlashAttention-2, and a KV cache
  memory profiler for single GPU deployment of a 7B parameter model.
* [Streaming QoS Analytics](./netflix-device-analytics). Quality of Service
  analytics on simulated streaming telemetry with non parametric bootstrap
  confidence intervals on buffering differences across device classes.
* [Latent Recommend](./latent-recommend). Content based music
  recommendation using PCA, cosine similarity, and K-Means on Spotify
  acoustic features, with Welch's t-tests quantifying popularity bias.
* [Global Trade Inequality](./global-trade-inequality-data). Group research
  project quantifying trade concentration using Gini and
  Herfindahl Hirschman indices, plus a follow on study on cross tier
  software pricing.
* [Capstone: KB27/799 HTR Reconciliation](./capstone). Capstone extension
  of the historical NLP project: two slot bipartite matching, a HTML
  scraper for ground truth, and social network analysis on a population of
  1,200+ medieval litigants.
* [Algorithmic Trading Research](./algorithmic-trading-research). Time
  series EDA and rule based backtesting on AAPL daily bars.
* [Coursework & Practice](./coursework-and-practice). Weekly Python
  exercises, R statistical computing labs, data ethics assignments, and a
  Genetic Algorithms heat exchanger optimization project (CSF 407).

---

## 📂 Project Structure

```text
.
├── credit-intelligence-platform/     # Production ML stack (FastAPI, Kafka, Redis, MLflow, SHAP)
├── bing-experimentation-suite/       # CUPED, post-stratification, novelty detection
├── media_mix_model/                  # Bayesian MMM (adstock, saturation, R-hat checks)
├── tv_attribution/                   # Per spot lift, response curves, CausalImpact
├── dataset-shift-analysis/           # 9 models x 8 shifts x 10 intensities, robustness scoring
├── historical-nlp-reconciliation/    # Hungarian bipartite matching, RapidFuzz, NetworkX
├── mistral-llm-optimised-inference/  # NF4 quant, FlashAttention-2, KV cache profiler
├── netflix-device-analytics/         # QoS analytics, bootstrap CI on buffering
├── latent-recommend/                 # PCA + cosine similarity music recommender
├── global-trade-inequality-data/     # Gini, HHI, software pricing study
├── algorithmic-trading-research/     # Time series EDA + rule based backtesting
├── capstone/                         # KB27/799 HTR reconciliation + social network
└── coursework-and-practice/          # Python, R, data ethics, GA optimization
```

---

## 🚀 Reproducibility

Every project ships a `requirements.txt`, a runnable entry point, and a
self contained README. Several (Credit Intelligence Platform, Historical
NLP Reconciliation, Netflix QoS, Media Mix Model, TV Attribution) include a
`Dockerfile` or `docker-compose.yml` for a clean run without local install.

---

## 📫 Let's Connect

* **Email:** [kov225@lehigh.edu](mailto:kov225@lehigh.edu)
* **Personal:** [vennelakantiavinash@gmail.com](mailto:vennelakantiavinash@gmail.com)
* **Mobile:** 484-935-7840
* **Location:** Bethlehem, PA. Open to relocation.

Open to full time roles in Data Science, Applied ML, and ML Engineering.
