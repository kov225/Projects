# 📊 Koushik Vennalakanti | Applied Data Science & ML Engineering Portfolio

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://python.org)
[![R](https://img.shields.io/badge/R-276DC3?logo=r&logoColor=white)](https://r-project.org)
[![GPA](https://img.shields.io/badge/GPA-3.94-success)]()
[![Status](https://img.shields.io/badge/Status-Actively_Seeking_Roles-blue)]()

> "Built scalable LLM inference pipelines, engineered Bayesian media mix models, and automated TV ad attribution for high-growth DTC brands."

I am a Data Science graduate student at Lehigh University (GPA: 3.94) specializing in applied statistics, Bayesian inference, and ML systems engineering. This repository showcases my ability to own the entire data lifecycle from raw telemetry simulation to production-grade causal measurement and optimization.

## 🏆 Featured Projects & Business Impact

### 1. [Bing Experimentation Suite](./bing-experimentation-suite)
* **The Business Problem:** Online platforms like Bing require rigorous statistical engines to measure experiment lift while minimizing variance.
* **Key Results:** Built a production-grade experimentation framework implementing CUPED (Controlled-experiment Using Pre-Experiment Data) and post-stratification. Achieved significant variance reduction on synthetic telemetry and implemented automated novelty effect detection using exponential decay models.
* **Tech Stack:** Python, SciPy, Statsmodels, Pandas, Pytest.

### 2. [Bayesian Media Mix Modeling (MMM)](./media_mix_model)
* **The Business Problem:** Brands often struggle to quantify the causal contribution of offline and online channels to conversions due to complex carryover effects and diminishing returns.
* **Key Results:** Developed a Bayesian MMM using PyMC-Marketing to decompose website conversions across five channels. Implemented geometric adstock and logistic saturation transformations with custom Gamma and Beta priors. Built a budget optimizer using SciPy SLSQP to find the optimal allocation across the posterior distribution.
* **Tech Stack:** PyMC-Marketing, PyMC, ArviZ, SciPy, Pandas.

### 3. [TV Ad Attribution Engine](./tv_attribution)
* **The Business Problem:** Measuring the immediate impact of linear TV airings requires high-resolution session data and sophisticated counterfactual estimation.
* **Key Results:** Engineered a minute-level attribution system that extracts incremental sessions by fitting parametric response curves against local linear baselines. Validated campaign-level lift using Bayesian Structural Time Series (CausalImpact) with correlated control markets. Detected missed airings with high precision using statistical z-score thresholds.
* **Tech Stack:** Python, TF-CausalImpact, SciPy (Non-linear Least Squares), Pandas.

### 4. [Real-Time Credit Intelligence Platform](./credit-intelligence-platform)
* **The Business Problem:** Regulated lenders require drift-aware and explainable risk scoring to ensure compliance and model stability.
* **Key Results:** Engineered a 9-service containerized system handling 2.2M loans with p99 latency under 100ms. Integrated automated retraining triggered by drift sensing and SHAP-based interpretability for high-stakes decisioning.
* **Tech Stack:** FastAPI, Kafka, Redis, MLflow, Prometheus, Grafana, SHAP.

### 5. [NLP Reconciliation of Historical Data](./historical-nlp-reconciliation)
* **The Business Problem:** Resolving entities across fragmented historical records is a massive manual bottleneck for researchers.
* **Key Results:** Automated the alignment of AI-generated text against ground-truth data using RapidFuzz and Bipartite matching. Successfully resolved 1,200 unique entities to reconstruct 15th-century legal social network graphs.
* **Tech Stack:** Python, RapidFuzz, SciPy (Bipartite Matching), NetworkX.

### 6. [Dataset Shift & Model Robustness](./dataset-shift-analysis)
* **The Business Problem:** Models trained on historical data often collapse in production due to environmental drift.
* **Key Results:** Assessed model resilience against controlled covariate and label shift. Implemented Kolmogorov-Smirnov divergence tracking and 95% CI bootstrapping to certify model stability.
* **Tech Stack:** Python, SciPy, Scikit-learn, Bootstrap Resampling.

## 📂 Project Structure

```text
├── bing-experimentation-suite/       # Variance Reduction, CUPED, Novelty Detection
├── media_mix_model/                  # Bayesian MMM, Geometric Adstock, Budget Optimization
├── tv_attribution/                    # TV Attribution, Response Curves, CausalImpact
├── netflix-device-analytics/         # High-scale Telemetry Simulations & Real-time Dashboards
├── credit-intelligence-platform/     # Production ML (FastAPI, Redis, Kafka, Monitoring)
├── dataset-shift-analysis/           # Model Robustness & Statistical Validation
├── mistral-llm-optimised-inference/  # LLM Deployment (Quantization, Batching)
├── historical-nlp-reconciliation/    # Entity Resolution & Social Network NLP
├── latent-recommend/                 # Audio Latent Space Search (VAE, PCA, K-Means)
└── algorithmic-trading-research/     # Time-series EDA & Rule-based Backtesting
```

## 🚀 Reproducibility
Every project is designed for production reproducibility. Each directory contains a `requirements.txt` and a dedicated README with setup instructions.

## 📫 Let's Connect
* **Email:** [kov225@lehigh.edu](mailto:kov225@lehigh.edu)
* **Mobile:** 484-935-7840
