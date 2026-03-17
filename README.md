# 📊 Koushik Vennalakanti | Applied Data Science & ML Engineering Portfolio

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://python.org)
[![R](https://img.shields.io/badge/R-276DC3?logo=r&logoColor=white)](https://r-project.org)
[![GPA](https://img.shields.io/badge/GPA-3.94-success)]()
[![Status](https://img.shields.io/badge/Status-Actively_Seeking_Roles-blue)]()

> **"Built scalable LLM inference pipelines, reconstructed 15th-century legal networks using NLP and bipartite matching, and engineered interpretable trading algorithms."**

I am a **Data Science graduate student at Lehigh University (GPA: 3.94)** specializing in **applied statistics, interpretable machine learning, and ML systems engineering**. This repository showcases my ability to own the entire data lifecycle - from raw data ingestion and heuristic modeling to hardware-aware LLM deployment.

---

## 🏆 Featured Projects & Business Impact

### 1. [Real-Time Credit Intelligence Platform](./credit-intelligence-platform) *(ML Engineering & Production AI)*
* **The Business Problem:** Regulated lenders need drift-aware, explainable risk scoring to prevent model decay and ensure ECOA compliance.
* **Key Results:** Built a 9-service containerized system handling 2.2M loans. Reached **p99 < 100ms** latency and automated retraining via drift sensing (PSI/KS).
* **Tech Stack:** FastAPI, Kafka, Redis, MLflow, Prometheus, Grafana, SHAP, Mistral-7B.

### 2. [Optimized Inference Pipeline for Mistral LLMs](./mistral-llm-optimised-inference)  *(ML Systems Engineering)*
* **The Business Problem:** Deploying LLMs is computationally expensive and latency-heavy.
* **Key Results:** Built a high-performance, batched inference engine on a single NVIDIA T4 GPU, achieving **>200 tokens/sec throughput** (handling quantization and LoRA adapters).
* **Tech Stack:** PyTorch, HuggingFace Accelerate, bitsandbytes.

### 3. [Automated NLP Reconciliation of Historical Data](./historical-nlp-reconciliation) *(End-to-End Data Science)*
* **The Business Problem:** Manual transcription and entity resolution of fragmented records costs thousands of human hours.
* **Key Results:** Automated alignment of AI-generated HTR text against human ground-truth data using RapidFuzz and Bipartite matching. Achieved scalable bipartite resolution for 1,200+ unique entities to build historical social network graphs.
* **Tech Stack:** Python, Flask, RapidFuzz, Pandas, SciPy, NetworkX.

### 4. [Algorithmic Trading & Interpretability Research](./algorithmic-trading-research) *(Financial Modeling)*
* **The Business Problem:** Black-box financial models fail during regime shifts; stakeholders require interpretable risk strategies.
* **Key Results:** Cleaned decades of raw AAPL market data to engineer structural features (support/resistance). Developed a robust, rule-based scoring framework prioritizing logic transparency over opaque neural networks.
* **Tech Stack:** Python, Pandas, Technical Indicators.

### 5. [latent-recommend: Music Search via Learned Audio Embeddings](./latent-recommend) *(Audio AI & Latent Space)*
* **The Business Problem:** Popularity bias in music recommendation systems traps indie artists; content-based similarity is needed for discovery.
* **Key Results:** Built a purely acoustic search engine extracting deep latent embeddings via VAE/Diffusion architectures. Validated 1.2M Spotify tracks via K-Means and 3D PCA to enable discovery strictly by "acoustic feeling."
* **Tech Stack:** Python, PyTorch, Scikit-learn, PCA, K-Means, SQLite.

### 6. [DSCI 441: Dataset Shift & Model Robustness](./dsci441-recommendation-analysis) *(ML Research & Statistical Validation)*
* **The Business Problem:** Models trained on static data collapse in dynamic environments; production systems need quantified resilience.
* **Key Results:** Assessed 7 ML architectures against controlled covariate and label shift. Implemented Kolmogorov-Smirnov (KS) divergence tracking and 95% CI bootstrapping to certify model stability under environmental drift.
* **Tech Stack:** Python, Streamlit, Scipy (KS-Test), Scikit-learn, Bootstrap Resampling.

---

## 📂 Project Structure

```text
├── credit-intelligence-platform/         # Production ML (FastAPI, Redis, Kafka, Monitoring)
├── mistral-llm-optimised-inference/      # LLM deployment (Quantization, Batching, Benchmarking)
├── historical-nlp-reconciliation/        # Bipartite matching v2, Web Dashboard, Fuzzy String NLP
├── latent-recommend/                     # Music latent space search (VAE, PCA, K-Means)
├── dsci441-recommendation-analysis/      # Dataset shift longitudinal study (KS-Test, Resampling)
├── capstone/capstone-project-kov225/     # Modular reconciliation scripts, unit tests, network graph
├── algorithmic-trading-research/         # Time-series EDA, Feature Engineering, Rule-based Backtesting
├── global-trade-inequality-data/         # Large-scale R/Python Statistical Analysis
└── coursework-and-practice/              # Genetic Algorithms, R Stats, Ethics, Coding exercises
```

## 🚀 Reproducibility & Code Quality
I prioritize code that works in production. Every major project folder contains a `requirements.txt` and explicit setup instructions. 

**Example (Running the NLP Reconciliation Pipeline):**
```bash
cd historical-nlp-reconciliation
pip install -r requirements.txt
python reconciliation.py
```


## 📫 Let's Connect
- **Email:** [kov225@lehigh.edu](mailto:kov225@lehigh.edu)
- **Mobile:** 484-935-7840
