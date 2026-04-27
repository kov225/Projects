# Koushik Vennalakanti, Data Science Portfolio

M.S. Data Science candidate at **Lehigh University** (GPA 3.94). Interests:
applied statistics, Bayesian inference, causal measurement, and ML systems
engineering. This repository collects the larger projects I have built during
my graduate program and in independent study, ranging from coursework
deliverables to longer running applied work.

I have tried to keep each project honest about what is finished, what is a
prototype, and what is a study exercise. The READMEs reflect that. They
describe what is implemented in code, the methods used, and the limitations I
am still working on.

## Education

- **M.S. Data Science**, Lehigh University. GPA 3.94.
- Coursework: Machine Learning (DSCI 441), Statistical Computing (R), Data
  Ethics, Optimization, Applied Statistics.

## How this repository is organized

The projects fall into three rough groups.

### Causal inference and experimentation
- [bing-experimentation-suite](./bing-experimentation-suite). Variance
  reduction estimators (CUPED, post-stratification) for online A/B tests, plus
  a novelty effect detector and a comparative benchmark harness.
- [media_mix_model](./media_mix_model). A Bayesian Media Mix Model built on
  PyMC-Marketing, with geometric adstock and logistic saturation transforms,
  posterior diagnostics, and a budget allocation prototype.
- [tv_attribution](./tv_attribution). Minute resolution incremental session
  attribution for linear TV airings, using local linear baselines, residual
  bootstrap, and a parametric response curve. CausalImpact is used at the
  campaign level for cross validation.

### Applied machine learning and modeling
- [credit-intelligence-platform](./credit-intelligence-platform). A
  containerized credit scoring stack (FastAPI, Redis, Kafka, MLflow,
  Prometheus) trained on 2.2M LendingClub loans joined to FRED macro features
  with a publication lag correct point in time merge. Includes drift triggered
  retraining and SHAP based adverse action notices.
- [dataset-shift-analysis](./dataset-shift-analysis). Coursework project (DSCI
  441, Milestone 2) that quantifies how nine classifiers degrade under eight
  shift families across ten intensities, with bootstrap CIs, PSI, and a
  Streamlit dashboard.
- [latent-recommend](./latent-recommend). PCA plus cosine similarity
  recommender on Spotify acoustic features, with Welch's t-tests motivating
  the move away from popularity ranked baselines.

### Systems, NLP, and analysis
- [historical-nlp-reconciliation](./historical-nlp-reconciliation). Entity
  reconciliation across 17th century King's Bench plea rolls using RapidFuzz
  scoring and Hungarian bipartite matching, with domain heuristics for
  occupational stop words.
- [mistral-llm-optimised-inference](./mistral-llm-optimised-inference).
  Mistral-7B inference setup with NF4 quantization, FlashAttention-2, and a
  KV cache memory profiler. Skeleton for benchmarking; numbers are works in
  progress.
- [netflix-device-analytics](./netflix-device-analytics). Quality of Service
  analytics with bootstrap significance tests for buffering differences across
  device classes. Analytics layer is implemented; ingestion is a work in
  progress.
- [global-trade-inequality-data](./global-trade-inequality-data). Notebook
  based study of trade concentration using Gini and Herfindahl Hirschman
  indices, plus a smaller statistical study on software pricing.
- [algorithmic-trading-research](./algorithmic-trading-research). Exploratory
  data analysis and rule based backtesting on AAPL daily bars; written as a
  self contained research project.
- [capstone](./capstone). Capstone version of the historical NLP work,
  producing a social network graph from reconciled litigants in King's Bench
  plea rolls.
- [coursework-and-practice](./coursework-and-practice). Weekly exercises,
  optimization assignments, R statistical computing, and certificates. Kept
  here for completeness rather than as portfolio pieces.

## Reading order

If you have limited time, the projects that best represent how I approach
problems are **dataset-shift-analysis**, **credit-intelligence-platform**, and
**bing-experimentation-suite**. Each has a self contained README, a runnable
entry point, and explicit notes on what is and is not finished.

## Reproducibility

Every project pins its dependencies in `requirements.txt` and documents an
entry point. Several projects ship a `Dockerfile` or `docker-compose.yml` for
a clean run without installing anything locally.

## Contact

- **Email:** [kov225@lehigh.edu](mailto:kov225@lehigh.edu)
- **Personal:** [vennelakantiavinash@gmail.com](mailto:vennelakantiavinash@gmail.com)
- **Phone:** 484-935-7840
