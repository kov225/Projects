# Dataset Shift: Assessing Model Robustness to Environmental Change

**Course:** DSCI 441 Machine Learning  
**Milestone:** 2  
**Dataset:** UCI Adult Income (OpenML ID 1590)

---

## Overview

Real-world machine learning systems routinely encounter data that looks nothing like what they were trained on. Sensor drift, demographic shifts, annotation policy changes, and silent data-pipeline failures are just a few mechanisms that cause the training and deployment distributions to diverge. This phenomenon is collectively called **Dataset Shift**, and it is one of the most underappreciated failure modes in production ML.

This project builds a rigorous, reproducible experimental suite to study how nine classical ML architectures degrade under eight mechanistically distinct shift families, across ten intensity levels each. The goal is not merely to observe degradation but to quantify it precisely using confidence intervals, distribution divergence metrics, and a composite robustness score so that actionable deployment decisions can be made.

---

## Streamlit Dashboard

The interactive research dashboard provides four analysis tabs accessible through a single command.

**Tab 1: Performance Curves**  
Multi-model accuracy, F1, and ROC-AUC trajectories as shift intensity increases from 0 to 1. Shaded bands show 95% bootstrap confidence intervals.

![Performance Curves Tab](docs/screenshots/dashboard_performance_tab.png)

**Tab 3: Robustness Analysis**  
Horizontal model ranking by composite Robustness Score, a global relative-drop heatmap across all shift types, and Brier Score calibration decay curves.

![Robustness Analysis Tab](docs/screenshots/dashboard_robustness_tab.png)

Launch the dashboard with:

```bash
streamlit run app.py
```

---

## Project Structure

```text
dataset-shift-project/
├── main.py                       Entry point for the full experiment pipeline
├── app.py                        Streamlit interactive dashboard
├── requirements.txt              Python package dependencies
│
├── src/
│   ├── data_loader.py            UCI Adult Income acquisition and preprocessing
│   ├── models.py                 Model registry (9 estimators including XGBoost)
│   ├── shift_simulators.py       Eight shift simulation algorithms
│   ├── evaluation.py             Metrics, bootstrapping, robustness scoring
│   ├── visualizations.py         Seven publication-quality plot functions
│   ├── run_experiments.py        Experiment orchestration loop
│   ├── statistics.py             KS test, CLT, LLN, hypothesis testing helpers
│   └── utils.py                  PSI, logging, seed management, path helpers
│
├── results/
│   └── experiment_results.csv    Tidy result table (auto-generated)
│
├── figures/                      Auto-saved PNG plots (auto-generated)
├── outputs/                      Supplementary artifacts (auto-generated)
└── docs/screenshots/             Dashboard screenshots for documentation
```

---

## Shift Taxonomy

Eight shift families are implemented with a consistent contract: every function accepts an `intensity` parameter in [0.0, 1.0], and intensity = 0.0 always returns the unmodified baseline data.

| ID | Family | Mechanism | What Changes |
|----|--------|-----------|--------------|
| A | **Covariate Shift** | Gaussian noise + multiplicative drift on continuous features | P(X) |
| A2 | **Scaling Drift** | Log-normal rescaling per feature (sensor calibration failure) | P(X) |
| B | **Label Shift** | Progressive minority-class undersampling | P(Y) |
| C | **Concept Drift** | Feature corruption + label flipping on top informative dimensions | P(Y given X) |
| D | **Gaussian Noise** | Pure additive i.i.d. Gaussian noise | P(X) |
| E | **MCAR Missingness** | Uniformly random cell masking | P(X) |
| E2 | **MAR Missingness** | Structured masking conditioned on an observed feature | P(X) |
| F | **Feature Removal** | Zeroing of top-N informative columns (data-pipeline failure) | P(X) |

---

## Model Suite

Nine estimators spanning diverse inductive biases are trained once on the clean data and evaluated repeatedly under all shift conditions.

| Model | Type |
|-------|------|
| Naive Baseline (DummyClassifier) | Mode predictor, performance floor |
| Naive Bayes | Generative, Gaussian likelihood |
| Logistic Regression | Linear discriminative |
| SVM (RBF) | Kernel method, non-linear |
| Decision Tree | Single deep tree, overfit-prone reference |
| Random Forest | Bagging ensemble |
| Gradient Boosting | Sequential boosting ensemble |
| AdaBoost | Adaptive boosting |
| XGBoost | Regularized gradient boosting (when installed) |

---

## Evaluation Framework

Every (model, shift-type, intensity) triple is evaluated with:

**Point Estimates**

- Accuracy, Precision, Recall, F1 Score (weighted average)
- ROC-AUC (requires at least one positive sample)
- Brier Score (proper probability calibration metric)
- Confusion matrix counts: TP, TN, FP, FN

**Uncertainty Quantification**

- 200-iteration bootstrap resampling with replacement
- 95% confidence intervals for all six metrics

**Distribution Divergence**

- Kolmogorov-Smirnov statistic (average and maximum across features)
- Population Stability Index (PSI), with standard thresholds: < 0.10 no action, 0.10 to 0.20 monitor, > 0.20 investigate

**Robustness Scoring**

Two complementary degradation indices are computed per (model, shift) pair:

```
Robustness Score  = (shifted_accuracy / baseline_accuracy) * (1 - KS_statistic)
Relative Drop (%) = (baseline_accuracy - shifted_accuracy) / baseline_accuracy * 100
```

**Statistical Significance**

A one-sided Welch t-test compares the bootstrapped accuracy distributions of each (model, shift) combination against the model's clean baseline. Results flagged with `Significant_Shift = True` confirm that the observed degradation exceeds what sampling variation alone can explain.

---

## Statistical Foundations

This project builds directly on four core statistical principles established in Milestone 1:

**Central Limit Theorem:** The 200-sample bootstrap distributions of Accuracy, F1, and ROC-AUC converge to normal distributions regardless of the underlying metric distribution, justifying the use of percentile-based confidence intervals.

**Law of Large Numbers:** As experiment iterations accumulate, observed mean performance converges to the true expected performance under each shift regime, reducing the influence of stochastic noise in the simulation process.

**Hypothesis Testing:** One-sided Welch t-tests (H0: no performance difference; Ha: baseline accuracy > shifted accuracy) provide a formal framework for declaring when degradation is statistically real rather than noise.

**Kolmogorov-Smirnov Test:** The KS two-sample test measures the maximum distance between the empirical CDFs of baseline and shifted feature distributions, providing a non-parametric, distribution-free divergence measure.

---

## Reproduction

**Step 1: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 2: Run the full experiment pipeline**

```bash
python main.py
```

This downloads the UCI Adult Income dataset, trains all nine models once on clean data, sweeps 8 shift families across 10 intensity levels, computes all metrics with 200 bootstrap iterations, saves `results/experiment_results.csv`, and auto-saves all figures to `figures/`.

Approximate runtime on a modern laptop: 15 to 25 minutes (dominated by SVM and Gradient Boosting fitting on ~33,000 samples).

**Step 3: Explore results interactively**

```bash
streamlit run app.py
```

The dashboard loads the pre-computed results file and requires no additional computation.

---

## Key Findings (Milestone 2)

The experiment suite produces a 25-column tidy result table with over 700 rows covering all (model, shift-type, intensity) combinations. Preliminary observations from the saved CSV:

- Ensemble methods (Gradient Boosting, Random Forest, XGBoost) consistently achieve the highest Robustness Scores across all shift types.
- Logistic Regression is surprisingly robust under Covariate Shift but collapses rapidly under Concept Drift because linear decision boundaries cannot adapt to P(Y given X) changes.
- Feature Removal is the most damaging shift type for tree-based models that rely on a small set of top features.
- Naive Bayes is the least calibrated model under shift: its Brier Score rises steeply even at low intensities.
- MCAR Missingness at intensity > 0.5 causes all models to converge toward the Naive Baseline performance, suggesting that imputation quality (mean-fill in standardized space) is the binding constraint.

---

## Milestone 2 Additions over Milestone 1

| Component | Milestone 1 | Milestone 2 |
|-----------|-------------|-------------|
| Shift families | 3 (Covariate, Prior, Concept-Adjacent) | 8 (full taxonomy) |
| Intensity levels | 5 | 10 |
| Models | 8 (no XGBoost) | 9 (XGBoost optional) |
| Metrics per eval | 4 | 11 + confusion matrix |
| Bootstrap iterations | 100 | 200 |
| PSI computation | No | Yes |
| Robustness Score | No | Yes |
| Relative drop | No | Yes |
| Auto-saved figures | No | Yes (40+ PNG files) |
| Dashboard tabs | 1 | 4 |
| Evaluation state management | Module-level globals | Explicit EvaluationContext class |
| Logging | print() calls | Structured logging.Logger |

---

*This project represents the completion of Milestone 2 for DSCI 441 Machine Learning.*
