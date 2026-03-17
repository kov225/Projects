# Dataset Shift: How Classical ML Models Respond to Distribution Changes

This project investigates the resilience of classical machine learning algorithms under three types of distribution change. We quantify performance degradation using the UCI Adult Income dataset to determine how models respond when their training and testing environments diverge.

## 1. Problem Explanation
In real-world machine learning applications, the assumption that training and testing data come from the same distribution often fails. This phenomenon, known as **Dataset Shift**, can lead to significant performance drops when a model is deployed. 

Key challenges include:
- **Model Fragility**: Models may rely on spurious correlations that don't hold in new environments.
- **Uncertainty Quantification**: Failing to account for variance in shifted environments can lead to overconfident, incorrect predictions.
- **Safety and Reliability**: In domains like healthcare or finance, undetected drift can have catastrophic consequences.

## 2. Goals and Proposed Approaches
The objective is to establish a rigorous benchmarking suite that measures model stability under stress. 

**Our approaches include:**
- **Controlled Corruption**: Systematically inducing Covariate, Prior, and Concept-Adjacent shifts at varying intensities.
- **Statistical Framework**: Moving beyond point estimates to include confidence intervals and divergence metrics.
- **Comparative Analysis**: Evaluating a diverse set of models (Linear, Tree-based, Ensemble, Probabilistic) to identify which architectures are inherently more robust.

## 3. Statistical Model Components
To ensure scientific rigor, we incorporate several core statistical principles:

- **Central Limit Theorem (CLT)**: We leverage CLT to justify the use of normal distributions for our performance confidence intervals. By bootstrapping model metrics 100 times, the mean of these resamples follows a normal distribution, allowing us to calculate 95% CIs.
- **Law of Large Numbers (LLN)**: This principle ensures that as we increase our experimental iterations or sample sizes, our observed model performance converges to its true value under a given shift regime.
- **Hypothesis Testing**: We use t-tests (specifically `ttest_ind`) to determine if the performance degradation observed under shift is statistically significant compared to the baseline (p < 0.05).
- **Non-Parametric Verification (KS Test)**: The Kolmogorov-Smirnov test is used to measure the physical divergence between original and shifted feature distributions without assuming a specific data distribution.
- **Bootstrapping**: We use resampling with replacement to quantify the uncertainty in our performance metrics (Accuracy, F1, ROC-AUC).

## 4. Baseline Solution
The project uses a **Naive Baseline** (`DummyClassifier`) which simply predicts the most frequent class from the training set. 
- **Purpose**: It serves as a performance "floor." 
- **Benchmark**: Any model that performs worse than the baseline is considered completely broken by the shift. 
- **Calibration**: It helps us understand the difficulty of the dataset; if a complex model only slightly outperforms this naive baseline, it suggests the features are weak or the problem is highly stochastic.

## 5. Milestone 2 Roadmap
Building on this foundation, Milestone 2 will focus on:
- **Advanced Shift Simulators**: Implementation of `Scaling Drift`, `Prior Oversampling`, and `Feature Permutation`.
- **Robustness Scoring**: Developing a unified `Robustness Score` that combines performance drop rate with stability across intensities.
- **Interpretability Layer**: Analyzing feature importance shifts to understand *why* certain models fail while others persist.
- **Automated Reporting**: Enhanced dashboard visualizations and automated figure generation for the final report.

---

### Project Structure
```text
dataset-shift-project/
├── app.py                # Research Dashboard (Streamlit)
├── src/
│   ├── data_loader.py    # UCI Census Data acquisition
│   ├── evaluation.py     # Bootstrapping and metric calculation
│   ├── models.py         # Model definitions and baseline fitting
│   ├── run_experiments.py # Pipeline orchestration
│   ├── shift_simulators.py # Synthetic drift algorithms
│   └── statistics.py     # CLT, LLN, and KS tests
└── results/              # Experimental logs and CSV artifacts
```

### Execution
To reproduce the experimental results and update the baseline:
```bash
python src/run_experiments.py
```

### Visualization
To explore the findings interactively:
```bash
streamlit run app.py
```

---
*Note: This work represents the completion of Milestone 1 for the DSCI 441 Machine Learning course.*
