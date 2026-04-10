# 🧪 Bing Experimentation Suite: Advanced Variance Reduction & Causal Inference

This suite implements production-grade statistical estimators designed to increase experiment sensitivity and reduce the time-to-decision in high-traffic online platforms. By utilizing pre-experiment data and demographic stratification, we can achieve up to 30-50% variance reduction, allowing for the detection of smaller treatment effects with the same sample size.

## 🚀 Key Methodologies

### 1. CUPED (Controlled-experiment Using Pre-Experiment Data)
Following **Deng et al. (2013)**, we implement CUPED to adjust the post-treatment metric $Y$ using a pre-experiment covariate $X$. 
- **Estimator**: $\hat{Y}_{CUPED} = Y - \theta(X - \mathbb{E}[X])$
- **Optimal $\theta$**: $\frac{Cov(Y, X)}{Var(X)}$
- **Impact**: Reduces variance by a factor of $(1 - \rho^2)$, where $\rho$ is the correlation between $X$ and $Y$.

### 2. Post-Stratification
We adjust for accidental imbalances in group assignment across user segments (e.g., Geo, Device, Tier).
- **Technique**: Weighted average of within-strata treatment effects.
- **Reference**: Miratrix, Sekhon, and Yu (2013).

### 3. Heterogeneous Treatment Effects (HTE)
Beyond the Global Average Treatment Effect (GATE), this suite identifies if specific segments respond differently to the treatment, enabling personalized product experiences.

## 🛠️ Project Structure

```text
├── experiments/
│   ├── ab_test.py            # Welch's T-Test & Non-parametric Bootstrap
│   ├── cuped.py              # CUPED variance reduction logic
│   ├── stratification.py     # Post-stratification & Regression Adjustment
│   ├── novelty.py            # Exponential decay model for novelty effects
│   └── variance_benchmark.py # Comparative performance framework
├── data/
│   └── generate.py           # Realistic telemetry simulation with noise
├── dashboard/               # Plotly/Dash visualization for experiment results
└── tests/                   # Statistical validation unit tests
```

## 📊 Quick Start: Running the Benchmark

To compare all estimators on fresh synthetic telemetry:

```bash
python -m experiments.variance_benchmark
```

## 🧪 Statistical Validation
We use a rigorous validation suite to ensure estimators are unbiased. This includes:
- **A/A Testing**: Verifying that p-values follow a Uniform distribution $U(0, 1)$ when no effect is present.
- **Power Analysis**: Benchmarking Minimum Detectable Effect (MDE) improvements across methods.
- **Bootstrap Robustness**: Comparing frequentist p-values against non-parametric distributions.

---
*Developed as part of my Applied Data Science & ML Engineering Portfolio.*
