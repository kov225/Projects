# 📈 Bayesian Media Mix Modeling (MMM)

This project implements a production-grade Bayesian Media Mix Model designed to quantify the causal impact of multi-channel marketing spend on website conversions. It addresses the fundamental marketing challenge of measuring carryover effects (adstock) and diminishing returns (saturation) in a non-linear environment.

## 🧠 Methodology and Statistical Framework

### 1. Data Generation and Simulation
The `simulator.py` module creates three years of weekly telemetry for a Direct-to-Consumer (DTC) brand. It incorporates realistic complexities:
- **Geometric Adstock**: Modeling the delayed impact of advertising where effects decay over time (e.g., 55% carryover for Linear TV).
- **Hill Saturation**: Implementing diminishing returns using S-shaped curves to identify the "sweet spot" of channel spend.
- **Seasonality and Controls**: Including Fourier terms for annual cycles, indicators for eight major US holidays, and a competitor suppression signal.

### 2. Bayesian Inference Engine
The model is built using the `PyMC-Marketing` framework, allowing for rigorous uncertainty quantification via Markov Chain Monte Carlo (MCMC) sampling.
- **Priors**: Informed priors ensure logical consistency, such as `HalfNormal` for coefficients to enforce positive contributions and `Beta` priors for adstock alpha.
- **Multiplicative Model**: A log-link structure is used to capture proportional rather than additive effects, matching the reality of marketing synergy.

### 3. Budget Optimization
An optimization module utilizes the posterior parameter distributions to solve the budget allocation problem. Using the SLSQP (Sequential Least Squares Programming) algorithm, the system finds the spend mix that maximizes expected conversions subject to total budget constraints and channel-level floors.

## 🚀 Key Features

- **Parameter Recovery**: Validated against ground truth to ensure the statistical engine correctly identifies adstock and saturation coefficients.
- **Incremental ROAS (iROAS)**: Calculation of marginal return on ad spend with full credible intervals.
- **Saturation Analysis**: Interactive visualizations identifying which channels have "headroom" for growth versus those that are deeply saturated.

## 🛠️ Tech Stack

- **Inference**: PyMC, PyMC-Marketing, ArviZ
- **Optimization**: SciPy (SLSQP)
- **Data Engineering**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## 📋 Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate simulated data:
   ```bash
   python simulator.py
   ```
3. Open the notebooks to view the analysis:
   ```bash
   jupyter notebook notebooks/02_model_fitting.ipynb
   ```
