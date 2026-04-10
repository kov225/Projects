# 📈 Bayesian Media Mix Modeling (MMM) for DTC Growth

This repository implements a production-grade Bayesian Media Mix Model (MMM) using **PyMC-Marketing**. The objective is to quantify the causal contribution of various marketing channels (Social, Search, TV, etc.) to website conversions, while accounting for adstock (carryover) and saturation (diminishing returns).

## 🧠 Methodology: The Bayesian Advantage

Unlike frequentist MMMs which often suffer from multicollinearity between channel spends, our Bayesian approach utilizes informative priors to stabilize estimates and provides a full posterior distribution of ROI.

### 1. Geometric Adstock Transformation
Marketing impact isn't instantaneous; it decays over time. We model this using a geometric decay parameter $\alpha \in [0, 1]$:
$$Y_t = X_t + \alpha Y_{t-1}$$
We use a **Beta(2, 2)** prior for $\alpha$, reflecting an expectation of moderate carryover for digital channels.

### 2. Hill / Logistic Saturation
To model diminishing returns, we apply a saturation transformation. This prevents the model from overstating ROI as spend scales to inefficient levels.

### 3. Posterior Diagnostics & Convergence
We enforce rigorous MCMC validation:
- **R-hat ($\hat{R}$)**: All parameters must satisfy $\hat{R} < 1.05$ to ensure chain convergence.
- **Effective Sample Size (ESS)**: Ensuring sufficient bulk and tail ESS for reliable inference.
- **Posterior Predictive Checks (PPC)**: Validating that the model captures the observed variance in conversions.

## 🛠️ Project Structure

```text
├── modeling.py        # MMMWrapper using PyMC-Marketing with Diagnostic Checks
├── simulator.py       # Realistic DTC data simulation (Seasonality + Trend + Noise)
├── optimizer.py       # Budget optimization using SLSQP over the posterior
├── notebooks/
│   └── analysis.ipynb # Detailed EDA and Model Interpretation
└── requirements.txt   # PyMC, ArviZ, PyMC-Marketing
```

## 🚀 Usage

### 1. Simulated Telemetry
Generate synthetic marketing spend and conversion data:
```bash
python simulator.py
```

### 2. Model Fitting & Diagnostics
Fit the Bayesian model and generate diagnostic plots:
```bash
python modeling.py
```

### 3. Budget Optimization
Find the optimal spend allocation across channels to maximize predicted conversions:
```bash
python optimizer.py
```

---
*Developed as part of my Applied Data Science & ML Engineering Portfolio.*
