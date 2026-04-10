# 📡 TV Ad Attribution Engine: Minute-Level Causal Inference

This engine measures the minute-level causal incremental impact of linear TV ad airings on website traffic. It addresses the lack of direct click-through data for traditional television advertising by implementing high-resolution counterfactual estimation and statistical significance testing.

## 🧠 Methodology: Parametric Response Recovery

Unlike simple "spike" detection, this engine models the underlying physics of TV response—where a spot triggers immediate awareness followed by a specific decay curve.

### 1. Counterfactual Estimation (Local Linear Baseline)
We estimate the "unobserved" baseline (what would have happened without the ad) by fitting a local linear trend to the 20 minutes of telemetry prior to the airing. This controls for intra-day seasonality and pre-existing trends.

### 2. Signal Extraction & Bootstrapping
Incremental sessions are isolated by subtracting the counterfactual baseline from the observed sessions. 
- **Bootstrap Resampling**: To account for the high variance in minute-level web traffic, we resample pre-airing residuals (Efron, 1979) to generate a full distribution of potential lifts.
- **Significance Criteria**: A spot is marked as "Significant" only if its 95% bootstrap confidence interval does not cross zero.

### 3. Parametric Curve Fitting
We fit a non-linear response model to the Isolated signal:
$$L(t) = A \cdot \frac{t}{\tau} \cdot e^{1 - \frac{t}{\tau}}$$
- **$A$ (Intensity)**: The peak response magnitude.
- **$\tau$ (Decay Rate)**: The time constant reflecting how quickly the response fades.

## 🛠️ Project Structure

```text
├── attribution.py    # Core Attribution Engine (Bootstrap + Curve Fitting)
├── simulator.py       # High-resolution Session Simulation (Seasonality + Noise)
├── notebooks/
│   └── analysis.ipynb # Performance Dashboard & Network Scorecards
└── data/             # Airing logs and minute-level telemetry
```

## 🚀 Quick Start

1. **Generate Simulated Data**:
   ```bash
   python simulator.py
   ```

2. **Run Attribution Pipeline**:
   ```bash
   python attribution.py
   ```

## 📈 Strategic KPIs
The engine outputs a network-level scorecard including:
- **CPIS (Cost Per Incremental Session)**: Causal efficiency of each network.
- **Response Half-Life**: Identifying which networks drive the most "durable" sessions.
- **Daypart Optimality**: Heatmaps identifying the most efficient airing windows.

---
*Developed as part of my Applied Data Science & ML Engineering Portfolio.*
