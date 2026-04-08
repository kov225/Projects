# 📡 TV Ad Attribution Engine

This engine measures the minute-level causal incremental impact of linear TV ad airings on website traffic. It addresses the lack of direct click-through data for traditional television advertising by implementing high-resolution counterfactual estimation and statistical significance testing.

## 🧠 Methodology and Performance Framework

### 1. High-Resolution Simulation
The `simulator.py` module generates a 90-day minute-level sessions dataset (129k mins) with complex organic behaviors:
- **Baseline Components**: Hourly and weekly seasonality, slow growth trends, and AR(1) noise reflecting the inherent volatility of web traffic.
- **Airing Response Model**: Injects non-linear response curves ($lift(t) = A \cdot (t/\tau) \cdot e^{1-t/\tau}$) with varying network reach, spot length (15s vs 30s), and creative effectiveness factor.
- **Frequency Fatigue**: Realistically decays response when multiple airings occur on the same network in a short time frame.

### 2. Spot-Level Attribution
The `attribution.py` system extracts the immediate incremental sessions for every ad spot:
- **Local Linear Baseline**: Estimates what traffic would have been without the ad using a 20-minute pre-airing window.
- **Statistical Significance**: Performs Z-score thresholding against the cumulative noise floor to separate true signal from random fluctuations.
- **Response Curve Fitting**: Uses non-linear least squares to fit the lift surface and extract peak response time and total incremental impact.

### 3. Campaign-Level Causal Impact
For aggregate measurement, the project uses Bayesian Structural Time Series (BSTS). By utilizing a correlated control market, the model constructs a robust counterfactual to identify the overall campaign-level lift with quantified 95% credible intervals.

## 🚀 Key Performance Indicators (KPIs)

- **Network Scorecard**: Ranking networks by Cost Per Incremental Session (CPIS) and average response magnitude.
- **Creative Efficiency**: Box plots comparisons and ANOVA testing to identify the most effective ad creative version.
- **Daypart Heatmaps**: 20x6 heatmaps (Network x Daypart) coloring efficiency to guide media buyer re-allocation.
- **Frequency Response**: Identifying the "saturation point" where additional daily airings yield diminishing returns.

## 🛠️ Tech Stack

- **Causal Analysis**: TF-CausalImpact, Statsmodels
- **Optimization**: SciPy (Curve Fit), LinearRegression
- **Data Engineering**: Pandas, NumPy
- **Diagnostics**: Scikit-learn (ANOVA, Tukey HSD)

## 📋 Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate simulated sessions and airings:
   ```bash
   python simulator.py
   ```
3. Open the notebooks to view the performance dashboard:
   ```bash
   jupyter notebook notebooks/04_performance_report.ipynb
   ```
