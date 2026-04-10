# 📈 Algorithmic Trading Research: Quantitative Alpha & Risk

This repository contains a quantitative research framework for developing and backtesting systematic trading strategies. It focuses on the application of statistical learning to identify predictive signals (alphas) while maintaining a rigorous risk-management framework.

## 🧠 Methodology: The Science of the Alpha

Quant research requires a move beyond simple visual backtesting into formal statistical validation.

### 1. Walk-Forward Cross-Validation (WF-CV)
To prevent look-ahead bias and overfitting, we utilize a sliding window model optimization.
- **Training**: Anchored to historical data.
- **Validation**: Rolling OOS (Out-of-Sample) period for parameter tuning.
- **Testing**: Pure walk-forward execution to simulate real-world deployment.

### 2. Risk-Adjusted Performance (Sharpe & MDD)
We prioritize risk-adjusted returns over absolute nominal gains.
- **Sharpe Ratio ($\sqrt{52} \cdot \frac{\mu}{\sigma}$)**: Annualized metric for consistency.
- **Maximum Drawdown (MDD)**: Measuring the peak-to-trough resilience of the equity curve to understand 'tail risk'.

### 3. Strategy Robustness (Monte Carlo)
A successful backtest may still be the result of 'lucky' market regimes.
- **Implementation**: We perform **Bootstrap Monte Carlo Simulations** (5,000+ trajectories) on empirical return distributions.
- **Assurance**: By simulating thousands of 'alternative histories', we quantify the **Probability of Ruin (PoR)** and ensure the strategy is robust to high-variance regimes.

## 🛠️ Project Structure

```text
├── strategy.py           # Core Backtesting & Monte Carlo Engine
├── evaluate_strategy.py  # Performance profiling suite
├── notebooks/
│   ├── 01_eda.ipynb      # Market regime analysis
│   ├── 03_baseline.ipynb # Benchmarking vs Buy-and-Hold
└── data/                 # Cleaned OHLCV market data (AAPL, SPY)
```

## 🚀 Usage

1. **Run Backtest & Risk Profile**:
   ```bash
   python strategy.py
   ```

The next phase of this research involves the implementation of a constraint generated optimization.

2. **Generate Visualization**:
   ```bash
   python generate_plot.py
   ```

---
*Developed as part of my Applied Data Science & ML Engineering Portfolio.*
