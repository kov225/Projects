# 📈 Algorithmic Trading Research: Systematic Alpha Extraction

![Apple Feature Correlation](assets/feature_correlation_heatmap.png)

This project represents an **independent quantitative research initiative** focused on the multi-stage development of a systematic trading framework for Apple (AAPL) market data. The research methodology prioritizes **market microstructure analysis**, price action dynamics, and structural feature engineering over opaque black-box modeling.

## 🧠 Research Philosophy

Real-world quantitative trading requires a move beyond simple indicators into a formal understanding of **market topology**. This project emphasizes:
- **Reasoning over Prediction**: Building a rule-based scoring logic that mimics professional discretionary trader logic.
- **Structural Integrity**: Identifying non-stationary market regimes (e.g., support/resistance zones) to contextualize signals.
- **Iterative Refinement**: Preserving the 'evolution of thought' through tiered research notebooks, from raw exploration to final strategy validation.

---

## 📂 Project Architecture

### 📁 Quantitative Data Store
- **[aapl.us.txt](file:///e:/Projects/Projects/algorithmic-trading-research/aapl.us.txt)**: Raw high-resolution historical Apple equity data.
- **[cleaned_apple.csv](file:///e:/Projects/Projects/algorithmic-trading-research/cleaned_apple.csv)**: Pre-processed, analysis-ready dataset with integrity verification.

### 📓 Tiered Research Notebooks
| Phase | Research Focus | Key Output |
| :--- | :--- | :--- |
| **01 EDA** | Initial feature distributions & cleaning | Signal-to-noise baseline |
| **02 Logic** | Feature engineering & momentum patterns | Normalized alpha features |
| **03 Zones** | Support and resistance structural mapping | Regime detection logic |
| **04 Scoring** | Final alpha signal & Tue-Thu framework | Integrated strategy engine |

---

## 🔍 The Research Pipeline

### 1. Feature Engineering & Normalization
We implement expanding-window normalization to transform non-stationary price data into Z-scored features, preventing look-ahead bias and ensuring scale independence across different market regimes.

### 2. Walk-Forward Cross-Validation
Rather than standard K-fold CV, we utilize a **sliding-window optimization** (WF-CV) to simulate real-world deployment. This approach validates the strategy's stability against temporal drift.

### 3. Systematic Scoring Framework
We develop a custom multi-objective scoring function that balances:
- **Precision**: Signal accuracy in the validation window.
- **Chattiness**: Signal frequency to ensure statistical significance.
- **Correctness**: Mean expected value of identified trades.

---

## 🛠️ Tech Stack & Methodology

- **Language**: Python 3.13
- **Analysis**: Pandas, NumPy, SciPy (Bipartite Matching & Statistical Testing)
- **Modelling**: Scikit-Learn (Decision Tree classifiers for non-linear boundary detection)
- **Validation**: Walk-forward backtesting with risk-adjusted performance profiling (Sharpe, MDD).

## 🔮 Future Directions
The next phase of this research involves the implementation of constrained-gradient optimization and a more granular transaction cost modeling layer to simulate high-frequency slippage.

---
*Developed under the research initiative of Professor Bilal Khan. Part of my Applied Data Science & ML Engineering Portfolio.*
