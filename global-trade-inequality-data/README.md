# 🌍 Global Trade Inequality: Distributional Economics

This project investigates the structural inequalities in international trade using longitudinal economic datasets. It combines traditional exploratory data analysis (EDA) with rigorous statistical measures of concentration and disparity to identify patterns of economic divergence.

## 🧠 Methodology: The Statistics of Inequality

Understanding global trade requires moving beyond aggregate means to analyze the full distribution of value across economic actors.

### 1. Inequality Quantification (Gini Coefficient)
We implement the **Gini Coefficient** to measure the statistical dispersion of trade value.
- **Formula**: $G = \frac{\sum_{i=1}^n \sum_{j=1}^n |x_i - x_j|}{2n^2\bar{x}}$
- **Interpretation**: Allows for a normalized comparison of inequality across different product categories and time periods, regardless of the absolute scale of trade.

### 2. Market Concentration (HHI)
To detect monopolistic patterns or trade dependencies, we utilize the **Herfindahl-Hirschman Index (HHI)**.
- **Application**: Identifying "Bottleneck" regions where global trade for specific commodities is concentrated in a handful of actors.

### 3. Hypothesis Testing: Pricing & Power
The repository includes comparative studies on software pricing strategies, utilizing **Welch’s T-Tests** to identify significant market shifts in licensing models across different economic tiers.

## 🛠️ Project Structure

```text
├── src/
│   ├── inequality_metrics.py  # Gini & HHI Calculation Engine
├── Cheema_Vennalakanti_Global Trade Inequality.ipynb  # Primary Research Notebook
├── Vennalakanti_Koushik_softwarePricing.ipynb        # Pricing Analysis Notebook
└── reports/                   # Formal PDF Research Summaries
```

## 🚀 Quick Start

1. **Calculate Concentration Metrics**:
   ```bash
   python src/inequality_metrics.py
   ```

2. **Run Interactive Analysis**:
   Open the Jupyter notebooks in the root directory to view the full longitudinal study.

---
*Developed as part of my Applied Data Science & ML Engineering Portfolio.*
