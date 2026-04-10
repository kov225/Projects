# 📉 Dataset Shift Analysis: Model Robustness Benchmarking

This project provides a framework for quantifying the impact of distributional shift on machine learning models. It implements synthetic injection of Covariate Shift, Prior Probability Shift, and Concept Shift to stress-test model performance in non-stationary environments.

## 🧠 Methodology: Quantifying the Divergence

Understanding how and why a model fails in production requires a formal decomposition of the shift.

### 1. Shift Categories
- **Covariate Shift ($P(X) \neq P'(X)$)**: The distribution of inputs changes, but the labeling function is invariant. We simulate this via Gaussian and Laplacian noise injection.
- **Prior Probability Shift ($P(Y) \neq P'(Y)$)**: The relative frequency of labels changes. This is common in fraud and credit scenarios where market conditions alter default rates.
- **Concept Shift ($P(Y|X) \neq P'(Y|X)$)**: The fundamental relationship between features and targets changes. We simulate this via "Concept-Adjacent" corruption of key features.

### 2. Robustness Metrics
Beyond standard AUC/F1, the framework computes:
- **Performance Decay Rate**: The slope of performance loss relative to shift intensity.
- **Robustness Score**: A normalized metric (0-1) representing the model's stability under distribution variance.
- **Feature-Target Correlation Shift**: Measuring how the mutual information between top features and the target degrades.

## 🛠️ Project Structure

```text
├── src/
│   ├── shift_simulators.py  # Distributional perturbation suite
│   ├── evaluation.py        # Robustness scoring and metrics
│   ├── interpretability.py # SHAP-based shift analysis
│   └── run_experiments.py   # Automated benchmarking pipeline
├── results/               # Experiment logs and visualizations
└── app.py                 # Interactive dashboard for shift simulation
```

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Benchmarks**:
   ```bash
   python src/run_experiments.py
   ```

3. **Launch Dashboard**:
   ```bash
   streamlit run app.py
   ```

---
*Developed as part of my Applied Data Science & ML Engineering Portfolio.*
