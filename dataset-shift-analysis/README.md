# Dataset Shift Analysis

Dataset Shift Analysis is a rigorous research project dedicated to quantifying the robustness of machine learning models in the presence of non stationary environments. In real world deployments, the distribution of features and labels often diverges from the training data, leading to silent performance degradation that is difficult to diagnose. This project provides a robust framework for simulating multiple categories of dataset shift, such as covariate shift and prior probability shift, and implements a unified robustness scoring system to identify which architectures are most resilient to environmental changes.

## Key Results

| Model Architecture | Shift Type | Median AUC ROC | Robustness Score | Rank |
|---|---|---:|---:|---|
| Random Forest | Covariate Shift | 0.9412 | 92.4 | 1 |
| XGBoost | Concept Adjacent Shift | 0.9385 | 89.1 | 2 |
| Logistic Regression | Prior Shift | 0.9124 | 81.6 | 3 |
| Naive Bayes | Scaling Drift | 0.8845 | 74.2 | 4 |

## Methodology

This study utilizes a modular simulation pipeline to systematically manipulate the test distribution across multiple dimensions. Covariate shift is induced through additive and multiplicative noise on continuous features while prior probability shift is simulated by biasing the label distribution. We also implement more advanced scenarios like feature scaling drift and value permutation to challenge the model's reliance on specific joint dependencies. To quantify the impact of these changes, we calculate a unified robustness score by integrating the performance decay curve over a range of shift intensities. This provides a single number that reflects a model's stability better than individual point estimates at single intensity levels.

## Implementation

The project is implemented in a modular Python framework with clearly separated layers for data simulation, model training, and evaluation. We use a standardized interface for all simulators to ensure that different types of shift can be applied consistently across multiple datasets. The interpretability layer provides detailed views of how feature importance and prediction confidence distributions evolve as the shift intensity increases. This allows us to identify whether a model is failing gracefully or if its decision logic is fundamentally breaking down under pressure.

## Quickstart

Initialize the virtual environment, install the project dependencies, and execute the benchmark script to run the multi model shift analysis.

```bash
cd dataset-shift-analysis/dataset-shift-project
python -m venv .venv
# On Windows PowerShell use: .\.venv\Scripts\Activate.ps1
# On Unix or Mac use: source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiments.py
```

## Reproducing Results

The key results table and the associated performance plots can be reproduced by running the main experiment script which iterates through all models and shift categories. The output metrics are saved to a summary file that is used to compute the final robustness rankings.
