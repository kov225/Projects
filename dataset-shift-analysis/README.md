# Dataset Shift: Assessing Model Robustness to Environmental Change

This repository contains a longitudinal study on the performance degradation of classical machine learning algorithms under various dataset shift regimes. Developed as part of a Machine Learning course project, the study quantifies model resilience using statistical benchmarks and divergence metrics.

**Current Project Phase**: Milestone 1 (Statistical Baselines & Benchmarking)

---

## Technical Overview
The project evaluates **seven classical ML architectures** against a **Naive Baseline** (`DummyClassifier`) to determine their inherent stability when the i.i.d. assumption is violated. We utilize the UCI Adult Income dataset to simulate environmental drift through controlled feature and label corruption.

### Key Deliverables
- **Statistical Benchmarking**: Inclusion of a majority-class baseline to provide a performance floor.
- **Uncertainty Quantification**: Implementation of statistical bootstrapping (95% CI) for all accuracy and calibration metrics.
- **Divergence Monitoring**: Integration of the Kolmogorov-Smirnov (KS) test to quantify physical data shift magnitude.
- **Interactive Dashboard**: A Streamlit application for multi-dimensional visualization of performance decay.

## Project Structure
The repository is organized into a modular package structure:
- `dataset-shift-project/`: The core research directory.
  - `src/`: Source code for simulators, models, and evaluation routines.
  - `results/`: Historical benchmark data and statistical logs.
  - `app.py`: Interface for the research dashboard.

## Next Steps: Milestone 2 Plan
The upcoming phase transitions from benchmarking to diagnostic analysis:
1. **Deeper Shift Simulation**: Scaling drift and feature value permutation.
2. **Robustness Scoring**: Unified ranking system based on AUC degradation rates.
3. **Interpretability**: Analysis of feature importance drift and decision boundary collapse.
