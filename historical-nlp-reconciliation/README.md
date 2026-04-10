# 📜 Historical NLP Reconciliation: Global Optimal Entity Matching

This project implements a sophisticated reconciliation pipeline to align historical legal records (King's Bench, 17th Century) between human-curated ground truth and noisy AI-extracted data. It addresses the challenges of OCR errors, historical spelling variations, and fragmented document structures.

## 🧠 Methodology: The Hungarian Reconciliation

Traditional fuzzy matching is insufficient for many-to-one record reconciliation where multiple fragmented AI extractions might map to a single historical record.

### 1. Global Optimization (Hungarian Algorithm)
Instead of greedy local matching, we model the problem as a Bipartite Matching task. We utilize the **Hungarian Algorithm (Kuhn-Munkres, 1955)** to minimize the global edit-distance between the ground truth set and the extracted set. This ensures the most mathematically optimal alignment across the entire dataset.

### 2. Domain-Specific Heuristics
- **Occupational Filtering**: Historical records are laden with occupational titles (e.g., *Yeoman*, *Husbandman*, *Spinster*) that act as semantic noise. Our pipeline implements a domain-aware stop-word filter and entity grouping logic to isolate core name tokens.
- **Weighted Attribute Scoring**: Matches are computed via a composite score of:
  - **Phonetic Name Similarity**: 75% weight (Rapidfuzz Token Sort Ratio).
  - **County Alignment**: +15 bonus for categorical matches.
  - **Plea Reconciliation**: +10 bonus for semantic alignment of legal pleas.
  - **Entity Count Penalty**: -20 penalty for significant divergence in the number of litigants.

## 🛠️ Project Structure

```text
├── src/
│   ├── engine.py       # Core Reconciliation Logic (Hungarian Matching)
│   ├── accuracy.py     # Evaluation Module (F1, Precision, Recall)
├── data/               # Raw and Processed JSON records
├── tests/              # Unit tests for matching heuristics
└── Dockerfile          # Containerized pipeline execution
```

## 🚀 Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Pipeline**:
   ```bash
   python -m src.engine
   ```

3. **Verify via Docker**:
   ```bash
   docker compose up
   ```

---
*Developed as part of my Applied Data Science & ML Engineering Portfolio.*
