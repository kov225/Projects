# King's Bench Plea Rolls (KB27/799) : HTR Reconciliation v2 (Flask Dashboard)

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)]()
[![NLP](https://img.shields.io/badge/NLP-FuzzyMatching-blueviolet)]()

> **Full-stack entity resolution pipeline with an interactive Flask web dashboard : reconciling AI-transcribed historical legal manuscripts against human ground truth at scale.**

---

## 🎯 Business Problem Solved

Manual reconciliation of AI-generated transcriptions against authoritative records is a bottleneck in large-scale digitization projects. This system **fully automates the pipeline** end-to-end, from data parsing to a production-ready web interface.

---

## 📊 Key Results / Impact

| Metric | Result |
|--------|--------|
| Unique Entities Identified | **1,200+** individuals |
| Split Cases Resolved | Automated via bipartite matching |
| Precision / Recall (Fuzzy) | 0.008 / 0.006 |
| Dashboard Tabs | 6 interactive views |
| Deployment | Flask web app at `localhost:5000` |

---

## 📂 Project Structure

```text
├── datasets/
│   ├── AI_HTR_Output.json           # Raw AI transcription source
│   ├── KB_Table.html                # Raw human ground truth
│   ├── KB_Table_Parsed.json         # Parsed GT schema
│   ├── standardized_ai_data.json    # Processed AI schema
│   ├── master_reconciliation.json   # ✅ Final reconciled output
│   └── FINAL_ACCURACY.json          # Aggregated F1 metrics
├── parse_kb_html_file.py            # GT scraper: HTML → JSON
├── parse_ai_json_file.py            # AI parser: nested JSON → flat schema
├── reconciliation.py                # Core: weighted bipartite matching engine
├── parse_accuracy_by_case.py        # Evaluation: Precision / Recall / F1
└── requirements.txt
```

---

## 🚀 Quick Start

### Option A: Explore Pre-Computed Results
Pre-computed JSON outputs are included. Start the dashboard immediately:
```bash
pip install -r requirements.txt
# open datasets/master_reconciliation.json directly
```

### Option B: Run Full Pipeline From Scratch
```bash
# 1. Parse human GT
python parse_kb_html_file.py

# 2. Parse AI HTR data
python parse_ai_json_file.py

# 3. Run reconciliation
python reconciliation.py

# 4. Generate accuracy report
python parse_accuracy_by_case.py
```

---

## 🔮 Future Work
- Integrate transformer-based name embeddings for improved fuzzy matching
- Deploy as a containerized FastAPI service for production use
- Expand coverage to full King's Bench archive (10,000+ records)
