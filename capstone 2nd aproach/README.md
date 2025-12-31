
# King's Bench Plea Rolls (KB 27/799) Reconciliation Project

**DSCI 321 Capstone Project**

This project implements an automated reconciliation pipeline to validate AI-transcribed historical legal documents (1461, 39 Henry VI) against human-curated ground truth data. It solves the "Split Case" problem using a weighted bipartite matching algorithm and visualizes the results through an interactive web dashboard.

## 📂 Key File Structure

```
capstone-project-ansc25/
├── datasets/
│   ├── AI_HTR_Output.json          # Raw AI transcription source
│   ├── KB_Table.html               # Raw Human Ground Truth (scraped HTML)
│   ├── KB_Table_Parsed.json        # Parsed Human Ground Truth
│   ├── standardized_ai_data.json   # Processed AI data (flat structure)
│   ├── master_reconciliation.json  # Final reconciled output (Human <-> AI pairs)
│   ├── FINAL_ACCURACY.json         # Aggregated and case-by-case accuracy metrics
│   └── network_stats.json          # Graph analysis data
├── webapp/
│   ├── app.py                      # Flask server backend
│   └── static/                     # Frontend (HTML/CSS/JS)
├── parse_ai_json_file.py           # Script: Preprocesses AI JSON
├── parse_kb_html_file.py           # Script: Scrapes/Parses Human HTML
├── reconciliation.py               # Core Algorithm: Similarity & Bipartite Matching
├── parse_accuracy_by_case.py       # Script: Generates final accuracy report
├── network_analysis.py             # Script: Generates social network graph & stats
├── versatile_digraph.py            # Helper class for graph visualization
└── requirements.txt                # Python dependencies
```
>

## 🚀 Setup & Running Instructions

### Prerequisites
- Python 3.8+
- Packages from `requirements.txt`

### Installation
1. Navigate to the project root directory:
   ```bash
   cd capstone 2nd aproach
   ```
2. Install Python dependencies:
(flask, beautifulsoup4, rapidfuzz, numpy, pandas, scipy, bokeh, graphviz)
   
---

### Option A: Run from Existing Data (Quick Start)
The repository comes with pre-computed JSON outputs. To explore the results immediately:

1. **Start the Web Application:**
   ```bash
   cd webapp
   python app.py
   ```
2. **Open in Browser:**
   Navigate to `http://localhost:5000`

3. **Explore the tabs**
---

### Option B: Run Pipeline from Scratch
To regenerate all datasets and statistics from the raw source files, execute the scripts in this order:

1. **Parse Human Ground Truth:**
   ```bash
   python parse_kb_html_file.py
   ```
   *Output: `datasets/KB_Table_Parsed.json`*

2. **Parse AI HTR Data:**
   ```bash
   python parse_ai_json_file.py
   ```
   *Output: `datasets/standardized_ai_data.json`*

3. **Run Reconciliation Algorithm:**
   ```bash
   python reconciliation.py
   ```
   *Output: `datasets/master_reconciliation.json`, `datasets/split_cases_report.json`*

4. **Run Network Analysis:**
   ```bash
   python network_analysis.py
   ```
   *Output: `datasets/network_stats.json`, `webapp/static/litigation_network.png`*

5. **Start Web App:**
   ```bash
   cd webapp
   python app.py
   ```

---

### 🖥️ UI Dashboard Guide

| Tab | Description |
|-----|-------------|
| **Presentation** | Interactive slide deck explaining the project methodology and results. |
| **Case Explorer** | Side-by-side inspection of Human GT vs. AI Output with search and filtering. |
| **Image Statistics** | Heatmap of case count discrepancies per image to identify split cases. |
| **Reconciliation** | Detailed view of matched pairs, split-merges, and confidence scores. |
| **Accuracy** | Performance metrics (Precision/Recall/F1) for every reconciled case. |
| **Network Analysis** | Interactive litigation graph and analysis of top plaintiffs/defendants. |

## 📁 Data Strategy

We bridge the gap between unstructured 15th-century nomenclature and modern structured data.

### 1. Human Ground Truth (`parse_kb_html_file.py`)
*   **Source:** Scraped from AALT website HTML tables.
*   **Mapping:** Converts text labels like `"f 38"` (face) or `"d 153"` (dorse) into integer `image_num` (38, 153).
*   **Parsing:** Splits comma-separated name lists into arrays.

### 2. AI HTR Output (`parse_ai_json_file.py`)
*   **Source:** Raw JSON from the AI text recognition model.
*   **Mapping:** Extracts integers from file paths like `"IMG_0038"` → `38`.
*   **Parsing:** Flattens complex nested objects (e.g., separate first/last name fields) into comparable string tokens.

### Dataset Files Reference

| Stage       | File                        | Description                                         | Serve Path (localhost:5000)         |
|-------------|-----------------------------|-----------------------------------------------------|--------------------------------------|
| **Raw**     | `KB_Table.html`             | Original scraped HTML source.                       | `http://localhost:5000/datasets/KB_Table.html`                        |
| **Raw**     | `AI_HTR_Output.json`        | Original AI model output.                           | `http://localhost:5000/datasets/AI_HTR_Output.json`                        |
| **Processed**| `KB_Table_Parsed.json`     | Standardized Human GT schema.                       | `http://localhost:5000/datasets/KB_Table_Parsed.json`    |
| **Processed**| `standardized_ai_data.json`| Standardized AI schema.                             | `http://localhost:5000/datasets/standardized_ai_data.json`|
| **Output**  | `master_reconciliation.json`| **Final Result**: The reconciled dataset with metrics.| `http://localhost:5000/datasets/master_reconciliation.json`|
