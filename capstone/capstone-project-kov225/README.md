# King’s Bench Reconciliation Project
## Validating 15th Century Handwritten Text Recognition

This project builds a complete reconciliation pipeline for medieval English legal manuscripts from the King’s Bench (KB27/799). The goal is to compare AI generated HTR output against human curated ground truth, evaluate accuracy, reconstruct split cases, and extract social networks.

The workflow contains four stages:

1. Scraping and preparing the data  
2. Scoring similarity between GT and HTR cases  
3. Two slot bipartite matching to handle split cases  
4. Evaluation and network analysis  

---

## 1. Project Overview

Medieval manuscripts create a difficult reconciliation problem.

Historians often record one case that spans two physical manuscript pages.  
The AI extracts cases page-by-page, which splits single cases into multiple fragments.  

Our pipeline aligns both datasets and determines which HTR records correspond to each GT case.

---

## 2. Data Sources

### Ground Truth (GT)
- Source: WAALT KB27/799 webpage  
- Format: HTML table  
- Parsing rule:  
  - “f 38” → 38  
  - “d 153” → 153  

### HTR Dataset
- Source: htr_dataset_799.json  
- Correct field: `source_image_directory`  
  - Example: “IMG_0038” → 38  
- Note: The `image` field must be ignored due to OCR errors.

### Page Window Rule  
A GT case on page **N** can match only HTR cases from pages:  
**N - 1**, **N**, **N + 1**

This accounts for digitization misalignment.

---

## 3. Pipeline Architecture

### Phase 1: Scraping & Normalization (scraper.py)
- Extract HTML table  
- Normalize nested name fields  
- Parse image numbers  
- Produce standardized dictionaries  

### Phase 2: Similarity Scoring (similarity.py)
Similarity is calculated using:
- RapidFuzz name scoring  
- Place scoring  
- List-size penalty for mismatch  

Weights used:
defendant_weight = 0.7
place_weight = 0.3

Penalty: min(gt_list_size, htr_list_size) / max(gt_list_size, htr_list_size)


### Phase 3: Two Slot Bipartite Matching (reconciliation.py)
- Each GT case is duplicated into two “slots”  
- Candidates filtered via N±1 image rule  
- Similarity scores inserted into a cost matrix  
- Hungarian algorithm assigns best matches  
- Low-score matches below threshold are removed  

This enables:
- 1 GT → 1 HTR  
- 1 GT → 2 HTR (split cases)

### Phase 4: Evaluation & Network Analysis (analysis.py)
- Strict name evaluation  
- Optional fuzzy evaluation  
- Build historical social network graph  
- Identify central individuals  

---

## 4. Tests
Located in: `tests/test_similarity.py`

Covers:
- identical matching  
- completely different cases  
- partial overlaps  
- empty HTR  
- size penalty behavior  

---

## 5. Running the Project

Download datasets, then:
pip install -r requirements.txt

python reconciliation.py or py reconciliation.py
python analysis.py or py analysis.py 
pytest -q


---

## 6. Results Summary

Strict matching:
- Precision: 0.003  
- Recall: 0.003  
- F1: 0.003  

Fuzzy evaluation:
- Precision: 0.008  
- Recall: 0.006  
- F1: 0.007  

Network:
- Over 1,200 unique individuals  
- Dense co-defendant graph  
- Most connected name: **John**

---

## 7. Reflection

Challenges:
- medieval spelling variation  
- OCR misalignment  
- page-number inconsistencies  
- split-case reconstruction  
- tuning similarity thresholds  

Key insight:  
Even with low raw scores, the pipeline correctly reconstructs case alignments and delivers historically meaningful network structures.




