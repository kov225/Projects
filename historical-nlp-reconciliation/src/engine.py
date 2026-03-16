import json
import logging
import os
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from typing import Dict, Any, List, Tuple
from .accuracy import calculate_final_accuracy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION - Central place for all thresholds
# ============================================================
CONFIG = {
    "similarity_threshold": 50.0,
    "county_bonus_threshold": 85,
    "plea_bonus_threshold": 70,
    "size_penalty_threshold": 4,
    "name_match_threshold": 90,
    "plea_containment_threshold": 70,
    "name_weight": 0.75,
    "county_bonus": 15,
    "plea_bonus": 10,
    "size_penalty": 20,
}

def group_tokens_into_entities(tokens: List[str]) -> List[str]:
    """Heuristic to group fragmented tokens into full person entities."""
    entities = []
    current_parts = []
    
    skip_next = False
    for t in tokens:
        if not t: continue
        if skip_next:
            skip_next = False
            continue
            
        low = t.lower().strip()
        if low == "of":
            skip_next = True
            continue
            
        if low in ["husbandman", "yeoman", "clerk", "gent", "esq", "knight", "widow", "laborer", "spinster", "chapman"]:
            continue
            
        if low.startswith("of ") or "together with" in low:
            continue
            
        current_parts.append(t.strip())
        if len(current_parts) >= 2:
            entities.append(f"{current_parts[1]} {current_parts[0]}")
            current_parts = []
            
    if current_parts:
        entities.append(" ".join(current_parts))
    return entities

def calculate_accuracy_metrics(human_data: Dict[str, Any], ai_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate accuracy metrics comparing human ground truth to AI prediction."""
    metrics = {"county_match": False, "plea_match": False, "name_precision": 0.0, "name_recall": 0.0, "name_f1": 0.0}
    
    # County
    gt_county = str(human_data.get('county', '')).strip().lower()
    ai_county = str(ai_data.get('county', '')).strip().lower()
    metrics["county_match"] = (gt_county == ai_county) if gt_county and ai_county else False
    
    # Plea
    gt_plea = str(human_data.get('plea', '')).strip().lower()
    ai_plea = str(ai_data.get('plea', '')).strip().lower()
    ai_plea_details = str(ai_data.get('plea_details', '')).strip().lower()
    
    if gt_plea and (ai_plea or ai_plea_details):
        score = max(fuzz.token_set_ratio(gt_plea, ai_plea), fuzz.token_set_ratio(gt_plea, ai_plea_details))
        metrics["plea_match"] = score >= CONFIG["plea_containment_threshold"]
    
    # Names
    gt_names = [str(n).strip().lower() for n in (human_data.get('plaintiffs', []) + human_data.get('defendants', [])) if str(n).strip()]
    ai_names = [str(n).strip().lower() for n in (ai_data.get('plaintiffs', []) + ai_data.get('defendants', [])) if str(n).strip()]
    
    if gt_names and ai_names:
        tp_p = sum(1 for a in ai_names if any(fuzz.ratio(a, g) >= CONFIG["name_match_threshold"] for g in gt_names))
        precision = tp_p / len(ai_names)
        tp_r = sum(1 for g in gt_names if any(fuzz.ratio(g, a) >= CONFIG["name_match_threshold"] for a in ai_names))
        recall = tp_r / len(gt_names)
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        metrics.update({"name_precision": round(precision, 4), "name_recall": round(recall, 4), "name_f1": round(f1, 4)})
    elif not gt_names and not ai_names:
        metrics.update({"name_precision": 1.0, "name_recall": 1.0, "name_f1": 1.0})
        
    return metrics

class KingsBenchReconciler:
    def __init__(self, gt_data: List[Dict[str, Any]], ai_data: List[Dict[str, Any]]):
        self.gt_data = gt_data
        self.ai_data = ai_data
        self.merge_id_counter = 10000
        self.ai_by_image = defaultdict(list)
        for idx, case in enumerate(ai_data):
            self.ai_by_image[case['image_num']].append((idx, case))

    def calculate_similarity(self, gt_case: Dict[str, Any], ai_case: Dict[str, Any]) -> float:
        gt_names = " ".join(group_tokens_into_entities(gt_case.get('plaintiffs', [])) + group_tokens_into_entities(gt_case.get('defendants', [])))
        ai_names = " ".join(group_tokens_into_entities(ai_case.get('plaintiffs', [])) + group_tokens_into_entities(ai_case.get('defendants', [])))
        name_score = fuzz.token_sort_ratio(gt_names, ai_names)
        
        bonus = 0
        if fuzz.partial_ratio(str(gt_case.get('county')).lower(), str(ai_case.get('county')).lower()) > CONFIG["county_bonus_threshold"]:
            bonus += CONFIG["county_bonus"]
        if max(fuzz.token_set_ratio(str(gt_case.get('plea')).lower(), str(ai_case.get('plea')).lower()), 
               fuzz.token_set_ratio(str(gt_case.get('plea')).lower(), str(ai_case.get('plea_details')).lower())) > CONFIG["plea_bonus_threshold"]:
            bonus += CONFIG["plea_bonus"]
            
        gt_len = len(gt_case.get('plaintiffs', [])) + len(gt_case.get('defendants', []))
        ai_len = len(ai_case.get('plaintiffs', [])) + len(ai_case.get('defendants', []))
        penalty = CONFIG["size_penalty"] if abs(gt_len - ai_len) > CONFIG["size_penalty_threshold"] else 0
        
        return max(0, min(100, (name_score * CONFIG["name_weight"]) + bonus - penalty))

    def run_reconciliation(self, threshold=None):
        threshold = threshold or CONFIG["similarity_threshold"]
        num_gt, num_ai = len(self.gt_data), len(self.ai_data)
        cost_matrix = np.full((num_gt * 2, num_ai), 10000.0)
        
        logger.info(f"Reconciling {num_gt} records...")
        for g_idx, g_case in enumerate(self.gt_data):
            img = g_case.get('image_num', 0)
            for i in range(img - 1, img + 2):
                for a_idx, a_case in self.ai_by_image[i]:
                    score = self.calculate_similarity(g_case, a_case)
                    if score >= threshold:
                        cost_matrix[g_idx, a_idx] = cost_matrix[g_idx + num_gt, a_idx] = 100.0 - score

        rows, cols = linear_sum_assignment(cost_matrix)
        matches = defaultdict(list)
        for r, c in zip(rows, cols):
            if cost_matrix[r, c] < 9000:
                matches[r % num_gt].append({"ai_case": self.ai_data[c], "score": 100.0 - cost_matrix[r, c]})
        
        return self.generate_reports(matches)

    def generate_reports(self, matches):
        split_report, master_recon = [], []
        for g_idx, ai_list in matches.items():
            gt = self.gt_data[g_idx]
            if len(ai_list) > 1:
                ai_list.sort(key=lambda x: x['ai_case']['image_num'])
                new_id = self.merge_id_counter
                self.merge_id_counter += 1
                
                merged = {
                    "reconciled_id": new_id, "is_split_merge": True,
                    "constituent_ai_cases": [x['ai_case']['ai_caseid'] for x in ai_list],
                    "image_num": ai_list[0]['ai_case']['image_num'],
                    "county": ai_list[0]['ai_case'].get('county'),
                    "plea": ai_list[0]['ai_case'].get('plea'),
                    "plaintiffs": list(set(sum([x['ai_case'].get('plaintiffs', []) for x in ai_list], []))),
                    "defendants": list(set(sum([x['ai_case'].get('defendants', []) for x in ai_list], []))),
                    "full_text": " [SPLIT] ".join([x['ai_case'].get('full_text', '') for x in ai_list]),
                    "score": {str(x['ai_case']['ai_caseid']): x['score'] for x in ai_list}
                }
                split_report.append({"human_id": gt.get('human_case_id'), "ai_ids": merged["constituent_ai_cases"], "merged_id": new_id})
                recon_ai = merged
            else:
                match = ai_list[0]
                recon_ai = match['ai_case'].copy()
                recon_ai.update({"reconciled_id": recon_ai['ai_caseid'], "is_split_merge": False, "score": {str(recon_ai['ai_caseid']): match['score']}})

            master_recon.append({"human_CaseId": gt.get('human_case_id'), "human_data": gt, "reconciled_ai_data": recon_ai, "accuracy_metrics_v1": calculate_accuracy_metrics(gt, recon_ai)})
        
        return split_report, master_recon

def main():
    logger.info("Starting professional reconciliation pipeline...")
    data_dir = "data"
    try:
        with open(f"{data_dir}/KB_Table_Parsed.json", "r", encoding="utf-8") as f: gt = json.load(f)
        with open(f"{data_dir}/standardized_ai_data.json", "r", encoding="utf-8") as f: ai = json.load(f)
    except Exception as e:
        logger.error(f"Data load failure: {e}")
        return

    reconciler = KingsBenchReconciler(gt, ai)
    splits, master = reconciler.run_reconciliation()

    with open(f"{data_dir}/split_cases_report.json", "w", encoding="utf-8") as f: json.dump(splits, f, indent=2)
    with open(f"{data_dir}/master_reconciliation.json", "w", encoding="utf-8") as f: json.dump(master, f, indent=2)
    
    logger.info("Pipeline complete. Running accuracy audit...")
    calculate_final_accuracy()

if __name__ == "__main__":
    main()