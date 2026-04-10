import json
import logging
import os
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from typing import Dict, Any, List, Tuple
from .accuracy import calculate_final_accuracy

# Professional Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# RECONCILIATION PARAMETERS
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
    """
    Heuristic to group fragmented tokens into person-entities.
    
    Handles historical honorifics and occupational suffixes (e.g., 'Husbandman', 'Yeoman') 
    that act as noise in semantic name matching.
    """
    entities = []
    current_parts = []
    
    # Historical occupational noise tokens
    STOP_WORDS = {
        "husbandman", "yeoman", "clerk", "gent", "esq", "knight", 
        "widow", "laborer", "spinster", "chapman", "of"
    }
    
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
            
        if low in STOP_WORDS or low.startswith("of "):
            continue
            
        current_parts.append(t.strip())
        if len(current_parts) >= 2:
            # Assumes format: [First, Last] -> Last First for sorting consistency
            entities.append(f"{current_parts[1]} {current_parts[0]}")
            current_parts = []
            
    if current_parts:
        entities.append(" ".join(current_parts))
    return entities

class KingsBenchReconciler:
    """
    Entity Reconciliation Engine for High-Variance Historical Records.
    
    Implements Global Optimal Matching using the Hungarian Algorithm 
    (Kuhn, 1955) to solve the bipartite matching problem between human-curated 
    ground truth and high-noise AI extractions.
    """
    
    def __init__(self, gt_data: List[Dict[str, Any]], ai_data: List[Dict[str, Any]]):
        self.gt_data = gt_data
        self.ai_data = ai_data
        self.merge_id_counter = 10000
        self.ai_by_image = defaultdict(list)
        for idx, case in enumerate(ai_data):
            self.ai_by_image[case['image_num']].append((idx, case))

    def calculate_similarity(self, gt_case: Dict[str, Any], ai_case: Dict[str, Any]) -> float:
        """
        Calculates a composite similarity score between two court records.
        
        Uses weighted Fuzzy String Matching (Token Sort Ratio) for robustness 
        against OCR character errors and historical spelling variations.
        """
        gt_names = " ".join(group_tokens_into_entities(gt_case.get('plaintiffs', [])) + 
                           group_tokens_into_entities(gt_case.get('defendants', [])))
        ai_names = " ".join(group_tokens_into_entities(ai_case.get('plaintiffs', [])) + 
                           group_tokens_into_entities(ai_case.get('defendants', [])))
        
        # Base phonetic/structural similarity
        name_score = fuzz.token_sort_ratio(gt_names, ai_names)
        
        bonus = 0
        # Categorical bonuses for attribute alignment
        if fuzz.partial_ratio(str(gt_case.get('county')).lower(), 
                              str(ai_case.get('county')).lower()) > CONFIG["county_bonus_threshold"]:
            bonus += CONFIG["county_bonus"]
            
        # Plea alignment (handling multi-field OCR targets)
        plea_score = max(
            fuzz.token_set_ratio(str(gt_case.get('plea')).lower(), str(ai_case.get('plea')).lower()),
            fuzz.token_set_ratio(str(gt_case.get('plea')).lower(), str(ai_case.get('plea_details')).lower())
        )
        if plea_score > CONFIG["plea_bonus_threshold"]:
            bonus += CONFIG["plea_bonus"]
            
        # Entity count penalty (penalize hallucinated or missing parties)
        gt_len = len(gt_case.get('plaintiffs', [])) + len(gt_case.get('defendants', []))
        ai_len = len(ai_case.get('plaintiffs', [])) + len(ai_case.get('defendants', []))
        penalty = CONFIG["size_penalty"] if abs(gt_len - ai_len) > CONFIG["size_penalty_threshold"] else 0
        
        return max(0, min(100, (name_score * CONFIG["name_weight"]) + bonus - penalty))

    def run_reconciliation(self, threshold=None):
        """
        Pipeline entry point. Resolves records using global cost optimization.
        """
        threshold = threshold or CONFIG["similarity_threshold"]
        num_gt, num_ai = len(self.gt_data), len(self.ai_data)
        
        # Initialize cost matrix (Distance = 100 - Similarity)
        # Using num_gt * 2 to account for potential split-case merges
        cost_matrix = np.full((num_gt * 2, num_ai), 10000.0)
        
        logger.info(f"Initiating global optimization for {num_gt} Ground Truth records...")
        
        for g_idx, g_case in enumerate(self.gt_data):
            img = g_case.get('image_num', 0)
            # Localized Search Window: Only compare to adjacent images to optimize compute
            for i in range(img - 1, img + 2):
                for a_idx, a_case in self.ai_by_image[i]:
                    score = self.calculate_similarity(g_case, a_case)
                    if score >= threshold:
                        cost_matrix[g_idx, a_idx] = 100.0 - score
                        cost_matrix[g_idx + num_gt, a_idx] = 100.0 - score

        # Kuhn-Munkres (Hungarian Algorithm) for bipartite matching
        rows, cols = linear_sum_assignment(cost_matrix)
        
        matches = defaultdict(list)
        for r, c in zip(rows, cols):
            if cost_matrix[r, c] < 9000:
                matches[r % num_gt].append({
                    "ai_case": self.ai_data[c], 
                    "score": 100.0 - cost_matrix[r, c]
                })
        
        return self.generate_reports(matches)

    def generate_reports(self, matches):
        """Processes matches into normalized output structures."""
        # ... logic remains similar but with professional logging ...
        from .engine import calculate_accuracy_metrics
        split_report, master_recon = [], []
        for g_idx, ai_list in matches.items():
            gt = self.gt_data[g_idx]
            if len(ai_list) > 1:
                recon_ai = self._merge_ai_cases(ai_list)
                split_report.append({
                    "human_id": gt.get('human_case_id'), 
                    "ai_ids": recon_ai["constituent_ai_cases"]
                })
            else:
                recon_ai = ai_list[0]['ai_case'].copy()
                recon_ai.update({"score": ai_list[0]['score']})

            master_recon.append({
                "human_CaseId": gt.get('human_case_id'),
                "reconciled_ai_data": recon_ai,
                "accuracy": calculate_accuracy_metrics(gt, recon_ai)
            })
        
        return split_report, master_recon

    def _merge_ai_cases(self, ai_list):
        """Merges multiple AI-extracted records that map to a single human entry."""
        new_id = self.merge_id_counter
        self.merge_id_counter += 1
        return {
            "reconciled_id": new_id,
            "is_split_merge": True,
            "constituent_ai_cases": [x['ai_case']['ai_caseid'] for x in ai_list],
            "plaintiffs": list(set(sum([x['ai_case'].get('plaintiffs', []) for x in ai_list], []))),
            "defendants": list(set(sum([x['ai_case'].get('defendants', []) for x in ai_list], [])))
        }

def main():
    logger.info("Executing NLP Reconciliation Pipeline...")
    # ... setup and run logic ...
    pass