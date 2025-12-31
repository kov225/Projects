import json
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from parse_accuracy_by_case import calculate_final_accuracy

# ============================================================
# CONFIGURATION - Central place for all thresholds
# ============================================================
CONFIG = {
    # --- Similarity Engine Thresholds (existing) ---
    "similarity_threshold": 50.0,          # Minimum score to consider a match
    "county_bonus_threshold": 85,          # partial_ratio threshold for county bonus
    "plea_bonus_threshold": 70,            # token_set_ratio threshold for plea bonus
    "size_penalty_threshold": 4,           # Difference in person count before penalty
    
    # --- Accuracy Metrics Thresholds (new) ---
    "name_match_threshold": 90,            # Fuzzy match threshold for name comparison (precision/recall)
    "plea_containment_threshold": 70,      # token_set_ratio threshold for plea accuracy
    
    # --- Weight Configuration (existing) ---
    "name_weight": 0.75,                   # Weight for name score in similarity calculation
    "county_bonus": 15,                    # Bonus points for county match
    "plea_bonus": 10,                      # Bonus points for plea match
    "size_penalty": 20,                    # Penalty for large size difference
}


# ============================================================
# ACCURACY METRICS CALCULATION
# ============================================================
def calculate_accuracy_metrics(human_data, ai_data):
    """
    Calculate accuracy metrics comparing human ground truth to AI prediction.
    
    Returns dict with:
    - county_match: boolean
    - plea_match: boolean  
    - name_precision: float (0-1)
    - name_recall: float (0-1)
    - name_f1: float (0-1)
    """
    metrics = {
        "county_match": False,
        "plea_match": False,
        "name_precision": 0.0,
        "name_recall": 0.0,
        "name_f1": 0.0,
    }
    
    # --- 1. County Match (case-insensitive exact match) ---
    gt_county = str(human_data.get('county', '')).strip().lower()
    ai_county = str(ai_data.get('county', '')).strip().lower()
    metrics["county_match"] = (gt_county == ai_county) if gt_county and ai_county else False
    
    # --- 2. Plea Match (containment check using token_set_ratio) ---
    gt_plea = str(human_data.get('plea', '')).strip().lower()
    ai_plea = str(ai_data.get('plea', '')).strip().lower()
    ai_plea_details = str(ai_data.get('plea_details', '')).strip().lower()
    
    if gt_plea and (ai_plea or ai_plea_details):
        # Check against both short plea and detailed plea
        score_short = fuzz.token_set_ratio(gt_plea, ai_plea) if ai_plea else 0
        score_long = fuzz.token_set_ratio(gt_plea, ai_plea_details) if ai_plea_details else 0
        best_score = max(score_short, score_long)
        metrics["plea_match"] = best_score >= CONFIG["plea_containment_threshold"]
    elif not gt_plea and not ai_plea:
        # Both empty = match
        metrics["plea_match"] = True
    
    # --- 3. Name Precision & Recall (fuzzy token matching) ---
    # Get all name tokens from both sources
    gt_plaintiffs = human_data.get('plaintiffs', []) or []
    gt_defendants = human_data.get('defendants', []) or []
    ai_plaintiffs = ai_data.get('plaintiffs', []) or []
    ai_defendants = ai_data.get('defendants', []) or []
    
    gt_names = [str(n).strip().lower() for n in (gt_plaintiffs + gt_defendants) if str(n).strip()]
    ai_names = [str(n).strip().lower() for n in (ai_plaintiffs + ai_defendants) if str(n).strip()]
    
    if not gt_names and not ai_names:
        # Both empty = perfect match
        metrics["name_precision"] = 1.0
        metrics["name_recall"] = 1.0
        metrics["name_f1"] = 1.0
    elif not ai_names:
        # AI found nothing, GT has names = 0 recall
        metrics["name_precision"] = 1.0  # No false positives (nothing predicted)
        metrics["name_recall"] = 0.0
        metrics["name_f1"] = 0.0
    elif not gt_names:
        # GT empty, AI has names = 0 precision
        metrics["name_precision"] = 0.0
        metrics["name_recall"] = 1.0  # No false negatives (nothing to find)
        metrics["name_f1"] = 0.0
    else:
        # Calculate precision: Of AI names, how many match GT?
        true_positives_precision = 0
        for ai_name in ai_names:
            # Check if this AI name fuzzy-matches any GT name
            for gt_name in gt_names:
                if fuzz.ratio(ai_name, gt_name) >= CONFIG["name_match_threshold"]:
                    true_positives_precision += 1
                    break  # Count each AI name only once
        
        precision = true_positives_precision / len(ai_names) if ai_names else 0
        
        # Calculate recall: Of GT names, how many were found by AI?
        true_positives_recall = 0
        for gt_name in gt_names:
            # Check if this GT name fuzzy-matches any AI name
            for ai_name in ai_names:
                if fuzz.ratio(gt_name, ai_name) >= CONFIG["name_match_threshold"]:
                    true_positives_recall += 1
                    break  # Count each GT name only once
        
        recall = true_positives_recall / len(gt_names) if gt_names else 0
        
        # Calculate F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        metrics["name_precision"] = round(precision, 4)
        metrics["name_recall"] = round(recall, 4)
        metrics["name_f1"] = round(f1, 4)
    
    return metrics


class KingsBenchReconciler:
    def __init__(self, gt_data, ai_data):
        self.gt_data = gt_data
        self.ai_data = ai_data
        
        # Counter for generating new Integer IDs for merged cases
        self.merge_id_counter = 10000
        
        # Pre-indexing AI data by image_num for O(1) lookup during candidate filtering
        self.ai_by_image = defaultdict(list)
        for idx, case in enumerate(ai_data):
            # Storing original index to track back later
            self.ai_by_image[case['image_num']].append((idx, case))

    def calculate_similarity(self, gt_case, ai_case):
        """
        Phase 2: The Art of Similarity
        Returns a score 0-100.
        Uses CONFIG thresholds for tunability.
        """
        # 1. Construct Comparison Strings
        # Join lists into single strings for fuzzy comparison
        gt_names = " ".join(gt_case.get('plaintiffs', []) + gt_case.get('defendants', []))
        ai_names = " ".join(ai_case.get('plaintiffs', []) + ai_case.get('defendants', []))
        
        # 2. Name Matching (Heavy Weight)
        name_score = fuzz.token_sort_ratio(gt_names, ai_names)
        
        # 3. Context Bonus (County & Plea)
        context_bonus = 0
        
        # --- A. County Check ---
        gt_county = str(gt_case.get('county', '')).lower()
        ai_county = str(ai_case.get('county', '')).lower()
        # Partial ratio helps matches like "Herts" vs "Hertfordshire"
        if gt_county and ai_county and fuzz.partial_ratio(gt_county, ai_county) > CONFIG["county_bonus_threshold"]:
            context_bonus += CONFIG["county_bonus"]
            
        # --- B. Plea Check ---
        gt_plea = str(gt_case.get('plea', '')).lower()
        
        # We check against both the short AI 'plea' and the long 'plea_details'
        ai_plea_short = str(ai_case.get('plea', '')).lower()
        ai_plea_long = str(ai_case.get('plea_details', '')).lower()
        
        if gt_plea and (ai_plea_short or ai_plea_long):
            # Use token_set_ratio for "needle in haystack" logic.
            score_short = fuzz.token_set_ratio(gt_plea, ai_plea_short) if ai_plea_short else 0
            score_long = fuzz.token_set_ratio(gt_plea, ai_plea_long) if ai_plea_long else 0
            
            best_plea_score = max(score_short, score_long)
            
            if best_plea_score > CONFIG["plea_bonus_threshold"]:
                context_bonus += CONFIG["plea_bonus"]
            
        # 4. Size Penalty
        gt_len = len(gt_case.get('plaintiffs', [])) + len(gt_case.get('defendants', []))
        ai_len = len(ai_case.get('plaintiffs', [])) + len(ai_case.get('defendants', []))
        size_diff = abs(gt_len - ai_len)
        
        if size_diff > CONFIG["size_penalty_threshold"]: 
            penalty = CONFIG["size_penalty"]
        else:
            penalty = 0
        
        # Final calculation
        # Base is Name Score (0-100 scaled by weight) + Context Bonus - Penalty
        final_score = (name_score * CONFIG["name_weight"]) + context_bonus - penalty
        
        return max(0, min(100, final_score))

    def run_reconciliation(self, similarity_threshold=None):
        """
        Phase 3: The Great Reconciliation (Two-Slot Assignment)
        Uses CONFIG threshold if not specified.
        """
        if similarity_threshold is None:
            similarity_threshold = CONFIG["similarity_threshold"]
            
        num_gt = len(self.gt_data)
        num_ai = len(self.ai_data)
        
        # 1. The Two-Slot Setup
        # We need 2 rows for every GT case (Slot A and Slot B) to allow splitting.
        # Matrix shape: (2 * GT) x (AI)
        # We initialize with a high cost (infinity) so unrelated cases are never picked.
        cost_matrix = np.full((num_gt * 2, num_ai), 10000.0)
        
        print(f"Building Cost Matrix for {num_gt} GT cases and {num_ai} AI candidates...")
        print(f"Using similarity threshold: {similarity_threshold}")
        
        # 2. Fill Matrix (Sparse approach using N-1, N, N+1)
        for gt_idx, gt_case in enumerate(self.gt_data):
            img_n = gt_case['image_num']
            
            # Candidate Filtering: Get indices of AI cases in N-1, N, N+1
            candidates = []
            for i in range(img_n - 1, img_n + 2):
                candidates.extend(self.ai_by_image[i])
            
            for ai_global_idx, ai_case in candidates:
                # Calculate Similarity
                score = self.calculate_similarity(gt_case, ai_case)
                
                if score >= similarity_threshold:
                    # Convert Score to Cost (100 - Score)
                    cost = 100.0 - score

                    # Assign cost to BOTH slots for this GT case
                    # Slot 1 (Primary)
                    cost_matrix[gt_idx, ai_global_idx] = cost
                    # Slot 2 (Secondary - for split cases)
                    cost_matrix[gt_idx + num_gt, ai_global_idx] = cost

        print("Solving Maximum Weight Bipartite Matching...")
        # 3. The Solver (Hungarian Algorithm)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # 4. Parse Results
        raw_matches = defaultdict(list)
        
        for row, col in zip(row_indices, col_indices):
            # Check the actual cost. If it's 10000, the solver forced a match 
            # where none existed (shouldn't happen with our filtering, but good safety).
            cost = cost_matrix[row, col]
            if cost >= 9000: 
                continue # Ignore forced matches between unrelated cases
                
            similarity = 100.0 - cost

            # Map the row back to the original GT index
            # If row is 500 and num_gt is 500, then actual_gt_idx is 0 (Slot 2)
            actual_gt_idx = row % num_gt
            
            match_data = {
                "ai_case": self.ai_data[col],
                "score": similarity
            }
            raw_matches[actual_gt_idx].append(match_data)
            
        return self.generate_reports(raw_matches)

    def generate_reports(self, raw_matches):
        """
        Generates the two JSON artifacts:
        - split_cases_report.json: A list of split cases with their constituent AI IDs and merged ID.
        - master_reconciliation.json: A list of all reconciled cases with their human and AI data.
        """
        split_cases_report = []
        master_reconciliation = []
        
        # Iterate through matched results
        for gt_idx, matched_ai_list in raw_matches.items():
            gt_case = self.gt_data[gt_idx]
            
            # Check for Split
            is_split = len(matched_ai_list) > 1
            
            if is_split:
                # --- LOGIC FOR SPLIT MERGE ---
                
                # 1. Sort matches by image number (Top -> Bottom)
                sorted_matches = sorted(matched_ai_list, key=lambda x: x['ai_case']['image_num'])
                
                # 2. Generate New Integer ID
                reconciled_id = self.merge_id_counter
                self.merge_id_counter += 1
                
                # 3. Create the Merged AI Object (Mirroring source schema)
                merged_plaintiffs = []
                merged_defendants = []
                merged_places = []
                text_segments = []
                constituent_ids = []
                score_map = {}
                
                # NEW: Store plaintiffs/defendants per constituent case for UI distinction
                plaintiffs_by_case = {}
                defendants_by_case = {}
                
                # Capture plea details from top segment (usually where the header is)
                merged_plea = sorted_matches[0]['ai_case'].get('plea', '')
                merged_plea_details = sorted_matches[0]['ai_case'].get('plea_details', '')

                for m in sorted_matches:
                    case = m['ai_case']
                    case_id_str = str(case['ai_caseid'])
                    
                    # Store per-case data for UI distinction
                    plaintiffs_by_case[case_id_str] = case.get('plaintiffs', [])
                    defendants_by_case[case_id_str] = case.get('defendants', [])
                    
                    merged_plaintiffs.extend(case.get('plaintiffs', []))
                    merged_defendants.extend(case.get('defendants', []))
                    merged_places.extend(case.get('places', []))
                    text_segments.append(case.get('full_text', ''))
                    constituent_ids.append(case['ai_caseid'])
                    score_map[str(case['ai_caseid'])] = m['score']
                    
                    # Fallback: if top segment didn't have plea info, check bottom
                    if not merged_plea and case.get('plea'):
                        merged_plea = case.get('plea')
                    if not merged_plea_details and case.get('plea_details'):
                        merged_plea_details = case.get('plea_details')

                reconciled_ai_object = {
                    "reconciled_id": reconciled_id,
                    "is_split_merge": True,
                    "constituent_ai_cases": constituent_ids,
                    "source": "ai_htr_merged",
                    
                    # Merged Data Fields
                    "image_num": sorted_matches[0]['ai_case']['image_num'], 
                    "county": sorted_matches[0]['ai_case'].get('county', ''),
                    "plea": merged_plea,
                    "plea_details": merged_plea_details,
                    "plaintiffs": merged_plaintiffs,
                    "defendants": merged_defendants,
                    "places": merged_places,
                    "full_text": " [OVERLAP_SPLIT] ".join(text_segments),
                    
                    # NEW: Per-case breakdown for UI
                    "plaintiffs_by_case": plaintiffs_by_case,
                    "defendants_by_case": defendants_by_case,
                    
                    # Consistent Score Structure
                    "score": score_map
                }
                
                # Add to Split Report
                split_cases_report.append({
                    "human_CaseId": gt_case.get('human_case_id'),
                    "AI_CaseIds": constituent_ids,
                    "merged_AI_CaseId": reconciled_id,
                    "match_scores": score_map
                })
                
            else:
                # --- LOGIC FOR SINGLE MATCH ---
                single_match = matched_ai_list[0]
                original_case = single_match['ai_case']
                
                # Create a copy to avoid mutating original data
                reconciled_ai_object = original_case.copy()
                
                # Add Metadata
                reconciled_ai_object['reconciled_id'] = original_case['ai_caseid']
                reconciled_ai_object['is_split_merge'] = False
                
                # Consistent Score Structure (Dictionary)
                reconciled_ai_object['score'] = {
                    str(original_case['ai_caseid']): single_match['score']
                }

            # Calculate Accuracy Metrics
            accuracy_metrics = calculate_accuracy_metrics(gt_case, reconciled_ai_object)
            
            # Add to Master JSON
            master_reconciliation.append({
                "human_CaseId": gt_case.get('human_case_id'),
                "human_data": gt_case, 
                "reconciled_ai_data": reconciled_ai_object,
                "accuracy_metrics_v1": accuracy_metrics
            })
            
        return split_cases_report, master_reconciliation

# ---  EXECUTION FOR DEMO ---

# 1. Load your standardized data from actual JSON files
with open("datasets/KB_Table_Parsed.json", "r", encoding="utf-8") as f:
    human_gt_sample = json.load(f)

with open("datasets/standardized_ai_data.json", "r", encoding="utf-8") as f:
    ai_htr_sample = json.load(f)

# 2. Initialize Reconciler
reconciler = KingsBenchReconciler(human_gt_sample, ai_htr_sample)

# 3. Run Pipeline
split_report, master_json = reconciler.run_reconciliation()

# 4. Save Outputs (Printing for preview)
# Save the split cases report to a JSON file
with open("datasets/split_cases_report.json", "w", encoding="utf-8") as f:
    json.dump(split_report, f, indent=2, ensure_ascii=False)
print("Split cases report saved to datasets/split_cases_report.json")

# Save the master reconciliation JSON to a file
with open("datasets/master_reconciliation.json", "w", encoding="utf-8") as f:
    json.dump(master_json, f, indent=2, ensure_ascii=False)
print("Master reconciliation JSON saved to datasets/master_reconciliation.json")

# 5. Generate Aggregated Accuracy Report
print("\nGenerating Final Accuracy Report...")
calculate_final_accuracy(input_path="datasets/master_reconciliation.json", output_path="datasets/FINAL_ACCURACY.json")