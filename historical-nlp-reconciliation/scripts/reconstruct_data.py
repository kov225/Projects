import json
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths relative to project root
BASE_DIR = Path(__file__).resolve().parent.parent
CAPSTONE_DATA = Path("e:/Projects/Projects/capstone/capstone-project-kov225/data")
DATA_DIR = BASE_DIR / "data"

def reconstruct():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)
        logger.info(f"Created data directory at {DATA_DIR}")

    # 1. Reconstruct AI Data
    htr_path = CAPSTONE_DATA / "htr_dataset_799.json"
    if htr_path.exists():
        logger.info(f"Loading raw AI data from {htr_path}")
        with open(htr_path, 'r', encoding='utf-8') as f:
            ai_data = json.load(f)
        
        standardized_ai = []
        for i, case in enumerate(ai_data.get("cases", [])):
            img_dir = case.get("source_image_directory", "")
            img_num = -1
            if "IMG_" in img_dir:
                try:
                    img_num = int(img_dir.split("_")[1])
                except: pass
            
            std_case = {
                "ai_caseid": i + 1,
                "source": "ai_htr",
                "image_num": img_num,
                "original_image_id": img_dir,
                "county": str(case.get("county", "") or ""),
                "places": [str(d.get("location")) for d in case.get("defendants", []) if isinstance(d, dict) and d.get("location")],
                "plaintiffs": [str(p.get("entityName") or p.get("lastName") or p.get("firstName") or "") for p in case.get("plaintiffs", []) if isinstance(p, dict)],
                "defendants": [str(d.get("firstName") or "") if j % 2 != 0 else str(d.get("lastName") or "") for d in case.get("defendants", []) for j in range(2) if isinstance(d, dict) and (d.get("firstName") or d.get("lastName"))],
                "plea": str(case.get("plea", {}).get("primary_charge", "") if isinstance(case.get("plea"), dict) else ""),
                "plea_details": str(case.get("plea", {}).get("details", "") if isinstance(case.get("plea"), dict) else ""),
            }
            
            full_text_parts = std_case["plaintiffs"] + std_case["defendants"] + std_case["places"]
            if std_case["plea_details"]:
                full_text_parts.append(std_case["plea_details"])
            std_case["full_text"] = " ".join(full_text_parts)
            
            if case.get("isCrownPlaintiff", True) and not std_case["plaintiffs"]:
                std_case["plaintiffs"].append("Rex")
                std_case["full_text"] = "Rex " + std_case["full_text"]
            
            standardized_ai.append(std_case)
            
        with open(DATA_DIR / "standardized_ai_data.json", 'w', encoding='utf-8') as f:
            json.dump(standardized_ai, f, indent=2)
        logger.info("Successfully reconstructed standardized_ai_data.json")
    else:
        logger.error(f"AI source data not found at {htr_path}")

    # 2. Reconstruct GT Data
    gt_path = CAPSTONE_DATA / "ground_truth_cache.json"
    if gt_path.exists():
        logger.info(f"Loading raw GT data from {gt_path}")
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        parsed_gt = []
        for i, (key, case) in enumerate(gt_data.items()):
            parsed_gt.append({
                "human_case_id": i + 1,
                "image_id": str(key),
                "image_num": case.get("image_num"),
                "county": str(case.get("county", "") or ""),
                "plaintiffs": [str(p) for p in case.get("plaintiffs", []) if p],
                "defendants": [str(d) for d in case.get("defendants", []) if d],
                "plea": str(case.get("plea", "") or "")
            })
            
        with open(DATA_DIR / "KB_Table_Parsed.json", 'w', encoding='utf-8') as f:
            json.dump(parsed_gt, f, indent=2)
        logger.info("Successfully reconstructed KB_Table_Parsed.json")
    else:
        logger.error(f"GT source data not found at {gt_path}")

if __name__ == "__main__":
    reconstruct()
