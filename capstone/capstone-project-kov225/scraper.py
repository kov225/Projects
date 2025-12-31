# scraper.py
"""Load and clean the Ground Truth (GT) and HTR datasets for this project.

GT data is already fairly structured. HTR data needs more work because
names and places are scattered across several fields. This file turns both
datasets into consistent, simple Python structures for downstream matching.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DATA_DIR = Path(__file__).parent / "data"


def load_gt_raw() -> Dict[str, Any]:
    """Load the GT JSON exactly as provided."""
    path = DATA_DIR / "ground_truth_cache.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_htr_raw() -> Dict[str, Any]:
    """Load the HTR dataset JSON produced by the handwriting model."""
    path = DATA_DIR / "htr_dataset_799.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_image_text(text: str) -> Tuple[Optional[int], Optional[str]]:
    """Parse image labels like 'f 38' or 'd 153' into (38, 'f')."""
    if not text:
        return None, None

    stripped = text.strip()
    if not stripped:
        return None, None

    side = stripped[0].lower()
    digits = "".join(ch for ch in stripped if ch.isdigit())

    if not digits:
        return None, side
    return int(digits), side


def normalize_gt(gt_raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize GT cases into a predictable structure."""
    cases: List[Dict[str, Any]] = []

    for key, case in gt_raw.items():
        img_num_from_text, side = parse_image_text(case.get("image_text", ""))

        cases.append(
            {
                "gt_id": key,
                "image_num": case.get("image_num", img_num_from_text),
                "side": side,
                "plaintiffs": case.get("plaintiffs", []),
                "defendants": case.get("defendants", []),
                "places": case.get("places", []),
                "county": case.get("county"),
            }
        )

    return cases


def parse_source_image_directory(text: str) -> Optional[int]:
    """Extract numbers from strings like 'IMG_0224' -> 224."""
    if not text:
        return None
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else None


def extract_defendant_fields(defendant: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Extract as much name and place information as possible from one HTR defendant."""
    name_parts: List[str] = []
    place_parts: List[str] = []

    name_fields = [
        "firstName",
        "lastName",
        "alias",
        "altName",
        "otherNames",
        "description",
        "parsedName",
        "additionalInfo",
        "role",
        "occupation",
    ]

    place_fields = ["location", "origin", "residence", "place"]

    for f in name_fields:
        val = defendant.get(f)
        if isinstance(val, str) and val.strip():
            name_parts.append(val.strip())

    for f in place_fields:
        val = defendant.get(f)
        if isinstance(val, str) and val.strip():
            place_parts.append(val.strip())

    final_name = " ".join(name_parts).strip()

    return final_name, place_parts


def normalize_htr(htr_raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize HTR cases into a structure consistent with GT."""

    cases_raw = htr_raw.get("cases", [])
    cases: List[Dict[str, Any]] = []

    for i, case in enumerate(cases_raw):
        image_num = parse_source_image_directory(
            case.get("source_image_directory", "")
        )

        names: List[str] = []
        places: List[str] = []

        for d in case.get("defendants", []):
            if isinstance(d, dict):
                nm, pls = extract_defendant_fields(d)
                if nm:
                    names.append(nm)
                places.extend(pls)

        plea = case.get("plea") or {}
        primary = plea.get("primary_charge")

        cases.append(
            {
                "htr_id": i,
                "image_num": image_num,
                "county": case.get("county"),
                "defendants": names,
                "places": places,
                "plea_primary": primary,
            }
        )

    return cases


def main() -> None:
    """Quick check when running this file manually."""
    print("Loading raw datasets...")
    gt_raw = load_gt_raw()
    htr_raw = load_htr_raw()

    print("Normalizing...")
    gt = normalize_gt(gt_raw)
    htr = normalize_htr(htr_raw)

    print(f"GT cases: {len(gt)}")
    print(f"HTR cases: {len(htr)}")

    if gt:
        print("\nExample GT:", gt[0])
    if htr:
        print("\nExample HTR:", htr[0])


if __name__ == "__main__":
    main()
