# reconciliation.py
"""Reconcile GT and HTR cases using the Hungarian algorithm.

Each GT case is represented by two "slots" so that a single GT case can
match two HTR fragments when the handwriting model splits a case across
two images.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from scraper import normalize_gt, normalize_htr, load_gt_raw, load_htr_raw
from similarity import calculate_similarity


def index_by_image(cases: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Group cases by their image number for fast lookup."""
    result: Dict[int, List[Dict[str, Any]]] = {}
    for c in cases:
        num = c.get("image_num")
        if num is not None:
            result.setdefault(num, []).append(c)
    return result


def htr_candidates_for_image(
    image_num: int,
    htr_index: Dict[int, List[Dict[str, Any]]],
    window: int = 1,
) -> List[Dict[str, Any]]:
    """Return HTR candidates within image_num +/- window."""
    candidates: List[Dict[str, Any]] = []
    seen: set[int] = set()

    for offset in range(-window, window + 1):
        arr = htr_index.get(image_num + offset, [])
        for c in arr:
            hid = c["htr_id"]
            if hid not in seen:
                seen.add(hid)
                candidates.append(c)

    return candidates


def build_cost_matrix(
    gt_cases: List[Dict[str, Any]],
    htr_candidates: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[Tuple[Dict[str, Any], int]]]:
    """Construct the cost matrix with two rows per GT case.

    Each GT case gets two "slots", corresponding to the possibility of
    being split into two HTR fragments.
    """
    slots: List[Tuple[Dict[str, Any], int]] = []
    for case in gt_cases:
        slots.append((case, 0))
        slots.append((case, 1))

    if not slots or not htr_candidates:
        return np.zeros((0, 0), dtype=float), slots

    rows = len(slots)
    cols = len(htr_candidates)
    matrix = np.full((rows, cols), 100.0, dtype=float)

    for r, (gt_case, _) in enumerate(slots):
        for c, htr_case in enumerate(htr_candidates):
            sim = calculate_similarity(gt_case, htr_case)
            matrix[r, c] = 100.0 - sim

    return matrix, slots


def reconcile_block(
    image_num: int,
    gt_cases_here: List[Dict[str, Any]],
    htr_index: Dict[int, List[Dict[str, Any]]],
    threshold: float,
    window: int = 1,
) -> List[Dict[str, Any]]:
    """Run Hungarian matching for a single image region."""
    candidates = htr_candidates_for_image(image_num, htr_index, window=window)
    if not candidates:
        return []

    matrix, slots = build_cost_matrix(gt_cases_here, candidates)
    if matrix.size == 0:
        return []

    row_ind, col_ind = linear_sum_assignment(matrix)

    matches: List[Dict[str, Any]] = []
    for r, c in zip(row_ind, col_ind):
        sim = 100.0 - float(matrix[r, c])
        if sim < threshold:
            # Reject low quality matches after Hungarian.
            continue

        gt_case, slot = slots[r]
        htr_case = candidates[c]

        matches.append(
            {
                "gt_id": gt_case["gt_id"],
                "htr_id": htr_case["htr_id"],
                "slot": slot,
                "gt_image_num": gt_case["image_num"],
                "htr_image_num": htr_case["image_num"],
                "similarity": sim,
            }
        )
    return matches


def reconcile_all(
    gt_cases: List[Dict[str, Any]],
    htr_cases: List[Dict[str, Any]],
    similarity_threshold: float = 50.0,
    window: int = 1,
) -> List[Dict[str, Any]]:
    """Reconcile all GT cases against HTR cases.

    Returns a flat list of match dictionaries. Each GT case can appear up
    to two times (one for each slot).
    """
    gt_index = index_by_image(gt_cases)
    htr_index = index_by_image(htr_cases)

    all_matches: List[Dict[str, Any]] = []
    for img in sorted(gt_index.keys()):
        matches = reconcile_block(
            img,
            gt_index[img],
            htr_index,
            threshold=similarity_threshold,
            window=window,
        )
        all_matches.extend(matches)
    return all_matches


def main() -> None:
    gt = normalize_gt(load_gt_raw())
    htr = normalize_htr(load_htr_raw())
    matches = reconcile_all(gt, htr, similarity_threshold=50.0, window=1)
    print("Sample matches:", matches[:5])


if __name__ == "__main__":
    main()
