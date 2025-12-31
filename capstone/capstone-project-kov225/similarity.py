# similarity.py
"""Similarity scoring between GT and HTR cases.

This scoring uses defendants and place names. Medieval spellings
vary a lot, so everything is fuzzy-matched. The score stays on a 0–100 scale.
"""

from __future__ import annotations

from typing import Any, Dict, List
from rapidfuzz import fuzz


def clean_text(name: str) -> str:
    """Remove parentheses and trim spaces for better fuzzy matching."""
    if not name:
        return ""
    return (
        name.replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .strip()
    )


def best_fuzzy_score(a: str, b: str) -> float:
    """Return the strongest fuzzy match among several strategies."""
    if not a or not b:
        return 0.0

    return max(
        fuzz.token_sort_ratio(a, b),
        fuzz.token_set_ratio(a, b),
        fuzz.partial_ratio(a, b),
    )


def avg_pairwise_best_match(gt_list: List[str], htr_list: List[str]) -> float:
    """For each GT item, find its best HTR match and average the scores.

    This is asymmetric on purpose:
    - For every GT element we ask: how well is this represented in HTR.
    """
    if not gt_list and not htr_list:
        return 100.0
    if not gt_list or not htr_list:
        # Some evidence, but one side is empty.
        return 20.0

    scores: List[float] = []
    for gt in gt_list:
        best = 0.0
        for htr in htr_list:
            s = best_fuzzy_score(gt, htr)
            if s > best:
                best = s
        scores.append(best)

    return sum(scores) / len(scores)


def soft_size_penalty(gt_list: List[str], htr_list: List[str]) -> float:
    """A gentle penalty when list sizes differ, since medieval records vary."""
    if not gt_list and not htr_list:
        return 1.0
    if not gt_list or not htr_list:
        return 0.6

    ratio = min(len(gt_list), len(htr_list)) / max(len(gt_list), len(htr_list))
    # Range from 0.5 to 1.0
    return 0.5 + 0.5 * ratio


def calculate_similarity(gt_case: Dict[str, Any], htr_case: Dict[str, Any]) -> float:
    """Compute the final similarity score between a GT and an HTR case.

    Current components:
    - defendants: main signal
    - places: secondary signal
    - soft penalty for list-size mismatch
    """

    gt_def = [clean_text(x) for x in gt_case.get("defendants", [])]
    gt_places = [clean_text(x) for x in gt_case.get("places", [])]

    htr_def_raw = htr_case.get("defendants", [])
    htr_places_raw = htr_case.get("places", [])

    htr_def = [clean_text(x) for x in htr_def_raw]
    htr_places = [clean_text(x) for x in htr_places_raw]

    def_score = avg_pairwise_best_match(gt_def, htr_def)
    place_score = avg_pairwise_best_match(gt_places, htr_places)

    # Weights can be tuned, but this is reasonable.
    combined = 0.7 * def_score + 0.3 * place_score

    penalty = soft_size_penalty(gt_def, htr_def)

    return float(min(combined * penalty, 100.0))
