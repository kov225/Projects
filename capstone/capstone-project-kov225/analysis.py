# analysis.py
"""Evaluate GT–HTR reconciliation, compute strict and fuzzy metrics, and visualize networks."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from rapidfuzz import fuzz

from scraper import load_gt_raw, load_htr_raw, normalize_gt, normalize_htr
from reconciliation import reconcile_all


def name_overlap_strict(gt_names: List[str], htr_names: List[str]) -> Tuple[int, int, int]:
    """Exact string matching."""
    gt_set = {n.strip().lower() for n in gt_names if n and n.strip()}
    htr_set = {n.strip().lower() for n in htr_names if n and n.strip()}

    tp = len(gt_set & htr_set)
    fp = len(htr_set - gt_set)
    fn = len(gt_set - htr_set)
    return tp, fp, fn


def name_overlap_fuzzy(gt_names: List[str], htr_names: List[str], threshold: int = 80) -> Tuple[int, int, int]:
    """Fuzzy matching for medieval spelling variation."""
    gt_clean = [n.strip().lower() for n in gt_names if n and n.strip()]
    htr_clean = [n.strip().lower() for n in htr_names if n and n.strip()]

    tp = 0
    matched_gt = set()
    matched_htr = set()

    # Count fuzzy true positives
    for i, g in enumerate(gt_clean):
        for j, h in enumerate(htr_clean):
            if j in matched_htr:
                continue
            if fuzz.token_sort_ratio(g, h) >= threshold:
                tp += 1
                matched_gt.add(i)
                matched_htr.add(j)
                break

    fp = len(htr_clean) - len(matched_htr)
    fn = len(gt_clean) - len(matched_gt)
    return tp, fp, fn


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def group_matches(matches: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = {}
    for m in matches:
        grouped.setdefault(m["gt_id"], []).append(m)
    return grouped


def merge_htr_defendants(htrs: List[Dict[str, Any]]) -> List[str]:
    seen = set()
    merged = []
    for h in htrs:
        for name in h.get("defendants", []):
            if not name:
                continue
            low = name.lower().strip()
            if low not in seen:
                seen.add(low)
                merged.append(name)
    return merged


def build_graph(groups: Dict[str, List[Dict[str, Any]]], gt_lookup: Dict[str, Any]) -> nx.Graph:
    graph = nx.Graph()
    for gid in groups:
        defendants = [
            n.strip() for n in gt_lookup[gid].get("defendants", []) if n and n.strip()
        ]

        for n in defendants:
            graph.add_node(n)

        for i in range(len(defendants)):
            for j in range(i + 1, len(defendants)):
                graph.add_edge(defendants[i], defendants[j])
    return graph


def run_analysis(threshold: float = 50.0) -> None:
    print("RECONCILIATION PIPELINE ANALYSIS\n")

    print("[1] Loading data...")
    gt_cases = normalize_gt(load_gt_raw())
    htr_cases = normalize_htr(load_htr_raw())
    print(f"GT cases:  {len(gt_cases)}")
    print(f"HTR cases: {len(htr_cases)}\n")

    gt_lookup = {c["gt_id"]: c for c in gt_cases}
    htr_lookup = {c["htr_id"]: c for c in htr_cases}

    print("[2] Running reconciliation...")
    matches = reconcile_all(gt_cases, htr_cases, similarity_threshold=threshold)
    print(f"Matches above threshold: {len(matches)}")

    groups = group_matches(matches)
    print(f"Unique GT cases matched: {len(groups)}\n")

    print("[3] Strict name evaluation...\n")
    strict_tp = strict_fp = strict_fn = 0

    for gid, match_list in groups.items():
        gt_def = gt_lookup[gid].get("defendants", [])
        htrs = [htr_lookup[m["htr_id"]] for m in match_list]
        htr_def = merge_htr_defendants(htrs)

        t, fp_i, fn_i = name_overlap_strict(gt_def, htr_def)
        strict_tp += t
        strict_fp += fp_i
        strict_fn += fn_i

    strict_precision = safe_div(strict_tp, strict_tp + strict_fp)
    strict_recall = safe_div(strict_tp, strict_tp + strict_fn)
    strict_f1 = safe_div(2 * strict_precision * strict_recall, strict_precision + strict_recall)

    print("STRICT MATCHING RESULTS")
    print(f"Strict Precision: {strict_precision:.3f}")
    print(f"Strict Recall:    {strict_recall:.3f}")
    print(f"Strict F1 Score:  {strict_f1:.3f}\n")

    print("[4] Fuzzy name evaluation...\n")
    fuzzy_tp = fuzzy_fp = fuzzy_fn = 0

    for gid, match_list in groups.items():
        gt_def = gt_lookup[gid].get("defendants", [])
        htrs = [htr_lookup[m["htr_id"]] for m in match_list]
        htr_def = merge_htr_defendants(htrs)

        t, fp_i, fn_i = name_overlap_fuzzy(gt_def, htr_def, threshold=80)
        fuzzy_tp += t
        fuzzy_fp += fp_i
        fuzzy_fn += fn_i

    fuzzy_precision = safe_div(fuzzy_tp, fuzzy_tp + fuzzy_fp)
    fuzzy_recall = safe_div(fuzzy_tp, fuzzy_tp + fuzzy_fn)
    fuzzy_f1 = safe_div(2 * fuzzy_precision * fuzzy_recall, fuzzy_precision + fuzzy_recall)

    print("FUZZY MATCHING RESULTS (Threshold = 80)")
    print(f"Fuzzy Precision: {fuzzy_precision:.3f}")
    print(f"Fuzzy Recall:    {fuzzy_recall:.3f}")
    print(f"Fuzzy F1 Score:  {fuzzy_f1:.3f}\n")

    print("[5] Building social network...\n")
    graph = build_graph(groups, gt_lookup)

    centrality = nx.degree_centrality(graph)
    sorted_people = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    most_connected = sorted_people[0][0]

    print("Most connected individual:", most_connected)

    ego = nx.ego_graph(graph, most_connected)
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(ego, seed=42)
    nx.draw(ego, pos, with_labels=True, node_size=400, font_size=8, edge_color="gray")
    plt.title(f"Sub-network of {most_connected}")
    plt.tight_layout()
    plt.show()

    print("\nDONE\n")


if __name__ == "__main__":
    run_analysis()
