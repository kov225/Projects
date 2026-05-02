"""Tests for experiment_analysis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src import experiment_analysis as exa


def test_cohen_h_zero_when_proportions_equal():
    assert exa.cohen_h(0.2, 0.2) == pytest.approx(0.0, abs=1e-9)


def test_cohen_h_clips_extreme_values():
    h = exa.cohen_h(0.0, 1.0)
    assert math.isfinite(h)
    assert h < 0


def test_two_proportion_test_detects_clear_winner():
    res = exa.two_proportion_test(1000, 100, 1000, 200)
    assert res.lift_abs == pytest.approx(0.10, rel=1e-3)
    assert res.p_value < 0.001
    assert res.cohen_h > 0
    assert res.power > 0.99


def test_two_proportion_test_handles_loss():
    res = exa.two_proportion_test(1000, 200, 1000, 100)
    assert res.lift_abs == pytest.approx(-0.10, rel=1e-3)
    assert res.p_value < 0.001


def test_two_proportion_test_inconclusive_when_small():
    res = exa.two_proportion_test(50, 5, 50, 6)
    assert res.p_value > 0.5


def test_two_proportion_test_rejects_empty_arm():
    with pytest.raises(ValueError):
        exa.two_proportion_test(0, 0, 100, 10)


def test_proportion_ci_returns_increasing_bounds():
    lo, hi = exa.proportion_ci(50, 1000)
    assert 0 < lo < hi < 1


def test_analyse_experiment_emits_control_and_variant_rows(tiny_results):
    arms = exa.analyse_experiment(tiny_results, "EXP_001")
    variants = set(arms["variant"])
    assert variants == {"control", "variant_a"}
    assert (arms["users"] > 0).all()


def test_summarise_experiments_one_row_per_experiment(tiny_experiments, tiny_results):
    summary = exa.summarise_experiments(tiny_results, tiny_experiments)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["experiment_id"] == "EXP_001"
    assert row["recommendation"] in {"ship", "kill", "iterate"}


def test_recommendation_for_routes_significant_winner_to_ship():
    row = pd.Series({"p_value": 0.001, "lift_abs": 0.05, "power": 0.9})
    assert exa.recommendation_for(row) == "ship"


def test_recommendation_for_routes_significant_loser_to_kill():
    row = pd.Series({"p_value": 0.001, "lift_abs": -0.05, "power": 0.9})
    assert exa.recommendation_for(row) == "kill"


def test_recommendation_for_routes_underpowered_to_iterate():
    row = pd.Series({"p_value": 0.4, "lift_abs": 0.01, "power": 0.2})
    assert exa.recommendation_for(row) == "iterate"


def test_pairwise_multivariate_returns_combinations():
    rng = np.random.default_rng(0)
    n = 200
    rows: list[dict] = []
    rates = {"control": 0.10, "variant_a": 0.18, "variant_b": 0.14}
    for _ in range(n):
        for variant, rate in rates.items():
            rows.append(
                {
                    "experiment_id": "EXP_X",
                    "variant": variant,
                    "converted": bool(rng.random() < rate),
                }
            )
    df = pd.DataFrame(rows)
    df["user_pseudo_id"] = [f"u{i}" for i in range(len(df))]
    df["revenue_impact"] = 0.0
    pairs = exa.pairwise_multivariate(df, "EXP_X")
    assert set(zip(pairs["arm_a"], pairs["arm_b"])).issubset(
        {("control", "variant_a"), ("control", "variant_b"), ("variant_a", "variant_b")}
    )
    assert len(pairs) == 3
