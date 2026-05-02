"""Statistical tests for A/B and multivariate experiments.

The functions here favour readability over micro-optimisation. They take
plain dataframes, return plain dataclasses, and avoid hidden state. The
experiment simulation helper at the bottom is what notebook 4 uses to
project realistic experiment outcomes onto the extended user base.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

LOGGER = logging.getLogger(__name__)


@dataclass
class TwoProportionResult:
    """Output of a two-sample proportion test."""

    variant: str
    control_n: int
    control_conv: int
    variant_n: int
    variant_conv: int
    control_rate: float
    variant_rate: float
    lift_abs: float
    lift_rel: float
    p_value: float
    chi2: float
    cohen_h: float
    ci_low: float
    ci_high: float
    power: float

    def to_row(self) -> dict:
        return asdict(self)


def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    p1 = min(max(p1, 1e-9), 1 - 1e-9)
    p2 = min(max(p2, 1e-9), 1 - 1e-9)
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def proportion_ci(success: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = success / trials
    denom = 1 + z**2 / trials
    centre = (p_hat + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denom
    return float(centre - half), float(centre + half)


def two_proportion_test(
    control_n: int,
    control_conv: int,
    variant_n: int,
    variant_conv: int,
    variant_label: str = "variant_a",
) -> TwoProportionResult:
    """Run a chi-squared test, compute effect size, CI, and observed power."""
    if control_n == 0 or variant_n == 0:
        raise ValueError("both arms must have at least one user")

    table = np.array(
        [[control_conv, control_n - control_conv], [variant_conv, variant_n - variant_conv]]
    )
    chi2, p_value, _, _ = stats.chi2_contingency(table, correction=False)

    p_c = control_conv / control_n
    p_v = variant_conv / variant_n
    lift_abs = p_v - p_c
    lift_rel = lift_abs / p_c if p_c > 0 else float("nan")
    h = cohen_h(p_v, p_c)

    diff_var = (p_c * (1 - p_c) / control_n) + (p_v * (1 - p_v) / variant_n)
    z = stats.norm.ppf(0.975)
    ci_low = lift_abs - z * np.sqrt(diff_var)
    ci_high = lift_abs + z * np.sqrt(diff_var)

    n_eff = (control_n + variant_n) / 2
    z_crit = stats.norm.ppf(0.975)
    power = float(1 - stats.norm.cdf(z_crit - abs(h) * np.sqrt(n_eff / 2)))

    return TwoProportionResult(
        variant=variant_label,
        control_n=int(control_n),
        control_conv=int(control_conv),
        variant_n=int(variant_n),
        variant_conv=int(variant_conv),
        control_rate=float(p_c),
        variant_rate=float(p_v),
        lift_abs=float(lift_abs),
        lift_rel=float(lift_rel),
        p_value=float(p_value),
        chi2=float(chi2),
        cohen_h=float(h),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        power=power,
    )


def analyse_experiment(
    results: pd.DataFrame,
    experiment_id: str,
) -> pd.DataFrame:
    """Per-variant analysis for a single experiment.

    Expects a long frame with columns: experiment_id, variant, converted,
    revenue_impact. Returns one row per non-control variant comparing it to
    control, plus a row for control itself for reference.
    """
    df = results[results["experiment_id"] == experiment_id]
    if df.empty:
        raise ValueError(f"no rows for experiment {experiment_id}")

    counts = df.groupby("variant")["converted"].agg(["sum", "count"]).rename(
        columns={"sum": "conversions", "count": "users"}
    )
    if "control" not in counts.index:
        raise ValueError("experiment has no control arm")

    control_n = int(counts.loc["control", "users"])
    control_c = int(counts.loc["control", "conversions"])

    revenue = df.groupby("variant")["revenue_impact"].sum()

    rows: list[dict] = []
    rows.append(
        {
            "experiment_id": experiment_id,
            "variant": "control",
            "users": control_n,
            "conversions": control_c,
            "conv_rate": control_c / control_n if control_n else 0,
            "revenue": float(revenue.get("control", 0.0)),
            "lift_abs": 0.0,
            "lift_rel": 0.0,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "cohen_h": 0.0,
            "power": np.nan,
        }
    )

    for variant in [v for v in counts.index if v != "control"]:
        v_n = int(counts.loc[variant, "users"])
        v_c = int(counts.loc[variant, "conversions"])
        res = two_proportion_test(control_n, control_c, v_n, v_c, variant_label=variant)
        rows.append(
            {
                "experiment_id": experiment_id,
                "variant": variant,
                "users": v_n,
                "conversions": v_c,
                "conv_rate": res.variant_rate,
                "revenue": float(revenue.get(variant, 0.0)),
                "lift_abs": res.lift_abs,
                "lift_rel": res.lift_rel,
                "p_value": res.p_value,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "cohen_h": res.cohen_h,
                "power": res.power,
            }
        )
    return pd.DataFrame(rows)


def pairwise_multivariate(
    results: pd.DataFrame,
    experiment_id: str,
) -> pd.DataFrame:
    """All pairwise comparisons between variants for a multivariate test."""
    df = results[results["experiment_id"] == experiment_id]
    counts = df.groupby("variant")["converted"].agg(["sum", "count"]).rename(
        columns={"sum": "conversions", "count": "users"}
    )

    rows: list[dict] = []
    variants = list(counts.index)
    for a, b in combinations(variants, 2):
        a_n, a_c = int(counts.loc[a, "users"]), int(counts.loc[a, "conversions"])
        b_n, b_c = int(counts.loc[b, "users"]), int(counts.loc[b, "conversions"])
        res = two_proportion_test(a_n, a_c, b_n, b_c, variant_label=b)
        rows.append(
            {
                "experiment_id": experiment_id,
                "arm_a": a,
                "arm_b": b,
                "rate_a": a_c / a_n if a_n else 0,
                "rate_b": b_c / b_n if b_n else 0,
                "lift_abs": res.lift_abs,
                "p_value": res.p_value,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "cohen_h": res.cohen_h,
            }
        )
    return pd.DataFrame(rows)


def recommendation_for(row: pd.Series, alpha: float = 0.05, min_power: float = 0.6) -> str:
    """Map a chi-squared row to a ship/iterate/kill recommendation."""
    if pd.isna(row.get("p_value")):
        return "control"
    if row["p_value"] < alpha and row["lift_abs"] > 0:
        return "ship"
    if row["p_value"] < alpha and row["lift_abs"] < 0:
        return "kill"
    if row.get("power", 1.0) < min_power:
        return "iterate"
    return "iterate"


def summarise_experiments(
    results: pd.DataFrame,
    experiments: pd.DataFrame,
) -> pd.DataFrame:
    """One row per experiment summarising the best variant and its verdict."""
    rows: list[dict] = []
    for exp_id in experiments["experiment_id"]:
        try:
            arms = analyse_experiment(results, exp_id)
        except ValueError:
            continue
        non_control = arms[arms["variant"] != "control"]
        if non_control.empty:
            continue
        best = non_control.sort_values("lift_abs", ascending=False).iloc[0]
        meta = experiments[experiments["experiment_id"] == exp_id].iloc[0]
        rows.append(
            {
                "experiment_id": exp_id,
                "experiment_name": meta["experiment_name"],
                "feature_area": meta["feature_area"],
                "experiment_type": meta["experiment_type"],
                "winner_variant": best["variant"],
                "control_rate": float(arms.iloc[0]["conv_rate"]),
                "winner_rate": float(best["conv_rate"]),
                "lift_abs": float(best["lift_abs"]),
                "lift_rel": float(best["lift_rel"]),
                "p_value": float(best["p_value"]),
                "ci_low": float(best["ci_low"]),
                "ci_high": float(best["ci_high"]),
                "power": float(best["power"]),
                "revenue": float(arms["revenue"].sum()),
                "recommendation": recommendation_for(best),
            }
        )
    return pd.DataFrame(rows)


def simulate_experiments(
    users: pd.DataFrame,
    seed: int = 41,
    n_ab: int = 15,
    n_mvt: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate experiment metadata, assignments, and results for the
    extended dataset. Outcomes follow a known truth so that the analysis
    notebook produces a believable mix of clear wins, clear losses, and
    inconclusive results.
    """
    rng = np.random.default_rng(seed)
    base_users = users.copy()
    base_users["first_seen_date"] = pd.to_datetime(base_users["first_seen_date"])

    feature_areas = [
        "checkout", "product_detail", "search", "homepage", "recommendations", "cart",
    ]
    hypothesis_pool = [
        ("simplified_checkout_v3", "checkout", "Removing the address autofill confirmation reduces checkout drop off."),
        ("recommendation_carousel", "recommendations", "Showing related items on PDP increases add to cart rate."),
        ("free_shipping_banner", "homepage", "A persistent free shipping banner lifts visit-to-add-to-cart."),
        ("search_typeahead", "search", "Inline typeahead suggestions surface popular SKUs faster."),
        ("sticky_buy_button", "product_detail", "A sticky buy button on mobile improves PDP conversion."),
        ("cart_upsell_block", "cart", "Adding a small upsell block raises units per order."),
        ("trust_badges_pdp", "product_detail", "Adding payment trust badges on PDP increases purchase rate."),
        ("guest_checkout_default", "checkout", "Defaulting to guest checkout reduces friction for first time buyers."),
        ("price_drop_email", "homepage", "Triggering price drop emails brings dormant users back."),
        ("color_swatch_pdp", "product_detail", "Swatch previews on PDP raise add to cart for apparel."),
        ("new_arrivals_module", "homepage", "Featuring new arrivals on the homepage drives engagement."),
        ("category_filters_v2", "search", "Refined filters in category pages improve discovery."),
        ("review_summary_card", "product_detail", "A short review summary lifts confidence and conversion."),
        ("shipping_calc_pdp", "product_detail", "Showing shipping cost on PDP avoids checkout surprise."),
        ("loyalty_signin_prompt", "homepage", "Prompting members to sign in increases repeat purchase rate."),
        ("checkout_progress_bar", "checkout", "A progress bar reduces abandonment during multi step checkout."),
        ("cart_save_for_later", "cart", "Save for later reduces accidental cart deletion."),
        ("recs_ranker_v4", "recommendations", "A new ranker order improves click through to PDP."),
        ("hero_video_homepage", "homepage", "An autoplay hero video raises homepage engagement."),
        ("pdp_color_filter", "product_detail", "Filtering review snippets by color helps high intent shoppers."),
    ]

    selected = hypothesis_pool[: n_ab + n_mvt]
    rng.shuffle(selected)

    experiments_rows: list[dict] = []
    assignments_rows: list[dict] = []
    results_rows: list[dict] = []

    eligible_users = base_users[
        (base_users["first_seen_date"] >= pd.Timestamp("2022-04-01"))
        & (base_users["first_seen_date"] <= pd.Timestamp("2024-09-01"))
    ].copy()

    for i, (name, area, hypothesis) in enumerate(selected):
        is_mvt = i >= n_ab
        n_variants = int(rng.integers(3, 5)) if is_mvt else 2
        variant_labels = ["control"] + [f"variant_{c}" for c in "abc"[: n_variants - 1]]

        start = pd.Timestamp("2022-06-01") + pd.Timedelta(days=int(rng.integers(0, 800)))
        weeks_run = int(rng.integers(4, 9))
        end = start + pd.Timedelta(weeks=weeks_run)

        outcome = rng.choice(
            ["clear_winner", "clear_winner", "inconclusive", "loser", "mixed"],
            p=[0.35, 0.20, 0.20, 0.15, 0.10],
        )

        experiments_rows.append(
            {
                "experiment_id": f"EXP_{i+1:03d}",
                "experiment_name": name,
                "feature_area": area,
                "start_date": start.date(),
                "end_date": end.date(),
                "experiment_type": "multivariate" if is_mvt else "a_b",
                "variant_count": n_variants,
                "hypothesis": hypothesis,
                "designed_outcome": outcome,
            }
        )

        eligible_pool = eligible_users[
            (eligible_users["first_seen_date"] >= start - pd.Timedelta(days=60))
            & (eligible_users["first_seen_date"] <= end)
        ]
        if len(eligible_pool) < 8_000:
            eligible_pool = eligible_users.sample(
                min(20_000, len(eligible_users)),
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
        sample_n = int(min(len(eligible_pool), rng.integers(12_000, 22_000)))
        sampled = eligible_pool.sample(sample_n, random_state=int(rng.integers(0, 2**31 - 1)))
        chosen_variants = rng.choice(variant_labels, size=sample_n)

        assignments_rows.extend(
            {
                "user_pseudo_id": uid,
                "experiment_id": f"EXP_{i+1:03d}",
                "variant": v,
                "assignment_date": start.date(),
            }
            for uid, v in zip(sampled["user_pseudo_id"], chosen_variants)
        )

        control_rate = float(np.clip(rng.normal(0.12, 0.012), 0.08, 0.16))
        variant_rates = {"control": control_rate}
        if outcome == "clear_winner":
            for v in variant_labels[1:]:
                variant_rates[v] = control_rate * float(rng.uniform(1.10, 1.32))
        elif outcome == "loser":
            for v in variant_labels[1:]:
                variant_rates[v] = control_rate * float(rng.uniform(0.78, 0.94))
        elif outcome == "inconclusive":
            for v in variant_labels[1:]:
                variant_rates[v] = control_rate * float(rng.uniform(0.97, 1.04))
        else:
            for v in variant_labels[1:]:
                variant_rates[v] = control_rate * float(rng.uniform(0.92, 1.20))

        for uid, v in zip(sampled["user_pseudo_id"], chosen_variants):
            converted = bool(rng.random() < variant_rates[v])
            revenue = float(np.round(rng.gamma(2.0, 24.0), 2)) if converted else 0.0
            engagement = float(np.clip(rng.normal(58 if converted else 42, 12), 0, 100))
            results_rows.append(
                {
                    "user_pseudo_id": uid,
                    "experiment_id": f"EXP_{i+1:03d}",
                    "variant": v,
                    "converted": converted,
                    "revenue_impact": revenue,
                    "engagement_score": engagement,
                }
            )

    experiments_df = pd.DataFrame(experiments_rows)
    assignments_df = pd.DataFrame(assignments_rows)
    results_df = pd.DataFrame(results_rows)
    LOGGER.info(
        "simulated %d experiments, %d assignments, %d result rows",
        len(experiments_df),
        len(assignments_df),
        len(results_df),
    )
    return experiments_df, assignments_df, results_df
