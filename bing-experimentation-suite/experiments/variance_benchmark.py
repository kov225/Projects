from __future__ import annotations

import pandas as pd
from .ab_test import TwoSampleTTest, ExperimentResult
from .cuped import CUPEDAdjuster
from .stratification import PostStratificationEstimator, RegressionAdjustedEstimator


class VarianceBenchmark:
    """A comprehensive evaluation of experimentation frameworks.

    This class compares four mainstream statistical methods on a single
    dataset. It quantifies the value of variance reduction techniques
    like CUPED and regression adjustment by measuring the narrowing of
    confidence intervals and the improvement in minimum detectable effects.
    """

    def run_benchmark(
        self, df: pd.DataFrame, outcome_col: str, covariate_col: str, stratum_col: str
    ) -> pd.DataFrame:
        """Execute all four methods and build a comparative summary table.

        We apply each method to the provided dataset and capture the
        key inference metrics. The resulting table allows for a direct
        assessment of which technique provides the most precise estimate
        for this specific metric and population.
        """
        results = {}

        # 1. Standard t test
        ctrl = df[df["treatment"] == 0][outcome_col].values
        trtm = df[df["treatment"] == 1][outcome_col].values
        results["Standard T Test"] = TwoSampleTTest().run(ctrl, trtm)

        # 2. CUPED
        cuped = CUPEDAdjuster(covariate_col, outcome_col).fit(df)
        df_cuped = df.copy()
        df_cuped[f"{outcome_col}_cuped"] = cuped.transform(df)
        c_cuped = df_cuped[df_cuped["treatment"] == 0][f"{outcome_col}_cuped"].values
        t_cuped = df_cuped[df_cuped["treatment"] == 1][f"{outcome_col}_cuped"].values
        results["CUPED"] = TwoSampleTTest().run(c_cuped, t_cuped)

        # 3. Post Stratification
        results["Post Stratification"] = PostStratificationEstimator().run(df, outcome_col, stratum_col)

        # 4. Regression Adjustment
        results["Regression Adjustment"] = RegressionAdjustedEstimator().run(df, outcome_col, [covariate_col])

        # Build comparison table
        data = []
        base_se = results["Standard T Test"].absolute_difference / results["Standard T Test"].t_statistic if results["Standard T Test"].t_statistic != 0 else 1.0

        for name, res in results.items():
            se = abs(res.absolute_difference / res.t_statistic) if res.t_statistic != 0 else 0.0
            var_reduction = (1 - (se / base_se) ** 2) * 100 if base_se > 0 else 0.0
            
            data.append({
                "Method": name,
                "Estimated Effect": res.absolute_difference,
                "Standard Error": se,
                "CI Width": res.confidence_interval_95[1] - res.confidence_interval_95[0],
                "Var Reduction Pct": var_reduction,
                "MDE (80% power)": 2.8 * se, # Approximated for a portfolio visual
                "p Value": res.p_value,
                "Recommendation": res.recommendation
            })

        return pd.DataFrame(data)


if __name__ == "__main__":
    import os
    if not os.path.exists("data/raw/synthetic_telemetry.parquet"):
        from data.generate import generate_synthetic_telemetry
        df = generate_synthetic_telemetry()
    else:
        df = pd.read_parquet("data/raw/synthetic_telemetry.parquet")
    
    benchmark = VarianceBenchmark()
    results = benchmark.run_benchmark(df, "clicked", "pre_experiment_engagement", "user_segment")
    print("\nVariance Benchmark Results:")
    print(results.to_string(index=False))
