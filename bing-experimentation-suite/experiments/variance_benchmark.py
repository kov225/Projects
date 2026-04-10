import logging
import pandas as pd
import numpy as np
from .ab_test import TwoSampleTTest, ExperimentResult
from .cuped import CUPEDAdjuster
from .stratification import PostStratificationEstimator, RegressionAdjustedEstimator

# Set up technical logging
logger = logging.getLogger(__name__)

class VarianceBenchmark:
    """
    A comparative evaluation framework for online experimentation estimators.

    This suite implements and compares four frequentist inference techniques:
    1. Standard Welch's T-Test (Baseline)
    2. CUPED (Controlled-experiment Using Pre-Experiment Data) - Deng et al. (2013)
    3. Post-Stratification - Miratrix et al. (2013)
    4. Regression Adjustment (CUPAC/OWLS variant)
    """

    def detect_hte(self, df: pd.DataFrame, outcome_col: str, segment_col: str) -> pd.DataFrame:
        """
        Detects Heterogeneous Treatment Effects (HTE) across segments.
        
        Significant variance in treatment impact across segments suggests that a 
        single Global Average Treatment Effect (GATE) may be misleading for product decisions.
        """
        logger.info(f"Analyzing HTE across {segment_col}")
        hte_results = []
        segments = df[segment_col].unique()
        
        for seg in segments:
            sub = df[df[segment_col] == seg]
            ctrl = sub[sub["treatment"] == 0][outcome_col].values
            trtm = sub[sub["treatment"] == 1][outcome_col].values
            
            if len(ctrl) > 10 and len(trtm) > 10:
                res = TwoSampleTTest().run(ctrl, trtm)
                hte_results.append({
                    "Segment": seg,
                    "Lift %": res.relative_lift_pct,
                    "p-Value": res.p_value,
                    "Significant": res.is_significant
                })
        
        return pd.DataFrame(hte_results)

    def run_benchmark(
        self, df: pd.DataFrame, outcome_col: str, covariate_col: str, stratum_col: str
    ) -> pd.DataFrame:
        """
        Executes all estimators and builds a comparative performance matrix.
        """
        results = {}
        logger.info("Initializing Variance Benchmark Suite...")

        # 1. Standard t test (Baseline)
        ctrl = df[df["treatment"] == 0][outcome_col].values
        trtm = df[df["treatment"] == 1][outcome_col].values
        results["Standard T Test"] = TwoSampleTTest().run(ctrl, trtm)

        # 2. CUPED (Deng et al. 2013)
        # Using pre-experiment covariate to reduce variance in the post-period outcome
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

        # Compilation of performance registry
        data = []
        # Calculate baseline SE safely
        res_baseline = results["Standard T Test"]
        base_se = (res_baseline.confidence_interval_95[1] - res_baseline.confidence_interval_95[0]) / (2 * 1.96)

        for name, res in results.items():
            se = (res.confidence_interval_95[1] - res.confidence_interval_95[0]) / (2 * 1.96)
            var_reduction = (1 - (se / base_se) ** 2) * 100 if base_se > 0 else 0.0
            
            data.append({
                "Method": name,
                "Point Estimate": res.absolute_difference,
                "Std Error": se,
                "Relative Lift %": res.relative_lift_pct,
                "Var reduction %": var_reduction,
                "MDE (80% power)": 2.8 * se, 
                "p-Value": res.p_value,
                "Status": res.recommendation
            })

        return pd.DataFrame(data)


if __name__ == "__main__":
    import os
    # Configure root logger for the script
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Data loading/generation logic
    data_path = "data/raw/synthetic_telemetry.parquet"
    if not os.path.exists(data_path):
        from data.generate import generate_synthetic_telemetry
        logger.info("Local data not found. Generating realistic telemetry...")
        df = generate_synthetic_telemetry()
    else:
        df = pd.read_parquet(data_path)
    
    # Execution
    benchmark = VarianceBenchmark()
    logger.info("--- STARTING STATISTICAL BENCHMARK ---")
    results = benchmark.run_benchmark(df, "clicked", "pre_experiment_engagement", "user_segment")
    
    print("\n" + "="*80)
    print("STRATEGIC ESTIMATOR PERFORMANCE REPORT")
    print("="*80)
    print(results.to_string(index=False))
    print("="*80)
    
    # HTE Deep Dive
    print("\nHETEROGENEOUS TREATMENT EFFECTS (HTE) ANALYSIS")
    hte_df = benchmark.detect_hte(df, "clicked", "user_segment")
    print(hte_df.to_string(index=False))
    
    best_method = results.loc[results["Var reduction %"].idxmax(), "Method"]
    print(f"\nANALYTICAL RECOMMENDATION: Use {best_method} for primary metric reporting.")
    print(f"Variance reduction of {results['Var reduction %'].max():.1f}% achieved.")
