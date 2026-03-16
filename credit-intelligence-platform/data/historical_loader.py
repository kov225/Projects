"""
historical_loader.py

Reads the raw LendingClub CSV, applies a disciplined cleaning pass, and
writes a Parquet file that everything downstream consumes. Keeping this
as its own script rather than inline pipeline code means the cleaning
logic is versioned, documented, and reproducible.

The LendingClub dataset has a few well-known annoyances: percent columns
stored as "12.5%" strings, employment length as "10+ years", the target
variable split across loan_status with multiple labels, and ~150 columns
most of which are useless post-origination (anything that describes what
happened *after* the loan was issued leaks the target). We drop all of
those prospectively enriched columns explicitly.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Columns that exist in the LendingClub data but describe post-origination
# behavior : keeping any of these would be severe label leakage.
POST_ORIGINATION_COLS = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high",
    "last_fico_range_low", "collections_12_mths_ex_med", "chargeoff_within_12_mths",
    "debt_settlement_flag", "settlement_status", "settlement_date",
    "settlement_amount", "settlement_percentage", "settlement_term",
]

KEEP_COLS = [
    "issue_d", "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "loan_status", "purpose", "addr_state",
    "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "initial_list_status", "application_type", "mort_acc", "pub_rec_bankruptcies",
]

# Loan statuses that cleanly represent a binary outcome. Everything else
# (Current, In Grace Period, Late) is censored and we drop it.
CHARGED_OFF_LABELS = {"Charged Off", "Default"}
FULLY_PAID_LABELS = {"Fully Paid"}


def parse_percent(series: pd.Series) -> pd.Series:
    """Strip trailing '%' and cast to float."""
    return series.astype(str).str.replace("%", "", regex=False).str.strip().astype(float)


def parse_emp_length(series: pd.Series) -> pd.Series:
    """Convert strings like '10+ years', '< 1 year', '3 years' → numeric years."""
    s = series.astype(str).str.lower()
    s = s.str.replace("10+ years", "11", regex=False)
    s = s.str.replace("< 1 year", "0", regex=False)
    s = s.str.extract(r"(\d+)").astype(float).squeeze()
    return s


def load_and_clean(csv_path: Path, sample_frac: float | None = None) -> pd.DataFrame:
    """Full cleaning pipeline for the LendingClub raw CSV.
    
    Args:
        csv_path: Path to the raw loans CSV (accepts gzip too).
        sample_frac: If set, sample this fraction for fast iteration
                     during development. Pass None for full dataset.
    
    Returns:
        Clean DataFrame with binary 'default' target.
    """
    logger.info(f"Loading raw data from {csv_path}")

    # The first two rows of some LendingClub CSVs are header junk; skiprows=1
    # handles the description row that precedes the actual header in older dumps.
    raw = pd.read_csv(csv_path, low_memory=False, skiprows=1)
    logger.info(f"Raw shape: {raw.shape}")

    # Only keep columns we know are safe at origination time
    available_keep = [c for c in KEEP_COLS if c in raw.columns]
    df = raw[available_keep].copy()

    # --- Target variable ---
    # Filter to the clean terminal states only; drop everything ambiguous
    df = df[df["loan_status"].isin(CHARGED_OFF_LABELS | FULLY_PAID_LABELS)].copy()
    df["default"] = df["loan_status"].isin(CHARGED_OFF_LABELS).astype(int)
    df.drop(columns=["loan_status"], inplace=True)

    # --- Date parsing ---
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df.dropna(subset=["issue_d"], inplace=True)

    # --- Numeric cleanup ---
    if "int_rate" in df.columns:
        df["int_rate"] = parse_percent(df["int_rate"])
    if "revol_util" in df.columns:
        df["revol_util"] = parse_percent(df["revol_util"])
    if "emp_length" in df.columns:
        df["emp_length"] = parse_emp_length(df["emp_length"])

    # FICO midpoint : more stable as a single feature than low/high range
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
        df.drop(columns=["fico_range_low", "fico_range_high"], inplace=True)

    # term: "36 months" → 36
    if "term" in df.columns:
        df["term_months"] = df["term"].str.extract(r"(\d+)").astype(float).squeeze()
        df.drop(columns=["term"], inplace=True)

    # --- Categorical encoding ---
    # Label-encode grade and sub_grade as ordinals since they have natural order
    grade_order = list("ABCDEFG")
    df["grade_enc"] = df["grade"].map({g: i for i, g in enumerate(grade_order)})
    df.drop(columns=["grade"], inplace=True)

    if "sub_grade" in df.columns:
        all_subgrades = [f"{g}{n}" for g in grade_order for n in range(1, 6)]
        sg_map = {sg: i for i, sg in enumerate(all_subgrades)}
        df["sub_grade_enc"] = df["sub_grade"].map(sg_map)
        df.drop(columns=["sub_grade"], inplace=True)

    # One-hot encode remaining low-cardinality categoricals
    cat_cols = ["home_ownership", "verification_status", "purpose",
                "initial_list_status", "application_type"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # addr_state has ~50 levels : we keep it as-is and let downstream handle it
    # (it's used in the fairness audit as the geo proxy)

    # --- Missing value imputation ---
    # For continuous features, fill with median using training-set statistics.
    # We compute median here across the whole dataset which is fine for the
    # static training pipeline; the serving layer uses the feature store instead.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    medians = df[numeric_cols].median()
    df[numeric_cols] = df[numeric_cols].fillna(medians)

    # --- Sanity checks ---
    assert df["default"].isin([0, 1]).all(), "Target must be binary"
    assert df["issue_d"].notna().all(), "All records must have a valid issue date"

    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled {sample_frac:.0%} of data: {len(df)} rows")

    logger.info(f"Clean shape: {df.shape}, default rate: {df['default'].mean():.3f}")
    return df


def save_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="snappy")
    logger.info(f"Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    import sys
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/loan.csv")
    out_path = Path("data/processed/loans_clean.parquet")
    df = load_and_clean(csv_path)
    save_parquet(df, out_path)
    print(df.dtypes)
    print(df.head())
