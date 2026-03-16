"""
fred_macro_pipeline.py

Pulls unemployment rate (UNRATE), CPI (CPIAUCSL), and the federal funds rate
(FEDFUNDS) from FRED and merges them onto the loan records using point-in-time
correct joins. "Point-in-time correct" means that for a loan originated in
month M, we only attach macro data available *before* month M : so if FRED
releases February data on March 5th, a loan originated on March 1 gets
January's macro values, not February's. This prevents any look-ahead bias
from leaking into training.

The typical way this goes wrong is a simple left-join on year-month, which
gives February data to a February origination even though that data wasn't
published yet. We avoid it by shifting all macro series forward by one month
before joining.
"""

import os
import logging
from datetime import date
from pathlib import Path

import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FRED_SERIES = {
    "unemployment_rate": "UNRATE",
    "cpi": "CPIAUCSL",
    "fed_funds_rate": "FEDFUNDS",
}

# Observation start covers the entire LendingClub dataset range
OBS_START = "2007-01-01"
OBS_END = date.today().isoformat()


def fetch_macro_series(fred: Fred) -> pd.DataFrame:
    """Pull each series and align to a monthly period index.
    
    FRED returns monthly data with the release date as the series index.
    We resample to month-end to get a clean monthly cadence before the
    look-ahead shift.
    """
    frames = []
    for col_name, series_id in FRED_SERIES.items():
        raw = fred.get_series(series_id, observation_start=OBS_START, observation_end=OBS_END)
        monthly = raw.resample("ME").last().rename(col_name)
        frames.append(monthly)
        logger.info(f"Fetched {series_id}: {len(monthly)} monthly observations")

    macro = pd.concat(frames, axis=1)
    macro.index = macro.index.to_period("M")
    macro.index.name = "macro_month"
    return macro


def apply_publication_lag(macro: pd.DataFrame, lag_months: int = 1) -> pd.DataFrame:
    """Shift macro data forward by lag_months to simulate publication delay.
    
    BLS and Fed typically publish prior-month data with a 3-5 week lag. A
    one-month shift is conservative but safe : it ensures that a loan
    originated on, say, March 15 only sees February's macro data (and not
    March's, which hasn't been released yet).
    """
    shifted = macro.copy()
    shifted.index = macro.index.shift(lag_months)
    return shifted


def build_macro_lookup(save_path: Path | None = None) -> pd.DataFrame:
    """Build the full macro feature table, save it, and return it."""
    fred = Fred(api_key=os.environ["FRED_API_KEY"])
    macro = fetch_macro_series(fred)
    macro_lagged = apply_publication_lag(macro, lag_months=1)

    # Compute month-over-month change rates as additional features : these
    # capture acceleration of economic conditions, not just levels.
    macro_lagged["unemployment_mom_change"] = macro_lagged["unemployment_rate"].diff()
    macro_lagged["cpi_yoy_pct"] = macro_lagged["cpi"].pct_change(periods=12) * 100

    macro_lagged = macro_lagged.dropna(subset=["unemployment_rate", "cpi", "fed_funds_rate"])
    logger.info(f"Macro table ready: {macro_lagged.shape[0]} months, {macro_lagged.shape[1]} features")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        macro_lagged.to_parquet(save_path)
        logger.info(f"Saved macro table to {save_path}")

    return macro_lagged


def join_macro_to_loans(loans: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Merge macro features onto the loan dataframe by origination month.
    
    We derive the origination period from the issue_d column and use it as
    the join key. Any loan whose origination month falls outside the macro
    table's coverage is dropped rather than filled : silence is safer than
    a bad imputation here.
    
    Args:
        loans: DataFrame with an 'issue_d' column (datetime).
        macro: The point-in-time macro table from build_macro_lookup().
    
    Returns:
        Loan DataFrame with macro features appended.
    """
    if not pd.api.types.is_datetime64_any_dtype(loans["issue_d"]):
        loans = loans.copy()
        loans["issue_d"] = pd.to_datetime(loans["issue_d"])

    loans["origination_period"] = loans["issue_d"].dt.to_period("M")

    before = len(loans)
    merged = loans.merge(
        macro.reset_index(),
        left_on="origination_period",
        right_on="macro_month",
        how="inner",
    ).drop(columns=["macro_month"])

    after = len(merged)
    if after < before:
        logger.warning(
            f"Dropped {before - after} loans whose origination month had no macro coverage"
        )

    logger.info(f"Joined macro features: {after} loans with {len(macro.columns)} macro columns")
    return merged


if __name__ == "__main__":
    out = Path("data/processed/macro_features.parquet")
    macro = build_macro_lookup(save_path=out)
    print(macro.tail(10))
