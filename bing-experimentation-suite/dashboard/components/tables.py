from __future__ import annotations

import pandas as pd
import streamlit as st


def display_results_table(df: pd.DataFrame) -> None:
    """Display a professional experiment results comparison table.

    The table highlights the chosen statistical method and its relative
    lift and significance. It provides a quick reference for product
    decision making and technical validation.
    """
    st.table(df.style.format({
        "Estimated Effect": "{:.4f}",
        "Standard Error": "{:.4f}",
        "CI Width": "{:.4f}",
        "Var Reduction Pct": "{:.2f}%",
        "MDE (80% power)": "{:.4f}",
        "p Value": "{:.6f}"
    }))


def display_health_summary(health_results: list[dict]) -> None:
    """Show a concise health summary for all monitored metrics.

    This table clearly identifies any quality alarms or sensitivity
    degradations, allowing the data team to act before they impact the
    reliability of experimental results.
    """
    df = pd.DataFrame(health_results)
    st.dataframe(df.style.applymap(
        lambda val: "color: red" if val in ["SENSITIVITY_ALARM", "VARIANCE_ALARM"] else "color: green",
        subset=["status"]
    ))
