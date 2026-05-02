"""Reusable plotting helpers. Keeps the notebooks focused on analysis instead
of styling boilerplate. The colour palette is the seaborn 'muted' set so
charts stay readable without primary-colour glare.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PALETTE = sns.color_palette("muted")
DEFAULT_FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def set_style() -> None:
    """Apply the project-wide matplotlib style. Call once per notebook."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("muted")
    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 110,
            "axes.grid.which": "major",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig(fig: plt.Figure, name: str, fig_dir: Path | None = None) -> Path:
    """Persist a figure to figures/. Returns the full path written."""
    target_dir = fig_dir or DEFAULT_FIG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / f"{name}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    return out


def line_with_band(
    ax: plt.Axes,
    x: Sequence,
    y: Sequence,
    lo: Sequence | None = None,
    hi: Sequence | None = None,
    label: str | None = None,
    color: str | None = None,
) -> None:
    """Line plot with an optional shaded confidence band."""
    color = color or PALETTE[0]
    ax.plot(x, y, color=color, linewidth=2, label=label)
    if lo is not None and hi is not None:
        ax.fill_between(x, lo, hi, color=color, alpha=0.18)


def retention_heatmap(
    cohort_matrix: pd.DataFrame,
    title: str = "Cohort retention",
    cmap: str = "BuPu",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a cohort week vs weeks-since-signup retention heatmap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(
        cohort_matrix * 100,
        cmap=cmap,
        annot=False,
        cbar_kws={"label": "Retention rate (%)"},
        ax=ax,
        linewidths=0.3,
        linecolor="white",
    )
    ax.set_title(title)
    ax.set_xlabel("Weeks since first visit")
    ax.set_ylabel("Cohort week")
    return ax


def funnel_chart(
    steps: Sequence[str],
    counts: Sequence[int],
    title: str = "Conversion funnel",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Horizontal funnel with conversion percentages annotated."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    counts_arr = np.array(counts)
    pct = counts_arr / counts_arr[0] * 100
    bars = ax.barh(list(steps)[::-1], counts_arr[::-1], color=PALETTE[2], edgecolor="white")
    for bar, p, c in zip(bars, pct[::-1], counts_arr[::-1]):
        ax.text(
            bar.get_width() * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{c:,} ({p:.1f}%)",
            va="center",
            fontsize=10,
            color="#333",
        )
    ax.set_title(title)
    ax.set_xlabel("Users")
    ax.margins(x=0.18)
    return ax


def segment_radar(
    profile: pd.DataFrame,
    title: str = "Segment behavioural profile",
) -> plt.Figure:
    """Radar chart comparing segments across normalised behavioural metrics."""
    metrics = list(profile.columns)
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    for i, (segment, row) in enumerate(profile.iterrows()):
        values = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, values, label=str(segment), color=PALETTE[i % len(PALETTE)], linewidth=2)
        ax.fill(angles, values, color=PALETTE[i % len(PALETTE)], alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title(title, pad=18)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.1), frameon=False)
    return fig
