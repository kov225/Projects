"""K-Means based behavioural segmentation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)

BEHAVIOUR_FEATURES: list[str] = [
    "total_events",
    "unique_event_types",
    "avg_engagement_time",
    "sessions_count",
    "pages_viewed",
    "items_viewed",
    "purchases_count",
    "total_revenue",
]


@dataclass
class SegmentationResult:
    """Container for fitted segmentation artefacts."""

    features: pd.DataFrame
    labels: pd.Series
    profile: pd.DataFrame
    k: int
    silhouette: float
    inertia_curve: list[tuple[int, float]]
    silhouette_curve: list[tuple[int, float]]


def build_user_features(events: pd.DataFrame, items: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate the event log into the per-user behavioural feature frame."""
    g = events.groupby("user_pseudo_id", as_index=False).agg(
        total_events=("event_timestamp", "count"),
        unique_event_types=("event_name", "nunique"),
        avg_engagement_time=("engagement_time_msec", "mean"),
        sessions_count=("session_id", "nunique"),
        pages_viewed=("event_name", lambda s: int((s == "page_view").sum())),
        items_viewed=("event_name", lambda s: int((s == "view_item").sum())),
        purchases_count=("event_name", lambda s: int((s == "purchase").sum())),
        total_revenue=("purchase_revenue", lambda s: float(s.fillna(0).sum())),
    )
    g["avg_engagement_time"] = g["avg_engagement_time"].fillna(0)

    if items is not None and not items.empty:
        cats = (
            items.groupby("user_pseudo_id")["item_category"]
            .nunique()
            .rename("item_categories")
            .reset_index()
        )
        g = g.merge(cats, on="user_pseudo_id", how="left")
        g["item_categories"] = g["item_categories"].fillna(0)
    return g


def select_k(
    features: pd.DataFrame,
    candidate_k: list[int] | None = None,
    sample_size: int = 6_000,
    seed: int = 11,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Sweep candidate k values, returning inertia and silhouette curves."""
    candidate_k = candidate_k or [3, 4, 5, 6, 7]
    rng = np.random.default_rng(seed)
    sample = features.sample(min(sample_size, len(features)), random_state=int(rng.integers(0, 2**31 - 1)))
    X = StandardScaler().fit_transform(sample[BEHAVIOUR_FEATURES].fillna(0))

    inertia: list[tuple[int, float]] = []
    silhouette: list[tuple[int, float]] = []
    for k in candidate_k:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        inertia.append((k, float(km.inertia_)))
        silhouette.append((k, float(silhouette_score(X, km.labels_))))
    return inertia, silhouette


def fit_segments(
    features: pd.DataFrame,
    k: int = 5,
    seed: int = 11,
) -> SegmentationResult:
    """Fit K-Means and produce a per-user label plus a profile table."""
    feat_cols = BEHAVIOUR_FEATURES
    X = StandardScaler().fit_transform(features[feat_cols].fillna(0))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    labels = pd.Series(km.labels_, index=features.index, name="segment_id")

    profile = features.assign(segment_id=labels.values).groupby("segment_id")[feat_cols].mean()
    profile["users"] = features.assign(segment_id=labels.values).groupby("segment_id").size()

    sample = StandardScaler().fit_transform(features[feat_cols].fillna(0))
    sil = float(silhouette_score(sample, labels.values, sample_size=min(5_000, len(features)), random_state=seed))
    inertia, sil_curve = select_k(features, seed=seed)

    return SegmentationResult(
        features=features.assign(segment_id=labels.values),
        labels=labels,
        profile=profile,
        k=k,
        silhouette=sil,
        inertia_curve=inertia,
        silhouette_curve=sil_curve,
    )


def label_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Assign human readable names so each segment has a unique label.

    Ranking is purely positional, which keeps the mapping stable across
    reruns and avoids name collisions when two clusters happen to look
    similar on a single feature. The fallback list always has at least as
    many candidates as there are clusters.
    """
    candidate_pool = [
        "high_value_buyers",
        "repeat_shoppers",
        "engaged_browsers",
        "window_shoppers",
        "one_and_done_visitors",
        "casual_visitors",
        "low_engagement_visitors",
    ]

    score = (
        profile["total_revenue"].rank(method="first")
        + profile["purchases_count"].rank(method="first") * 0.5
        + profile["total_events"].rank(method="first") * 0.25
    )
    ordered = score.sort_values(ascending=False).index.tolist()

    names: dict[int, str] = {}
    for rank, idx in enumerate(ordered):
        names[idx] = candidate_pool[rank] if rank < len(candidate_pool) else f"segment_{idx}"
    return names
