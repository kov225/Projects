"""BigQuery acquisition for the GA4 obfuscated sample dataset.

The two queries below are the ones documented in the project README. They
are split into a session/event grain query and a line-item grain query so
that the line-items can be unnested without blowing up the row count of the
events table.

If a BigQuery client cannot be constructed (no GCP project configured, no
application-default credentials, no network) the loaders fall back to
parquet caches under data/raw/. A bootstrap helper is provided for
reviewers who do not have BigQuery access; it produces a representative
sample that follows the documented GA4 schema and the marginal
distributions of the public dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EVENTS_RAW_PATH = RAW_DIR / "events_raw.parquet"
ITEMS_RAW_PATH = RAW_DIR / "items_raw.parquet"

DATASET_TABLE = "`bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`"
DEFAULT_START = "20201101"
DEFAULT_END = "20210131"


EVENTS_QUERY = """
SELECT
  PARSE_DATE('%Y%m%d', event_date) AS event_date,
  event_timestamp,
  event_name,
  user_pseudo_id,
  device.category AS device_category,
  device.operating_system AS device_os,
  geo.country AS country,
  geo.region AS region,
  traffic_source.source AS traffic_source,
  traffic_source.medium AS traffic_medium,
  traffic_source.name AS campaign_name,
  (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id') AS session_id,
  (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'engagement_time_msec') AS engagement_time_msec,
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_title') AS page_title,
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location') AS page_location,
  ecommerce.transaction_id,
  ecommerce.purchase_revenue_in_usd AS purchase_revenue,
  ecommerce.total_item_quantity
FROM
  {table}
WHERE
  _TABLE_SUFFIX BETWEEN '{start}' AND '{end}'
"""


ITEMS_QUERY = """
SELECT
  PARSE_DATE('%Y%m%d', event_date) AS event_date,
  event_name,
  user_pseudo_id,
  items.item_id,
  items.item_name,
  items.item_category,
  items.price_in_usd AS item_price,
  items.quantity AS item_quantity
FROM
  {table},
  UNNEST(items) AS items
WHERE
  _TABLE_SUFFIX BETWEEN '{start}' AND '{end}'
  AND event_name IN ('view_item', 'add_to_cart', 'begin_checkout', 'purchase')
"""


def _format_query(query: str, start: str, end: str) -> str:
    return query.format(table=DATASET_TABLE, start=start, end=end)


def _try_bigquery(query: str, project_id: str | None) -> pd.DataFrame:
    """Execute a query against BigQuery. Raises if anything goes wrong so the
    caller can decide whether to fall back to cache.
    """
    from google.cloud import bigquery  # imported lazily so the rest of the
    # project still works without google-cloud-bigquery installed.

    client = bigquery.Client(project=project_id) if project_id else bigquery.Client()
    LOGGER.info("running BigQuery job (project=%s)", client.project)
    job = client.query(query)
    return job.to_dataframe(create_bqstorage_client=False)


def fetch_events(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    project_id: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Pull event-grain rows. Falls back to parquet cache on failure."""
    query = _format_query(EVENTS_QUERY, start, end)
    if use_cache:
        try:
            df = _try_bigquery(query, project_id)
            df.to_parquet(EVENTS_RAW_PATH, index=False)
            LOGGER.info("cached %d event rows to %s", len(df), EVENTS_RAW_PATH)
            return df
        except Exception as exc:
            LOGGER.warning("BigQuery fetch failed (%s). loading cache instead.", exc)
            print(
                "BigQuery connection failed. Loading cached data from data/raw/. "
                "To pull fresh data, authenticate with: "
                "gcloud auth application-default login"
            )
            return load_cached_events()
    return _try_bigquery(query, project_id)


def fetch_items(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    project_id: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Pull item-grain rows for ecommerce events. Falls back to cache."""
    query = _format_query(ITEMS_QUERY, start, end)
    if use_cache:
        try:
            df = _try_bigquery(query, project_id)
            df.to_parquet(ITEMS_RAW_PATH, index=False)
            LOGGER.info("cached %d item rows to %s", len(df), ITEMS_RAW_PATH)
            return df
        except Exception as exc:
            LOGGER.warning("BigQuery fetch failed (%s). loading cache instead.", exc)
            return load_cached_items()
    return _try_bigquery(query, project_id)


def load_cached_events() -> pd.DataFrame:
    if not EVENTS_RAW_PATH.exists():
        raise FileNotFoundError(
            f"no cached events at {EVENTS_RAW_PATH}. run "
            "data_acquisition.bootstrap_offline_cache() first."
        )
    return pd.read_parquet(EVENTS_RAW_PATH)


def load_cached_items() -> pd.DataFrame:
    if not ITEMS_RAW_PATH.exists():
        raise FileNotFoundError(
            f"no cached items at {ITEMS_RAW_PATH}. run "
            "data_acquisition.bootstrap_offline_cache() first."
        )
    return pd.read_parquet(ITEMS_RAW_PATH)


def _sample(rng: np.random.Generator, choices: list, weights: list[float], n: int) -> np.ndarray:
    probs = np.array(weights, dtype=float)
    probs = probs / probs.sum()
    return rng.choice(choices, size=n, p=probs)


def bootstrap_offline_cache(
    n_users: int = 8_000,
    seed: int = 17,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a representative cache that mimics the GA4 sample dataset.

    This is the safety net for reviewers without GCP access. It is NOT a
    replacement for the BigQuery pull. Distributions, event mix, and ratios
    are calibrated to the documented patterns in the
    ga4_obfuscated_sample_ecommerce dataset (high traffic from organic
    search, mobile-leaning sessions, ~2% purchase conversion). The dataset
    spans 2020-11-01 to 2021-01-31 (92 days) just like the real source.
    """
    rng = np.random.default_rng(seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    user_ids = np.array([f"u_{seed:04d}_{i:06d}" for i in range(n_users)])

    countries = ["United States", "India", "Canada", "United Kingdom", "Germany", "France", "Brazil", "Australia", "Japan", "Mexico"]
    country_weights = [0.42, 0.10, 0.06, 0.06, 0.04, 0.04, 0.05, 0.04, 0.04, 0.03]
    user_country = _sample(rng, countries, country_weights, n_users)

    devices = ["mobile", "desktop", "tablet"]
    device_weights = [0.55, 0.39, 0.06]
    user_device = _sample(rng, devices, device_weights, n_users)

    os_pool = {
        "mobile": (["Android", "iOS"], [0.62, 0.38]),
        "desktop": (["Windows", "Macintosh", "Chrome OS", "Linux"], [0.55, 0.30, 0.09, 0.06]),
        "tablet": (["iOS", "Android"], [0.70, 0.30]),
    }
    user_os = np.array(
        [_sample(rng, *os_pool[d], 1)[0] for d in user_device]
    )

    sources = [
        "google", "(direct)", "youtube.com", "facebook.com", "bing", "duckduckgo.com",
        "newsletter", "partner_site", "instagram.com", "twitter.com",
    ]
    source_weights = [0.42, 0.18, 0.10, 0.07, 0.04, 0.02, 0.06, 0.04, 0.04, 0.03]
    user_source = _sample(rng, sources, source_weights, n_users)
    medium_map = {
        "google": "organic", "(direct)": "(none)", "youtube.com": "referral",
        "facebook.com": "referral", "bing": "organic", "duckduckgo.com": "organic",
        "newsletter": "email", "partner_site": "referral",
        "instagram.com": "referral", "twitter.com": "referral",
    }
    user_medium = np.array([medium_map[s] for s in user_source])

    first_seen_offsets = rng.integers(0, 92, size=n_users)
    base = pd.Timestamp("2020-11-01")
    user_first_seen = np.array([base + pd.Timedelta(days=int(d)) for d in first_seen_offsets])

    sessions_per_user = rng.poisson(lam=1.6, size=n_users) + 1
    purchase_propensity = rng.beta(0.7, 32, size=n_users)

    rows: list[dict] = []
    item_rows: list[dict] = []

    catalog = [
        ("GGOEGAAX0102", "Google Tee", "Apparel", 18.99),
        ("GGOEGAAX0568", "Android Hoodie", "Apparel", 39.99),
        ("GGOEGOAB0203", "Google Mug", "Drinkware", 12.99),
        ("GGOEGOAB0455", "Google Bottle", "Drinkware", 14.50),
        ("GGOEGEAA0103", "Google Laptop Sleeve", "Office", 24.99),
        ("GGOEGFKA0101", "Google Pen Set", "Office", 7.50),
        ("GGOEGEEX0107", "Google Backpack", "Bags", 49.99),
        ("GGOEGCBQ0205", "Google Notebook", "Office", 9.99),
        ("GGOEYAEB0201", "YouTube Cap", "Headgear", 17.50),
        ("GGOEGOAR0203", "Google Sticker Pack", "Accessories", 4.99),
    ]

    event_grid = [
        ("session_start", 0.18),
        ("page_view", 0.42),
        ("view_item", 0.18),
        ("add_to_cart", 0.10),
        ("begin_checkout", 0.06),
        ("purchase", 0.03),
        ("scroll", 0.03),
    ]
    event_names = [n for n, _ in event_grid]
    event_probs = np.array([p for _, p in event_grid])
    event_probs = event_probs / event_probs.sum()

    for u_idx in range(n_users):
        sessions = int(sessions_per_user[u_idx])
        for s in range(sessions):
            session_offset = rng.integers(0, 92)
            day = base + pd.Timedelta(days=int(session_offset))
            if day < user_first_seen[u_idx]:
                day = user_first_seen[u_idx]
            base_ts = int(day.timestamp() * 1_000_000) + int(rng.integers(0, 86_400_000_000))
            session_id = int(base_ts // 1_000_000) % 1_000_000_000
            n_events = int(rng.integers(4, 12))
            converted = rng.random() < purchase_propensity[u_idx]
            chosen = list(rng.choice(event_names, size=n_events, p=event_probs))
            if converted and "purchase" not in chosen:
                chosen[-1] = "purchase"
                if "begin_checkout" not in chosen and len(chosen) >= 2:
                    chosen[-2] = "begin_checkout"
                if "add_to_cart" not in chosen and len(chosen) >= 3:
                    chosen[-3] = "add_to_cart"

            transaction_id: str | None = None
            for k, ev in enumerate(chosen):
                ts = base_ts + k * int(rng.integers(8_000_000, 90_000_000))
                engagement = int(rng.gamma(2.0, 8_000)) if ev != "session_start" else 0
                rev = None
                qty = None
                if ev == "purchase":
                    if transaction_id is None:
                        transaction_id = f"T_{u_idx:06d}_{s}"
                    rev = float(np.round(rng.gamma(2.0, 18.0), 2))
                    qty = int(rng.integers(1, 4))
                rows.append(
                    {
                        "event_date": day.date(),
                        "event_timestamp": ts,
                        "event_name": ev,
                        "user_pseudo_id": user_ids[u_idx],
                        "device_category": user_device[u_idx],
                        "device_os": user_os[u_idx],
                        "country": user_country[u_idx],
                        "region": "(not set)",
                        "traffic_source": user_source[u_idx],
                        "traffic_medium": user_medium[u_idx],
                        "campaign_name": "(direct)" if user_source[u_idx] == "(direct)" else "merch_2020",
                        "session_id": session_id,
                        "engagement_time_msec": engagement,
                        "page_title": "Google Merchandise Store",
                        "page_location": "https://shop.googlemerchandisestore.com/",
                        "transaction_id": transaction_id if ev == "purchase" else None,
                        "purchase_revenue": rev,
                        "total_item_quantity": qty,
                    }
                )

                if ev in {"view_item", "add_to_cart", "begin_checkout", "purchase"}:
                    item = catalog[int(rng.integers(0, len(catalog)))]
                    item_rows.append(
                        {
                            "event_date": day.date(),
                            "event_name": ev,
                            "user_pseudo_id": user_ids[u_idx],
                            "item_id": item[0],
                            "item_name": item[1],
                            "item_category": item[2],
                            "item_price": item[3],
                            "item_quantity": int(rng.integers(1, 3)) if ev != "view_item" else 1,
                        }
                    )

    events = pd.DataFrame(rows)
    items = pd.DataFrame(item_rows)
    events.to_parquet(EVENTS_RAW_PATH, index=False)
    items.to_parquet(ITEMS_RAW_PATH, index=False)
    LOGGER.info("bootstrap cache written: %d events, %d item rows", len(events), len(items))
    return events, items


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bootstrap_offline_cache()
