"""
kafka_producer.py

Replays historical loan records from the clean Parquet file as a stream
of JSON messages to a Kafka topic. You configure the rate via REPLAY_RATE_PER_SEC
in the environment. This is used to stress-test the serving layer and to
drive the drift simulation scenarios.

The producer deliberately adds a small amount of jitter to the inter-message
sleep so the stream looks like real traffic rather than a perfect metronome.
"""

import json
import logging
import os
import random
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaProducer

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_producer(bootstrap_servers: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        acks="all",                    # wait for all ISR replicas
        retries=5,
        linger_ms=10,                  # small batch window for throughput
        compression_type="gzip",
    )


def stream_loans(
    parquet_path: Path,
    topic: str,
    bootstrap_servers: str,
    rate_per_sec: float = 10.0,
    loop: bool = False,
    drift_scenario: str | None = None,
) -> None:
    """Stream loan records to Kafka, optionally injecting drift.
    
    Args:
        parquet_path: Path to the clean loans Parquet file.
        topic: Kafka topic name.
        bootstrap_servers: Kafka broker address(es).
        rate_per_sec: Target message rate. We sleep between messages accordingly.
        loop: If True, restart from the beginning when exhausted.
        drift_scenario: One of 'economic_shock', 'population_shift', 'concept_drift',
                        or None for clean replay.
    """
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} loan records from {parquet_path}")

    df = apply_drift_scenario(df, drift_scenario) if drift_scenario else df

    producer = build_producer(bootstrap_servers)
    sleep_base = 1.0 / rate_per_sec
    sent = 0

    def replay():
        nonlocal sent
        for _, row in df.iterrows():
            record = row.to_dict()
            # Attach a simulated applicant ID so the serving layer can use
            # the feature store keyed lookup path
            record["applicant_id"] = f"sim_{sent:08d}"
            producer.send(topic, value=record)
            sent += 1
            if sent % 500 == 0:
                logger.info(f"Sent {sent} records")
            # Jitter ±20% of base sleep to look like real traffic
            time.sleep(sleep_base * random.uniform(0.8, 1.2))

    try:
        while True:
            replay()
            if not loop:
                break
            logger.info("Loop complete, restarting stream")
    finally:
        producer.flush()
        producer.close()
        logger.info(f"Producer done. Total sent: {sent}")


def apply_drift_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Inject a drift scenario into the loan records before streaming.
    
    Scenario 'economic_shock': spike unemployment_rate and cpi to 2008 recession
    peak values (10.0% and high CPI) for all records. This tests whether the
    PSI monitor fires on macro feature drift.

    Scenario 'population_shift': gradually skew annual_inc and loan_amnt toward
    lower values over the course of the dataframe, simulating a slow demographic
    change over 60 days of streaming.

    Scenario 'concept_drift': artificially flip 15% of the 'default' labels from
    0 to 1 without touching any input feature. PSI will not fire; only AUC
    monitoring should catch this via the rolling holdout degradation.
    """
    df = df.copy()

    if scenario == "economic_shock":
        logger.info("Injecting economic shock: unemployment=10.0, CPI spike")
        if "unemployment_rate" in df.columns:
            df["unemployment_rate"] = 10.0
        if "cpi" in df.columns:
            df["cpi"] = df["cpi"] * 1.15
        if "fed_funds_rate" in df.columns:
            df["fed_funds_rate"] = 0.25  # ZIRP era

    elif scenario == "population_shift":
        logger.info("Injecting population shift: gradual income / loan_amnt skew")
        n = len(df)
        # Scale factor ramps from 1.0 at the start to 0.6 at the end
        # to simulate a 40% drop in borrower income over the simulated 60 days
        ramp = 1.0 - 0.4 * (df.reset_index().index / n)
        if "annual_inc" in df.columns:
            df["annual_inc"] = df["annual_inc"] * ramp.values
        if "loan_amnt" in df.columns:
            df["loan_amnt"] = df["loan_amnt"] * ramp.values

    elif scenario == "concept_drift":
        logger.info("Injecting concept drift: flipping 15% of default labels")
        # Only flip currently non-defaulted records to default : this raises
        # the true default rate without any change to X, which PSI cannot detect.
        non_default_idx = df[df["default"] == 0].sample(frac=0.15, random_state=42).index
        df.loc[non_default_idx, "default"] = 1
        logger.info(
            f"New default rate after concept drift: {df['default'].mean():.3f}"
        )

    else:
        raise ValueError(f"Unknown drift scenario: {scenario}")

    return df


if __name__ == "__main__":
    import sys
    scenario = sys.argv[1] if len(sys.argv) > 1 else None
    stream_loans(
        parquet_path=Path("data/processed/loans_clean.parquet"),
        topic=os.environ.get("KAFKA_TOPIC", "loan_applications"),
        bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        rate_per_sec=float(os.environ.get("REPLAY_RATE_PER_SEC", "20")),
        loop=False,
        drift_scenario=scenario,
    )
