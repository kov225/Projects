"""
kafka_consumer.py

Consumes loan application events from Kafka, writes features to the Redis
feature store, and immediately scores them via the FastAPI /score endpoint.
This is how we test the full streaming loop: producer → Kafka → consumer
→ feature store → scorer → PostgreSQL.

In production this would be a long-running service. Locally it runs until
you kill it or until max_records is reached for testing purposes.
"""

import json
import logging
import os
from typing import Any

import httpx
import redis
from dotenv import load_dotenv
from kafka import KafkaConsumer

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class LoanApplicationConsumer:
    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        score_api_url: str,
        redis_host: str,
        redis_port: int,
        redis_ttl: int,
    ) -> None:
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            consumer_timeout_ms=30_000,  # stop after 30s idle : useful for tests
        )
        self.score_api_url = score_api_url
        self.http = httpx.Client(timeout=5.0)
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.redis_ttl = redis_ttl
        self.processed = 0

    def cache_features(self, applicant_id: str, record: dict[str, Any]) -> None:
        """Write the raw loan record into Redis so the API can retrieve it.
        
        We store the whole record rather than just engineered features because
        the serving API runs the feature pipeline at scoring time. Redis here
        is a lookup cache, not a precomputed-feature store for batch jobs.
        The TTL prevents stale entries from accumulating indefinitely.
        """
        key = f"applicant:{applicant_id}:raw"
        self.redis.set(key, json.dumps(record, default=str), ex=self.redis_ttl)

    def score_application(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Hit the /score endpoint synchronously and return the response."""
        try:
            # Pull out the non-feature metadata before sending to the scorer
            payload = {k: v for k, v in record.items() if k != "default"}
            resp = self.http.post(f"{self.score_api_url}/score", json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.warning(f"Scoring failed for {record.get('applicant_id')}: {e}")
            return None

    def run(self, max_records: int | None = None) -> None:
        logger.info("Consumer started, waiting for messages")
        try:
            for msg in self.consumer:
                record = msg.value
                applicant_id = record.get("applicant_id", f"anon_{self.processed}")
                self.cache_features(applicant_id, record)
                result = self.score_application(record)
                if result:
                    logger.debug(
                        f"{applicant_id} → score={result.get('risk_score'):.4f} "
                        f"decision={result.get('decision')} model={result.get('model_version')}"
                    )
                self.processed += 1
                if self.processed % 200 == 0:
                    logger.info(f"Processed {self.processed} messages")
                if max_records and self.processed >= max_records:
                    logger.info(f"Reached max_records={max_records}, stopping")
                    break
        finally:
            self.consumer.close()
            self.http.close()
            logger.info(f"Consumer done. Total processed: {self.processed}")


if __name__ == "__main__":
    LoanApplicationConsumer(
        bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        topic=os.environ.get("KAFKA_TOPIC", "loan_applications"),
        group_id=os.environ.get("KAFKA_CONSUMER_GROUP", "credit_scorer"),
        score_api_url=os.environ.get("SCORE_API_URL", "http://localhost:8000"),
        redis_host=os.environ.get("REDIS_HOST", "localhost"),
        redis_port=int(os.environ.get("REDIS_PORT", "6379")),
        redis_ttl=int(os.environ.get("REDIS_TTL_SECONDS", "3600")),
    ).run()
