"""
feature_store.py

Redis-backed feature cache with TTL-based expiry and version-aware namespacing.
The serving layer uses this to avoid recomputing features for the same
applicant ID more than once within the TTL window. The Kafka consumer writes
raw records here; the API reads from here and runs the pipeline transform
only if a cache miss occurs.

Key schema:
    applicant:{applicant_id}:features:{model_version}  →  JSON-encoded feature dict
    applicant:{applicant_id}:raw                        →  JSON-encoded raw record

We namespace by model_version so that when we promote a new model, feature
cache entries from the old version aren't silently reused (the feature schema
may have changed between versions).
"""

import json
import logging
from typing import Any

import redis
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class FeatureStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        ttl_seconds: int = 3600,
        db: int = 0,
    ) -> None:
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.ttl = ttl_seconds
        # Verify connectivity at construction time so failures are loud and early
        self.redis.ping()
        logger.info(f"Feature store connected: {host}:{port} (TTL={ttl_seconds}s)")

    def _feature_key(self, applicant_id: str, model_version: str) -> str:
        return f"applicant:{applicant_id}:features:{model_version}"

    def _raw_key(self, applicant_id: str) -> str:
        return f"applicant:{applicant_id}:raw"

    def put_features(
        self,
        applicant_id: str,
        features: dict[str, Any],
        model_version: str,
    ) -> None:
        """Cache a feature dict for `applicant_id` under the given model version."""
        key = self._feature_key(applicant_id, model_version)
        self.redis.set(key, json.dumps(features, default=float), ex=self.ttl)

    def get_features(
        self,
        applicant_id: str,
        model_version: str,
    ) -> dict[str, Any] | None:
        """Return cached features or None on a cache miss."""
        key = self._feature_key(applicant_id, model_version)
        raw = self.redis.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def put_raw(self, applicant_id: str, record: dict[str, Any]) -> None:
        """Store the raw loan application record for later feature computation."""
        self.redis.set(self._raw_key(applicant_id), json.dumps(record, default=str), ex=self.ttl)

    def get_raw(self, applicant_id: str) -> dict[str, Any] | None:
        raw = self.redis.get(self._raw_key(applicant_id))
        if raw is None:
            return None
        return json.loads(raw)

    def invalidate(self, applicant_id: str, model_version: str) -> None:
        """Drop the cached features for this applicant, forcing recomputation."""
        self.redis.delete(self._feature_key(applicant_id, model_version))

    def flush_model_version(self, model_version: str) -> int:
        """Delete all cached entries for a specific model version.
        
        Call this after promoting a new model to production so the serving
        layer doesn't serve stale features from the old schema.
        """
        pattern = f"applicant:*:features:{model_version}"
        keys = list(self.redis.scan_iter(match=pattern))
        if keys:
            self.redis.delete(*keys)
        logger.info(f"Flushed {len(keys)} keys for model version '{model_version}'")
        return len(keys)

    def health_check(self) -> bool:
        try:
            return self.redis.ping()
        except redis.ConnectionError:
            return False
