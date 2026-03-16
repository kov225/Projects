"""
schema_registry.py

Tracks which feature columns belong to which model version. A version
mismatch between the schema the model was trained on and the features
the serving layer computes is one of the sneakiest production bugs in ML
systems. This registry makes that mismatch an explicit, loud error rather
than a silent one.

The registry is a versioned JSON file. Each entry maps a model version
string to the ordered list of feature column names. The serving layer
loads this at startup and validates every inference request against the
registered schema for the current model version.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path("features/feature_schema_registry.json")


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    if not path.exists():
        return {"versions": {}}
    with open(path) as f:
        return json.load(f)


def save_registry(registry: dict, path: Path = REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def register_schema(version: str, feature_columns: list[str], notes: str = "") -> None:
    """Record the feature schema for `version`.
    
    Raises ValueError if this version already exists, because overwriting a
    deployed schema silently is how you corrupt a production model.
    """
    registry = load_registry()
    if version in registry["versions"]:
        raise ValueError(
            f"Schema for version '{version}' already exists. "
            "Create a new version instead of overwriting."
        )
    registry["versions"][version] = {
        "feature_columns": feature_columns,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "n_features": len(feature_columns),
    }
    save_registry(registry)
    logger.info(f"Registered schema for version '{version}': {len(feature_columns)} features")


def get_schema(version: str) -> list[str]:
    """Return the ordered feature column list for `version`.
    
    Raises KeyError if the version has no registered schema : fail loud.
    """
    registry = load_registry()
    if version not in registry["versions"]:
        available = list(registry["versions"].keys())
        raise KeyError(
            f"No schema registered for model version '{version}'. "
            f"Available versions: {available}"
        )
    return registry["versions"][version]["feature_columns"]


def validate_features(version: str, feature_dict: dict) -> None:
    """Assert that feature_dict contains exactly the columns registered for version.
    
    Missing features are a model bug. Extra features are silently ignored
    by most frameworks but indicate a pipeline drift, so we flag both.
    """
    expected = set(get_schema(version))
    actual = set(feature_dict.keys())
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ValueError(f"Missing features for version '{version}': {sorted(missing)}")
    if extra:
        logger.warning(f"Extra features not in registered schema: {sorted(extra)}")


def list_versions() -> list[dict]:
    registry = load_registry()
    return [
        {"version": v, **meta}
        for v, meta in registry["versions"].items()
    ]
