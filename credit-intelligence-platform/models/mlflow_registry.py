"""
mlflow_registry.py

Wrappers around the MLflow tracking and model registry APIs. The goal is to
keep all the MLflow-specific boilerplate out of train.py so that the training
script reads cleanly as a data science workflow rather than an MLflow tutorial.

Model lifecycle:
    None → Staging (automatic after passing validation)
    Staging → Production (requires beating incumbent AUC)
    Production -> Archived (after being superseded)
"""

import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def get_client() -> MlflowClient:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    return MlflowClient(tracking_uri=tracking_uri)


def log_training_run(
    metrics: dict,
    params: dict,
    artifact_dir: Path,
    feature_columns: list[str],
) -> str:
    """Log a training run and register the model. Returns the version string."""
    model_name = os.environ.get("CHAMPION_MODEL_NAME", "credit_risk_champion")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "credit_intelligence")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"feature_columns": feature_columns}, "feature_schema.json")

        # Log all serialized artifacts (pipeline, models, scaler)
        mlflow.log_artifacts(str(artifact_dir), artifact_path="model_artifacts")

        run_id = run.info.run_id
        logger.info(f"MLflow run {run_id} logged with AUC={metrics.get('test_auc', 'N/A'):.4f}")

    # Register the model in the Model Registry
    model_uri = f"runs:/{run_id}/model_artifacts"
    try:
        registered = mlflow.register_model(model_uri, model_name)
        version = registered.version
        logger.info(f"Registered model '{model_name}' version {version}")
    except Exception as e:
        logger.warning(f"Model registration failed (MLflow server may not be running): {e}")
        version = run_id[:8]  # fall back to run ID prefix for local dev

    return version


def get_production_model_version(model_name: str) -> str | None:
    """Return the version string of the current Production model, or None."""
    client = get_client()
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            return versions[0].version
    except Exception as e:
        logger.warning(f"Could not fetch production model version: {e}")
    return None


def get_production_auc(model_name: str) -> float | None:
    """Return the test AUC of the current production model from MLflow metadata."""
    client = get_client()
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            return None
        run_id = versions[0].run_id
        run = client.get_run(run_id)
        return float(run.data.metrics.get("test_auc", 0.0))
    except Exception as e:
        logger.warning(f"Could not fetch production AUC: {e}")
        return None


def promote_model(
    version: str,
    new_auc: float,
    model_name: str | None = None,
    min_improvement: float = 0.005,
) -> bool:
    """Promote `version` to Production if it beats the incumbent AUC.
    
    The current production model is archived first : we never have two
    production versions at the same time. The incumbent must be beaten by
    at least `min_improvement` AUC points, not just equal, to avoid
    constantly redeploying statistically identical models.
    
    Returns True if promotion happened.
    """
    model_name = model_name or os.environ.get("CHAMPION_MODEL_NAME", "credit_risk_champion")
    client = get_client()

    incumbent_auc = get_production_auc(model_name)
    if incumbent_auc and new_auc < incumbent_auc + min_improvement:
        logger.info(
            f"New model AUC {new_auc:.4f} does not beat incumbent "
            f"{incumbent_auc:.4f} by {min_improvement:.4f} : NOT promoting"
        )
        return False

    try:
        # Archive the current production version first
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        for v in prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=v.version,
                stage="Archived",
                archive_existing_versions=False,
            )
            logger.info(f"Archived previous production version {v.version}")

        # Move new version through Staging -> Production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
        )
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
        )
        client.update_model_version(
            name=model_name,
            version=version,
            description=f"Promoted to Production. AUC={new_auc:.4f}",
        )
        logger.info(f"Promoted version {version} to Production (AUC={new_auc:.4f})")
        return True

    except Exception as e:
        logger.error(f"Promotion failed for version {version}: {e}")
        return False


def load_production_artifacts(model_name: str, local_dir: Path) -> Path | None:
    """Download the production model's artifacts to local_dir.
    
    Returns the path to the downloaded artifacts, or None if no production
    model exists yet (useful during initial setup).
    """
    client = get_client()
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            logger.warning(f"No production version found for '{model_name}'")
            return None
        run_id = versions[0].run_id
        artifact_path = client.download_artifacts(run_id, "model_artifacts", str(local_dir))
        logger.info(f"Downloaded production artifacts to {artifact_path}")
        return Path(artifact_path)
    except Exception as e:
        logger.warning(f"Could not load production artifacts: {e}")
        return None
