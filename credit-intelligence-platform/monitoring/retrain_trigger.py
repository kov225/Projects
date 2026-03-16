"""
retrain_trigger.py

Scheduled job that runs every RETRAIN_CHECK_INTERVAL_HOURS hours, polls drift
metrics from PostgreSQL, and kicks off a retraining job if thresholds are
breached. After retraining, the new model is validated against the incumbent
and promoted to production only if it improves AUC by at least MIN_AUC_IMPROVEMENT.

We use APScheduler for the scheduler because it's lighter than Celery for a
single-machine setup and doesn't need a separate broker. In a real deployment
this would be an Airflow DAG or a Kubernetes CronJob.
"""

import importlib
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

INTERVAL_HOURS = float(os.environ.get("RETRAIN_CHECK_INTERVAL_HOURS", "6"))
PSI_THRESHOLD = float(os.environ.get("PSI_THRESHOLD", "0.2"))
AUC_DROP_THRESHOLD = float(os.environ.get("AUC_DROP_THRESHOLD", "0.03"))
MIN_AUC_IMPROVEMENT = float(os.environ.get("MIN_AUC_IMPROVEMENT", "0.005"))
DATA_PATH = os.environ.get("TRAINING_DATA_PATH", "data/processed/loans_clean.parquet")


def should_retrain() -> tuple[bool, str]:
    """Check drift metrics and return (should_retrain, reason).
    
    We check both covariate drift (PSI) and concept drift (AUC drop).
    Either one is sufficient to trigger retraining.
    """
    try:
        import psycopg2
        conn_args = dict(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            dbname=os.environ.get("POSTGRES_DB", "credit_intelligence"),
            user=os.environ.get("POSTGRES_USER", "credit_app"),
            password=os.environ.get("POSTGRES_PASSWORD", "changeme"),
        )
        conn = psycopg2.connect(**conn_args)
        try:
            with conn.cursor() as cur:
                # Check most recent drift report for covariate drift
                cur.execute("""
                    SELECT any_drifted, drifted_features
                    FROM drift_reports
                    ORDER BY checked_at DESC
                    LIMIT 1
                """)
                row = cur.fetchone()
                if row and row[0]:
                    features = row[1]
                    return True, f"Covariate drift in features: {features}"

                # Check most recent concept drift report
                cur.execute("""
                    SELECT drift_detected, auc_drop, model_version
                    FROM concept_drift_reports
                    ORDER BY checked_at DESC
                    LIMIT 1
                """)
                row = cur.fetchone()
                if row and row[0]:
                    return True, f"Concept drift: AUC drop={row[1]:.4f} for model {row[2]}"

        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error checking drift metrics: {e}")

    return False, "No drift detected"


def run_retraining() -> float | None:
    """Invoke the training script as a subprocess and return the new model's test AUC.
    
    Running as a subprocess rather than calling train.run_training() directly
    ensures a clean Python environment for each run and makes the resource usage
    visible in the process list - useful for monitoring in production.
    """
    logger.info("Starting retraining job...")
    start = datetime.now(timezone.utc)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "models.train", "--data", DATA_PATH],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute hard limit
        )
        if result.returncode != 0:
            logger.error(f"Retraining failed:\n{result.stderr}")
            return None

        # Parse the AUC from stdout: train.py logs "Test AUC: X.XXXX"
        for line in result.stdout.splitlines():
            if "Test AUC:" in line:
                try:
                    auc = float(line.split("Test AUC:")[-1].strip().split()[0])
                    elapsed = (datetime.now(timezone.utc) - start).total_seconds() / 60
                    logger.info(f"Retraining complete in {elapsed:.1f}m. New AUC: {auc:.4f}")
                    return auc
                except ValueError:
                    pass

        logger.warning("Could not parse AUC from retraining output")
        return None

    except subprocess.TimeoutExpired:
        logger.error("Retraining timed out after 10 minutes")
        return None


def promotion_check_and_promote(new_auc: float) -> bool:
    """Fetch the production model's AUC and promote if new model is better."""
    from models.mlflow_registry import get_production_auc, promote_model
    import os

    model_name = os.environ.get("CHAMPION_MODEL_NAME", "credit_risk_champion")
    prod_auc = get_production_auc(model_name)

    if prod_auc and new_auc < prod_auc + MIN_AUC_IMPROVEMENT:
        logger.info(
            f"New model AUC {new_auc:.4f} does not improve over production "
            f"{prod_auc:.4f} by {MIN_AUC_IMPROVEMENT} : not promoting"
        )
        return False

    # Get the latest staged version to promote
    from mlflow.tracking import MlflowClient
    client = MlflowClient(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    staged = client.get_latest_versions(model_name, stages=["Staging", "None"])
    if not staged:
        logger.warning("No staged model version found for promotion")
        return False

    latest_version = sorted(staged, key=lambda v: int(v.version))[-1].version
    promoted = promote_model(
        version=latest_version,
        new_auc=new_auc,
        model_name=model_name,
        min_improvement=MIN_AUC_IMPROVEMENT,
    )
    return promoted


def check_and_retrain() -> None:
    """Main entry point for the scheduled job."""
    logger.info(f"Running drift check at {datetime.now(timezone.utc).isoformat()}")

    should, reason = should_retrain()
    if not should:
        logger.info(f"No retraining needed: {reason}")
        return

    logger.warning(f"Drift detected : triggering retraining: {reason}")
    new_auc = run_retraining()

    if new_auc is None:
        logger.error("Retraining produced no usable model")
        return

    promoted = promotion_check_and_promote(new_auc)
    if promoted:
        logger.info(f"New model promoted to production with AUC={new_auc:.4f}")
    else:
        logger.info("Retraining complete but new model not promoted (insufficient improvement)")


def main() -> None:
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        check_and_retrain,
        "interval",
        hours=INTERVAL_HOURS,
        id="retrain_check",
        max_instances=1,  # prevent overlapping runs
        coalesce=True,
    )
    logger.info(f"Retraining scheduler started. Interval: {INTERVAL_HOURS}h")
    # Run immediately at startup in addition to the scheduled interval
    check_and_retrain()
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
