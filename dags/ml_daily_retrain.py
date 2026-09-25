from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator

# Базовые параметры выполнения DAG
DEFAULT_ARGS = {
    "owner": "estate-intelligence",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


@dag(
    dag_id="ml_daily_retrain",
    default_args=DEFAULT_ARGS,
    description="Ежесуточное автоматическое переобучение CatBoost со сравнением Challenger vs Champion в MLflow",
    schedule="0 3 * * *",  # Каждый день в 03:00 по московскому времени
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "catboost", "retrain", "daily"],
)
def ml_daily_retrain_pipeline():

    @task
    def check_data_readiness() -> dict[str, int | str]:
        """
        Проверяет доступность сервера MLflow и наличие достаточного объема данных в PostgreSQL.
        """
        import psycopg
        import urllib.request

        # 1. Проверка доступности MLflow Tracking Server
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        try:
            req = urllib.request.urlopen(f"{mlflow_uri}/health", timeout=10)
            status_code = req.getcode()
            print(f"MLflow health check passed: {status_code}")
        except Exception as exc:
            raise RuntimeError(f"MLflow Tracking Server is unreachable at {mlflow_uri}: {exc}")

        # 2. Проверка объема обучающих данных в PostgreSQL
        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://estate:estate@postgres:5432/estate",
        )
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT count(*) FROM apartments
                    WHERE monthly_rent IS NOT NULL
                      AND monthly_rent > 0
                      AND location IS NOT NULL;
                    """
                )
                row = cur.fetchone()
                valid_count = int(row[0]) if row else 0

        print(f"Found {valid_count:,} valid training records in PostgreSQL.")
        if valid_count < 500:
            raise ValueError(
                f"Insufficient training samples in database: {valid_count} < 500. Retraining skipped."
            )

        return {"valid_samples": valid_count, "checked_at": datetime.now().isoformat()}

    # Запуск основного скрипта переобучения и валидации
    run_retraining = BashOperator(
        task_id="run_challenger_retrain",
        bash_command=(
            "set -e\n"
            "cd /opt/estate/services/ml\n"
            "/opt/airflow/venvs/ml/bin/python retrain.py --iterations 2500\n"
        ),
    )

    @task.external_python(python="/opt/airflow/venvs/ml/bin/python")
    def report_retrain_outcome():
        """
        Запрашивает актуальный статус модели-чемпиона из MLflow Model Registry и выводит отчет.
        """
        import os
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        client = MlflowClient(tracking_uri=mlflow_uri)

        model_name = os.getenv("MLFLOW_MODEL_NAME", "estate_rent_catboost")
        alias = os.getenv("MLFLOW_CHAMPION_ALIAS", "champion")

        print("==================================================")
        print(" 🏆 ML MODEL DEPLOYMENT STATUS")
        print("==================================================")
        try:
            champion_mv = client.get_model_version_by_alias(name=model_name, alias=alias)
            run_id = champion_mv.run_id or ""
            metrics: dict[str, float] = {}
            if run_id:
                run = client.get_run(run_id)
                metrics = {str(k): float(v) for k, v in run.data.metrics.items()}

            print(f"  Model Name:       {model_name}")
            print(f"  Champion Version: v{champion_mv.version}")
            print(f"  Run ID:           {champion_mv.run_id}")
            print(f"  Aliases:          {champion_mv.aliases}")
            print(f"  Last Promotion:   {champion_mv.tags.get('promoted_at', 'N/A')}")
            print(f"  Reason:           {champion_mv.tags.get('promotion_reason', 'N/A')}")
            print("  Champion Test Metrics:")
            for m_key, m_val in metrics.items():
                print(f"    - {m_key}: {m_val}")
        except Exception as exc:
            print(f"  Could not fetch champion info: {exc}")
        print("==================================================")

    @task
    def notify_inference_reload():
        """
        Отправляет запрос на горячую перезагрузку модели-чемпиона в микросервис инференса.
        """
        import json
        import urllib.request

        inference_url = os.getenv("INFERENCE_URL", "http://inference:8000")
        try:
            req = urllib.request.Request(
                f"{inference_url}/model/reload",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                print(f"Inference service reloaded successfully: version v{data.get('version')}")
        except Exception as exc:
            print(f"Warning: Could not notify inference service at {inference_url}: {exc}")

    # Определение пайплайна задач DAG
    data_status = check_data_readiness()
    reload_task = notify_inference_reload()
    _ = data_status >> run_retraining >> report_retrain_outcome() >> reload_task


ml_daily_retrain_pipeline()
