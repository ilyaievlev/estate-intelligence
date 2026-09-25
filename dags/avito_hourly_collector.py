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
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(minutes=45),
}


@dag(
    dag_id="avito_hourly_collector",
    default_args=DEFAULT_ARGS,
    description="Почасовой сбор объявлений аренды квартир в Москве с Avito в PostgreSQL и ClickHouse",
    schedule="0 * * * *",  # Каждый час в 00 минут
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["collector", "avito", "hourly"],
)
def avito_hourly_collector_pipeline():

    @task
    def check_db_readiness() -> dict[str, int]:
        """
        Проверяет доступность PostgreSQL и ClickHouse перед запуском сборщика.
        Возвращает текущее количество записей в таблицах.
        """
        import psycopg
        import clickhouse_connect

        # 1. Проверка доступности PostgreSQL
        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://estate:estate@postgres:5432/estate",
        )
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM apartments;")
                row = cur.fetchone()
                pg_count = int(row[0]) if row else 0

        # 2. Проверка доступности ClickHouse
        ch_client = clickhouse_connect.get_client(
            host=os.getenv("CLICKHOUSE_HOST", "clickhouse"),
            port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
            database=os.getenv("CLICKHOUSE_DATABASE", "estate"),
            username=os.getenv("CLICKHOUSE_USER", "estate"),
            password=os.getenv("CLICKHOUSE_PASSWORD", "estate"),
        )
        ch_raw = ch_client.command("SELECT count() FROM apartment_snapshots")
        ch_count = int(ch_raw) if isinstance(ch_raw, (int, str, float)) else 0
        ch_client.close()

        print(f"Pre-flight DB check passed. Postgres: {pg_count:,} items, ClickHouse: {ch_count:,} snapshots.")
        return {"pg_count_before": pg_count, "ch_count_before": ch_count}

    # Запуск основного скрипта сбора через изолированное виртуальное окружение
    run_collector = BashOperator(
        task_id="collect_avito_ads",
        bash_command=(
            "set -e\n"
            "cd /opt/estate/services/collector\n"
            "/opt/airflow/venvs/collector/bin/python main.py\n"
        ),
    )

    @task
    def log_collection_summary(pre_counts: dict[str, int]):
        """
        Сравнивает количество записей до и после сбора и выводит сводный отчет.
        """
        import psycopg
        import clickhouse_connect

        # Получение актуального количества записей в PostgreSQL
        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://estate:estate@postgres:5432/estate",
        )
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM apartments;")
                row = cur.fetchone()
                pg_count_after = int(row[0]) if row else 0

        # Получение актуального количества слепков в ClickHouse
        ch_client = clickhouse_connect.get_client(
            host=os.getenv("CLICKHOUSE_HOST", "clickhouse"),
            port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
            database=os.getenv("CLICKHOUSE_DATABASE", "estate"),
            username=os.getenv("CLICKHOUSE_USER", "estate"),
            password=os.getenv("CLICKHOUSE_PASSWORD", "estate"),
        )
        ch_raw = ch_client.command("SELECT count() FROM apartment_snapshots")
        ch_count_after = int(ch_raw) if isinstance(ch_raw, (int, str, float)) else 0
        ch_client.close()

        new_snapshots = int(ch_count_after) - int(pre_counts.get("ch_count_before", 0))
        total_apartments = pg_count_after

        print("==================================================")
        print(" 📊 HOURLY COLLECTION SUMMARY")
        print("==================================================")
        print(f"  New snapshots appended: +{new_snapshots:,}")
        print(f"  Total unique listings in Postgres: {total_apartments:,}")
        print(f"  Total history snapshots in ClickHouse: {ch_count_after:,}")
        print("==================================================")

    # Определение пайплайна задач DAG
    pre_counts = check_db_readiness()
    _ = pre_counts >> run_collector >> log_collection_summary(pre_counts)


avito_hourly_collector_pipeline()
