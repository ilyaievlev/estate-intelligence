from __future__ import annotations

import os
import time
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from loguru import logger

# Поиск и подгрузка основного .env проекта
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if (_PROJECT_ROOT / ".env").is_file():
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
else:
    load_dotenv(find_dotenv(usecwd=True), override=False)

from domain import Apartment
from sources.avito.client import AvitoClient
from storage.clickhouse import save_snapshots
from storage.postgres import upsert_apartments


def collect_avito() -> list[Apartment]:
    """Сбор актуальных объявлений с площадки Avito."""
    client = AvitoClient()
    return client.fetch()


def run_collection() -> None:
    """Выполняет один цикл сбора, сохранения в PostgreSQL и ClickHouse."""
    apartments: list[Apartment] = []

    # Сбор данных с Avito
    avito_data = collect_avito()
    apartments += avito_data

    # Сохранение актуального состояния в PostgreSQL и снимков в ClickHouse
    upserted = upsert_apartments(apartments)
    snapshots = save_snapshots(apartments)
    logger.info(
        f"Fetched: {len(apartments)}, Upserted in Postgres: {upserted}, Snapshots in ClickHouse: {snapshots}"
    )


def main() -> None:
    """Основная функция: поддержка разового вызова и циклического режима демона."""
    interval_sec = int(os.getenv("COLLECTOR_INTERVAL_SEC", "0"))
    if interval_sec > 0:
        logger.info(f"Starting collector in continuous daemon mode (interval={interval_sec}s)...")
        while True:
            try:
                run_collection()
            except Exception as err:
                logger.error(f"Collector run failed: {err}")
            time.sleep(interval_sec)
    else:
        run_collection()


if __name__ == "__main__":
    main()
