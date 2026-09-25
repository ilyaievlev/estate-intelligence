from __future__ import annotations

import os
import time
from loguru import logger

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
