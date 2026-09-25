"""
Collector entrypoint: Avito → Apartment → Postgres (current state) + ClickHouse (snapshots).
"""

from __future__ import annotations

from loguru import logger

from sources.avito.client import AvitoClient
from domain import Apartment
from storage.clickhouse import save_snapshots
from storage.postgres import upsert_apartments

def collect_avito() -> list[Apartment]:
    client = AvitoClient()
    return client.fetch()

def main() -> None:
    apartments = []

    avito_data = collect_avito()
    apartments += avito_data # для расширения источников

    upserted = upsert_apartments(apartments)
    snapshots = save_snapshots(apartments)
    logger.info(
        f"fetched={len(apartments)} upserted={upserted} snapshots={snapshots}"
    )


if __name__ == "__main__":
    main()
