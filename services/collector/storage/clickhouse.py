from __future__ import annotations

import os

import clickhouse_connect
from clickhouse_connect.driver.client import Client

from domain import Apartment

_SNAPSHOT_COLUMNS = ["source", "external_id", "monthly_rent", "collected_at"]


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"{name} is not set. See .env.example")
    return value


def get_client() -> Client:
    return clickhouse_connect.get_client(
        host=_require_env("CLICKHOUSE_HOST"),
        port=int(_require_env("CLICKHOUSE_PORT")),
        database=_require_env("CLICKHOUSE_DATABASE"),
        username=_require_env("CLICKHOUSE_USER"),
        password=_require_env("CLICKHOUSE_PASSWORD"),
    )


def save_snapshots(apartments: list[Apartment]) -> int:
    """Append-only: одна строка на объявление за run, одним batch insert."""
    if not apartments:
        return 0

    rows = [
        [apt.source, str(apt.external_id), apt.monthly_rent, apt.collected_at]
        for apt in apartments
    ]
    client = get_client()
    try:
        client.insert("apartment_snapshots", rows, column_names=_SNAPSHOT_COLUMNS)
    finally:
        client.close()
    return len(rows)
