from __future__ import annotations

import os
from functools import lru_cache

import psycopg

from domain import Apartment

# first_seen_at не входит в SET: при повторной встрече остаётся первоначальным.
_UPSERT_SQL = """
INSERT INTO apartments (
    source,
    external_id,
    url,
    monthly_rent,
    rooms,
    area,
    floor,
    floors_total,
    address,
    location,
    metro,
    metro_distance_m,
    seller_type,
    description,
    published_at,
    first_seen_at,
    last_seen_at
) VALUES (
    %(source)s,
    %(external_id)s,
    %(url)s,
    %(monthly_rent)s,
    %(rooms)s,
    %(area)s,
    %(floor)s,
    %(floors_total)s,
    %(address)s,
    ST_SetSRID(ST_MakePoint(%(longitude)s, %(latitude)s), 4326),
    %(metro)s,
    %(metro_distance_m)s,
    %(seller_type)s,
    %(description)s,
    %(published_at)s,
    %(collected_at)s,
    %(collected_at)s
)
ON CONFLICT (source, external_id) DO UPDATE SET
    url              = EXCLUDED.url,
    monthly_rent     = EXCLUDED.monthly_rent,
    rooms            = EXCLUDED.rooms,
    area             = EXCLUDED.area,
    floor            = EXCLUDED.floor,
    floors_total     = EXCLUDED.floors_total,
    address          = EXCLUDED.address,
    location         = EXCLUDED.location,
    metro            = EXCLUDED.metro,
    metro_distance_m = EXCLUDED.metro_distance_m,
    seller_type      = EXCLUDED.seller_type,
    description      = EXCLUDED.description,
    published_at     = EXCLUDED.published_at,
    last_seen_at     = EXCLUDED.last_seen_at
"""


@lru_cache(maxsize=1)
def get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. "
            "Example: postgresql://estate:estate@localhost:5432/estate"
        )
    return url


def connect() -> psycopg.Connection:
    return psycopg.connect(get_database_url())


def upsert_apartments(apartments: list[Apartment]) -> int:
    """Текущее состояние объявлений. Возвращает число обработанных строк."""
    if not apartments:
        return 0

    rows = [
        apt.model_dump() | {"external_id": str(apt.external_id)}
        for apt in apartments
    ]
    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(_UPSERT_SQL, rows)
        conn.commit()
    return len(rows)
