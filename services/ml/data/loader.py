from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from loguru import logger

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from config import DATABASE_URL, MOSCOW_CENTER_LAT, MOSCOW_CENTER_LON

_SQL_QUERY = """
SELECT
    a.source,
    a.external_id,
    a.monthly_rent,
    a.rooms,
    a.area,
    a.floor,
    a.floors_total,
    a.address,
    ST_Y(a.location) AS latitude,
    ST_X(a.location) AS longitude,
    COALESCE(nm.station_name, a.metro) AS metro,
    COALESCE(nm.line_name, 'unknown') AS metro_line,
    COALESCE(nm.transport_type, 'metro') AS transport_type,
    COALESCE(nm.distance_m, a.metro_distance_m) AS metro_distance_m,
    ROUND(
        ST_Distance(
            a.location::geography,
            ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
        )
    )::integer AS distance_to_center_m,
    a.seller_type,
    a.description,
    a.published_at,
    a.first_seen_at,
    a.last_seen_at
FROM apartments a
LEFT JOIN apartment_nearest_metro nm
    ON a.source = nm.source AND a.external_id = nm.external_id
WHERE a.monthly_rent IS NOT NULL
  AND a.monthly_rent > 0
  AND a.location IS NOT NULL
ORDER BY a.published_at DESC NULLS LAST, a.first_seen_at DESC
"""


def load_apartments_from_db(
    database_url: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Загружает актуальный датасет объявлений из PostgreSQL.
    Расстояние до центра и до ближайшей станции рассчитываются на стороне БД (PostGIS).

    - metro_distance_m: расстояние до ближайшей станции в метрах (из PostGIS / справочника);
    - distance_to_center_m: точное геодезическое расстояние до центра Москвы (Красная пл.) в метрах.
    """
    url = database_url or DATABASE_URL
    logger.info("Connecting to PostgreSQL to load apartments dataset...")

    query = _SQL_QUERY.strip().rstrip(";")
    params: list[Any] = [MOSCOW_CENTER_LON, MOSCOW_CENTER_LAT]
    if limit is not None and limit > 0:
        query += " LIMIT %s"
        params.append(limit)
    query += ";"

    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            if cur.description is None:
                raise RuntimeError("Failed to retrieve query columns description from database")
            cols = [desc[0] for desc in cur.description]

    df = pd.DataFrame(rows, columns=cols)
    logger.info(f"Loaded {len(df):,} apartment records from PostgreSQL")

    if df.empty:
        raise ValueError("No apartment records found in database with valid rent and location")

    return df
