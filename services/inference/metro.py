from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

EARTH_RADIUS_M = 6371000.0

_STATIONS_SQL = """
SELECT name, line_name, transport_type, ST_Y(location) AS lat, ST_X(location) AS lon
FROM metro_stations
"""


@dataclass(frozen=True)
class NearestStation:
    name: str
    line_name: str
    transport_type: str
    distance_m: float


class MetroIndex:
    """
    Справочник станций метро/МЦК/МЦД в памяти для поиска ближайшей станции.
    Повторяет логику VIEW apartment_nearest_metro, по которому строится
    обучающая выборка, чтобы признаки в inference совпадали с обучением.
    """

    def __init__(self, database_url: str, retry_after_sec: float = 60.0) -> None:
        self._database_url = database_url
        self._retry_after_sec = retry_after_sec
        self._lock = threading.Lock()
        self._names: list[str] = []
        self._lines: list[str] = []
        self._types: list[str] = []
        self._lat_rad: np.ndarray | None = None
        self._lon_rad: np.ndarray | None = None
        self._last_attempt = 0.0

    @property
    def loaded(self) -> bool:
        return self._lat_rad is not None

    def _ensure_loaded(self) -> bool:
        if self.loaded:
            return True
        with self._lock:
            if self.loaded:
                return True
            if time.monotonic() - self._last_attempt < self._retry_after_sec and self._last_attempt:
                return False
            self._last_attempt = time.monotonic()
            try:
                import psycopg

                with psycopg.connect(self._database_url, connect_timeout=3) as conn:
                    rows = conn.execute(_STATIONS_SQL).fetchall()
            except Exception as exc:
                logger.warning(f"Could not load metro stations from PostgreSQL: {exc}")
                return False
            if not rows:
                logger.warning("metro_stations table is empty")
                return False

            self._names = [str(r[0]) for r in rows]
            self._lines = [str(r[1]) for r in rows]
            self._types = [str(r[2]) for r in rows]
            self._lat_rad = np.radians(np.array([float(r[3]) for r in rows]))
            self._lon_rad = np.radians(np.array([float(r[4]) for r in rows]))
            logger.info(f"Loaded {len(rows)} metro stations for nearest-station lookup")
            return True

    def nearest(self, lat: float, lon: float) -> NearestStation | None:
        if not self._ensure_loaded():
            return None
        assert self._lat_rad is not None and self._lon_rad is not None

        phi = np.radians(lat)
        lam = np.radians(lon)
        a = (
            np.sin((self._lat_rad - phi) / 2.0) ** 2
            + np.cos(phi) * np.cos(self._lat_rad) * np.sin((self._lon_rad - lam) / 2.0) ** 2
        )
        dist = 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
        idx = int(np.argmin(dist))
        return NearestStation(
            name=self._names[idx],
            line_name=self._lines[idx],
            transport_type=self._types[idx],
            distance_m=float(round(dist[idx])),
        )
