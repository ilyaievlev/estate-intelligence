"""Item/ML-dict из fork → наша Apartment."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from .models import Apartment

_METRO_M_RE = re.compile(r"(\d+)\s*м", re.IGNORECASE)


def _parse_metro_distance_m(after: str | None) -> int | None:
    if not after:
        return None
    normalized = after.replace("\xa0", " ").replace("\u202f", " ")
    match = _METRO_M_RE.search(normalized)
    return int(match.group(1)) if match else None


def _parse_published_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def ml_dict_to_apartment(row: dict[str, Any]) -> Apartment:
    """
    dict из apartment_ml.to_ml_dict → Apartment.

    Пропускаемые объявления (raise):
    - нет id / price / coords / rooms|area|floor
    """
    external_id = row.get("id")
    price = row.get("price")
    rooms = row.get("rooms")
    area = row.get("area_m2")
    floor = row.get("floor")
    floors_total = row.get("floors_total")
    lat = row.get("lat")
    lng = row.get("lng")
    url = row.get("url")

    if external_id is None:
        raise ValueError("missing id")
    if price is None:
        raise ValueError("missing price")
    if rooms is None or area is None or floor is None or floors_total is None:
        raise ValueError("missing rooms/area/floor from title")
    if lat is None or lng is None:
        raise ValueError("missing lat/lng")
    if not url:
        raise ValueError("missing url")

    return Apartment(
        source="avito",
        external_id=int(external_id),
        url=str(url),
        monthly_rent=int(price),
        rooms=int(rooms),
        area=float(area),
        floor=int(floor),
        floors_total=int(floors_total),
        address=row.get("address") or row.get("address_user"),
        latitude=float(lat),
        longitude=float(lng),
        metro=row.get("metro_nearest"),
        metro_distance_m=_parse_metro_distance_m(row.get("metro_nearest_after")),
        seller_type=None,
        description=row.get("description"),
        published_at=_parse_published_at(row.get("published_at")),
        collected_at=datetime.now(timezone.utc),
    )
