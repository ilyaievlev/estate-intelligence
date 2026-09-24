from datetime import datetime
from pydantic import BaseModel


class Apartment(BaseModel):
    source: str
    external_id: int
    url: str

    monthly_rent: int

    rooms: int
    area: float
    floor: int
    floors_total: int

    address: str | None = None
    latitude: float
    longitude: float

    metro: str | None = None
    metro_distance_m: int | None = None

    seller_type: str | None = None
    description: str | None = None

    published_at: datetime | None = None
    collected_at: datetime