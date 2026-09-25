from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class ApartmentPredictRequest(BaseModel):
    """
    Параметры квартиры для оценки стоимости долгосрочной аренды.
    """
    rooms: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Количество комнат (0 = студия, 1..10)",
        examples=[2],
    )
    area: float = Field(
        default=45.0,
        gt=5.0,
        le=500.0,
        description="Общая площадь квартиры в м²",
        examples=[54.5],
    )
    floor: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Этаж расположения квартиры",
        examples=[5],
    )
    floors_total: int = Field(
        default=16,
        ge=1,
        le=100,
        description="Общая этажность дома",
        examples=[12],
    )
    metro: str = Field(
        default="Белорусская",
        description="Название ближайшей станции метро",
        examples=["Белорусская"],
    )
    metro_distance_m: float = Field(
        default=500.0,
        ge=0.0,
        description="Расстояние до ближайшей станции метро в метрах",
        examples=[650.0],
    )
    metro_line: str = Field(
        default="Замоскворецкая",
        description="Линия метрополитена",
        examples=["Замоскворецкая"],
    )
    transport_type: str = Field(
        default="walk",
        description="Тип передвижения до метро: walk (пешком), transport (на транспорте), metro",
        examples=["walk"],
    )
    latitude: float | None = Field(
        default=None,
        description="Географическая широта (WGS84)",
        examples=[55.777],
    )
    longitude: float | None = Field(
        default=None,
        description="Географическая долгота (WGS84)",
        examples=[37.583],
    )
    distance_to_center_m: float | None = Field(
        default=None,
        description="Расстояние до центра Москвы (Кремля) в метрах. Если переданы latitude и longitude, рассчитывается автоматически.",
        examples=[3488.6],
    )
    seller_type: str = Field(
        default="realtor",
        description="Тип арендодателя / автора: owner, realtor, agency, developer",
        examples=["realtor"],
    )
    source: str = Field(
        default="avito",
        description="Источник объявления: avito, cian",
        examples=["avito"],
    )
    description: str | None = Field(
        default=None,
        description="Текст описания квартиры",
        examples=["Светлая уютная квартира рядом с метро."],
    )


class BatchPredictRequest(BaseModel):
    """
    Пакетный запрос оценки нескольких квартир.
    """
    apartments: list[ApartmentPredictRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Список квартир для оценки",
    )


class PredictionResult(BaseModel):
    """
    Результат предикта рыночной ставки аренды квартиры.
    """
    model_config = ConfigDict(protected_namespaces=())

    predicted_rent_rub: int = Field(
        description="Рекомендованная ставка аренды (округлена до 500 ₽)",
        examples=[134500],
    )
    predicted_rent_raw: float = Field(
        description="Точное значение модели в рублях",
        examples=[134484.73],
    )
    currency: str = Field(default="RUB", description="Валюта оценки")
    rooms: int
    area: float
    metro: str
    metro_distance_m: float
    distance_to_center_m: float
    model_name: str
    model_version: str | None = None
    model_source: str = Field(
        description="Источник загруженной модели: 'mlflow_registry' или 'local_cache'",
    )


class BatchPredictionResult(BaseModel):
    """
    Результат пакетной оценки.
    """
    items: list[PredictionResult]
    total_count: int
    took_ms: float


class ModelInfoResponse(BaseModel):
    """
    Метаданные активной ML-модели.
    """
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    alias: str
    version: str | None = None
    run_id: str | None = None
    source: str
    status: str
    metrics: dict[str, float] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    loaded_at: str


class HealthResponse(BaseModel):
    """
    Статус работоспособности сервиса.
    """
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(examples=["healthy", "degraded"])
    service: str = "estate-inference"
    model_loaded: bool
    model_info: ModelInfoResponse | None = None
