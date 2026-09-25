from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import httpx

# Добавление путей сервисов в системный путь sys.path
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from services.inference.schemas import (
        ApartmentPredictRequest,
        BatchPredictionResult,
        BatchPredictRequest,
        HealthResponse,
        ModelInfoResponse,
        PredictionResult,
    )
except ImportError:
    from schemas import (  # type: ignore
        ApartmentPredictRequest,
        BatchPredictionResult,
        BatchPredictRequest,
        HealthResponse,
        ModelInfoResponse,
        PredictionResult,
    )


class EstateInferenceClient:
    """
    Клиент для взаимодействия с микросервисом Estate Intelligence Inference API по HTTP.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> HealthResponse:
        """Проверяет состояние сервиса."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return HealthResponse(**resp.json())

    def get_model_info(self) -> ModelInfoResponse:
        """Получает метаданные активной модели."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/model")
            resp.raise_for_status()
            return ModelInfoResponse(**resp.json())

    def predict(self, apartment: ApartmentPredictRequest | dict[str, Any]) -> PredictionResult:
        """
        Отправляет параметры одной квартиры на оценку стоимости аренды.
        """
        payload: dict[str, Any]
        if isinstance(apartment, ApartmentPredictRequest):
            payload = apartment.model_dump()
        else:
            payload = dict(apartment)

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/predict", json=payload)
            resp.raise_for_status()
            return PredictionResult(**resp.json())

    def predict_batch(
        self, apartments: list[ApartmentPredictRequest] | list[dict[str, Any]]
    ) -> BatchPredictionResult:
        """
        Отправляет пакет квартир на массовую оценку стоимости аренды.
        """
        items: list[dict[str, Any]] = []
        for item in apartments:
            if isinstance(item, ApartmentPredictRequest):
                items.append(item.model_dump())
            else:
                items.append(dict(item))

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/predict/batch",
                json={"apartments": items},
            )
            resp.raise_for_status()
            return BatchPredictionResult(**resp.json())

    def reload_model(self) -> ModelInfoResponse:
        """Запрашивает горячую перезагрузку модели-чемпиона из MLflow."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/model/reload")
            resp.raise_for_status()
            return ModelInfoResponse(**resp.json())
