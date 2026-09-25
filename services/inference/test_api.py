from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# Подавление предупреждений сторонних библиотек при тестировании
warnings.filterwarnings("ignore")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from fastapi.testclient import TestClient

# Добавление путей сервисов в системный путь sys.path
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
ML_DIR = PROJECT_ROOT / "services" / "ml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

try:
    from services.inference.main import app
except ImportError:
    from main import app  # type: ignore

client = TestClient(app)


def test_health_endpoint():
    """Проверка доступности эндпоинта /health и статуса загрузки модели."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["service"] == "estate-inference"
    assert "model_loaded" in data


def test_model_endpoint():
    """Проверка получения метаданных активной модели-чемпиона из /model."""
    response = client.get("/model")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "estate_rent_catboost"
    assert "alias" in data
    assert "status" in data


def test_predict_single_apartment():
    """Проверка инференса для одиночного объекта недвижимости."""
    payload = {
        "rooms": 2,
        "area": 55.0,
        "floor": 7,
        "floors_total": 14,
        "metro": "Белорусская",
        "metro_distance_m": 450.0,
        "metro_line": "Замоскворецкая",
        "transport_type": "walk",
        "latitude": 55.777,
        "longitude": 37.583,
        "seller_type": "realtor",
        "source": "avito",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_rent_rub" in data
    assert data["predicted_rent_rub"] > 10000
    assert data["currency"] == "RUB"
    assert data["rooms"] == 2
    assert data["area"] == 55.0
    assert data["metro"] == "Белорусская"
    assert "distance_to_center_m" in data
    assert data["distance_to_center_m"] > 0
    # Проверка наличия заголовка времени обработки
    assert "x-process-time-ms" in response.headers


def test_predict_batch():
    """Проверка пакетного инференса для нескольких квартир."""
    payload = {
        "apartments": [
            {
                "rooms": 1,
                "area": 35.0,
                "floor": 3,
                "floors_total": 9,
                "metro": "Сокол",
                "metro_distance_m": 300.0,
                "metro_line": "Замоскворецкая",
                "transport_type": "walk",
                "latitude": 55.805,
                "longitude": 37.515,
            },
            {
                "rooms": 3,
                "area": 85.0,
                "floor": 10,
                "floors_total": 22,
                "metro": "Арбатская",
                "metro_distance_m": 200.0,
                "metro_line": "Арбатско-Покровская",
                "transport_type": "walk",
                "latitude": 55.752,
                "longitude": 37.601,
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 2
    assert len(data["items"]) == 2
    assert data["took_ms"] >= 0
    # Проверка зависимости: 3-комнатная квартира на Арбате дороже 1-комнатной на Соколе
    assert data["items"][1]["predicted_rent_rub"] > data["items"][0]["predicted_rent_rub"]


def test_predict_validation_error():
    """Проверка валидации Pydantic (площадь менее 5 м² недопустима)."""
    payload = {
        "rooms": 1,
        "area": 2.0,
        "floor": 1,
        "floors_total": 5,
        "metro": "Белорусская",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_model_reload():
    """Проверка эндпоинта горячей перезагрузки модели."""
    response = client.post("/model/reload")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "estate_rent_catboost"
    assert data["status"] == "READY"


if __name__ == "__main__":
    print("Running test_health_endpoint...")
    test_health_endpoint()
    print("Running test_model_endpoint...")
    test_model_endpoint()
    print("Running test_predict_single_apartment...")
    test_predict_single_apartment()
    print("Running test_predict_batch...")
    test_predict_batch()
    print("Running test_predict_validation_error...")
    test_predict_validation_error()
    print("Running test_model_reload...")
    test_model_reload()
    print("ALL TESTS PASSED SUCCESSFULLY!")
