from __future__ import annotations

import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# Ищем .env в текущей папке или выше по иерархии (в корне проекта)
load_dotenv(find_dotenv(usecwd=True))

ML_DIR = Path(__file__).resolve().parent
REPO_ROOT = ML_DIR.parent.parent

# PostgreSQL connection string
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://estate:estate@localhost:5433/estate",
)

# Каталоги для сохранения артефактов моделей
ARTIFACTS_DIR = ML_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

# Координаты центра Москвы (Красная площадь / Кремль) для гео-признака
MOSCOW_CENTER_LAT = 55.753930
MOSCOW_CENTER_LON = 37.620795

# Параметры разбиения и валидации
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Целевая метрика для отбора лучшей модели (чем меньше MAE, тем лучше)
PRIMARY_METRIC = "mae"
# Порог улучшения (0.1%), чтобы новая модель признавалась лучше текущей
MIN_IMPROVEMENT_RATIO = 0.001

# MLflow и MinIO (S3) настройки
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "estate_rent_prediction")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9010")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
MINIO_DEFAULT_BUCKET = os.getenv("MINIO_DEFAULT_BUCKET", "mlflow")
