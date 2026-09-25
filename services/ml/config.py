from __future__ import annotations

import os
import warnings
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# Отключение информационных хинтов и предупреждений сторонних библиотек
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

ML_DIR = Path(__file__).resolve().parent
REPO_ROOT = ML_DIR.parent.parent

# Поиск и загрузка общего файла .env из корня проекта
if (REPO_ROOT / ".env").is_file():
    load_dotenv(REPO_ROOT / ".env", override=False)
else:
    load_dotenv(find_dotenv(usecwd=True), override=False)

# PostgreSQL connection string
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://estate:estate@localhost:5433/estate",
)

# Каталоги для сохранения локальных артефактов
ARTIFACTS_DIR = ML_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Координаты центра Москвы (Красная площадь / Кремль) для гео-признака
MOSCOW_CENTER_LAT = 55.753930
MOSCOW_CENTER_LON = 37.620795

# Параметры разбиения и валидации
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Целевая метрика для отбора лучшей модели (чем меньше MAE, тем лучше)
PRIMARY_METRIC = "mae"
# Минимальный порог относительного улучшения (0.1%), чтобы Challenger заменил Champion
MIN_IMPROVEMENT_RATIO = 0.001

# MLflow и MinIO (S3) настройки
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "estate_rent_prediction")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "estate_rent_catboost")
MLFLOW_CHAMPION_ALIAS = os.getenv("MLFLOW_CHAMPION_ALIAS", "champion")

MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9010")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MINIO_DEFAULT_BUCKET = os.getenv("MINIO_DEFAULT_BUCKET", "mlflow")

# Экспортируем в os.environ, чтобы MLflow и boto3 гарантированно видели креды и эндпоинты
os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", MLFLOW_S3_ENDPOINT_URL)
os.environ.setdefault("AWS_ACCESS_KEY_ID", AWS_ACCESS_KEY_ID)
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", AWS_SECRET_ACCESS_KEY)
os.environ.setdefault("AWS_DEFAULT_REGION", AWS_DEFAULT_REGION)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
