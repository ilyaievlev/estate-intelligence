from __future__ import annotations

import os
import warnings
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

BASE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BASE_DIR.parent.parent

# Поиск и загрузка общего .env из корня проекта
if (PROJECT_ROOT / ".env").is_file():
    load_dotenv(PROJECT_ROOT / ".env", override=False)
else:
    load_dotenv(find_dotenv(usecwd=True), override=False)

# Отключение информационных сообщений и предупреждений сторонних библиотек
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", message=".*protected namespace.*")
warnings.filterwarnings("ignore", message=".*StarletteDeprecationWarning.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Координаты географического центра Москвы (Кремль / Красная площадь)
MOSCOW_CENTER_LAT: float = float(os.getenv("MOSCOW_CENTER_LAT", "55.751244"))
MOSCOW_CENTER_LON: float = float(os.getenv("MOSCOW_CENTER_LON", "37.618423"))

# Параметры HTTP-сервера инференса
INFERENCE_HOST: str = os.getenv("INFERENCE_HOST", "0.0.0.0")
_raw_port: str = os.getenv("INFERENCE_SERVER_PORT") or os.getenv("INFERENCE_PORT", "8000")
if _raw_port.startswith("tcp://"):
    _raw_port = _raw_port.split(":")[-1]
INFERENCE_PORT: int = int(_raw_port)

# Параметры подключения к реестру MLflow Model Registry
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5001",
)
MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "estate_rent_catboost")
MLFLOW_CHAMPION_ALIAS: str = os.getenv("MLFLOW_CHAMPION_ALIAS", "champion")

# Параметры хранилища артефактов S3 / MinIO
MLFLOW_S3_ENDPOINT_URL: str = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9010")
MLFLOW_S3_IGNORE_TLS: str = os.getenv("MLFLOW_S3_IGNORE_TLS", "true")

# Локальные пути к резервной копии модели
DEFAULT_LOCAL_MODEL_PATH: Path = (
    PROJECT_ROOT / "services" / "ml" / "artifacts" / "models" / "catboost_latest.cbm"
)
