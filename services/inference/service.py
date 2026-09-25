from __future__ import annotations

import math
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.catboost as mlflow_catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from loguru import logger
from mlflow.tracking import MlflowClient

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
    from services.inference.config import (
        DEFAULT_LOCAL_MODEL_PATH,
        MLFLOW_CHAMPION_ALIAS,
        MLFLOW_MODEL_NAME,
        MLFLOW_TRACKING_URI,
        MOSCOW_CENTER_LAT,
        MOSCOW_CENTER_LON,
    )
    from services.inference.schemas import (
        ApartmentPredictRequest,
        ModelInfoResponse,
        PredictionResult,
    )
except ImportError:
    from .config import (  # type: ignore
        DEFAULT_LOCAL_MODEL_PATH,
        MLFLOW_CHAMPION_ALIAS,
        MLFLOW_MODEL_NAME,
        MLFLOW_TRACKING_URI,
        MOSCOW_CENTER_LAT,
        MOSCOW_CENTER_LON,
    )
    from .schemas import (  # type: ignore
        ApartmentPredictRequest,
        ModelInfoResponse,
        PredictionResult,
    )

try:
    from features.engineering import prepare_features
except ImportError:
    from services.ml.features.engineering import prepare_features  # type: ignore


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Вычисляет ортодромическое расстояние в метрах между двумя гео-координатами
    по формуле гаверсинуса (WGS-84 сфера).
    """
    earth_radius = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * earth_radius * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


class InferenceService:
    """
    Потокобезопасная прослойка инференса для CatBoost модели.
    Управляет загрузкой, кэшированием в RAM, обращением к реестру MLflow (@champion),
    fallback-загрузкой с диска и выполнением предсказаний.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: CatBoostRegressor | None = None
        self._model_version: str | None = None
        self._model_run_id: str | None = None
        self._model_source: str = "none"
        self._loaded_at: str = ""
        self._metrics: dict[str, float] = {}
        self._tags: dict[str, str] = {}

    def get_model(self, force_reload: bool = False) -> CatBoostRegressor:
        """
        Возвращает загруженную модель или загружает её при первом вызове.
        """
        if self._model is None or force_reload:
            with self._lock:
                if self._model is None or force_reload:
                    self._load_model()
        if self._model is None:
            raise RuntimeError("ML model could not be loaded into memory.")
        return self._model

    def _load_model(self) -> None:
        """
        Загружает модель из MLflow Model Registry по алиасу @champion.
        При ошибке или недоступности сети используется резервная копия .cbm с диска.
        """
        model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_CHAMPION_ALIAS}"
        logger.info(f"Connecting to MLflow Model Registry at '{MLFLOW_TRACKING_URI}'...")

        # 1. Попытка загрузки модели из MLflow Model Registry
        try:
            import urllib.request
            try:
                urllib.request.urlopen(f"{MLFLOW_TRACKING_URI}/health", timeout=2)
            except Exception as conn_err:
                raise ConnectionError(f"MLflow server not reachable: {conn_err}")

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

            logger.info(f"Fetching champion model version for '{model_uri}'...")
            mv = client.get_model_version_by_alias(
                name=MLFLOW_MODEL_NAME,
                alias=MLFLOW_CHAMPION_ALIAS,
            )

            loaded_model: CatBoostRegressor = mlflow_catboost.load_model(model_uri)
            self._model = loaded_model
            self._model_version = str(mv.version)
            self._model_run_id = str(mv.run_id) if mv.run_id else None
            self._model_source = "mlflow_registry"
            self._loaded_at = datetime.now(timezone.utc).isoformat()

            # Сбор метрик и тегов из зарегистрированного запуска MLflow
            if mv.run_id:
                try:
                    run = client.get_run(mv.run_id)
                    self._metrics = {str(k): float(v) for k, v in run.data.metrics.items()}
                    self._tags = {str(k): str(v) for k, v in (mv.tags or {}).items()}
                except Exception as e:
                    logger.warning(f"Could not load run metadata for run_id={mv.run_id}: {e}")
                    self._metrics = {}
                    self._tags = {}
            else:
                self._metrics = {}
                self._tags = {}

            logger.success(
                f"Successfully loaded model '{MLFLOW_MODEL_NAME}' v{self._model_version} from MLflow Registry!"
            )
            return

        except Exception as exc:
            logger.warning(
                f"Could not load model from MLflow ({model_uri}): {exc}. "
                f"Falling back to local cache at '{DEFAULT_LOCAL_MODEL_PATH}'..."
            )

        # 2. Резервная загрузка из локального файла артефактов
        if DEFAULT_LOCAL_MODEL_PATH.is_file():
            fallback_model = CatBoostRegressor()
            fallback_model.load_model(str(DEFAULT_LOCAL_MODEL_PATH))
            self._model = fallback_model
            self._model_version = "local_latest"
            self._model_run_id = "local_file"
            self._model_source = "local_cache"
            self._loaded_at = datetime.now(timezone.utc).isoformat()
            self._metrics = {}
            self._tags = {"fallback": "true", "path": str(DEFAULT_LOCAL_MODEL_PATH)}
            logger.success(f"Successfully loaded model from local backup: {DEFAULT_LOCAL_MODEL_PATH}")
            return

        raise RuntimeError(
            f"Failed to load model from both MLflow Registry ({model_uri}) and local backup ({DEFAULT_LOCAL_MODEL_PATH})."
        )

    def reload(self) -> ModelInfoResponse:
        """
        Принудительно перезагружает активную модель из реестра MLflow.
        """
        logger.info("Executing hot reload of production model...")
        self.get_model(force_reload=True)
        return self.get_model_info()

    def get_model_info(self) -> ModelInfoResponse:
        """
        Возвращает метаданные о текущей активной модели в памяти.
        """
        # Прогрев модели при первом обращении
        if self._model is None:
            try:
                self.get_model()
            except Exception as e:
                return ModelInfoResponse(
                    model_name=MLFLOW_MODEL_NAME,
                    alias=MLFLOW_CHAMPION_ALIAS,
                    version=None,
                    run_id=None,
                    source="unavailable",
                    status=f"error: {e}",
                    metrics={},
                    tags={},
                    loaded_at=datetime.now(timezone.utc).isoformat(),
                )

        return ModelInfoResponse(
            model_name=MLFLOW_MODEL_NAME,
            alias=MLFLOW_CHAMPION_ALIAS,
            version=self._model_version,
            run_id=self._model_run_id,
            source=self._model_source,
            status="READY",
            metrics=self._metrics,
            tags=self._tags,
            loaded_at=self._loaded_at,
        )

    def predict(
        self,
        requests: list[ApartmentPredictRequest] | ApartmentPredictRequest,
    ) -> list[PredictionResult]:
        """
        Принимает параметры квартир, формирует признаки и возвращает результат прогноза.
        """
        items: list[ApartmentPredictRequest] = (
            [requests] if isinstance(requests, ApartmentPredictRequest) else list(requests)
        )

        if not items:
            return []

        # Преобразование входных данных в pandas DataFrame
        records = [item.model_dump() for item in items]
        df = pd.DataFrame(records)

        # Расчет расстояния до центра Москвы по формуле гаверсинуса (при наличии координат)
        has_dist_col = "distance_to_center_m" in df.columns
        needs_calc = (not has_dist_col) or bool(df["distance_to_center_m"].isna().any())

        if needs_calc and "latitude" in df.columns and "longitude" in df.columns:
            computed_dist: list[float] = []
            for _, row in df.iterrows():
                lat_val = row.get("latitude")
                lon_val = row.get("longitude")
                if (
                    lat_val is not None
                    and lon_val is not None
                    and not pd.isna(lat_val)
                    and not pd.isna(lon_val)
                ):
                    dist = haversine_distance_m(
                        float(lat_val), float(lon_val), MOSCOW_CENTER_LAT, MOSCOW_CENTER_LON
                    )
                    computed_dist.append(round(dist, 1))
                else:
                    computed_dist.append(float("nan"))

            if not has_dist_col:
                df["distance_to_center_m"] = computed_dist
            else:
                dist_series = pd.Series(computed_dist, index=df.index, dtype=float)
                df["distance_to_center_m"] = df["distance_to_center_m"].fillna(dist_series)

        # Подготовка признаков (Feature Engineering)
        X, _ = prepare_features(df, is_training=False)

        # Инференс обученной модели CatBoost
        model = self.get_model()
        raw_predictions = model.predict(X)

        results: list[PredictionResult] = []
        for idx, pred_val in enumerate(raw_predictions):
            raw_price = float(pred_val)
            # Округление цены до ближайших 500 рублей для реалистичности ставки
            rounded_price = int(round(raw_price / 500.0) * 500)
            row_features = X.iloc[idx]

            results.append(
                PredictionResult(
                    predicted_rent_rub=rounded_price,
                    predicted_rent_raw=round(raw_price, 2),
                    currency="RUB",
                    rooms=int(row_features["rooms"]),
                    area=float(row_features["area"]),
                    metro=str(row_features["metro"]),
                    metro_distance_m=float(row_features["metro_distance_m"]),
                    distance_to_center_m=float(row_features["distance_to_center_m"]),
                    model_name=MLFLOW_MODEL_NAME,
                    model_version=self._model_version,
                    model_source=self._model_source,
                )
            )

        return results


# Глобальный экземпляр сервиса для использования в эндпоинтах FastAPI
inference_service = InferenceService()
