from __future__ import annotations

import math
import sys
import threading
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.catboost as mlflow_catboost
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
        DATABASE_URL,
        DEFAULT_LOCAL_MODEL_PATH,
        MLFLOW_CHAMPION_ALIAS,
        MLFLOW_MODEL_NAME,
        MLFLOW_TRACKING_URI,
        MODEL_POLL_INTERVAL_SEC,
        MOSCOW_CENTER_LAT,
        MOSCOW_CENTER_LON,
    )
    from services.inference.metro import MetroIndex
    from services.inference.schemas import (
        ApartmentPredictRequest,
        ModelInfoResponse,
        PredictionResult,
    )
except ImportError:
    from .config import (  # type: ignore
        DATABASE_URL,
        DEFAULT_LOCAL_MODEL_PATH,
        MLFLOW_CHAMPION_ALIAS,
        MLFLOW_MODEL_NAME,
        MLFLOW_TRACKING_URI,
        MODEL_POLL_INTERVAL_SEC,
        MOSCOW_CENTER_LAT,
        MOSCOW_CENTER_LON,
    )
    from .metro import MetroIndex  # type: ignore
    from .schemas import (  # type: ignore
        ApartmentPredictRequest,
        ModelInfoResponse,
        PredictionResult,
    )

try:
    from features.engineering import align_features_to_model, prepare_features
except ImportError:
    from services.ml.features.engineering import (  # type: ignore
        align_features_to_model,
        prepare_features,
    )


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


@dataclass(frozen=True)
class _LoadedModel:
    model: CatBoostRegressor
    version: str | None
    run_id: str | None
    source: str
    loaded_at: str
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


class InferenceService:
    """
    Потокобезопасная прослойка инференса для CatBoost модели.
    Управляет загрузкой, кэшированием в RAM, обращением к реестру MLflow (@champion),
    fallback-загрузкой с диска и выполнением предсказаний.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: _LoadedModel | None = None
        self._metro = MetroIndex(DATABASE_URL)
        self._watcher: threading.Thread | None = None
        self._stop = threading.Event()
        self.on_model_change: Callable[[ModelInfoResponse], None] | None = None

    @property
    def model_loaded(self) -> bool:
        return self._state is not None

    @property
    def metro_index_loaded(self) -> bool:
        return self._metro.loaded

    def get_model(self, force_reload: bool = False) -> CatBoostRegressor:
        """
        Возвращает загруженную модель или загружает её при первом вызове.
        """
        return self._get_state(force_reload).model

    def _get_state(self, force_reload: bool = False) -> _LoadedModel:
        if self._state is None or force_reload:
            with self._lock:
                if self._state is None or force_reload:
                    # Если модель уже есть, при сбое реестра оставляем её, а не
                    # подменяем локальным .cbm.
                    self._state = self._load_model(allow_fallback=self._state is None)
                    self._notify_model_change()
        return self._state

    def _load_model(self, allow_fallback: bool) -> _LoadedModel:
        """
        Загружает модель из MLflow Model Registry по алиасу @champion.
        При ошибке или недоступности сети используется резервная копия .cbm с диска.
        """
        model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_CHAMPION_ALIAS}"
        logger.info(f"Connecting to MLflow Model Registry at '{MLFLOW_TRACKING_URI}'...")

        # 1. Попытка загрузки модели из MLflow Model Registry
        try:
            client = self._mlflow_client()
            logger.info(f"Fetching champion model version for '{model_uri}'...")
            mv = client.get_model_version_by_alias(
                name=MLFLOW_MODEL_NAME,
                alias=MLFLOW_CHAMPION_ALIAS,
            )
            loaded_model: CatBoostRegressor = mlflow_catboost.load_model(f"models:/{MLFLOW_MODEL_NAME}/{mv.version}")

            # Сбор метрик и тегов из зарегистрированного запуска MLflow
            metrics: dict[str, float] = {}
            tags: dict[str, str] = {}
            if mv.run_id:
                try:
                    run = client.get_run(mv.run_id)
                    metrics = {str(k): float(v) for k, v in run.data.metrics.items()}
                    tags = {str(k): str(v) for k, v in (mv.tags or {}).items()}
                except Exception as e:
                    logger.warning(f"Could not load run metadata for run_id={mv.run_id}: {e}")

            logger.success(
                f"Successfully loaded model '{MLFLOW_MODEL_NAME}' v{mv.version} from MLflow Registry!"
            )
            return _LoadedModel(
                model=loaded_model,
                version=str(mv.version),
                run_id=str(mv.run_id) if mv.run_id else None,
                source="mlflow_registry",
                loaded_at=datetime.now(timezone.utc).isoformat(),
                metrics=metrics,
                tags=tags,
            )
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(f"Could not load model from MLflow ({model_uri}): {exc}") from exc
            logger.warning(
                f"Could not load model from MLflow ({model_uri}): {exc}. "
                f"Falling back to local cache at '{DEFAULT_LOCAL_MODEL_PATH}'..."
            )

        # 2. Резервная загрузка из локального файла артефактов
        if DEFAULT_LOCAL_MODEL_PATH.is_file():
            fallback_model = CatBoostRegressor()
            fallback_model.load_model(str(DEFAULT_LOCAL_MODEL_PATH))
            logger.success(f"Successfully loaded model from local backup: {DEFAULT_LOCAL_MODEL_PATH}")
            return _LoadedModel(
                model=fallback_model,
                version="local_latest",
                run_id="local_file",
                source="local_cache",
                loaded_at=datetime.now(timezone.utc).isoformat(),
                tags={"fallback": "true", "path": str(DEFAULT_LOCAL_MODEL_PATH)},
            )

        raise RuntimeError(
            f"Failed to load model from both MLflow Registry ({model_uri}) and local backup ({DEFAULT_LOCAL_MODEL_PATH})."
        )

    @staticmethod
    def _mlflow_client() -> MlflowClient:
        try:
            urllib.request.urlopen(f"{MLFLOW_TRACKING_URI}/health", timeout=2)
        except Exception as conn_err:
            raise ConnectionError(f"MLflow server not reachable: {conn_err}") from conn_err
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    def _notify_model_change(self) -> None:
        if self.on_model_change is None or self._state is None:
            return
        try:
            self.on_model_change(self.get_model_info())
        except Exception as exc:
            logger.warning(f"on_model_change callback failed: {exc}")

    def reload(self) -> ModelInfoResponse:
        """
        Принудительно перезагружает активную модель из реестра MLflow.
        """
        logger.info("Executing hot reload of production model...")
        self._get_state(force_reload=True)
        return self.get_model_info()

    def start_watcher(self) -> None:
        """
        Запускает фоновую сверку версии @champion. Каждый под сам подтягивает
        новую модель, поэтому /model/reload через Service не нужен для всех реплик.
        """
        if MODEL_POLL_INTERVAL_SEC <= 0 or (self._watcher and self._watcher.is_alive()):
            return
        self._stop.clear()
        self._watcher = threading.Thread(target=self._watch_loop, name="champion-watcher", daemon=True)
        self._watcher.start()
        logger.info(f"Champion watcher started (interval: {MODEL_POLL_INTERVAL_SEC}s)")

    def stop_watcher(self) -> None:
        self._stop.set()

    def _watch_loop(self) -> None:
        while not self._stop.wait(MODEL_POLL_INTERVAL_SEC):
            try:
                if self._state is None:
                    self._get_state()
                    continue
                mv = self._mlflow_client().get_model_version_by_alias(
                    name=MLFLOW_MODEL_NAME, alias=MLFLOW_CHAMPION_ALIAS
                )
                if str(mv.version) != self._state.version:
                    logger.info(
                        f"Champion alias moved: {self._state.version} -> {mv.version}, reloading model"
                    )
                    self._get_state(force_reload=True)
            except Exception as exc:
                logger.debug(f"Champion watcher check skipped: {exc}")

    def get_model_info(self) -> ModelInfoResponse:
        """
        Возвращает метаданные о текущей активной модели в памяти.
        """
        state = self._state
        if state is None:
            try:
                state = self._get_state()
            except Exception as e:
                logger.error(f"Model is unavailable: {e}")
                return ModelInfoResponse(
                    model_name=MLFLOW_MODEL_NAME,
                    alias=MLFLOW_CHAMPION_ALIAS,
                    version=None,
                    run_id=None,
                    source="unavailable",
                    status="error: model is not loaded",
                    metrics={},
                    tags={},
                    loaded_at=datetime.now(timezone.utc).isoformat(),
                )

        return ModelInfoResponse(
            model_name=MLFLOW_MODEL_NAME,
            alias=MLFLOW_CHAMPION_ALIAS,
            version=state.version,
            run_id=state.run_id,
            source=state.source,
            status="READY",
            metrics=state.metrics,
            tags=state.tags,
            loaded_at=state.loaded_at,
        )

    def _fill_nearest_metro(self, df: pd.DataFrame) -> None:
        """
        Определяет ближайшую станцию по координатам так же, как это делает
        VIEW apartment_nearest_metro для обучающей выборки.
        """
        for idx, row in df.iterrows():
            lat_val, lon_val = row.get("latitude"), row.get("longitude")
            if lat_val is None or lon_val is None or pd.isna(lat_val) or pd.isna(lon_val):
                continue
            station = self._metro.nearest(float(lat_val), float(lon_val))
            if station is None:
                continue
            df.at[idx, "metro"] = station.name
            df.at[idx, "metro_line"] = station.line_name
            df.at[idx, "transport_type"] = station.transport_type
            df.at[idx, "metro_distance_m"] = station.distance_m

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
        df = pd.DataFrame(records).astype({"metro": object, "metro_line": object, "transport_type": object})

        self._fill_nearest_metro(df)

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

        # Инференс обученной модели CatBoost (снимок состояния, чтобы версия
        # в ответе совпадала с моделью, даже если параллельно идёт reload)
        state = self._get_state()
        raw_predictions = state.model.predict(align_features_to_model(X, state.model))

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
                    metro_line=str(row_features["metro_line"]),
                    transport_type=str(row_features["transport_type"]),
                    metro_distance_m=float(row_features["metro_distance_m"]),
                    distance_to_center_m=float(row_features["distance_to_center_m"]),
                    model_name=MLFLOW_MODEL_NAME,
                    model_version=state.version,
                    model_source=state.source,
                )
            )

        return results


# Глобальный экземпляр сервиса для использования в эндпоинтах FastAPI
inference_service = InferenceService()
