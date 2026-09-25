from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.catboost as mlflow_catboost
import pandas as pd
from catboost import CatBoostRegressor
from loguru import logger
from mlflow.exceptions import MlflowException, RestException
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.tracking import MlflowClient

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from config import (
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
)

DEFAULT_LOCAL_MODEL_PATH = MODELS_DIR / "catboost_latest.cbm"


def setup_mlflow(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> tuple[MlflowClient, str]:
    """
    Инициализирует подключение к серверу MLflow Tracking Server,
    проверяет наличие эксперимента и возвращает кортеж (MlflowClient, experiment_id).
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.info(f"Creating MLflow experiment: '{experiment_name}'")
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return client, experiment_id


def get_champion_info(
    client: MlflowClient | None = None,
    model_name: str = MLFLOW_MODEL_NAME,
    alias: str = MLFLOW_CHAMPION_ALIAS,
) -> dict[str, Any] | None:
    """
    Возвращает информацию о текущей модели-чемпионе из реестра MLflow Model Registry.
    При отсутствии модели или алиаса возвращает None.
    """
    ml_client = client if client is not None else MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    try:
        model_version = ml_client.get_model_version_by_alias(name=model_name, alias=alias)
    except (MlflowException, RestException) as exc:
        logger.warning(
            f"Champion alias '@{alias}' not found for model '{model_name}': {exc}"
        )
        return None

    run_id = model_version.run_id or ""
    metrics: dict[str, float] = {}
    if run_id:
        try:
            run = ml_client.get_run(run_id)
            metrics = {str(k): float(v) for k, v in run.data.metrics.items()}
        except Exception as exc:
            logger.warning(f"Could not fetch metrics for run {run_id}: {exc}")
            metrics = {}

    return {
        "name": model_name,
        "version": str(model_version.version),
        "run_id": run_id,
        "metrics": metrics,
        "aliases": [str(a) for a in (model_version.aliases or [])],
        "tags": {str(k): str(v) for k, v in (model_version.tags or {}).items()},
        "status": str(model_version.status),
    }


def promote_to_champion(
    version: str | int,
    client: MlflowClient | None = None,
    model: CatBoostRegressor | None = None,
    model_name: str = MLFLOW_MODEL_NAME,
    alias: str = MLFLOW_CHAMPION_ALIAS,
    reason: str | None = None,
) -> None:
    """
    Назначает указанной версии модели алиас '@champion', обновляет метаданные в MLflow
    и синхронизирует локальный файл catboost_latest.cbm.
    """
    ml_client = client if client is not None else MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    version_str = str(version)
    logger.info(
        f"Promoting model '{model_name}' v{version_str} to alias '@{alias}'..."
    )
    ml_client.set_registered_model_alias(name=model_name, alias=alias, version=version_str)

    now_iso = datetime.now(timezone.utc).isoformat()
    ml_client.set_model_version_tag(
        name=model_name,
        version=version_str,
        key="promoted_at",
        value=now_iso,
    )
    ml_client.set_model_version_tag(
        name=model_name,
        version=version_str,
        key="role",
        value=alias,
    )
    if reason:
        ml_client.set_model_version_tag(
            name=model_name,
            version=version_str,
            key="promotion_reason",
            value=reason,
        )

    # Синхронизация локальной резервной копии модели
    if model is not None:
        save_local_model(model, DEFAULT_LOCAL_MODEL_PATH)

    logger.success(f"Model '{model_name}' v{version_str} successfully promoted to '@{alias}'!")


def save_local_model(model: CatBoostRegressor, filepath: Path | str = DEFAULT_LOCAL_MODEL_PATH) -> Path:
    """
    Сохраняет бинарный артефакт CatBoost в локальный файл (.cbm).
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info(f"Saved local model artifact to: {path}")
    return path


def load_local_model(
    filepath: Path | str = DEFAULT_LOCAL_MODEL_PATH,
) -> CatBoostRegressor | None:
    """
    Загружает модель из локального файла .cbm. При отсутствии файла возвращает None.
    """
    path = Path(filepath)
    if not path.is_file():
        return None
    model = CatBoostRegressor()
    model.load_model(str(path))
    logger.info(f"Loaded model from local backup: {path}")
    return model


def log_and_register_model(
    model: CatBoostRegressor,
    model_name: str = MLFLOW_MODEL_NAME,
    input_example: pd.DataFrame | None = None,
    signature: ModelSignature | None = None,
) -> tuple[str, str]:
    """
    Логирует модель CatBoost в активный запуск MLflow и регистрирует в реестре Model Registry.
    Сохраняет версионированную локальную копию. Возвращает кортеж (run_id, version).
    """
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("No active MLflow run found. Call inside with mlflow.start_run().")

    run_id = active_run.info.run_id

    # Автоматический расчет сигнатуры входных признаков
    if signature is None and input_example is not None:
        example_pred = model.predict(input_example)
        signature = infer_signature(input_example, example_pred)

    log_kwargs: dict[str, Any] = {}
    if input_example is not None:
        log_kwargs["input_example"] = input_example
    if signature is not None:
        log_kwargs["signature"] = signature

    model_info = mlflow_catboost.log_model(
        cb_model=model,
        name="model",
        registered_model_name=model_name,
        **log_kwargs,
    )

    # Получение номера зарегистрированной версии модели
    version = str(model_info.registered_model_version or "")
    if not version:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.search_model_versions(f"name = '{model_name}'")
        for mv in versions:
            if mv.run_id == run_id:
                version = str(mv.version)
                break

    # Сохранение версионированной копии на диск
    if version:
        save_local_model(model, MODELS_DIR / f"catboost_v{version}.cbm")

    return run_id, version


def load_production_model(
    model_name: str = MLFLOW_MODEL_NAME,
    alias: str = MLFLOW_CHAMPION_ALIAS,
) -> CatBoostRegressor:
    """
    Загружает модель-чемпион из MLflow Model Registry (models:/{model_name}@{alias}).
    При недоступности сервера MLflow используется резервный локальный файл .cbm.
    """
    model_uri = f"models:/{model_name}@{alias}"
    try:
        logger.info(f"Loading production model from MLflow: '{model_uri}'...")
        model: CatBoostRegressor = mlflow_catboost.load_model(model_uri)
        return model
    except Exception as exc:
        logger.warning(
            f"Failed to load model from MLflow Registry ({model_uri}): {exc}. "
            f"Falling back to local cache..."
        )
        local_model = load_local_model(DEFAULT_LOCAL_MODEL_PATH)
        if local_model is not None:
            return local_model
        raise RuntimeError(
            f"Unable to load model from both MLflow Registry ({model_uri}) and local cache ({DEFAULT_LOCAL_MODEL_PATH})."
        ) from exc
