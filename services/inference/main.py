from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

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
    from services.inference.metrics import (
        ACTIVE_MODEL_VERSION,
        BATCH_SIZE,
        INFERENCE_LATENCY_SECONDS,
        MODEL_RELOAD_TOTAL,
        PREDICTED_RENT_RUB,
        PREDICTIONS_TOTAL,
    )
    from services.inference.schemas import (
        ApartmentPredictRequest,
        BatchPredictionResult,
        BatchPredictRequest,
        HealthResponse,
        ModelInfoResponse,
        PredictionResult,
    )
    from services.inference.service import inference_service
except ImportError:
    from metrics import (  # type: ignore
        ACTIVE_MODEL_VERSION,
        BATCH_SIZE,
        INFERENCE_LATENCY_SECONDS,
        MODEL_RELOAD_TOTAL,
        PREDICTED_RENT_RUB,
        PREDICTIONS_TOTAL,
    )
    from schemas import (  # type: ignore
        ApartmentPredictRequest,
        BatchPredictionResult,
        BatchPredictRequest,
        HealthResponse,
        ModelInfoResponse,
        PredictionResult,
    )
    from service import inference_service  # type: ignore


def update_active_model_metric(info: ModelInfoResponse) -> None:
    """Обновляет значение Prometheus Gauge для текущей активной модели."""
    ACTIVE_MODEL_VERSION.clear()
    ACTIVE_MODEL_VERSION.labels(
        model_name=info.model_name,
        model_version=info.version or "unknown",
        model_source=info.source,
    ).set(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения:
    при старте прогревает ML-модель в памяти (MLflow или local fallback)
    и регистрирует начальные метрики.
    """
    logger.info("Initializing Estate Intelligence Inference API...")
    inference_service.on_model_change = update_active_model_metric
    try:
        _ = inference_service.get_model()
        info = inference_service.get_model_info()
        logger.success(
            f"Production model '{info.model_name}' (version: {info.version}, source: {info.source}) loaded successfully."
        )
    except Exception as exc:
        logger.error(f"Failed to preload model on startup: {exc}. Background watcher will retry.")
    inference_service.start_watcher()
    yield
    inference_service.stop_watcher()
    logger.info("Inference API is shutting down.")


app = FastAPI(
    title="Estate Intelligence - Rent Inference API",
    description=(
        "Production REST API для оценки рыночной стоимости долгосрочной аренды квартир в Москве. "
        "Построено на базе CatBoost Regressor, обученного на данных Avito, с версионированием через MLflow Model Registry (@champion)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware для подключения веб-клиентов и дашбордов.
# API без cookie-авторизации, поэтому credentials не нужны (и с "*" они небезопасны).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Автоматическая инструментация стандартных HTTP-метрик для Prometheus
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/health", "/livez", "/readyz"],
    inprogress_name="estate_http_requests_inprogress",
    inprogress_labels=True,
)
instrumentator.instrument(app).expose(app, endpoint="/metrics", tags=["Monitoring"])


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Логирует входящие запросы и замеряет время обработки в миллисекундах.
    """
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(process_time_ms)
    logger.debug(f"{request.method} {request.url.path} -> {response.status_code} in {process_time_ms}ms")
    return response


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Health check и статус активной модели",
)
def health_check() -> HealthResponse:
    """
    Возвращает статус сервиса и информацию о модели в памяти.
    Модель здесь не загружается: этим занимается старт приложения и фоновый watcher.
    """
    model_loaded = inference_service.model_loaded
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        service="estate-inference",
        model_loaded=model_loaded,
        metro_index_loaded=inference_service.metro_index_loaded,
        model_info=inference_service.get_model_info() if model_loaded else None,
    )


@app.get("/livez", tags=["Monitoring"], summary="Liveness probe")
def liveness() -> dict[str, str]:
    """Процесс жив и отвечает. Не зависит от MLflow и модели."""
    return {"status": "alive"}


@app.get("/readyz", tags=["Monitoring"], summary="Readiness probe")
def readiness(response: Response) -> dict[str, str]:
    """Под готов принимать трафик, только когда модель уже в памяти."""
    if not inference_service.model_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "model not loaded"}
    return {"status": "ready"}


@app.get(
    "/model",
    response_model=ModelInfoResponse,
    tags=["Model Registry"],
    summary="Метаданные текущей модели-чемпиона",
)
def get_model_info() -> ModelInfoResponse:
    """
    Возвращает информацию о текущей активной модели в памяти:
    название, версия, MLflow Run ID, метрики теста и теги промоушна.
    """
    return inference_service.get_model_info()


@app.post(
    "/predict",
    response_model=PredictionResult,
    tags=["Inference"],
    summary="Оценка стоимости аренды квартиры",
)
def predict_apartment(request: ApartmentPredictRequest) -> PredictionResult:
    """
    Принимает параметры квартиры (комнатность, площадь, этаж, метро, гео-координаты)
    и возвращает прогноз арендной ставки в рублях/месяц.
    """
    try:
        t_start = time.perf_counter()
        predictions = inference_service.predict(request)
        duration = time.perf_counter() - t_start

        if not predictions:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Inference generated empty result.",
            )

        res = predictions[0]

        # Фиксация бизнес-метрик инференса в Prometheus
        INFERENCE_LATENCY_SECONDS.observe(duration)
        PREDICTIONS_TOTAL.labels(
            status="success",
            rooms=str(res.rooms),
            model_version=res.model_version or "unknown",
            model_source=res.model_source,
        ).inc()
        PREDICTED_RENT_RUB.observe(float(res.predicted_rent_rub))

        return res
    except HTTPException:
        raise
    except Exception as exc:
        PREDICTIONS_TOTAL.labels(
            status="error",
            rooms=str(request.rooms),
            model_version="error",
            model_source="error",
        ).inc()
        logger.exception(f"Inference error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference execution failed.",
        ) from exc


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResult,
    tags=["Inference"],
    summary="Пакетная оценка нескольких квартир",
)
def predict_batch(request: BatchPredictRequest) -> BatchPredictionResult:
    """
    Принимает массив объектов квартир и возвращает предсказания стоимости аренды для каждого.
    """
    t0 = time.perf_counter()
    try:
        batch_len = len(request.apartments)
        BATCH_SIZE.observe(batch_len)

        t_model_start = time.perf_counter()
        predictions = inference_service.predict(request.apartments)
        model_duration = time.perf_counter() - t_model_start
        INFERENCE_LATENCY_SECONDS.observe(model_duration)

        # Фиксация метрик по каждому элементу пакета
        for p in predictions:
            PREDICTIONS_TOTAL.labels(
                status="success",
                rooms=str(p.rooms),
                model_version=p.model_version or "unknown",
                model_source=p.model_source,
            ).inc()
            PREDICTED_RENT_RUB.observe(float(p.predicted_rent_rub))

        took_ms = round((time.perf_counter() - t0) * 1000, 2)
        return BatchPredictionResult(
            items=predictions,
            total_count=len(predictions),
            took_ms=took_ms,
        )
    except Exception as exc:
        PREDICTIONS_TOTAL.labels(
            status="error",
            rooms="batch",
            model_version="error",
            model_source="error",
        ).inc()
        logger.exception(f"Batch inference error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch inference execution failed.",
        ) from exc


@app.post(
    "/model/reload",
    response_model=ModelInfoResponse,
    tags=["Model Registry"],
    summary="Горячая перезагрузка модели из реестра MLflow",
)
def reload_model() -> ModelInfoResponse:
    """
    Принудительно подтягивает новую модель-чемпион из MLflow Model Registry без перезапуска контейнера.
    Вызывается оркестратором (Airflow) сразу после успешного промоушна нового чемпиона.
    """
    try:
        info = inference_service.reload()
        MODEL_RELOAD_TOTAL.labels(status="success").inc()
        return info
    except Exception as exc:
        MODEL_RELOAD_TOTAL.labels(status="error").inc()
        logger.exception(f"Model reload failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload model.",
        ) from exc
