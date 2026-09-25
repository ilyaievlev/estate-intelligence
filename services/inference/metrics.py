from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Метрика количества выполненных прогнозов стоимости аренды
PREDICTIONS_TOTAL = Counter(
    "estate_predictions_total",
    "Общее количество выполненных прогнозов стоимости аренды",
    ["status", "rooms", "model_version", "model_source"],
)

# Метрика распределения прогнозируемой стоимости аренды (в рублях)
PREDICTED_RENT_RUB = Histogram(
    "estate_predicted_rent_rub",
    "Распределение прогнозируемой ставки аренды в рублях",
    buckets=[
        30_000.0,
        50_000.0,
        75_000.0,
        100_000.0,
        150_000.0,
        200_000.0,
        300_000.0,
        500_000.0,
        1_000_000.0,
    ],
)

# Метрика длительности чистого ML-инференса (подготовка признаков + CatBoost)
INFERENCE_LATENCY_SECONDS = Histogram(
    "estate_inference_latency_seconds",
    "Время выполнения ML-инференса (секунды)",
    buckets=[0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Метрика размера пакетов при пакетной оценке
BATCH_SIZE = Histogram(
    "estate_batch_predict_size",
    "Размер пакета запросов в эндпоинте /predict/batch",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500],
)

# Метрика активной версии модели в оперативной памяти инференса
ACTIVE_MODEL_VERSION = Gauge(
    "estate_active_model_info",
    "Информация об активной модели в памяти микросервиса",
    ["model_name", "model_version", "model_source"],
)

# Метрика количества горячих перезагрузок модели
MODEL_RELOAD_TOTAL = Counter(
    "estate_model_reload_total",
    "Количество запросов на горячую перезагрузку модели-чемпиона",
    ["status"],
)
