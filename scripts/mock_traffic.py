"""
Генератор мок-нагрузки и тестовых данных для демонстрации мониторинга.

Скрипт выполняет:
1. Наполнение ClickHouse новыми снапшотами за текущий час для актуализации графиков.
2. Имитацию разнообразного входящего трафика на FastAPI Inference:
   - Одиночные предсказания для разных категорий квартир (студии, 1-4 комнатные, элитные).
   - Пакетные запросы с разным размером батча (/predict/batch).
   - Запросы с ошибками валидации (/predict с некорректными данными -> HTTP 422).
   - Запросы на горячую перезагрузку модели (/model/reload).
   - Фоновые health-check запросы.
3. Выполнение аналитических запросов к ClickHouse для генерации QPS и профилирования.
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import clickhouse_connect
from loguru import logger
import requests

# Конфигурация эндпоинтов
INFERENCE_URL = "http://localhost:8000"
CLICKHOUSE_HOST = "localhost"
CLICKHOUSE_PORT = 8123
CLICKHOUSE_USER = "estate"
CLICKHOUSE_PASSWORD = "estate"
CLICKHOUSE_DB = "estate"

# Список станций метро для генерации реалистичных запросов
METRO_STATIONS = [
    {"metro": "Охотный Ряд", "line": "Сокольническая", "lat": 55.757, "lon": 37.616, "dist": 300.0},
    {"metro": "Тверская", "line": "Замоскворецкая", "lat": 55.765, "lon": 37.604, "dist": 1400.0},
    {"metro": "Арбатская", "line": "Арбатско-Покровская", "lat": 55.752, "lon": 37.602, "dist": 1100.0},
    {"metro": "Белорусская", "line": "Кольцевая", "lat": 55.777, "lon": 37.583, "dist": 3500.0},
    {"metro": "Павелецкая", "line": "Замоскворецкая", "lat": 55.730, "lon": 37.639, "dist": 3200.0},
    {"metro": "Сокол", "line": "Замоскворецкая", "lat": 55.805, "lon": 37.516, "dist": 8700.0},
    {"metro": "Университет", "line": "Сокольническая", "lat": 55.692, "lon": 37.534, "dist": 8200.0},
    {"metro": "ВДНХ", "line": "Калужско-Рижская", "lat": 55.821, "lon": 37.641, "dist": 8100.0},
    {"metro": "Митино", "line": "Арбатско-Покровская", "lat": 55.846, "lon": 37.363, "dist": 17500.0},
    {"metro": "Выхино", "line": "Таганско-Краснопресненская", "lat": 55.716, "lon": 37.818, "dist": 14200.0},
]


def generate_apartment(rooms: int | None = None) -> dict[str, object]:
    """Генерирует реалистичный запрос оценки квартиры."""
    if rooms is None:
        rooms = random.choice([0, 1, 2, 3, 4])

    st = random.choice(METRO_STATIONS)
    floor_total = random.randint(9, 32)
    floor = random.randint(1, floor_total)

    if rooms == 0:
        area = round(random.uniform(18.0, 32.0), 1)
    elif rooms == 1:
        area = round(random.uniform(33.0, 48.0), 1)
    elif rooms == 2:
        area = round(random.uniform(50.0, 75.0), 1)
    elif rooms == 3:
        area = round(random.uniform(76.0, 115.0), 1)
    else:
        area = round(random.uniform(116.0, 220.0), 1)

    return {
        "rooms": rooms,
        "area": area,
        "floor": floor,
        "floors_total": floor_total,
        "metro": st["metro"],
        "metro_line": st["line"],
        "metro_distance_m": round(random.uniform(150.0, 1600.0), 1),
        "transport_type": random.choice(["walk", "transport"]),
        "latitude": st["lat"] + random.uniform(-0.005, 0.005),
        "longitude": st["lon"] + random.uniform(-0.005, 0.005),
        "distance_to_center_m": st["dist"] + random.uniform(-300.0, 300.0),
        "seller_type": random.choice(["owner", "realtor", "agency"]),
    }


def seed_clickhouse_recent_data() -> None:
    """Генерирует свежие снапшоты в ClickHouse за последние 2 часа до текущего момента."""
    logger.info("Подключение к ClickHouse для генерации свежих снапшотов...")
    try:
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            username=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASSWORD,
            database=CLICKHOUSE_DB,
        )

        # Выбираем случайные существующие ID и цены для реалистичных повторных снапшотов
        existing = client.query("SELECT external_id, monthly_rent FROM estate.apartment_snapshots LIMIT 1500").result_rows
        if not existing:
            logger.warning("Таблица apartment_snapshots пуста, генерация пропущена.")
            return

        now = datetime.now()
        new_rows: list[list[object]] = []

        # Создаем снапшоты с шагом 5 минут за последние 90 минут
        for minutes_ago in range(90, 0, -5):
            snapshot_time = now - timedelta(minutes=minutes_ago)
            # Берем случайную выборку из 100-200 объявлений для каждого среза
            sample = random.sample(existing, min(150, len(existing)))
            for ext_id, rent in sample:
                # Небольшая вариация цены (+- 5%)
                drift = random.choice([0.95, 1.0, 1.0, 1.0, 1.05])
                adj_rent = int(round(rent * drift / 500) * 500)
                new_rows.append(["avito", str(ext_id), max(adj_rent, 20000), snapshot_time])

        client.insert(
            "estate.apartment_snapshots",
            new_rows,
            column_names=["source", "external_id", "monthly_rent", "collected_at"],
        )
        logger.success(f"Успешно вставлено {len(new_rows)} новых снапшотов в ClickHouse!")

    except Exception as exc:
        logger.error(f"Ошибка при работе с ClickHouse: {exc}")


def run_clickhouse_analytics_queries() -> None:
    """Выполняет серию агрегационных запросов в ClickHouse для генерации QPS и профилирования."""
    try:
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            username=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASSWORD,
            database=CLICKHOUSE_DB,
        )
        queries = [
            "SELECT count(), avg(monthly_rent), quantile(0.5)(monthly_rent) FROM estate.apartment_snapshots",
            "SELECT toStartOfHour(collected_at) AS hr, uniqExact(external_id) FROM estate.apartment_snapshots GROUP BY hr",
            "SELECT monthly_rent, count() FROM estate.apartment_snapshots WHERE monthly_rent > 100000 GROUP BY monthly_rent LIMIT 50",
            "SELECT multiIf(monthly_rent < 50000, 'low', 'high'), count() FROM estate.apartment_snapshots GROUP BY 1",
        ]
        for q in queries:
            client.query(q)
    except Exception as exc:
        logger.warning(f"Ошибка запроса ClickHouse: {exc}")


def send_single_predict() -> None:
    """Отправляет одиночный валидный запрос на /predict."""
    payload = generate_apartment()
    try:
        resp = requests.post(f"{INFERENCE_URL}/predict", json=payload, timeout=5)
        if resp.status_code == 200:
            pass
        else:
            logger.warning(f"Predict returned {resp.status_code}")
    except Exception as exc:
        logger.warning(f"Predict error: {exc}")


def send_batch_predict() -> None:
    """Отправляет пакетный запрос на /predict/batch со случайным размером батча."""
    size = random.choice([3, 5, 12, 20, 35])
    apartments = [generate_apartment() for _ in range(size)]
    try:
        resp = requests.post(f"{INFERENCE_URL}/predict/batch", json={"apartments": apartments}, timeout=10)
        if resp.status_code == 200:
            pass
        else:
            logger.warning(f"Batch predict returned {resp.status_code}")
    except Exception as exc:
        logger.warning(f"Batch predict error: {exc}")


def send_validation_error() -> None:
    """Отправляет заведомо некорректный запрос для генерации ответов HTTP 422 в метриках."""
    # Площадь 2.0 м² (меньше валидационного минимума 5.0)
    payload = generate_apartment()
    payload["area"] = 2.0
    try:
        requests.post(f"{INFERENCE_URL}/predict", json=payload, timeout=5)
    except Exception:
        pass


def send_model_reload() -> None:
    """Отправляет запрос на горячую перезагрузку модели."""
    try:
        resp = requests.post(f"{INFERENCE_URL}/model/reload", timeout=10)
        if resp.status_code == 200:
            logger.info("Модель успешно перезагружена через /model/reload")
    except Exception as exc:
        logger.warning(f"Ошибка перезагрузки модели: {exc}")


def main() -> None:
    logger.info("=== Запуск генератора мок-нагрузки Estate Intelligence ===")

    # 1. Заполняем свежими данными ClickHouse
    seed_clickhouse_recent_data()

    # 2. Выполняем предварительный reload модели
    send_model_reload()

    # 3. Генерируем волновой трафик в течение ~45 секунд
    logger.info("Начало эмуляции реального пользовательского трафика (RPS, Latency, Errors, Batches)...")
    start_time = time.time()
    iteration = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        while time.time() - start_time < 50:
            iteration += 1
            # Чередование интенсивности (пики и спокойные фазы)
            intensity = 15 if (iteration % 6 in [2, 3]) else 5

            futures = []
            for _ in range(intensity):
                futures.append(executor.submit(send_single_predict))

            # Периодически отправляем батчи
            if iteration % 3 == 0:
                futures.append(executor.submit(send_batch_predict))

            # Периодически отправляем запросы с 422 ошибкой валидации
            if iteration % 4 == 0:
                futures.append(executor.submit(send_validation_error))

            # Периодически опрашиваем ClickHouse для поддержания QPS
            if iteration % 2 == 0:
                futures.append(executor.submit(run_clickhouse_analytics_queries))

            # Ждем завершения пачки
            for f in futures:
                f.result()

            time.sleep(0.8)

    # 4. Финальный reload для проверки регистрации счетчика
    send_model_reload()

    logger.success("=== Генерация мок-данных успешно завершена! ===")
    logger.info("Проверьте дашборды в Grafana: http://localhost:3000")


if __name__ == "__main__":
    main()
