from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config import MOSCOW_CENTER_LAT, MOSCOW_CENTER_LON

# Числовые признаки
NUMERIC_FEATURES = [
    "rooms",
    "area",
    "floor",
    "floors_total",
    "floor_ratio",
    "is_first_floor",
    "is_last_floor",
    "area_per_room",
    "metro_distance_m",
    "log_metro_distance",
    "distance_to_center_m",
    "log_distance_to_center",
    "latitude",
    "longitude",
    "has_description",
    "description_len",
]

# Категориальные признаки (обрабатываются CatBoost нативно без OHE)
CATEGORICAL_FEATURES = [
    "metro",
    "metro_line",
    "transport_type",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Признаки, которые были у старых моделей и убраны из набора (в данных они
# константны: парсер пишет seller_type=NULL, источник только avito).
# Значения совпадают с тем, что эти модели видели при обучении.
LEGACY_FEATURE_DEFAULTS: dict[str, object] = {
    "seller_type": "unknown",
    "source": "avito",
}


def align_features_to_model(X: pd.DataFrame, model: object) -> pd.DataFrame:
    """
    Приводит матрицу признаков к набору и порядку колонок конкретной модели.
    Нужно, чтобы champion, обученный на другом наборе признаков, продолжал
    работать в inference и честно оценивался при переобучении.
    """
    expected = list(getattr(model, "feature_names_", None) or [])
    if not expected or expected == list(X.columns):
        return X

    aligned = X.copy()
    missing = [name for name in expected if name not in aligned.columns]
    unknown = [name for name in missing if name not in LEGACY_FEATURE_DEFAULTS]
    if unknown:
        raise ValueError(f"Model expects features that cannot be derived: {unknown}")
    for name in missing:
        aligned[name] = LEGACY_FEATURE_DEFAULTS[name]
    return pd.DataFrame(aligned[expected])


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка сырых данных от аномалий и артефактов перед обучением:
    - Цена аренды: от 15 000 до 1 500 000 ₽/мес.
    - Площадь: от 10 до 400 м²
    - Комнатность: от 0 (студия) до 6
    - Координаты: границы Москвы и ближнего Подмосковья
    - Коррекция этажей: если floor > floors_total, выравниваем floors_total
    """
    initial_count = len(df)

    rent = pd.Series(df["monthly_rent"], dtype=float)
    area = pd.Series(df["area"], dtype=float)
    rooms = pd.Series(df["rooms"], dtype=float)
    lat = pd.Series(df["latitude"], dtype=float)
    lon = pd.Series(df["longitude"], dtype=float)

    mask = (
        rent.notna()
        & (rent >= 15_000)
        & (rent <= 1_500_000)
        & area.notna()
        & (area >= 10.0)
        & (area <= 400.0)
        & rooms.notna()
        & (rooms >= 0)
        & (rooms <= 6)
        & lat.notna()
        & (lat >= 55.0)
        & (lat <= 56.5)
        & lon.notna()
        & (lon >= 36.8)
        & (lon <= 38.5)
    )
    cleaned = df.loc[mask].copy()

    floor_col = pd.Series(cleaned["floor"], dtype=float).fillna(1).astype(int)
    floors_total_col = pd.Series(cleaned["floors_total"], dtype=float).fillna(1).astype(int)
    cleaned["floor"] = floor_col
    cleaned["floors_total"] = np.maximum(floors_total_col.to_numpy(), floor_col.to_numpy())

    dropped = initial_count - len(cleaned)
    logger.info(f"Data cleaning: kept {len(cleaned):,} of {initial_count:,} rows (dropped {dropped:,} anomalies)")
    return pd.DataFrame(cleaned)


def prepare_features(
    df: pd.DataFrame,
    is_training: bool = False,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Feature Engineering: формирование матрицы признаков X и вектора таргета y.

    Работает как для обучения (is_training=True с очисткой выбросов),
    так и для инференса/предсказания одной квартиры (is_training=False).
    """
    data = clean_dataset(df) if is_training else df.copy()

    # Таргет (monthly_rent)
    y: pd.Series | None = None
    if "monthly_rent" in data.columns and bool(data["monthly_rent"].notna().any()):
        y = pd.Series(data["monthly_rent"], index=data.index, dtype=float)

    # 1. Признаки планировки и этажности
    rooms_raw = pd.Series(data["rooms"] if "rooms" in data.columns else 1, index=data.index, dtype=float)
    rooms = rooms_raw.fillna(1).clip(lower=0, upper=10)

    area_raw = pd.Series(data["area"] if "area" in data.columns else 45.0, index=data.index, dtype=float)
    area = area_raw.fillna(45.0)

    floor_raw = pd.Series(data["floor"] if "floor" in data.columns else 1, index=data.index, dtype=float)
    floor = floor_raw.fillna(1)

    floors_total_raw = pd.Series(
        data["floors_total"] if "floors_total" in data.columns else 1, index=data.index, dtype=float
    ).fillna(1)
    floors_total = pd.Series(np.maximum(floors_total_raw.to_numpy(), floor.to_numpy()), index=data.index)

    floor_ratio = floor / np.maximum(floors_total.to_numpy(), 1.0)
    is_first_floor = (floor == 1).astype(int)
    is_last_floor = ((floor == floors_total) & (floors_total > 1)).astype(int)
    area_per_room = area / np.maximum(rooms.to_numpy(), 1.0)

    # 2. Географические и транспортные признаки
    metro_dist_raw = pd.Series(
        data["metro_distance_m"] if "metro_distance_m" in data.columns else 1000.0,
        index=data.index,
        dtype=float,
    )
    metro_dist = metro_dist_raw.fillna(1000.0)
    log_metro_dist = np.log1p(np.maximum(metro_dist.to_numpy(), 0.0))

    dist_center_raw = pd.Series(
        data["distance_to_center_m"] if "distance_to_center_m" in data.columns else 10000.0,
        index=data.index,
        dtype=float,
    )
    dist_center = dist_center_raw.fillna(10000.0)
    log_dist_center = np.log1p(np.maximum(dist_center.to_numpy(), 0.0))

    lat_raw = pd.Series(
        data["latitude"] if "latitude" in data.columns else MOSCOW_CENTER_LAT,
        index=data.index,
        dtype=float,
    )
    lat = lat_raw.fillna(MOSCOW_CENTER_LAT)

    lon_raw = pd.Series(
        data["longitude"] if "longitude" in data.columns else MOSCOW_CENTER_LON,
        index=data.index,
        dtype=float,
    )
    lon = lon_raw.fillna(MOSCOW_CENTER_LON)

    # 3. Текстовые метаданные (описание)
    desc_raw = pd.Series(
        data["description"] if "description" in data.columns else None,
        index=data.index,
    )
    has_desc = desc_raw.notna().astype(int)
    desc_len = desc_raw.fillna("").astype(str).map(len).astype(int)

    # 4. Категориальные признаки
    metro = pd.Series(
        data["metro"] if "metro" in data.columns else "unknown", index=data.index
    ).fillna("unknown").astype(str)
    metro_line = pd.Series(
        data["metro_line"] if "metro_line" in data.columns else "unknown", index=data.index
    ).fillna("unknown").astype(str)
    trans_type = pd.Series(
        data["transport_type"] if "transport_type" in data.columns else "metro", index=data.index
    ).fillna("metro").astype(str)

    X_dict = {
        "rooms": rooms,
        "area": area,
        "floor": floor,
        "floors_total": floors_total,
        "floor_ratio": floor_ratio,
        "is_first_floor": is_first_floor,
        "is_last_floor": is_last_floor,
        "area_per_room": area_per_room,
        "metro_distance_m": metro_dist,
        "log_metro_distance": log_metro_dist,
        "distance_to_center_m": dist_center,
        "log_distance_to_center": log_dist_center,
        "latitude": lat,
        "longitude": lon,
        "has_description": has_desc,
        "description_len": desc_len,
        "metro": metro,
        "metro_line": metro_line,
        "transport_type": trans_type,
    }

    X = pd.DataFrame(X_dict, index=data.index)
    X_features = pd.DataFrame(X[ALL_FEATURES])

    return X_features, y

