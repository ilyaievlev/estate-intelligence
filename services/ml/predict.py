from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from loguru import logger

# Подключение модулей services/ml в системный путь
ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from config import (
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_MODEL_NAME,
    MOSCOW_CENTER_LAT,
    MOSCOW_CENTER_LON,
)
from features.engineering import prepare_features
from models.registry import load_production_model

# Кэшированный экземпляр модели в оперативной памяти
_CACHED_MODEL: CatBoostRegressor | None = None


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Вычисляет расстояние в метрах между двумя гео-координатами по формуле гаверсинуса."""
    earth_radius = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * earth_radius * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def get_model(force_reload: bool = False) -> CatBoostRegressor:
    """Возвращает кэшированный экземпляр модели или загружает его."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None or force_reload:
        _CACHED_MODEL = load_production_model(MLFLOW_MODEL_NAME, MLFLOW_CHAMPION_ALIAS)
    return _CACHED_MODEL


def predict_rent(
    data: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame,
    model: CatBoostRegressor | None = None,
) -> list[dict[str, Any]]:
    """
    Принимает словарь, список словарей или DataFrame с параметрами квартир
    и возвращает список предсказаний с рассчитанной стоимостью аренды.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(f"Unsupported data type for predict_rent: {type(data)}")

    if df.empty:
        return []

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

    # Формирование и нормализация признаков (Feature Engineering)
    X, _ = prepare_features(df, is_training=False)

    # Загрузка и инференс обученной модели CatBoost
    current_model = model or get_model()
    predictions = current_model.predict(X)

    results: list[dict[str, Any]] = []
    for idx, pred_val in enumerate(predictions):
        raw_price = float(pred_val)
        rounded_price = int(round(raw_price / 500.0) * 500)  # Округление до 500 ₽
        item_features = {col: X.iloc[idx][col] for col in X.columns}

        results.append(
            {
                "predicted_rent_rub": rounded_price,
                "predicted_rent_raw": round(raw_price, 2),
                "rooms": int(item_features["rooms"]),
                "area": float(item_features["area"]),
                "metro": str(item_features["metro"]),
                "metro_distance_m": float(item_features["metro_distance_m"]),
                "distance_to_center_m": float(item_features["distance_to_center_m"]),
            }
        )

    return results


def parse_cli_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки CLI."""
    parser = argparse.ArgumentParser(
        description="Predict rental price for Moscow apartments using CatBoost production model."
    )
    # Параметры одиночной квартиры
    parser.add_argument("--rooms", type=int, default=1, help="Number of rooms (0=studio, 1, 2, ...)")
    parser.add_argument("--area", type=float, default=40.0, help="Total area in m²")
    parser.add_argument("--floor", type=int, default=5, help="Floor number")
    parser.add_argument("--floors-total", type=int, default=16, help="Total floors in building")
    parser.add_argument("--metro", type=str, default="Белорусская", help="Nearest metro station name")
    parser.add_argument("--metro-distance", type=float, default=500.0, help="Distance to metro in meters")
    parser.add_argument("--metro-line", type=str, default="Замоскворецкая", help="Metro line name")
    parser.add_argument(
        "--transport-type",
        type=str,
        default="metro",
        choices=["metro", "walk", "transport"],
        help="Transport type to metro",
    )
    parser.add_argument("--distance-to-center", type=float, default=None, help="Distance to Kremlin in meters")
    parser.add_argument("--latitude", type=float, default=None, help="Latitude")
    parser.add_argument("--longitude", type=float, default=None, help="Longitude")
    parser.add_argument("--seller-type", type=str, default="realtor", help="Seller type (owner, realtor, etc.)")
    parser.add_argument("--source", type=str, default="avito", help="Listing source (avito, cian)")

    # Пакетный режим
    parser.add_argument("--json", type=str, default=None, help="JSON string with apartment data or list of apartments")
    parser.add_argument("--json-file", type=str, default=None, help="Path to JSON file with input apartment(s)")
    parser.add_argument("--output", type=str, default=None, help="Path to save prediction results as JSON")

    return parser.parse_args()


def main() -> None:
    """Точка входа CLI для получения предсказаний аренды."""
    args = parse_cli_args()

    input_data: list[dict[str, Any]] | dict[str, Any]

    if args.json_file:
        file_path = Path(args.json_file)
        if not file_path.exists():
            logger.error(f"Input file not found: {file_path}")
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    elif args.json:
        input_data = json.loads(args.json)
    else:
        # Сборка объекта квартиры из отдельных аргументов CLI
        apartment: dict[str, Any] = {
            "rooms": args.rooms,
            "area": args.area,
            "floor": args.floor,
            "floors_total": args.floors_total,
            "metro": args.metro,
            "metro_distance_m": args.metro_distance,
            "metro_line": args.metro_line,
            "transport_type": args.transport_type,
            "seller_type": args.seller_type,
            "source": args.source,
        }
        if args.distance_to_center is not None:
            apartment["distance_to_center_m"] = args.distance_to_center
        if args.latitude is not None:
            apartment["latitude"] = args.latitude
        if args.longitude is not None:
            apartment["longitude"] = args.longitude

        input_data = apartment

    results = predict_rent(input_data)

    print("\n" + "=" * 60)
    print(" 🏢 ESTATE INTELLIGENCE: RENTAL PRICE ESTIMATION")
    print("=" * 60)
    for i, res in enumerate(results, 1):
        rooms_label = "Студия" if res["rooms"] == 0 else f"{res['rooms']}-к. кв."
        print(f"\n[Объект #{i}]")
        print(f"  Параметры:        {rooms_label}, {res['area']} м²")
        print(f"  Метро:            ст. {res['metro']} (~{res['metro_distance_m']:.0f} м)")
        print(f"  До центра Москвы: ~{res['distance_to_center_m'] / 1000.0:.1f} км")
        print(f"  ➡️ ОЦЕНКА АРЕНДЫ: {res['predicted_rent_rub']:,.0f} ₽/мес.")
        print(f"     (точное значение модели: {res['predicted_rent_raw']:,.2f} ₽)")
    print("=" * 60 + "\n")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
