from __future__ import annotations

from loguru import logger

from data.loader import load_apartments_from_db
from features.engineering import prepare_features
from models.trainer import train_catboost_model


def main() -> None:
    """Точка входа: выгружает данные, чистит, готовит фичи и обучает модель."""
    logger.info("Загрузка данных из PostgreSQL...")
    df_raw = load_apartments_from_db()

    logger.info("Генерация признаков (Feature Engineering)...")
    X, y = prepare_features(df_raw, is_training=True)

    if y is None or len(y) == 0:
        logger.error("Нет валидных данных целевой переменной monthly_rent!")
        return

    logger.info(f"Сформирована матрица признаков: X={X.shape}, y={len(y)}")

    model, metrics, fi = train_catboost_model(X, y)

    print("\n" + "=" * 55)
    print("🏆 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ МОДЕЛИ (TEST SET):")
    print(f"  • MAE (средняя ошибка):       {metrics['mae']:,.0f} ₽")
    print(f"  • MAPE (процентная ошибка):    {metrics['mape']:.2f}%")
    print(f"  • Коэффициент детерминации R²: {metrics['r2']:.4f}")
    print(f"  • Baseline MAE (по медиане):  {metrics['baseline_mae']:,.0f} ₽")
    print(f"  • Улучшение относительно базы: {metrics['improvement_pct']:.1f}%")
    print("-" * 55)
    print("📊 ТОП-10 НАИБОЛЕЕ ВАЖНЫХ ПРИЗНАКОВ:")
    for feat, val in fi.head(10).items():
        print(f"  • {feat:<22}: {val:.2f}%")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
