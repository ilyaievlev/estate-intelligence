from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from loguru import logger

# Гарантируем доступ к модулям services/ml
ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from config import (
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    RANDOM_STATE,
    TEST_SIZE,
)
from data.loader import load_apartments_from_db
from features.engineering import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    clean_dataset,
    prepare_features,
)
from models.registry import (
    log_and_register_model,
    promote_to_champion,
    setup_mlflow,
)
from models.trainer import train_catboost_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CatBoost model for Moscow rental price prediction with MLflow tracking."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2500,
        help="Number of CatBoost boosting iterations (default: 2500)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.04,
        help="Learning rate (default: 0.04)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=7,
        help="Tree depth (default: 7)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help=f"Fraction of test set (default: {TEST_SIZE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows loaded from database (optional, for debug)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (optional)",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Do not promote the trained model to champion alias in MLflow Model Registry",
    )
    return parser.parse_args()


def run_training(
    iterations: int = 2500,
    learning_rate: float = 0.04,
    depth: int = 7,
    test_size: float = TEST_SIZE,
    limit: int | None = None,
    run_name: str | None = None,
    promote: bool = True,
) -> dict[str, Any]:
    """
    Основной пайплайн первоначального обучения модели.
    Возвращает словарь с run_id, model_version и метриками.
    """
    client, experiment_id = setup_mlflow()
    logger.info(
        f"MLflow initialized. Tracking URI: {mlflow.get_tracking_uri()}, Experiment: {MLFLOW_EXPERIMENT_NAME} (id={experiment_id})"
    )

    # 1. Загрузка данных
    logger.info("Loading apartment records from PostgreSQL...")
    raw_df = load_apartments_from_db(limit=limit)
    if raw_df.empty:
        raise ValueError("No data loaded from PostgreSQL. Check database connection and tables.")
    logger.info(f"Loaded {len(raw_df):,} raw records.")

    # 2. Очистка выбросов
    clean_df = clean_dataset(raw_df)
    logger.info(
        f"Dataset cleaned: {len(clean_df):,} records remaining ({len(clean_df)/len(raw_df):.1%})."
    )

    # 3. Извлечение и подготовка признаков
    X, y = prepare_features(clean_df, is_training=True)
    if y is None:
        raise ValueError("Target variable 'monthly_rent' is missing!")

    logger.info(
        f"Feature matrix prepared: {X.shape[0]:,} samples x {X.shape[1]} features "
        f"({len(NUMERIC_FEATURES)} numeric, {len(CATEGORICAL_FEATURES)} categorical)."
    )

    # 4. Обучение модели и логирование в MLflow
    output_summary: dict[str, Any] = {}
    custom_params = {
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
    }

    with mlflow.start_run(run_name=run_name or "train_catboost_baseline") as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run: {run_id}")

        # Логируем теги
        mlflow.set_tags(
            {
                "pipeline": "train",
                "model_family": "catboost",
                "target": "monthly_rent",
                "raw_samples": len(raw_df),
                "clean_samples": len(clean_df),
            }
        )

        # Обучаем CatBoost
        result = train_catboost_model(
            X=X,
            y=y,
            params=custom_params,
            test_size=test_size,
            random_state=RANDOM_STATE,
        )

        # Логируем гиперпараметры
        mlflow.log_params(result.params)
        mlflow.log_params(
            {
                "test_size": test_size,
                "random_state": RANDOM_STATE,
                "n_features": len(ALL_FEATURES),
                "features_list": ",".join(ALL_FEATURES),
            }
        )

        # Логируем ключевые метрики
        mlflow.log_metrics(result.metrics)
        mlflow.log_metrics(
            {
                "train_samples": float(len(result.X_train)),
                "test_samples": float(len(result.X_test)),
            }
        )

        logger.info("Evaluation results on holdout test set:")
        for metric_name, val in result.metrics.items():
            logger.info(f"  - {metric_name}: {val}")

        # Сохраняем и логируем артефакт важности признаков
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            fi_file = tmp_path / "feature_importances.csv"
            result.feature_importances.to_frame(name="importance").to_csv(fi_file)
            mlflow.log_artifact(str(fi_file), artifact_path="feature_analysis")

            metrics_file = tmp_path / "metrics.json"
            metrics_file.write_text(json.dumps(result.metrics, indent=2, ensure_ascii=False))
            mlflow.log_artifact(str(metrics_file), artifact_path="metrics")

        # Логирование топ-5 признаков в теги MLflow
        top_5_features = [str(feat) for feat in result.feature_importances.head(5).index]
        mlflow.set_tag("top_features", ", ".join(top_5_features))

        # Логируем и регистрируем модель в MLflow Model Registry и S3 (MinIO)
        logger.info(
            f"Logging CatBoost model to S3 / MinIO and registering as '{MLFLOW_MODEL_NAME}'..."
        )
        _, model_version = log_and_register_model(
            model=result.model,
            model_name=MLFLOW_MODEL_NAME,
            input_example=result.X_test.head(5),
        )

        # Назначаем статус Champion
        if promote and model_version:
            promote_to_champion(
                version=model_version,
                client=client,
                model=result.model,
                model_name=MLFLOW_MODEL_NAME,
                alias=MLFLOW_CHAMPION_ALIAS,
                reason="Initial baseline training",
            )
            logger.success(
                f"Model version {model_version} is set as Champion (@{MLFLOW_CHAMPION_ALIAS})!"
            )

        output_summary = {
            "run_id": run_id,
            "model_version": model_version,
            "metrics": result.metrics,
            "promoted": promote,
        }

        logger.success(
            f"Training pipeline finished successfully! "
            f"Run ID: {run_id}, Version: {model_version}, MAE: {result.metrics['mae']} rub."
        )

    return output_summary


def main() -> None:
    args = parse_args()
    run_training(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        test_size=args.test_size,
        limit=args.limit,
        run_name=args.run_name,
        promote=not args.no_promote,
    )


if __name__ == "__main__":
    main()
