from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# Гарантируем доступ к модулям services/ml
ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from config import (
    MIN_IMPROVEMENT_RATIO,
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    PRIMARY_METRIC,
    RANDOM_STATE,
    TEST_SIZE,
)
from data.loader import load_apartments_from_db
from features.engineering import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    align_features_to_model,
    clean_dataset,
    prepare_features,
)
from models.registry import (
    get_champion_info,
    load_production_model,
    log_and_register_model,
    promote_to_champion,
    setup_mlflow,
)
from models.trainer import train_catboost_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain CatBoost model and run Challenger vs Champion evaluation."
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
        "--min-improvement",
        type=float,
        default=MIN_IMPROVEMENT_RATIO,
        help=f"Minimum relative improvement threshold for Champion replacement (default: {MIN_IMPROVEMENT_RATIO})",
    )
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="Force promote Challenger even if metric didn't beat Champion",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows loaded from database (optional)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (default: 'retrain_challenger_vs_champion')",
    )
    return parser.parse_args()


def run_retraining(
    iterations: int = 2500,
    learning_rate: float = 0.04,
    depth: int = 7,
    min_improvement_ratio: float = MIN_IMPROVEMENT_RATIO,
    force_promote: bool = False,
    limit: int | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """
    Пайплайн периодического переобучения:
    1. Загрузка актуальных данных из PostgreSQL.
    2. Поиск текущего Champion в MLflow Model Registry.
    3. Обучение Challenger (кандидата) на train.
    4. Оценка Champion и Challenger на единой отложенной тестовой выборке (Head-to-Head).
    5. Если Challenger лучше на >= min_improvement_ratio — перевод в Champion.
       Иначе — сохранение метрик с отклонением кандидата.
    """
    client, experiment_id = setup_mlflow()
    logger.info(
        f"MLflow initialized. Tracking URI: {mlflow.get_tracking_uri()}, Experiment: {MLFLOW_EXPERIMENT_NAME} (id={experiment_id})"
    )

    # 1. Поиск информации о действующем Champion
    champion_info = get_champion_info(client, MLFLOW_MODEL_NAME, MLFLOW_CHAMPION_ALIAS)
    champion_model = None
    if champion_info:
        logger.info(
            f"Found current Champion: Version {champion_info['version']} (run_id: {champion_info['run_id']})"
        )
        try:
            champion_model = load_production_model(MLFLOW_MODEL_NAME, MLFLOW_CHAMPION_ALIAS)
        except Exception as exc:
            logger.warning(f"Could not load Champion model instance: {exc}")
    else:
        logger.info("No active Champion found in Model Registry. Challenger will become Champion automatically.")

    # 2. Загрузка свежих данных
    logger.info("Loading fresh dataset from PostgreSQL...")
    raw_df = load_apartments_from_db(limit=limit)
    if raw_df.empty:
        raise ValueError("No data loaded from PostgreSQL.")

    clean_df = clean_dataset(raw_df)
    logger.info(
        f"Fresh dataset loaded and cleaned: {len(clean_df):,} records (raw: {len(raw_df):,})."
    )

    # 3. Извлечение признаков
    X, y = prepare_features(clean_df, is_training=True)
    if y is None:
        raise ValueError("Target variable 'monthly_rent' is missing!")

    custom_params = {
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
    }

    # 4. Обучение Challenger
    retrain_summary: dict[str, Any] = {}
    actual_run_name = run_name or "retrain_challenger_vs_champion"
    with mlflow.start_run(run_name=actual_run_name) as run:
        run_id = run.info.run_id
        logger.info(f"Started retrain MLflow run: {run_id}")

        mlflow.set_tags(
            {
                "pipeline": "retrain",
                "model_family": "catboost",
                "target": "monthly_rent",
                "raw_samples": len(raw_df),
                "clean_samples": len(clean_df),
                "champion_version_at_start": champion_info["version"] if champion_info else "none",
            }
        )

        challenger_res = train_catboost_model(
            X=X,
            y=y,
            params=custom_params,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        challenger_mae = challenger_res.metrics["mae"]
        challenger_mape = challenger_res.metrics["mape"]
        challenger_r2 = challenger_res.metrics["r2"]

        # Логируем параметры Challenger
        mlflow.log_params(challenger_res.params)
        mlflow.log_params(
            {
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "n_features": len(ALL_FEATURES),
                "min_improvement_threshold": min_improvement_ratio,
            }
        )

        # 5. Head-to-Head оценка против Champion на тестовом наборе
        y_test_arr = np.asarray(challenger_res.y_test)
        champion_mae = None
        champion_mape = None
        improvement_ratio = 0.0

        if champion_model is not None:
            try:
                champion_X_test = align_features_to_model(challenger_res.X_test, champion_model)
                champion_test_pred = np.asarray(champion_model.predict(champion_X_test))
                champion_mae = round(float(mean_absolute_error(y_test_arr, champion_test_pred)), 2)
                champion_mape = round(
                    float(mean_absolute_percentage_error(y_test_arr, champion_test_pred) * 100.0), 2
                )
                champion_r2 = round(float(r2_score(y_test_arr, champion_test_pred)), 4)

                champion_version = str(champion_info.get("version", "?")) if champion_info else "?"
                logger.info(f"Head-to-Head on identical test set ({len(y_test_arr):,} samples):")
                logger.info(f"  - Champion (v{champion_version}) MAE: {champion_mae:,.2f} ₽, MAPE: {champion_mape}%")
                logger.info(f"  - Challenger (new)        MAE: {challenger_mae:,.2f} ₽, MAPE: {challenger_mape}%")

                improvement_ratio = (champion_mae - challenger_mae) / champion_mae

                mlflow.log_metrics(
                    {
                        "champion_test_mae": champion_mae,
                        "champion_test_mape": champion_mape,
                        "champion_test_r2": champion_r2,
                        "mae_improvement_ratio": round(improvement_ratio, 5),
                        "mae_diff_rub": round(champion_mae - challenger_mae, 2),
                    }
                )
            except Exception as exc:
                logger.warning(f"Failed to evaluate Champion on new test set: {exc}")
                champion_mae = None

        # Логируем метрики Challenger
        mlflow.log_metrics(challenger_res.metrics)
        mlflow.log_metrics(
            {
                "train_samples": float(len(challenger_res.X_train)),
                "test_samples": float(len(challenger_res.X_test)),
            }
        )

        # Решение: принимаем ли мы Challenger?
        is_champion_beaten = False
        decision_reason = ""

        if champion_model is None or champion_mae is None:
            is_champion_beaten = True
            decision_reason = "No active Champion was present or verifiable. Challenger becomes initial Champion."
        elif force_promote:
            is_champion_beaten = True
            decision_reason = f"Force promote flag set. Improvement ratio: {improvement_ratio:.2%}"
        elif improvement_ratio >= min_improvement_ratio:
            is_champion_beaten = True
            decision_reason = (
                f"Challenger improved MAE by {improvement_ratio:.2%} "
                f"(threshold: {min_improvement_ratio:.2%}). "
                f"MAE dropped from {champion_mae:,.2f} to {challenger_mae:,.2f} ₽."
            )
        else:
            is_champion_beaten = False
            decision_reason = (
                f"Challenger did NOT beat threshold: improvement {improvement_ratio:.2%} < {min_improvement_ratio:.2%}. "
                f"Champion MAE: {champion_mae:,.2f} ₽ vs Challenger: {challenger_mae:,.2f} ₽."
            )

        # 6. Сохранение артефактов
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            fi_file = tmp_path / "feature_importances.csv"
            challenger_res.feature_importances.to_frame(name="importance").to_csv(fi_file)
            mlflow.log_artifact(str(fi_file), artifact_path="feature_analysis")

            comparison_summary = {
                "decision": "ACCEPTED" if is_champion_beaten else "REJECTED",
                "reason": decision_reason,
                "improvement_ratio": round(improvement_ratio, 5),
                "challenger_metrics": challenger_res.metrics,
                "champion_version": champion_info["version"] if champion_info else None,
                "champion_test_mae": champion_mae,
                "min_improvement_threshold": min_improvement_ratio,
            }
            summary_file = tmp_path / "challenger_evaluation.json"
            summary_file.write_text(json.dumps(comparison_summary, indent=2, ensure_ascii=False))
            mlflow.log_artifact(str(summary_file), artifact_path="evaluation")

        mlflow.set_tag("decision", "ACCEPTED" if is_champion_beaten else "REJECTED")
        mlflow.set_tag("decision_reason", decision_reason)

        # 7. Логируем модель; в реестр попадает только принятый кандидат
        logger.info(
            f"Logging candidate model (register in '{MLFLOW_MODEL_NAME}': {is_champion_beaten})..."
        )
        _, new_version = log_and_register_model(
            model=challenger_res.model,
            model_name=MLFLOW_MODEL_NAME,
            input_example=challenger_res.X_test.head(5),
            register=is_champion_beaten,
        )

        if is_champion_beaten and new_version:
            logger.success(f"PROMOTION: {decision_reason}")
            promote_to_champion(
                version=new_version,
                client=client,
                model=challenger_res.model,
                model_name=MLFLOW_MODEL_NAME,
                alias=MLFLOW_CHAMPION_ALIAS,
                reason=decision_reason,
            )
        else:
            logger.warning(f"REJECTION: {decision_reason}")
            logger.info(
                f"Current Champion v{champion_info['version'] if champion_info else 'none'} remains active."
            )

        retrain_summary = {
            "run_id": run_id,
            "challenger_version": new_version,
            "accepted": is_champion_beaten,
            "reason": decision_reason,
            "metrics": challenger_res.metrics,
            "champion_mae": champion_mae,
        }

    return retrain_summary


def main() -> None:
    args = parse_args()
    result = run_retraining(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        min_improvement_ratio=args.min_improvement,
        force_promote=args.force_promote,
        limit=args.limit,
        run_name=args.run_name,
    )

    if result["accepted"]:
        logger.success(f"Retraining successful: New model v{result['challenger_version']} deployed!")
        sys.exit(0)
    else:
        logger.info("Retraining finished: Champion remains unchanged.")
        sys.exit(0)


if __name__ == "__main__":
    main()
