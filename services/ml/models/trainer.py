from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from config import RANDOM_STATE, TEST_SIZE
from features.engineering import CATEGORICAL_FEATURES


@dataclass
class TrainResult:
    """Результат обучения модели CatBoost."""

    model: CatBoostRegressor
    metrics: dict[str, float]
    feature_importances: pd.Series
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    params: dict[str, Any]

    def __iter__(self):
        """Поддержка распаковки кортежа: model, metrics, importances = result"""
        return iter((self.model, self.metrics, self.feature_importances))


def train_catboost_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any] | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> TrainResult:
    """
    Обучение модели CatBoostRegressor на признаках с валидацией на тесте.
    Возвращает TrainResult с моделью, метриками, важностью признаков и тестовыми выборками.
    """
    default_params = {
        "iterations": 2500,
        "learning_rate": 0.04,
        "depth": 7,
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "random_seed": random_state,
        "early_stopping_rounds": 150,
        "verbose": 200,
        "allow_writing_files": False,
    }
    model_params = {**default_params, **(params or {})}

    # Разбиение train / test
    X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    X_train = pd.DataFrame(X_train_df)
    X_test = pd.DataFrame(X_test_df)
    y_train = pd.Series(y_train_s)
    y_test = pd.Series(y_test_s)

    # Дополнительно выделяем validation выборку из train для early stopping
    X_tr_df, X_val_df, y_tr_s, y_val_s = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=random_state,
    )
    X_tr = pd.DataFrame(X_tr_df)
    X_val = pd.DataFrame(X_val_df)
    y_tr = pd.Series(y_tr_s)
    y_val = pd.Series(y_val_s)

    logger.info(
        f"Training CatBoost on {len(X_tr):,} samples, val={len(X_val):,}, test={len(X_test):,}..."
    )

    model = CatBoostRegressor(
        cat_features=CATEGORICAL_FEATURES,
        **model_params,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    # Оценка на отложенном тесте
    y_pred = np.asarray(model.predict(X_test))
    y_test_arr = np.asarray(y_test)

    mae = float(mean_absolute_error(y_test_arr, y_pred))
    mape = float(mean_absolute_percentage_error(y_test_arr, y_pred) * 100.0)
    r2 = float(r2_score(y_test_arr, y_pred))

    # Бейзлайн по медиане для сравнения
    baseline_val = float(np.median(np.asarray(y_train)))
    baseline_pred = np.full(len(y_test_arr), baseline_val)
    baseline_mae = float(mean_absolute_error(y_test_arr, baseline_pred))

    metrics = {
        "mae": round(mae, 2),
        "mape": round(mape, 2),
        "r2": round(r2, 4),
        "baseline_mae": round(baseline_mae, 2),
        "improvement_pct": round(((baseline_mae - mae) / baseline_mae) * 100.0, 2),
    }

    feature_importances = pd.Series(
        model.get_feature_importance(),
        index=X.columns,
    ).sort_values(ascending=False)

    return TrainResult(
        model=model,
        metrics=metrics,
        feature_importances=feature_importances,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        params=model_params,
    )
