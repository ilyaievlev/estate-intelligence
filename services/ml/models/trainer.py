from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TEST_SIZE
from features.engineering import CATEGORICAL_FEATURES


def train_catboost_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any] | None = None,
) -> tuple[CatBoostRegressor, dict[str, float], pd.Series]:
    """
    Обучение модели CatBoostRegressor на признаках с валидацией на тесте.
    Возвращает (модель, словарь_метрик, важность_признаков).
    """
    default_params = {
        "iterations": 2500,
        "learning_rate": 0.04,
        "depth": 7,
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "random_seed": RANDOM_STATE,
        "early_stopping_rounds": 150,
        "verbose": 200,
        "allow_writing_files": False,
    }
    model_params = {**default_params, **(params or {})}

    # Разбиение 80% train / 20% test
    X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
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
        random_state=RANDOM_STATE,
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
    y_pred = model.predict(X_test)
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

    return model, metrics, feature_importances
