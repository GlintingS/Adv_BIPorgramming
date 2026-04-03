import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("xgboost not installed – XGBoost model will be skipped.")


def get_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=10000),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
    return models


TREE_MODELS = {"Random Forest", "Gradient Boosting", "XGBoost"}


def train_all_models(
    models, X_train, y_train, X_train_scaled, X_test, X_test_scaled, y_test
):
    results = {}
    for name, model in models.items():
        if name in TREE_MODELS:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2,
            "MAPE (%)": mape,
            "Predictions": y_pred,
        }
        logger.info(
            "%s  MAE=%.2f  RMSE=%.2f  R²=%.4f  MAPE=%.2f%%", name, mae, rmse, r2, mape
        )

    return results


def save_models(models, output_dir=None):
    output_dir = Path(output_dir) if output_dir else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        safe_name = name.replace(" ", "_").lower()
        path = output_dir / f"{safe_name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Saved %s to %s", name, path)


def load_model(name, model_dir=None):
    model_dir = Path(model_dir) if model_dir else MODELS_DIR
    safe_name = name.replace(" ", "_").lower()
    path = model_dir / f"{safe_name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def cross_validate_models(models, X_train, y_train, X_train_scaled, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = []
    for name, model in models.items():
        if name in TREE_MODELS:
            scores = cross_val_score(
                model, X_train, y_train, cv=tscv, scoring="r2", n_jobs=-1
            )
        else:
            scores = cross_val_score(
                model, X_train_scaled, y_train, cv=tscv, scoring="r2", n_jobs=-1
            )
        cv_results.append(
            {"Model": name, "CV R² Mean": scores.mean(), "CV R² Std": scores.std()}
        )
        logger.info("%s  CV R² = %.4f ± %.4f", name, scores.mean(), scores.std())
    return pd.DataFrame(cv_results).sort_values("CV R² Mean", ascending=False)
