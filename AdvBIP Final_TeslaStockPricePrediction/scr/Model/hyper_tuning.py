"""Hyperparameter tuning for all models using GridSearchCV with TimeSeriesSplit."""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

logger = logging.getLogger(__name__)

TREE_MODELS = {"Random Forest", "Gradient Boosting", "XGBoost"}

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("xgboost not installed – XGBoost tuning will be skipped.")


# ── Parameter grids ──────────────────────────────────────────────────
def get_param_grids():
    """Return a dict of model name → (estimator, param_grid)."""
    grids = {
        "Ridge Regression": (
            Ridge(),
            {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        ),
        "Lasso Regression": (
            Lasso(max_iter=10000),
            {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            },
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.8, 1.0],
            },
        ),
    }

    if HAS_XGB:
        grids["XGBoost"] = (
            XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        )

    return grids


# ── Single model tuning ──────────────────────────────────────────────
def tune_model(name, estimator, param_grid, X_train, y_train, n_splits=5, scoring="r2"):
    """Run GridSearchCV for one model and return the best estimator + results dict."""
    logger.info("Tuning %s (%d combinations) ...", name, _grid_size(param_grid))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    grid = GridSearchCV(
        estimator,
        param_grid,
        cv=tscv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    grid.fit(X_train, y_train)

    logger.info(
        "%s  best %s=%.4f  params=%s",
        name,
        scoring,
        grid.best_score_,
        grid.best_params_,
    )

    return {
        "best_estimator": grid.best_estimator_,
        "best_score": grid.best_score_,
        "best_params": grid.best_params_,
        "cv_results": grid.cv_results_,
    }


# ── Tune all models ──────────────────────────────────────────────────
def tune_all_models(X_train, y_train, X_train_scaled, n_splits=5, scoring="r2"):
    """Tune every model in the param-grid dict.

    Tree-based models use unscaled features; linear models use scaled features.

    Returns
    -------
    tuned_models : dict
        name → best fitted estimator
    summary_df : pd.DataFrame
        One row per model with best score & params.
    """
    grids = get_param_grids()
    tuned_models = {}
    summary_rows = []

    for name, (estimator, param_grid) in grids.items():
        if name in TREE_MODELS:
            X = X_train
        else:
            X = X_train_scaled

        result = tune_model(name, estimator, param_grid, X, y_train, n_splits, scoring)
        tuned_models[name] = result["best_estimator"]
        summary_rows.append(
            {
                "Model": name,
                "Best CV R²": result["best_score"],
                "Best Params": result["best_params"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("Best CV R²", ascending=False)
    logger.info("\n%s", summary_df.to_string(index=False))
    return tuned_models, summary_df


# ── Helper ────────────────────────────────────────────────────────────
def _grid_size(param_grid):
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size
