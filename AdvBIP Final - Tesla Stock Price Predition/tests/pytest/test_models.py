"""Tests for scr.Model.train_models, predict_models, and hyper_tuning modules."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scr.Model.train_models import (
    get_models,
    save_models,
    load_model,
    TREE_MODELS,
    MODELS_DIR,
)
from scr.Model.predict_models import (
    evaluate_model,
    build_results_table,
    get_best_model,
    directional_accuracy,
    regime_error_analysis,
    monthly_error_analysis,
)
from scr.Model.hyper_tuning import get_param_grids, tune_model, _grid_size


# ── train_models tests ───────────────────────────────────────────────
class TestGetModels:
    """Tests for get_models()."""

    def test_returns_dict(self):
        models = get_models()
        assert isinstance(models, dict)

    def test_contains_at_least_five_models(self):
        models = get_models()
        assert len(models) >= 5

    def test_expected_model_names(self):
        models = get_models()
        expected = {
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Random Forest",
            "Gradient Boosting",
        }
        assert expected.issubset(set(models.keys()))

    def test_all_models_have_fit_predict(self):
        models = get_models()
        for name, model in models.items():
            assert hasattr(model, "fit"), f"{name} missing fit()"
            assert hasattr(model, "predict"), f"{name} missing predict()"


class TestTrainAllModels:
    """Tests for train_all_models() results."""

    def test_returns_dict(self, trained_models):
        _, results = trained_models
        assert isinstance(results, dict)

    def test_all_models_have_results(self, trained_models):
        models, results = trained_models
        assert set(results.keys()) == set(models.keys())

    def test_results_have_required_metrics(self, trained_models):
        _, results = trained_models
        required = {"MAE", "RMSE", "R²", "MAPE (%)", "Predictions"}
        for name, metrics in results.items():
            assert required.issubset(set(metrics.keys())), f"{name} missing metrics"

    def test_r2_positive_for_linear_models(self, trained_models):
        _, results = trained_models
        for name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
            if name in results:
                assert results[name]["R²"] > 0.5, f"{name} R² too low"

    def test_predictions_have_correct_length(self, trained_models, split_data_fixture):
        _, results = trained_models
        _, X_test, _, y_test, _, _ = split_data_fixture
        for name, metrics in results.items():
            assert len(metrics["Predictions"]) == len(
                y_test
            ), f"{name} prediction count mismatch"

    def test_mae_and_rmse_positive(self, trained_models):
        _, results = trained_models
        for name, metrics in results.items():
            assert metrics["MAE"] > 0, f"{name} MAE not positive"
            assert metrics["RMSE"] > 0, f"{name} RMSE not positive"
            assert metrics["RMSE"] >= metrics["MAE"], f"{name} RMSE < MAE"


class TestSaveLoadModels:
    """Tests for save_models() and load_model()."""

    def test_save_creates_pkl_files(self, trained_models, tmp_path):
        models, _ = trained_models
        save_models(models, output_dir=tmp_path)
        pkl_files = list(tmp_path.glob("*.pkl"))
        assert len(pkl_files) == len(models)

    def test_load_model_returns_fitted(self, trained_models, tmp_path):
        models, _ = trained_models
        save_models(models, output_dir=tmp_path)
        loaded = load_model("Ridge Regression", model_dir=tmp_path)
        assert hasattr(loaded, "predict")

    def test_pkl_files_exist_in_models_dir(self):
        pkl_files = list(MODELS_DIR.glob("*.pkl"))
        assert len(pkl_files) >= 5, f"Only {len(pkl_files)} model files in models/"


# ── predict_models tests ─────────────────────────────────────────────
class TestEvaluateModel:
    """Tests for evaluate_model()."""

    def test_returns_dict_with_metrics(self):
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        result = evaluate_model(y_true, y_pred)
        assert set(result.keys()) == {"MAE", "RMSE", "R²", "MAPE (%)"}

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        result = evaluate_model(y, y)
        assert result["MAE"] == 0.0
        assert result["RMSE"] == 0.0
        assert result["R²"] == 1.0
        assert result["MAPE (%)"] == 0.0


class TestBuildResultsTable:
    """Tests for build_results_table()."""

    def test_returns_dataframe(self, trained_models):
        _, results = trained_models
        table = build_results_table(results)
        assert isinstance(table, pd.DataFrame)

    def test_sorted_by_rmse(self, trained_models):
        _, results = trained_models
        table = build_results_table(results)
        rmse_vals = table["RMSE"].values
        assert (rmse_vals[1:] >= rmse_vals[:-1]).all()

    def test_no_predictions_column(self, trained_models):
        _, results = trained_models
        table = build_results_table(results)
        assert "Predictions" not in table.columns


class TestGetBestModel:
    """Tests for get_best_model()."""

    def test_returns_three_elements(self, trained_models):
        models, results = trained_models
        name, model, preds = get_best_model(results, models)
        assert isinstance(name, str)
        assert hasattr(model, "predict")
        assert isinstance(preds, np.ndarray)

    def test_best_model_has_lowest_rmse(self, trained_models):
        models, results = trained_models
        name, _, _ = get_best_model(results, models)
        table = build_results_table(results)
        assert name == table.index[0]


class TestDirectionalAccuracy:
    """Tests for directional_accuracy()."""

    def test_perfect_direction(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        acc = directional_accuracy(y_true, y_pred)
        assert acc == 100.0

    def test_returns_percentage(self, trained_models, split_data_fixture):
        models, results = trained_models
        _, _, _, y_test, _, _ = split_data_fixture
        _, _, preds = get_best_model(results, models)
        acc = directional_accuracy(y_test, preds)
        assert 0 <= acc <= 100


class TestRegimeErrorAnalysis:
    """Tests for regime_error_analysis()."""

    def test_returns_dataframe(self, trained_models, split_data_fixture):
        models, results = trained_models
        _, _, _, _, _, test_df = split_data_fixture
        name, _, preds = get_best_model(results, models)
        regime = regime_error_analysis(test_df, preds, name)
        assert isinstance(regime, pd.DataFrame)
        assert len(regime) == 2
        assert "MAE" in regime.columns


class TestMonthlyErrorAnalysis:
    """Tests for monthly_error_analysis()."""

    def test_returns_dataframe(self, trained_models, split_data_fixture):
        models, results = trained_models
        _, _, _, _, _, test_df = split_data_fixture
        _, _, preds = get_best_model(results, models)
        monthly = monthly_error_analysis(test_df, preds)
        assert isinstance(monthly, pd.DataFrame)
        assert "MAE" in monthly.columns
        assert "MAPE" in monthly.columns


# ── hyper_tuning tests ───────────────────────────────────────────────
class TestGetParamGrids:
    """Tests for get_param_grids()."""

    def test_returns_dict(self):
        grids = get_param_grids()
        assert isinstance(grids, dict)

    def test_contains_expected_models(self):
        grids = get_param_grids()
        expected = {
            "Ridge Regression",
            "Lasso Regression",
            "Random Forest",
            "Gradient Boosting",
        }
        assert expected.issubset(set(grids.keys()))

    def test_each_entry_has_estimator_and_grid(self):
        grids = get_param_grids()
        for name, (estimator, param_grid) in grids.items():
            assert hasattr(estimator, "fit"), f"{name} estimator missing fit()"
            assert isinstance(param_grid, dict), f"{name} param_grid not a dict"
            assert len(param_grid) > 0, f"{name} empty param grid"


class TestGridSize:
    """Tests for _grid_size()."""

    def test_calculates_correctly(self):
        grid = {"a": [1, 2, 3], "b": [4, 5]}
        assert _grid_size(grid) == 6

    def test_single_param(self):
        grid = {"alpha": [0.01, 0.1, 1.0]}
        assert _grid_size(grid) == 3


class TestTuneModel:
    """Tests for tune_model() — uses Ridge only (fast)."""

    def test_tune_ridge(self, split_data_fixture, scaled_data):
        from sklearn.linear_model import Ridge

        _, _, y_train, _, _, _ = split_data_fixture
        X_train_scaled, _, _ = scaled_data
        result = tune_model(
            "Ridge Regression",
            Ridge(),
            {"alpha": [0.1, 1.0, 10.0]},
            X_train_scaled,
            y_train,
            n_splits=3,
        )
        assert "best_estimator" in result
        assert "best_score" in result
        assert "best_params" in result
        assert result["best_score"] > 0
