import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

logger = logging.getLogger(__name__)


def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    return {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE (%)": mape}


def build_results_table(results):
    table = pd.DataFrame(
        {
            name: {k: v for k, v in metrics.items() if k != "Predictions"}
            for name, metrics in results.items()
        }
    ).T.sort_values("RMSE")
    return table


def get_best_model(results, models):
    table = build_results_table(results)
    best_name = table.index[0]
    return best_name, models[best_name], results[best_name]["Predictions"]


def directional_accuracy(y_test, y_pred):
    actual_dir = np.sign(np.diff(np.asarray(y_test)))
    pred_dir = np.sign(np.diff(np.asarray(y_pred)))
    return np.mean(actual_dir == pred_dir) * 100


def regime_error_analysis(test_df, y_pred, best_name):
    eval_df = test_df[["Date", "TSLA_Close", "HighVol_Regime"]].copy()
    eval_df["Predicted"] = y_pred
    eval_df["Abs_Error"] = (eval_df["TSLA_Close"] - eval_df["Predicted"]).abs()
    eval_df["Pct_Error"] = (eval_df["Abs_Error"] / eval_df["TSLA_Close"]) * 100

    regime_stats = (
        eval_df.groupby("HighVol_Regime")
        .agg(
            Count=("Abs_Error", "count"),
            MAE=("Abs_Error", "mean"),
            Mean_Pct_Error=("Pct_Error", "mean"),
            Max_Abs_Error=("Abs_Error", "max"),
        )
        .round(4)
    )
    regime_stats.index = ["Normal Volatility", "High Volatility"]
    return regime_stats


def monthly_error_analysis(test_df, y_pred):
    eval_df = test_df[["Date", "TSLA_Close"]].copy()
    eval_df["Predicted"] = y_pred
    eval_df["Abs_Error"] = (eval_df["TSLA_Close"] - eval_df["Predicted"]).abs()
    eval_df["Pct_Error"] = (eval_df["Abs_Error"] / eval_df["TSLA_Close"]) * 100
    eval_df["YearMonth"] = eval_df["Date"].dt.to_period("M")
    monthly_err = (
        eval_df.groupby("YearMonth")
        .agg(
            MAE=("Abs_Error", "mean"),
            MAPE=("Pct_Error", "mean"),
            Count=("Abs_Error", "count"),
        )
        .round(2)
    )
    return monthly_err
