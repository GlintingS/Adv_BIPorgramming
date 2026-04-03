import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from scr.data.data_download import download_all, merge_raw, START_DATE, END_DATE

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_COLS = [
    "SP500_Close",
    "NASDAQ_Close",
    "VIX_Close",
    "Fed_Funds_Rate",
    "CPI",
    "Unemployment_Rate",
    "TSLA_Daily_Return",
    "SP500_Daily_Return",
    "NASDAQ_Daily_Return",
    "TSLA_Return_Lag1",
    "TSLA_Return_Lag3",
    "TSLA_MA_7",
    "TSLA_MA_30",
    "TSLA_Vol_30",
    "Month",
    "Quarter",
    "Year",
    "HighVol_Regime",
]
TARGET_COL = "TSLA_Close"


# ── Phase 2: engineer features ───────────────────────────────────────
def _engineer_features(merged):
    df = merged.copy()

    # Daily returns
    df["TSLA_Daily_Return"] = df["TSLA_Close"].pct_change() * 100
    df["SP500_Daily_Return"] = df["SP500_Close"].pct_change() * 100
    df["NASDAQ_Daily_Return"] = df["NASDAQ_Close"].pct_change() * 100

    # Calendar
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Year"] = df["Date"].dt.year

    # Lag & rolling features
    df["TSLA_Return_Lag1"] = df["TSLA_Daily_Return"].shift(1)
    df["TSLA_Return_Lag3"] = df["TSLA_Daily_Return"].shift(3)
    df["TSLA_MA_7"] = df["TSLA_Close"].rolling(7).mean()
    df["TSLA_MA_30"] = df["TSLA_Close"].rolling(30).mean()
    df["TSLA_Vol_30"] = df["TSLA_Daily_Return"].rolling(30).std()

    vol_threshold = df["TSLA_Vol_30"].median(skipna=True)
    df["HighVol_Regime"] = (df["TSLA_Vol_30"] > vol_threshold).astype(int)

    logger.info("Feature engineering complete: %s", df.shape)
    return df


# ── Public entry point (same interface as before) ─────────────────────
def load_phase2_data(start=START_DATE, end=END_DATE, fred_api_key=None):
    logger.info(
        "Downloading & saving data (external → data/external/, merged → data/raw/) ..."
    )
    datasets = download_all(start, end, fred_api_key=fred_api_key)
    merged = merge_raw(datasets)
    df = _engineer_features(merged)
    df = df.sort_values("Date").reset_index(drop=True)
    logger.info(
        "Dataset ready: %d rows, date range %s to %s",
        len(df),
        df["Date"].min().date(),
        df["Date"].max().date(),
    )
    return df


def prepare_features(df, feature_cols=None, target_col=None):
    feature_cols = feature_cols or FEATURE_COLS
    target_col = target_col or TARGET_COL

    subset = df[["Date"] + feature_cols + [target_col]]
    nan_cols = {c: int(subset[c].isna().sum()) for c in subset.columns if subset[c].isna().any()}
    if nan_cols:
        logger.info("NaN counts before dropna: %s", nan_cols)

    model_df = subset.dropna().reset_index(drop=True)
    dropped = len(df) - len(model_df)
    logger.info("Rows after dropping NaN: %d (dropped %d)", len(model_df), dropped)

    if len(model_df) == 0:
        all_nan = [c for c in feature_cols + [target_col] if df[c].isna().all()]
        raise ValueError(
            f"All {len(df)} rows dropped by dropna(). "
            f"Columns that are entirely NaN: {all_nan}"
        )

    # Save cleaned & processed dataset
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "tesla_processed_dataset.csv"
    model_df.to_csv(out_path, index=False)
    logger.info("Saved processed dataset → %s (%d rows)", out_path, len(model_df))

    return model_df


def split_data(model_df, feature_cols=None, target_col=None, test_size=0.20):
    feature_cols = feature_cols or FEATURE_COLS
    target_col = target_col or TARGET_COL
    split_idx = int(len(model_df) * (1 - test_size))

    train_df = model_df.iloc[:split_idx].copy()
    test_df = model_df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    logger.info(
        "Train: %d rows (%s to %s)",
        len(train_df),
        train_df["Date"].min().date(),
        train_df["Date"].max().date(),
    )
    logger.info(
        "Test : %d rows (%s to %s)",
        len(test_df),
        test_df["Date"].min().date(),
        test_df["Date"].max().date(),
    )
    return X_train, X_test, y_train, y_test, train_df, test_df


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Feature scaling complete.")
    return X_train_scaled, X_test_scaled, scaler
