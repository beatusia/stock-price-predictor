# src/feature_engineering.py
# This script adds technical indicators, calendar features, lag features, and more to stock price CSVs.

import os
from pathlib import Path

import pandas as pd
import pandas_ta as ta

# Paths
input_folder = Path(
    "/Users/beatawyspianska/Desktop/AIML_Projects/predict_stock_price/stock-price-predictor/data/raw/modified"
)
output_folder = Path(
    "/Users/beatawyspianska/Desktop/AIML_Projects/predict_stock_price/stock-price-predictor/data/processed"
)
output_folder.mkdir(parents=True, exist_ok=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Ensure 'Date' column exists and is converted to datetime
    if "Date" not in df.columns:
        if df.index.name == "Date":
            df["Date"] = df.index
            df.reset_index(drop=True, inplace=True)
        else:
            raise KeyError("No 'Date' column or index found.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isnull().all():
        raise ValueError("'Date' column could not be converted to datetime.")

    # 2. Sort by date (keep column format, not index)
    df.sort_values("Date", inplace=True)

    # 3. Check for required columns
    required_cols = ["High", "Low", "Close", "Open", "Volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ----------------------------- #
    # BASIC PRICE RELATION FEATURES
    # ----------------------------- #
    df["High-Low"] = df["High"] - df["Low"]
    df["Price-Open"] = df["Close"] - df["Open"]
    df["open_to_close_return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["high_to_close_return"] = (df["Close"] - df["High"]) / df["High"]

    # ---------------------------------------- #
    # CALENDAR / DATE FEATURES from 'Date' col
    # ---------------------------------------- #
    df["day_of_week"] = df["Date"].dt.weekday + 1
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    df["is_quarter_start"] = df["Date"].dt.is_quarter_start.astype(int)
    df["is_quarter_end"] = df["Date"].dt.is_quarter_end.astype(int)
    df["is_year_start"] = df["Date"].dt.is_year_start.astype(int)
    df["is_year_end"] = df["Date"].dt.is_year_end.astype(int)

    # -------------------------------------- #
    # VOLUME FEATURES
    # -------------------------------------- #
    df["avg_volume_20"] = df["Volume"].rolling(window=20).mean()
    df["volume_norm"] = df["Volume"] / df["avg_volume_20"]

    # VWAP: volume-weighted average price (cumulative form)
    df["vwap"] = (
        df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3
    ).cumsum() / df["Volume"].cumsum()

    # -------------------------------------- #
    # ROLLING STATS / VOLATILITY
    # -------------------------------------- #
    for window in [5, 10, 20]:
        df[f"RollingMean_{window}"] = df["Close"].rolling(window=window).mean()
        df[f"RollingStd_{window}"] = df["Close"].rolling(window=window).std()
        df[f"volatility_{window}"] = df["Close"].rolling(window=window).std()

    # -------------------------------------- #
    # MOMENTUM RETURNS & ACCELERATION
    # -------------------------------------- #
    for period in [1, 5, 10, 20]:
        df[f"Return_{period}"] = df["Close"].pct_change(period)

    for length in [5, 10, 20]:
        df[f"roc_{length}"] = ta.roc(df["Close"], length=length)
        df[f"acceleration_{length}"] = df[f"roc_{length}"].diff()

    # -------------------------------------- #
    # DIFFERENCING
    # -------------------------------------- #
    df["close_diff1"] = df["Close"].diff()
    df["close_diff2"] = df["close_diff1"].diff()

    # -------------------------------------- #
    # MOVING AVERAGES: SMA & EMA
    # -------------------------------------- #
    for length in [5, 10, 20, 50, 200]:
        df[f"sma_{length}"] = ta.sma(df["Close"], length=length)

    for length in [5, 10, 20]:
        df[f"ema_{length}"] = ta.ema(df["Close"], length=length)

    # -------------------------------------- #
    # TECHNICAL INDICATORS
    # -------------------------------------- #
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    df["adx_14"] = ta.adx(df["High"], df["Low"], df["Close"], length=14)["ADX_14"]

    # Bollinger Bands
    bbands = ta.bbands(df["Close"])
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)

    # Volume-based indicators
    df["OBV"] = ta.obv(df["Close"], df["Volume"])
    df["cmf_20"] = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"], length=20)

    # Others: ATR, CCI, Stochastic, Williams %R, PSAR
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["cci_20"] = ta.cci(df["High"], df["Low"], df["Close"], length=20)

    stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
    if stoch is not None:
        df = pd.concat([df, stoch], axis=1)

    df["williams_r"] = ta.willr(df["High"], df["Low"], df["Close"], length=14)

    psar_df = ta.psar(df["High"], df["Low"], df["Close"])
    if psar_df is not None and not psar_df.empty:
        df["psar"] = psar_df.iloc[:, 0]

    # -------------------------------------- #
    # LAG FEATURES FOR 'Close'
    # -------------------------------------- #
    for i in range(1, 21):
        df[f"Close_lag{i}"] = df["Close"].shift(i)

    # -------------------------------------- #
    # TARGET VARIABLE
    # -------------------------------------- #
    df["Target"] = df["Close"].shift(-1)

    return df


# ------------------- #
# BATCH FILE PROCESS
# ------------------- #
for file in input_folder.glob("*.csv"):
    print(f"\n🔄 Processing {file.name}...")
    try:
        df = pd.read_csv(file)
        df_features = engineer_features(df)
        output_file = output_folder / file.name
        df_features.to_csv(output_file, index=False)
        print(f"✅ Saved processed file to: {output_file}")
        print(f"📊 Final shape: {df_features.shape}")
    except Exception as e:
        print(f"❌ Error processing {file.name}: {e}")
