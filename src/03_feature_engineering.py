# src/03_feature_engineering.py
# This script adds technical indicators, calendar features, lag features, and more to stock price CSVs.

from pathlib import Path

import numpy as np
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

    # VWAP: volume-weighted average price (rolling 20 days)
    df["rolling_vwap_20"] = (
        (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).rolling(20).sum()
        / df["Volume"].rolling(20).sum()
    ).fillna(0)

    # -------------------------------------- #
    # ROLLING STATS / VOLATILITY
    # -------------------------------------- #
    for window in [5, 10, 20, 30, 60, 90, 180]:
        df[f"RollingMean_{window}"] = df["Close"].rolling(window=window).mean()
        df[f"RollingStd_{window}"] = df["Close"].rolling(window=window).std()

    # -------------------------------------- #
    # MOMENTUM RETURNS & ACCELERATION
    # -------------------------------------- #
    for period in [1, 5, 10, 20]:
        df[f"Return_{period}"] = df["Close"].pct_change(period)

    for length in [7, 14, 36]:
        df[f"roc_{length}"] = ta.roc(df["Close"], length=length)
        df[f"acceleration_{length}"] = df[f"roc_{length}"].diff()

    # Long-term momentum: rate of change over longer periods
    df["long_roc_30"] = ta.roc(df["Close"], length=30)
    df["long_roc_60"] = ta.roc(df["Close"], length=60)
    df["long_roc_90"] = ta.roc(df["Close"], length=90)

    # Momentum: relative to 10 and 20 days ago
    df["Momentum_10d"] = df["Close"] / df.groupby("TICKER")["Close"].shift(10) - 1
    df["Momentum_20d"] = df["Close"] / df.groupby("TICKER")["Close"].shift(20) - 1

    # -------------------------------------- #
    # DIFFERENCING
    # -------------------------------------- #
    df["close_diff1"] = df["Close"].diff()
    df["close_diff2"] = df["close_diff1"].diff()

    # -------------------------------------- #
    # MOVING AVERAGES: SMA & EMA
    # -------------------------------------- #
    for length in [5, 10, 20, 50, 200]:
        df[f"SMA_{length}"] = ta.sma(df["Close"], length=length)

    df["SMA_5_minus_SMA_20"] = df["SMA_5"] - df["SMA_20"]
    df["SMA_50_minus_SMA_200"] = df["SMA_50"] - df["SMA_200"]
    df["price_over_SMA_200"] = df["Close"] / df["SMA_200"] - 1

    for length in [5, 20, 50, 200]:
        df[f"EMA_{length}"] = ta.ema(df["Close"], length=length)

    # -------------------------------------- #
    # TECHNICAL INDICATORS
    # -------------------------------------- #
    # Add ADX (Average Directional Index)
    df["ADX_14"] = ta.adx(df["High"], df["Low"], df["Close"], length=14)["ADX_14"]

    # RSI
    df["RSI_14"] = ta.rsi(df["Close"], length=14)

    # Bollinger Bands
    bbands = ta.bbands(df["Close"])
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)

    # Add Bollinger Bandwidth and Long Volatility for longer consolidation
    bb_50 = ta.bbands(df["Close"], length=50)
    if bb_50 is not None:
        df["bb_bandwidth_50"] = (bb_50["BBU_50_2.0"] - bb_50["BBL_50_2.0"]) / bb_50[
            "BBM_50_2.0"
        ]

    df["long_volatility_50"] = df["Close"].rolling(50).std()

    # Add MACD features
    macd_df = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        df = pd.concat([df, macd_df], axis=1)

    df["macd_diff"] = df["MACDh_12_26_9"]
    df["macd_signal"] = df["MACDs_12_26_9"]  # Signal line

    # Add Stochastic Oscillator features
    stoch_df = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
    if stoch_df is not None and not stoch_df.empty:
        df = pd.concat([df, stoch_df], axis=1)

    # Volume-based indicators
    df["OBV"] = ta.obv(df["Close"], df["Volume"])
    df["cmf_20"] = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"], length=20)

    # Others: ATR, CCI, Stochastic, Williams %R
    # ATR
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df.groupby("TICKER")["Close"].shift(1))
    low_close = np.abs(df["Low"] - df.groupby("TICKER")["Close"].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["ATR_14"] = true_range.groupby(df["TICKER"]).transform(
        lambda x: x.rolling(14).mean()
    )

    # CCI (Commodity Channel Index)
    df["cci_20"] = ta.cci(df["High"], df["Low"], df["Close"], length=20)

    # Williams %R
    df["williams_r"] = ta.willr(df["High"], df["Low"], df["Close"], length=14)

    # -------------------------------------- #
    # SWING DIRECTION LABEL
    # -------------------------------------- #

    # Define thresholds (e.g., 0.5% swing range)
    up_threshold = 0.005
    down_threshold = -0.005

    # Calculate daily return
    df["daily_return"] = df["Close"].pct_change()

    # Label direction
    def classify_swing(r):
        if pd.isna(r):
            return "grind"  # or np.nan
        elif r > up_threshold:
            return "up"
        elif r < down_threshold:
            return "down"
        else:
            return "grind"

    df["swing_direction"] = df["daily_return"].apply(classify_swing)

    # -------------------------------------- #
    # MULTI-DAY SWING TREND DETECTION
    # -------------------------------------- #

    trend_window = 5  # lookback period in days
    up_threshold = 0.01  # +1% over 5 days → uptrend
    down_threshold = -0.01  # -1% over 5 days → downtrend
    grind_volatility_threshold = 0.005  # max relative swing to be called sideways

    # Calculate rolling return
    df["trend_return"] = (df["Close"] - df["Close"].shift(trend_window)) / df[
        "Close"
    ].shift(trend_window)

    # Calculate relative price range over the window
    df["trend_range"] = (
        df["Close"].rolling(window=trend_window).max()
        - df["Close"].rolling(window=trend_window).min()
    ) / df["Close"].rolling(window=trend_window).mean()

    # Classify swing trend
    def classify_trend(row):
        r = row["trend_return"]
        v = row["trend_range"]

        if pd.isna(r) or pd.isna(v):
            return "unknown"
        elif abs(r) < up_threshold and v < grind_volatility_threshold:
            return "sideways"
        elif r >= up_threshold:
            return "uptrend"
        elif r <= down_threshold:
            return "downtrend"
        else:
            return "sideways"

    df["swing_trend"] = df.apply(classify_trend, axis=1)

    # -------------------------------------- #
    # TREND DURATION AND TREND IDs
    # -------------------------------------- #

    # Shift trend to detect changes
    df["swing_trend_shift"] = df["swing_trend"].shift(1)
    df["trend_change"] = df["swing_trend"] != df["swing_trend_shift"]

    # Assign a unique ID to each trend
    df["trend_id"] = df["trend_change"].cumsum()

    # Calculate trend duration for each row
    df["trend_duration"] = df.groupby("trend_id").cumcount() + 1

    # Flag when a new trend starts (e.g., for signal generation)
    df["is_trend_start"] = df["trend_duration"] == 1

    # -------------------------------------- #
    # W.D Gann's Features
    # -------------------------------------- #

    # 3 Consecutive Days Up/Down based on Close
    df["3_consecutive_up"] = (
        (df["Close"] > df["Close"].shift(1)).astype(int).rolling(window=3).sum()
    )
    df["3_consecutive_down"] = (
        (df["Close"] < df["Close"].shift(1)).astype(int).rolling(window=3).sum()
    )
    df["3_consecutive_up"] = df["3_consecutive_up"].fillna(0).astype(int)
    df["3_consecutive_down"] = df["3_consecutive_down"].fillna(0).astype(int)

    # 6-11 Days Up/Down based on Close
    for n in range(6, 12):
        df[f"{n}_up"] = (
            df["Close"]
            .rolling(window=n)
            .apply(lambda x: all(x[i] < x[i + 1] for i in range(n - 1)), raw=True)
            .fillna(0)
            .astype(int)
        )
        df[f"{n}_down"] = (
            df["Close"]
            .rolling(window=n)
            .apply(lambda x: all(x[i] > x[i + 1] for i in range(n - 1)), raw=True)
            .fillna(0)
            .astype(int)
        )

    # 3 Consecutive Closes at the Same Price
    df["same_close_3"] = (
        (df["Close"] == df["Close"].shift(1)) & (df["Close"] == df["Close"].shift(2))
    ).astype(int)
    df["same_close_3"] = df["same_close_3"].fillna(0).astype(int)

    # 3 Consecutive Closes at the Same Price (with 1% tolerance)
    df["same_close_3_tolerance"] = (
        (df["Close"].between(df["Close"].shift(1) * 0.99, df["Close"].shift(1) * 1.01))
        & (
            df["Close"].between(
                df["Close"].shift(2) * 0.99, df["Close"].shift(2) * 1.01
            )
        )
    ).astype(int)

    # Gann Seasonanl Dates
    gann_dates = [
        "02-04",
        "03-21",
        "05-06",
        "06-22",
        "08-08",
        "09-23",
        "11-07",
        "12-22",
    ]
    df["Date"] = pd.to_datetime(df["Date"])
    df["is_gann_date"] = df["Date"].dt.strftime("%m-%d").isin(gann_dates).astype(int)

    # -------------------------------------- #
    # Trend Decay Diagnostics
    # -------------------------------------- #
    df["days_since_high"] = (
        df["High"].expanding().apply(lambda x: len(x) - x.argmax() - 1)
    ).astype(int)

    df["days_since_low"] = (
        df["Low"].expanding().apply(lambda x: len(x) - x.argmin() - 1)
    ).astype(int)

    # long-term trend decay
    df["days_since_high_50"] = (
        df["High"].rolling(50, min_periods=1).apply(lambda x: len(x) - x.argmax() - 1)
    )

    df["days_since_low_50"] = (
        df["Low"].rolling(50, min_periods=1).apply(lambda x: len(x) - x.argmax() - 1)
    )

    # -------------------------------------- #
    # LAG FEATURES FOR 'Close'
    # -------------------------------------- #
    for i in [1, 5, 10, 20]:
        df[f"Close_lag{i}"] = df["Close"].shift(i)

    # -------------------------------------- #
    # TARGET VARIABLE
    # -------------------------------------- #
    df["Target_Raw_Close_daily"] = df["Close"].shift(-1)
    df["Target_Log_Return_daily"] = np.log(df["Close"].shift(-1) / df["Close"])
    df["Target_%_Return_daily"] = (df["Close"].shift(-1) - df["Close"]) / df["Close"]
    df["Target_Direction_daily"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Longer-term target variables
    df["Target_Raw_Close_7d"] = df["Close"].shift(-7)
    df["Target_Raw_Close_14d"] = df["Close"].shift(-14)
    df["Target_Raw_Close_21d"] = df["Close"].shift(-21)

    for horizon in [7, 14, 21]:
        # Raw future price
        df[f"Target_Raw_Close_{horizon}d"] = df["Close"].shift(-horizon)
        # % Return
        df[f"Target_%_Return_{horizon}d"] = (
            df["Close"].shift(-horizon) - df["Close"]
        ) / df["Close"]
        # Log Return
        df[f"Target_Log_Return_{horizon}d"] = np.log(
            df["Close"].shift(-horizon) / df["Close"]
        )
        # Direction: 1 if up, 0 if not
        df[f"Target_Direction_{horizon}d"] = (
            df["Close"].shift(-horizon) > df["Close"]
        ).astype(int)

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
