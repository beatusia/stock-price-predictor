# src/00_indicies_download.py
# This script downloads SP500 and NASDAQ data from Yahoo Finance, cleans them,
# adds lag features and returns, and saves processed versions.

import os
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

# Constants
INDEX_TICKERS = {"SP500": "^GSPC", "NASDAQ": "^IXIC"}

START_DATE = "1999-12-01"
END_DATE = date.today().strftime("%Y-%m-%d")

# Updated base directory for benchmark index data
BENCHMARK_DIR = Path(
    "/Users/beatawyspianska/Desktop/AIML_Projects/predict_stock_price/stock-price-predictor/data/benchmark"
)
TRUE_RAW_DIR = BENCHMARK_DIR / "true_raw"
MODIFIED_DIR = BENCHMARK_DIR / "modified"
PROCESSED_DIR = BENCHMARK_DIR / "processed"

# Create necessary directories
TRUE_RAW_DIR.mkdir(parents=True, exist_ok=True)
MODIFIED_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_index_data():
    for name, ticker in INDEX_TICKERS.items():
        print(f"📥 Downloading {name} ({ticker}) from Yahoo Finance...")
        df = yf.download(ticker, start=START_DATE, end=END_DATE)
        if df.empty:
            print(f"⚠️ No data found for {name}")
            continue
        out_path = TRUE_RAW_DIR / f"{name}.csv"
        df.to_csv(out_path)
        print(f"✅ Saved raw data to: {out_path}")


def clean_and_format_index_data():
    for file in os.listdir(TRUE_RAW_DIR):
        if file.endswith(".csv") and file.replace(".csv", "") in INDEX_TICKERS:
            input_path = os.path.join(TRUE_RAW_DIR, file)

            df = pd.read_csv(input_path, index_col=0)
            df = df.iloc[2:].copy()

            ticker = os.path.splitext(file)[0]
            df["TICKER"] = ticker

            df.insert(0, "Date", df.index)
            df.reset_index(drop=True, inplace=True)

            # ✅ Include Adj Close
            desired_order = [
                "Date",
                "TICKER",
                "Adj Close",
                "Close",
                "High",
                "Low",
                "Open",
                "Volume",
            ]
            df = df[[col for col in desired_order if col in df.columns]]

            output_path = os.path.join(MODIFIED_DIR, file)
            df.to_csv(output_path, index=False)
            print(f"🧹 Cleaned and saved: {output_path}")


def engineer_index_features():
    for file in os.listdir(MODIFIED_DIR):
        if file.endswith(".csv") and file.replace(".csv", "") in INDEX_TICKERS:
            file_path = os.path.join(MODIFIED_DIR, file)
            df = pd.read_csv(file_path)
            df["Date"] = pd.to_datetime(df["Date"])
            df.sort_values("Date", inplace=True)

            ticker_key = os.path.splitext(file)[0]
            close_col = "Close"

            # Define consistent periods for returns and lags
            return_periods = [1, 5, 7, 10, 14, 20, 30, 60]
            lag_periods = return_periods

            # Create return features
            for period in return_periods:
                df[f"{ticker_key}_return_{period}"] = df[close_col].pct_change(period)

            # Create lag features
            for i in lag_periods:
                df[f"{ticker_key}_lag_{i}"] = df[close_col].shift(i)

            # Reorder columns logically
            base_cols = [
                "Date",
                "TICKER",
                "Close",
                "Adj Close",
                "High",
                "Low",
                "Open",
                "Volume",
            ]
            return_cols = [f"{ticker_key}_return_{i}" for i in return_periods]
            lag_cols = [f"{ticker_key}_lag_{i}" for i in lag_periods]

            df = df[
                [col for col in base_cols if col in df.columns]
                + [col for col in return_cols if col in df.columns]
                + [col for col in lag_cols if col in df.columns]
            ]

            output_path = PROCESSED_DIR / file
            df.to_csv(output_path, index=False)
            print(f"🚀 Engineered and saved features to: {output_path}")


if __name__ == "__main__":
    download_index_data()
    clean_and_format_index_data()
    engineer_index_features()
