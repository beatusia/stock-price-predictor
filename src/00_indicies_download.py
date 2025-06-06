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

            # ✅ Read CSV and always drop the first two rows
            df = pd.read_csv(input_path, index_col=0)
            df = df.iloc[2:].copy()

            ticker = os.path.splitext(file)[0]
            df["TICKER"] = ticker

            # Move index to column
            df.insert(0, "Date", df.index)
            df.reset_index(drop=True, inplace=True)

            # ✅ Keep column naming consistent with the rest of the pipeline
            desired_order = ["Date", "TICKER", "Close", "High", "Low", "Open", "Volume"]
            df = df[[col for col in desired_order if col in df.columns]]

            # Save to modified dir
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

            # Define ticker based on filename
            ticker = os.path.splitext(file)[0]

            # Return features: % change over 1, 5, 10, 20 days
            for period in [1, 5, 10, 20]:
                df[f"{ticker}_return_{period}"] = df["Close"].pct_change(period)

            # Lag features for Close price (1–20 days)
            for i in range(1, 21):
                df[f"{ticker}_lag_{i}"] = df["Close"].shift(i)

            # Reorder columns logically
            base_cols = ["Date", "TICKER", "Close", "High", "Low", "Open", "Volume"]
            return_cols = [f"{ticker}_return_{i}" for i in [1, 5, 10, 20]]
            lag_cols = [f"{ticker}_lag_{i}" for i in range(1, 21)]

            # Retain only columns that exist (in case some are missing)
            df = df[
                [col for col in base_cols if col in df.columns] + return_cols + lag_cols
            ]

            # df.dropna(inplace=True)

            output_path = PROCESSED_DIR / file
            df.to_csv(output_path, index=False)
            print(f"🚀 Engineered and saved features to: {output_path}")


if __name__ == "__main__":
    download_index_data()
    clean_and_format_index_data()
    engineer_index_features()
