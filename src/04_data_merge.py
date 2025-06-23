# src/04_data_merge.py
# This script merges stock data with benchmark indices (SP500, NASDAQ) and processes the merged data for further analysis.

import os
import pandas as pd
from pathlib import Path
from functools import reduce

# Define paths
STOCKS_DIR = Path(
    "/Users/beatawyspianska/Desktop/AIML_Projects/predict_stock_price/stock-price-predictor/data/processed"
)
BENCHMARK_DIR = Path(
    "/Users/beatawyspianska/Desktop/AIML_Projects/predict_stock_price/stock-price-predictor/data/benchmark/processed"
)
MERGED_DIR = Path(
    "/Users/beatawyspianska/Desktop/AIML_Projects/predict_stock_price/stock-price-predictor/data/merged"
)
MERGED_DIR.mkdir(parents=True, exist_ok=True)

# Load all benchmark files into a single DataFrame (on 'Date')
benchmark_dfs = []
for file in os.listdir(BENCHMARK_DIR):
    if file.endswith(".csv"):
        benchmark = pd.read_csv(BENCHMARK_DIR / file)
        if "Date" not in benchmark.columns:
            continue
        benchmark["Date"] = pd.to_datetime(benchmark["Date"])
        benchmark_dfs.append(benchmark)

# Merge all benchmark DataFrames on "Date"
benchmark_merged = reduce(
    lambda left, right: pd.merge(left, right, on="Date", how="outer"), benchmark_dfs
)

# Define target columns to appear last
target_cols_to_move = [
    "Target_Raw_Close",
    "Target_Log_Return",
    "Target_%_Return",
    "Target_Direction",
]

# Merge with each stock file
for file in os.listdir(STOCKS_DIR):
    if not file.endswith(".csv"):
        continue

    stock_df = pd.read_csv(STOCKS_DIR / file)
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])

    # Merge with benchmark data
    merged_df = pd.merge(stock_df, benchmark_merged, on="Date", how="left")

    # Remove columns with _x and _y suffixes (from merge conflicts)
    merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith(("_x", "_y"))]

    # Define preferred base column order (guaranteed order if present)
    base_cols = [
        "Date",
        "TICKER",
        "Sector",
        "MarketCap",
        "MarketCapBin",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
    ]

    # Determine benchmark columns (e.g., SP500, NASDAQ) and preserve order
    benchmark_cols = [
        col
        for col in merged_df.columns
        if any(idx in col for idx in ["SP500", "NASDAQ"])
    ]

    # Determine other features
    lag_cols = [col for col in merged_df.columns if col.startswith("lag_")]
    target_cols = [col for col in target_cols_to_move if col in merged_df.columns]

    # All known columns to exclude from "other"
    known_cols = set(base_cols + benchmark_cols + lag_cols + target_cols)
    other_cols = [col for col in merged_df.columns if col not in known_cols]

    # Final column order
    ordered_cols = (
        [col for col in base_cols if col in merged_df.columns]
        + benchmark_cols
        + other_cols
        + lag_cols
        + target_cols
    )

    # Apply order
    merged_df = merged_df[[col for col in ordered_cols if col in merged_df.columns]]

    # Save merged file
    merged_df.to_csv(MERGED_DIR / file, index=False)
    print(f"✅ Merged and saved: {MERGED_DIR / file}")
