# src/04_data_merge.py
# This script merges stock price data with benchmark index features.

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

# Merge with each stock file
for file in os.listdir(STOCKS_DIR):
    if not file.endswith(".csv"):
        continue

    stock_df = pd.read_csv(STOCKS_DIR / file)
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])

    # Merge with benchmark data
    merged_df = pd.merge(stock_df, benchmark_merged, on="Date", how="left")

    # Reorganize columns: preserve all existing columns, just reposition main groups
    all_cols = merged_df.columns.tolist()

    base_cols = [
        col
        for col in ["Date", "TICKER", "Close", "High", "Low", "Open", "Volume"]
        if col in merged_df.columns
    ]
    benchmark_cols = [
        col for col in all_cols if any(x in col for x in ["SP500", "NASDAQ"])
    ]
    lag_cols = [col for col in all_cols if col.startswith("lag_")]
    target_col = ["Target"] if "Target" in all_cols else []

    # Determine remaining columns (all others)
    exclude = set(base_cols + benchmark_cols + lag_cols + target_col)
    other_cols = [col for col in all_cols if col not in exclude]

    # Final column order
    ordered_cols = base_cols + benchmark_cols + other_cols + lag_cols + target_col
    merged_df = merged_df[[col for col in ordered_cols if col in merged_df.columns]]

    # Save
    merged_df.to_csv(MERGED_DIR / file, index=False)
    print(f"✅ Merged and saved: {MERGED_DIR / file}")
