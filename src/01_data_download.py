# src/01_data_download.py
# This script downloads stock data from Yahoo Finance and saves it to CSV files
import os
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

# Define your stock tickers by sector
us_market_sectors = {
    "Information Technology": ["AAPL", "MSFT"],
    "Health Care": ["JNJ", "UNH"],
    "Financials": ["JPM", "BAC"],
    "Consumer Discretionary": ["AMZN", "TSLA"],
    "Communication Services": ["GOOGL", "META"],
    "Industrials": ["UNP", "RTX"],
    "Consumer Staples": ["PG", "KO"],
    "Energy": ["XOM", "CVX"],
    "Utilities": ["NEE", "DUK"],
    "Real Estate": ["AMT", "PLD"],
    "Materials": ["LIN", "SHW"],
}

# Output directory
OUTPUT_DIR = Path(
    "/Users/beatawyspianska/Desktop/AIML_Projects/predict_stock_price/stock-price-predictor/data/raw/true_raw"
)


def download_data(tickers, start="1999-12-01", end=None, output_dir=OUTPUT_DIR):
    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    output_dir.mkdir(parents=True, exist_ok=True)

    for sector, symbols in tickers.items():
        for symbol in symbols:
            print(f"📥 Downloading {symbol} from {start} to {end}...")
            try:
                data = yf.download(symbol, start=start, end=end)
                if data.empty:
                    print(f"⚠️ No data found for {symbol}")
                    continue
                file_path = output_dir / f"{symbol}.csv"
                data.to_csv(file_path)
                print(f"✅ Saved {symbol} to {file_path}")
            except Exception as e:
                print(f"❌ Failed to download {symbol}: {e}")


if __name__ == "__main__":
    download_data(us_market_sectors)
