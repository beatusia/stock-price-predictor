# src/data_prep.py

import yfinance as yf
import os
import pandas as pd

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
    "Materials": ["LIN", "SHW"]
}

def download_data(tickers, start="2000-01-01", end="2025-05-05"):
    os.makedirs("data/raw", exist_ok=True)

    for sector, symbols in tickers.items():
        for symbol in symbols:
            print(f"Downloading {symbol} from {start} to {end}...")
            data = yf.download(symbol, start=start, end=end)
            file_path = f"data/raw/{symbol}.csv"
            data.to_csv(file_path)
            print(f"Saved {symbol} to {file_path}")

if __name__ == "__main__":
    download_data(us_market_sectors)
