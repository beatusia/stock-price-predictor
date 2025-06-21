# src/01_data_download.py
# This script downloads stock data from Yahoo Finance, cleans it,
# processes the downloaded data by removing unnecessary rows, adding ticker and sector columns,
# and reordering columns for consistency.
# The processed data is saved in a separate directory for further analysis.

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

# Invert the mapping for easy lookup
ticker_to_sector = {
    ticker: sector
    for sector, tickers in us_market_sectors.items()
    for ticker in tickers
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

# Define paths
TRUE_RAW_DIR = "/Users/beatawyspianska/Desktop/AIML_Projects/predict_stock_price/stock-price-predictor/data/raw/true_raw"
MODIFIED_DIR = "/Users/beatawyspianska/Desktop/AIML_Projects/predict_stock_price/stock-price-predictor/data/raw/modified"

# Ensure the output directory exists
Path(MODIFIED_DIR).mkdir(parents=True, exist_ok=True)

# Process each CSV in true_raw
for file in os.listdir(TRUE_RAW_DIR):
    if file.endswith(".csv"):
        input_path = os.path.join(TRUE_RAW_DIR, file)

        # Read CSV and remove first two rows
        df = pd.read_csv(input_path, index_col=0)
        df = df.iloc[2:]

        # Extract ticker from filename
        ticker = os.path.splitext(file)[0]
        sector = ticker_to_sector.get(ticker, "Unknown")

        # Add TICKER and Sector columns
        df["TICKER"] = ticker
        df["Sector"] = sector

        # Move index (Date) to a column
        df.insert(0, "Date", df.index)
        df.reset_index(drop=True, inplace=True)

        # Reorder columns (only include those that exist)
        desired_order = [
            "Date",
            "TICKER",
            "Sector",
            "Close",
            "High",
            "Low",
            "Open",
            "Volume",
        ]
        final_columns = [col for col in desired_order if col in df.columns]
        df = df[final_columns]

        # Save cleaned file to the modified directory
        output_path = os.path.join(MODIFIED_DIR, file)
        df.to_csv(output_path, index=False)

        print(f"✅ Processed and saved: {output_path}")
