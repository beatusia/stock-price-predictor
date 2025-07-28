# src/02_data_download.py
# This script downloads market capitalization data for US stocks by sector and updates existing processed feature files with this data.

import yfinance as yf
import pandas as pd
from pathlib import Path

# --- Step 1: Define tickers by sector ---
us_market_sectors = {
    "Information Technology": ["AAPL", "MSFT", "NVDA"],
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


# --- Step 2: Define market cap binning function ---
def classify_market_cap(cap):
    if cap is None:
        return "Unknown"
    elif cap >= 200_000_000_000:
        return "Large Cap"
    elif cap >= 10_000_000_000:
        return "Mid Cap"
    else:
        return "Small Cap"


# --- Step 3: Download live market cap data ---
market_cap_data = []
for sector, tickers in us_market_sectors.items():
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            market_cap = info.get("marketCap", None)
            print(f"{ticker}: {market_cap if market_cap else '⚠️ Not available'}")
            market_cap_data.append(
                {
                    "Ticker": ticker,
                    "Sector": sector,
                    "MarketCap": market_cap,
                    "MarketCapBin": classify_market_cap(market_cap),
                }
            )
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            market_cap_data.append(
                {
                    "Ticker": ticker,
                    "Sector": sector,
                    "MarketCap": None,
                    "MarketCapBin": "Error",
                }
            )

# --- Step 4: Load processed feature files and update them ---
processed_folder = Path(
    "/Users/beatawyspianska/Desktop/AIML_Projects/predict_stock_price/stock-price-predictor/data/raw/modified"
)
df_cap_info = pd.DataFrame(market_cap_data)

# --- Step 4: Load processed feature files and update them ---
for file in processed_folder.glob("*.csv"):
    try:
        df = pd.read_csv(file)
        ticker = df["TICKER"].iloc[0] if "TICKER" in df.columns else None

        if ticker:
            cap_row = df_cap_info[df_cap_info["Ticker"] == ticker]
            if not cap_row.empty:
                market_cap = cap_row["MarketCap"].values[0]
                cap_bin = cap_row["MarketCapBin"].values[0]
            else:
                market_cap, cap_bin = None, "Unknown"

            # Try to insert after Sector, else after TICKER, else at end
            if "Sector" in df.columns:
                insert_idx = df.columns.get_loc("Sector") + 1
            elif "TICKER" in df.columns:
                insert_idx = df.columns.get_loc("TICKER") + 1
            else:
                insert_idx = len(df.columns)

            df.insert(insert_idx, "MarketCap", market_cap)
            df.insert(insert_idx + 1, "MarketCapBin", cap_bin)

            df.to_csv(file, index=False)
            print(f"✅ Updated {file.name} with market cap info")
        else:
            print(f"⚠️ TICKER column missing in {file.name}")

    except Exception as e:
        print(f"❌ Error processing {file.name}: {e}")
