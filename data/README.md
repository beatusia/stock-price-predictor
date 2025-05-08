# 📊 Data Directory

This folder contains the datasets used in the **Stock Price Predictor** project.

## 🔍 Overview

The data consists of historical daily stock prices for 22 companies across 11 major sectors of the U.S. market. The dataset was retrieved using the [`yfinance`](https://pypi.org/project/yfinance/) Python package.

| Sector                     | Tickers           |
|----------------------------|-------------------|
| Information Technology     | AAPL, MSFT        |
| Health Care                | JNJ, UNH          |
| Financials                 | JPM, BAC          |
| Consumer Discretionary     | AMZN, TSLA        |
| Communication Services     | GOOGL, META       |
| Industrials                | UNP, RTX          |
| Consumer Staples           | PG, KO            |
| Energy                     | XOM, CVX          |
| Utilities                  | NEE, DUK          |
| Real Estate                | AMT, PLD          |
| Materials                  | LIN, SHW          |

## 🗂 Structure

- `raw/`: Contains the original, unprocessed CSV files for each stock (one file per ticker).
- `processed/`: Will contain cleaned and feature-engineered datasets for modeling.

## 📅 Date Range

- Start Date: **2000-01-01**
- End Date: **2025-05-08**
- Frequency: **Daily**

## 📥 Source

Data is pulled using `yfinance.download()` which queries Yahoo Finance APIs. No proprietary or paid data sources are used.

## ⚠️ Notes

- Ensure you have an internet connection when downloading the data.
- Yahoo Finance data can occasionally change or become unavailable; always store local copies for reproducibility.

## 🔁 Recreate Raw Data

To regenerate the raw data files, run:

```bash
python src/data_download.py
