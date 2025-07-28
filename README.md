# 📈 Stock Price Predictor

Predict next-day stock price movement using historical OHLCV data.  
This project is part of my AI/ML apprenticeship and focuses on applying core learning techniques to a real-world finance problem.

---

## 🧠 Overview

This project uses historical stock data to predict whether a stock's closing price will go up or down the next day. It includes:

- Feature engineering on time-series data (lag, returns, volatility)
- Linear Regression model, tree based methods such as Random Forest, LightGBM and XGBoost
- Model evaluation 
- Reproducible, modular Python code
- Ready-to-deploy project structure

---

## 📁 Project Structure

```bash
stock_price_prediction/
│
├── data/
│   ├── raw/                       # Raw Data in .csv format (for each stock)
│   ├── processed/                 # Cleaned Data - ready for model building
│   └── README.md                  # Information about the data, preprocessing steps, and dataset sources
│
├── notebooks/                     # Jupyter notebooks for exploration and prototyping
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   ├── 02_model_building.ipynb
│   └── 03_model_tuning.ipynb
│
├── src/
│   ├── __init__.py
│   ├── 01_data_download.py        # Downloads raw data to data/raw
│   ├── 02_data_modification.py    # Removes two first empty rows from csv, adds Date and Ticker columns and saves modified files
│   ├── 03_feature_engineering.py  # Adds technical indicators, calendar features, lag features, and more to stock price data
│   ├── 04_model.py                # ML model definitions
│   └── 05_evaluation.py           # Model evaluation
│
├── requirements.txt               # Python dependencies
├── PROJECT_PLAN.md                # Detailed project plan
├── README.md                      # Overview of the project
└── .gitignore                     # Git ignore file

