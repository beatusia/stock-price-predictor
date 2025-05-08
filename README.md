# 📈 Stock Price Predictor

Predict next-day stock price movement using historical OHLCV data.  
This project is part of my AI/ML apprenticeship and focuses on applying core learning techniques to a real-world finance problem.

---

## 🧠 Overview

This project uses historical stock data to predict whether a stock's closing price will go up or down the next day. It includes:

- Feature engineering on time-series data (lag, returns, volatility)
- Linear Regression model
- Model evaluation 
- Reproducible, modular Python code
- Ready-to-deploy project structure

---

## 📁 Project Structure

```bash
stock_price_prediction/
│
├── data/
│   ├── raw/                  # Raw Data in .csv format (for each stock)
│   ├── processed/            # Cleaned Data - ready for model building
│   └── README.md             # Information about the data, preprocessing steps, and dataset sources
│
├── notebooks/                # Jupyter notebooks for exploration and prototyping
│   ├── EDA.ipynb             # Exploratory data analysis
│   ├── model_building.ipynb
│   └── model_tuning.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_download.py       # Downloads raw data to data/raw
│   ├── data_preprocessing.py  # Data cleaning and processing scripts
│   ├── feature_engineering.py # Creative Feature engineering 
│   ├── model.py               # ML model definitions
│   └── evaluation.py          # Model evaluation
│
├── requirements.txt           # Python dependencies
├── PROJECT_PLAN.md            # Detailed project plan
├── README.md                  # Overview of the project
└── .gitignore                 # Git ignore file

