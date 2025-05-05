# 📈 Stock Price Predictor

Predict next-day stock price movement (up/down) using historical OHLCV data.  
This project is part of my AI/ML apprenticeship and focuses on applying core supervised learning techniques to a real-world finance problem.

---

## 🧠 Overview

This project uses historical stock data to predict whether a stock's closing price will go up or down the next day. It includes:

- Feature engineering on time-series data (lag, returns, volatility)
- Classification models (Logistic Regression, Decision Trees, Random Forest, XGBoost)
- Model evaluation with metrics like ROC-AUC, precision, recall
- Reproducible, modular Python code
- Ready-to-deploy project structure

---

## 📁 Project Structure

```bash
stock_price_prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md  # Information about the data, preprocessing steps, and dataset sources
│
├── notebooks/          # Jupyter notebooks for exploration and prototyping
│   ├── EDA.ipynb       # Exploratory data analysis
│   ├── model_building.ipynb
│   └── model_tuning.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Data cleaning and processing scripts
│   ├── feature_engineering.py
│   ├── model.py          # ML model definitions
│   └── evaluation.py     # Model evaluation
│
├── requirements.txt  # Python dependencies
├── PROJECT_PLAN.md    # Detailed project plan
├── README.md          # Overview of the project
└── .gitignore         # Git ignore file

