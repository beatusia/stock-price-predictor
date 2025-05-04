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
stock-price-predictor/
├── data/                        # Raw and processed CSV files
├── notebooks/
│   ├── exploratory_data_analysis.ipynb   # Initial EDA notebook
├── src/
│   ├── data_prep.py             # Feature engineering, data splits
│   ├── model.py                 # Model training and saving
│   └── evaluate.py              # Evaluation metrics and plots
├── models/                      # Saved trained models
├── outputs/                     # Figures, logs, and evaluation results
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore
