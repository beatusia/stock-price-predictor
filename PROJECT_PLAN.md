# 📊 Stock Price Prediction Project Plan

This project aims to predict next-day stock prices using historical data and supervised machine learning techniques. Below is a breakdown of the tasks by week, with checkboxes to track progress.

---

## ✅ Week 1: Setup & Data Acquisition

- [ ] Create Python virtual environment
- [ ] Install and freeze libraries (`requirements.txt`)
- [ ] Download stock price data using `yfinance`
- [ ] Save raw data to `data/` directory
- [ ] Create `data_prep.py` with download functions
- [ ] Commit: "Project setup and data download"

---

## ✅ Week 2: Exploratory Data Analysis (EDA)

- [ ] Open and create `exploratory_data_analysis.ipynb`
- [ ] Plot stock prices, volume, moving averages
- [ ] Identify trends, volatility, seasonality
- [ ] Save plots to `outputs/`
- [ ] Commit: "EDA notebook"

---

## ✅ Week 3: Feature Engineering

- [ ] Add lag features and moving averages
- [ ] Add technical indicators (e.g. RSI, MACD)
- [ ] Encode time-based features (weekday, month)
- [ ] Save processed dataset to `data/processed/`
- [ ] Commit: "Feature engineering completed"

---

## ✅ Week 4: Model Building (Baseline)

- [ ] Create `model.py` and `evaluate.py`
- [ ] Train models: Linear Regression, Decision Tree, Random Forest
- [ ] Evaluate models using RMSE, MAE
- [ ] Save predictions to `outputs/`
- [ ] Commit: "Baseline models and evaluation results"

---

## ✅ Week 5: Model Tuning & Validation

- [ ] Perform cross-validation / GridSearch
- [ ] Save best model as `.pkl`
- [ ] Document model selection rationale in notebook
- [ ] Commit: "Model tuning and selection"

---

## ✅ Week 6: Finalization & Deployment Prep

- [ ] Create `predict.py` or minimal Flask API
- [ ] Freeze `requirements.txt`
- [ ] Update README with usage instructions
- [ ] Tag and push final version
- [ ] Commit: "Final version with prediction pipeline"

---

## 🚀 Stretch Goals (Optional)

- [ ] Streamlit app frontend
- [ ] Scheduled daily run (e.g. cron + script)
- [ ] Backtest model to simulate strategy performance

---
