# 📊 Stock Price Prediction Project Plan & Checklist

This project aims to predict next-day stock prices using historical data and supervised machine learning techniques. Below is a weekly breakdown of the tasks with checkboxes, which I will be using to track progress.

---

## ✅ Week 1: Setup & Data Acquisition

- ✅ Create project folder `stock-price-predictor` using standard ML directory structure
- ✅ Initialize Git repository and connect to GitHub
- ✅ Create `.gitignore` file
- ✅ Create and activate Python virtual environment
- ✅ Install required libraries (`yfinance`, `pandas`, `numpy`, etc.)
- ✅ Freeze requirements to `requirements.txt`
- ✅ Define stock tickers across 11 major U.S. sectors
- ✅ Implement `data_prep.py` to download historical data (2000-01-01 to 2025-05-05)
- ✅ Save raw data to `data/raw/` directory
- ✅ Create `data/README.md` describing data sources, tickers, and folder structure
- ✅ Commit: "✅ Project setup, environment, data download script, and data README completed"

---

## ✅ Week 2: Exploratory Data Analysis (EDA)

- ✅ Open and create `exploratory_data_analysis.ipynb`
- ✅ Plot stock prices, volume, moving averages
- ✅ Identify trends, volatility, seasonality
- ✅ Commit: "EDA notebook"

---

## ✅ Week 3: Feature Engineering

- ✅ Add lag features and moving averages
- ✅ Add technical indicators (e.g. RSI, MACD)
- ✅ Encode time-based features (weekday, month)
- ✅ Save processed dataset to `data/processed/`
- ✅ Commit: "Feature engineering completed"

---

## ✅ Week 4: Model Building (Baseline)

- ✅ Create `model.py` and `evaluate.py`
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
