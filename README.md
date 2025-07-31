# 📊 NIFTYPREDICT

**NIFTYPREDICT** is a stock prediction dashboard that uses real-time stock data (via Yahoo Finance) and forecasts future prices using four machine learning models: ARIMA, Prophet, Random Forest, and XGBoost.

## 🔍 Features

- Upload historical stock CSV files
- View actual vs predicted prices
- Get model-wise predictions for the next 30 days
- Ask questions using a ChatBot (e.g., "What is the price on 2025-08-20?")
- Sector-wise stock categorization and charts

## 🧠 Models Used

- ARIMA
- Facebook Prophet
- Random Forest Regressor
- XGBoost Regressor

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
