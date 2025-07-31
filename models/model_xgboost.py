import pandas as pd
import numpy as np
from xgboost import XGBRegressor

def predict_xgboost(df: pd.DataFrame, days: int = 30):
    df = df[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df["Close"] = df["Close"].astype(float)

    for lag in range(1, 8):
        df[f"lag_{lag}"] = df["Close"].shift(lag)
    
    df.dropna(inplace=True)
    
    X = df[[f"lag_{i}" for i in range(1, 8)]]
    y = df["Close"]

    model = XGBRegressor()
    model.fit(X, y)

    last_lags = X.iloc[-1].values.reshape(1, -1)
    preds = []

    for _ in range(days):
        pred = model.predict(last_lags)[0]
        preds.append(pred)
        last_lags = np.roll(last_lags, -1)
        last_lags[0, -1] = pred

    future_dates = pd.date_range(start=df["Date"].max() + pd.Timedelta(days=1), periods=days)
    return pd.DataFrame({"yhat": preds}, index=future_dates)
