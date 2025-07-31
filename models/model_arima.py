import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def predict_arima(df: pd.DataFrame, days: int = 30):
    df = df[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    model = ARIMA(df["Close"], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=days)

    forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days)
    return pd.DataFrame({"yhat": forecast.values}, index=forecast_index)
