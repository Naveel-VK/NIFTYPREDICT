from prophet import Prophet
import pandas as pd

def predict_prophet(df: pd.DataFrame, days: int = 30):
    df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    
    return forecast[["ds", "yhat"]].set_index("ds").tail(days)
