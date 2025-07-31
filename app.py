import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from models.model_arima import predict_arima
from models.model_prophet import predict_prophet
from models.model_rf import predict_rf
from models.model_xgboost import predict_xgboost
from chatbot import chatbot_section



st.markdown("<h1 style='text-align: center;'>NIFTY PREDICT</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'> Intelligent forecasting for NIFTY stocks</h4>", unsafe_allow_html=True)



# --- Sidebar ---
st.sidebar.header("Filter Options")

sector_map = {
    "RELIANCE.NS": "Energy", "TCS.NS": "IT", "HDFCBANK.NS": "Banking", "HDFCLIFE.NS": "Financial Services",
    "BHARTIARTL.NS": "Telecom", "ICICIBANK.NS": "Banking", "INFY.NS": "IT", "HINDUNILVR.NS": "Consumer Goods",
    "BAJFINANCE.NS": "Financial Services", "ITC.NS": "Consumer Goods", "LT.NS": "Industrials",
    "KOTAKBANK.NS": "Banking", "SUNPHARMA.NS": "Pharmaceuticals", "HCLTECH.NS": "IT", "MARUTI.NS": "Automobile",
    "M&M.NS": "Automobile", "ULTRACEMCO.NS": "Cement", "AXISBANK.NS": "Banking", "NTPC.NS": "Power",
    "BAJAJFINSV.NS": "Financial Services", "ONGC.NS": "Energy", "ADANIPORTS.NS": "Logistics", "ADANIENT.NS": "Conglomerate",
    "BEL.NS": "Defense", "POWERGRID.NS": "Power", "WIPRO.NS": "IT", "DABUR.NS": "Consumer Goods",
    "GRASIM.NS": "Conglomerate", "SBIN.NS": "Banking", "TITAN.NS": "Consumer Goods", "DRREDDY.NS": "Pharmaceuticals",
    "SHRIRAMFIN.NS": "NBFC", "TECHM.NS": "IT", "BRITANNIA.NS": "Consumer Goods", "EICHERMOT.NS": "Automobile",
    "TATACONSUM.NS": "Consumer Goods", "DIVISLAB.NS": "Pharmaceuticals", "TRENT.NS": "Retail", "VEDL.NS": "Metals",
    "IRFC.NS": "Finance", "COLPAL.NS": "Consumer Goods", "EIDPARRY.NS": "Agro Products", "UBL.NS": "Beverages",
    "ADANIGREEN.NS": "Renewable Energy", "LTI.NS": "IT", "PIDILITIND.NS": "Chemicals", "HDFC.NS": "Finance",
    "UPL.NS": "Agrochemicals", "COALINDIA.NS": "Mining", "ADANIPOWER.NS": "Power", "BPCL.NS": "Energy",
    "JSWSTEEL.NS": "Metals", "DMART.NS": "Retail", "LTIMINDTREE.NS": "IT", "ICICIPRULI.NS": "Insurance",
    "HAVELLS.NS": "Electricals", "TVSMOTOR.NS": "Automobile", "PNB.NS": "Banking", "BANKBARODA.NS": "Banking",
    "GAIL.NS": "Energy", "IRCTC.NS": "Tourism", "NHPC.NS": "Power", "RIL.NS": "Energy", "SBILIFE.NS": "Insurance",
    "SRF.NS": "Chemicals", "IDFCFIRSTB.NS": "Banking", "TORNTPHARM.NS": "Pharmaceuticals", "VEDANTA.NS": "Metals",
    "ZEEL.NS": "Media", "ASHOKLEY.NS": "Automobile", "BANDHANBNK.NS": "Banking", "CIPLA.NS": "Pharmaceuticals",
    "HINDZINC.NS": "Metals", "SHREECEM.NS": "Cement", "TATAMOTORS.NS": "Automobile", "BHARATFORG.NS": "Automobile",
    "JIOFIN.NS": "Finance", "PFC.NS": "Finance", "RBLBANK.NS": "Banking", "MUTHOOTFIN.NS": "NBFC",
    "CANBK.NS": "Banking", "SRTRANSFIN.NS": "NBFC", "BALKRISIND.NS": "Tyres", "PAGEIND.NS": "Textiles",
    "CADILAHC.NS": "Pharmaceuticals", "LICI.NS": "Insurance", "AMBUJACEM.NS": "Cement", "BOSCHLTD.NS": "Automobile",
    "AUROPHARMA.NS": "Pharmaceuticals"
}

csv_files = [f for f in os.listdir("backend/data/raw") if f.endswith(".csv")]
all_sectors = sorted(set(sector_map.get(f.replace(".csv", ""), "Unknown") for f in csv_files))
selected_sector = st.sidebar.selectbox("Select Sector", ["All"] + all_sectors)

if selected_sector != "All":
    csv_files = [f for f in csv_files if sector_map.get(f.replace(".csv", ""), "Unknown") == selected_sector]

file = st.sidebar.selectbox("Choose Stock", csv_files)

# Load selected CSV
df = pd.read_csv(f"backend/data/raw/{file}")
df["Date"] = pd.to_datetime(df["Date"])
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df.dropna(subset=["Close"], inplace=True)

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    st.error("Please select a valid start and end date.")
    st.stop()

df_filtered = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Historical Trends", "🔮 Predictions"])

# ---------------------------------------
# Tab 1 - Overview
# ---------------------------------------
with tab1:
    st.subheader(f"Summary for {file.replace('.csv', '')}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Start Price", f"{df_filtered['Close'].iloc[0]:.2f}")
        st.metric("End Price", f"{df_filtered['Close'].iloc[-1]:.2f}")
    with col2:
        if len(df_filtered) >= 2:
            start_price = df_filtered.iloc[0]["Close"]
            end_price = df_filtered.iloc[-1]["Close"]
            num_years = (df_filtered["Date"].iloc[-1] - df_filtered["Date"].iloc[0]).days / 365
            cagr = ((end_price / start_price) ** (1 / num_years)) - 1
            st.metric("📈CAGR", f"{cagr * 100:.2f}% over {num_years:.2f} years")
        else:
            st.warning("Not enough data to calculate CAGR.")

    st.divider()

# ---------------------------------------
# Tab 2 - Historical Trends
# ---------------------------------------
with tab2:
    st.subheader("📅 Daily Close Price")
    st.line_chart(df_filtered.set_index("Date")["Close"])

    

# ---------------------------------------
# Tab 3 - Predictions
# ---------------------------------------
with tab3:
    st.subheader("🔮 Forecasting Models")
    days = st.slider("Prediction Horizon (days)", 1, 60, 30)

    models = {
        "ARIMA": predict_arima,
        "Prophet": predict_prophet,
        "Random Forest": predict_rf,
        "XGBoost": predict_xgboost,
    }

    predictions = {}
    model_tabs = st.tabs(list(models.keys()))
    for i, (name, model_func) in enumerate(models.items()):
        with model_tabs[i]:
            try:
                forecast = model_func(df_filtered, days)
                st.line_chart(forecast)
                predictions[name.lower()] = forecast
            except Exception as e:
                st.error(f"{name} failed: {e}")
                predictions[name.lower()] = pd.DataFrame()

# ---------------------------------------
# Chatbot Section
# ---------------------------------------
predictions_dict = {}
for name, model_func in models.items():
    try:
        predictions_dict[name] = model_func(df_filtered, days)
    except:
        predictions_dict[name] = pd.DataFrame()

chatbot_section(
    file=file,
    df=df_filtered,
    cagr=cagr,
    days=days,
    predictions_dict=predictions_dict
)
