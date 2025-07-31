import streamlit as st
import pandas as pd
from dateutil.parser import parse

sector_map = {
 'ADANIENT.csv':'Conglomerate','ADANIGREEN.csv':'Renewable Energy','ADANIPORTS.csv':'Logistics','ADANIPOWER.csv':'Power','AMBUJACEM.csv':'Cement',
 'ASHOKLEY.csv':'Automobile','AUROPHARMA.csv':'Pharmaceuticals','AXISBANK.csv':'Banking','BAJAJFINSV.csv':'NBFC','BAJFINANCE.csv':'NBFC',
 'BALKRISIND.csv':'Tyres','BANDHANBNK.csv':'Banking','BANKBARODA.csv':'Banking','BEL.csv':'Defense','BHARATFORG.csv':'Automobile',
 'BHARTIARTL.csv':'Telecom','BOSCHLTD.csv':'Automobile','BPCL.csv':'Oil & Gas','BRITANNIA.csv':'Consumer Goods','CADILAHC.csv':'Pharmaceuticals',
 'CANBK.csv':'Banking','CIPLA.csv':'Pharmaceuticals','COALINDIA.csv':'Mining','COLPAL.csv':'Consumer Goods','DABUR.csv':'Consumer Goods',
 'DIVISLAB.csv':'Pharmaceuticals','DMART.csv':'Retail','DRREDDY.csv':'Pharmaceuticals','EICHERMOT.csv':'Automobile','EIDPARRY.csv':'Agro Products',
 'GAIL.csv':'Oil & Gas','GRASIM.csv':'Conglomerate','HAVELLS.csv':'Electricals','HCLTECH.csv':'IT Services','HDFC.csv':'Finance',
 'HDFCBANK.csv':'Banking','HDFCLIFE.csv':'Insurance','HINDUNILVR.csv':'Consumer Goods','HINDZINC.csv':'Metals','ICICIBANK.csv':'Banking',
 'ICICIPRULI.csv':'Insurance','IDFCFIRSTB.csv':'Banking','INFY.csv':'IT Services','IRCTC.csv':'Tourism','IRFC.csv':'Finance',
 'ITC.csv':'Consumer Goods','JIOFIN.csv':'Finance','JSWSTEEL.csv':'Metals','KOTAKBANK.csv':'Banking','LICI.csv':'Insurance',
 'LTI.csv':'IT Services','LT.csv':'Infrastructure','LTIMINDTREE.csv':'IT Services','M&M.csv':'Automobile','MARUTI.csv':'Automobile',
 'MINDTREE.csv':'IT Services','MUTHOOTFIN.csv':'NBFC','NHPC.csv':'Power','NTPC.csv':'Power','ONGC.csv':'Oil & Gas',
 'PAGEIND.csv':'Textiles','PFC.csv':'Finance','PIDILITIND.csv':'Chemicals','PNB.csv':'Banking','POWERGRID.csv':'Power',
 'RBLBANK.csv':'Banking','RELIANCE.csv':'Oil & Gas','RIL.csv':'Oil & Gas','SBILIFE.csv':'Insurance','SBIN.csv':'Banking',
 'SHREECEM.csv':'Cement','SHRIRAMFIN.csv':'NBFC','SRF.csv':'Chemicals','SRTRANSFIN.csv':'NBFC','SUNPHARMA.csv':'Pharmaceuticals',
 'TATACONSUM.csv':'Consumer Goods','TATAMOTORS.csv':'Automobile','TCS.csv':'IT Services','TECHM.csv':'IT Services','TORNTPHARM.csv':'Pharmaceuticals',
 'TRENT.csv':'Retail','TVSMOTOR.csv':'Automobile','UBL.csv':'Beverages','ULTRACEMCO.csv':'Cement','UPL.csv':'Agrochemicals',
 'VEDANTA.csv':'Metals','VEDL.csv':'Metals','WIPRO.csv':'IT Services','ZEEL.csv':'Media'
}


def chatbot_section(file, df, cagr, days, predictions_dict):
    st.header("🗣️ Chatbot Assistant")

    with st.expander("💡 Sample questions you can ask"):
        st.markdown("""
        - What is the stock name?
        - Which sector does this stock belong to?
        - What is the latest price?
        - What is the CAGR?
        - Show prediction using ARIMA
        - Explain Prophet
        - What is the predicted price on 2025-08-28
        """)

    question = st.text_input("Ask me anything about the stock or app:")

    if question:
        q = question.lower()

        if "stock name" in q or "which stock" in q:
            st.success(f"This stock is **{file.replace('.csv', '')}**.")

        elif "sector" in q:
            sector = sector_map.get(file, "Unknown")
            st.success(f"This stock belongs to the **{sector}** sector.")

        elif "latest price" in q or "last close" in q:
            st.success(f"The latest closing price is **₹{df['Close'].iloc[-1]:.2f}**.")

        elif "start price" in q:
            st.success(f"Start price: **₹{df['Close'].iloc[0]:.2f}**")

        elif "end price" in q:
            st.success(f"End price: **₹{df['Close'].iloc[-1]:.2f}**")

        elif "cagr" in q:
            if "what is" in q or "explain" in q:
                st.info("**CAGR** (Compound Annual Growth Rate) is the rate at which an investment grows annually over time.")
            else:
                st.success(f"The CAGR over the selected period is **{cagr * 100:.2f}%**.")

        elif "prediction" in q or "forecast" in q:
            model = None
            for m in predictions_dict.keys():
                if m.lower() in q:
                    model = m
                    break

            if model:
                pred_df = predictions_dict.get(model)
                if not pred_df.empty:
                    st.success(f"Prediction using **{model}**:")
                    st.dataframe(pred_df.tail())
                else:
                    st.warning(f"No predictions available for {model}.")
            else:
                st.success("Predicted value on final day from all models:")
                for name, pred_df in predictions_dict.items():
                    if not pred_df.empty:
                        st.markdown(f"**{name}**: ₹{pred_df['yhat'].iloc[-1]:.2f}")

        elif "on" in q:
            import re
            match = re.search(r"on (\d{4}-\d{2}-\d{2})", q)
            if match:
                try:
                    date = parse(match.group(1)).date()
                    responses = []
                    for name, pred_df in predictions_dict.items():
                        if not pred_df.empty:
                            pred_df = pred_df.copy()
                            if "ds" in pred_df.columns:
                                pred_df["Date"] = pd.to_datetime(pred_df["ds"]).dt.date
                            else:
                                pred_df["Date"] = pd.to_datetime(pred_df.index).date
                            row = pred_df[pred_df["Date"] == date]
                            if not row.empty:
                                responses.append(f"**{name}**: ₹{row.iloc[0]['yhat']:.2f}")
                    if responses:
                        st.success(" | ".join(responses))
                    else:
                        st.warning("No prediction found for that date.")
                except Exception:
                    st.warning("Date format is incorrect.")
            else:
                st.warning("Please ask like: prediction on 2025-08-20")

        elif "arima" in q:
            st.info("**ARIMA** (AutoRegressive Integrated Moving Average) is a time series forecasting model that captures trends and autocorrelations in historical stock data.")

        elif "prophet" in q:
            st.info("**Prophet** is a forecasting model developed by Meta (Facebook) that handles seasonality, holidays, and trend shifts in time series data.")

        elif "xgboost" in q:
            st.info("**XGBoost** is a high-performance machine learning algorithm based on gradient boosting, widely used for structured data prediction.")

        elif "random forest" in q:
            st.info("**Random Forest** is an ensemble learning method that builds multiple decision trees and merges them to get a more accurate and stable prediction.")

        else:
            st.warning("Sorry, I don't have an answer to that question about this stock or app.")
