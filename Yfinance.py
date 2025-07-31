import yfinance as yf
import pandas as pd
import os

symbols = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "BHARTIARTL.NS", 
    "ICICIBANK.NS", "INFY.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "ITC.NS",
    "LT.NS", "KOTAKBANK.NS", "SUNPHARMA.NS", "HCLTECH.NS", "MARUTI.NS",
    "M&M.NS", "ULTRACEMCO.NS", "AXISBANK.NS", "NTPC.NS", "BAJAJFINSV.NS",
    "ONGC.NS", "ADANIPORTS.NS", "ADANIENT.NS", "BEL.NS", "POWERGRID.NS",
    "WIPRO.NS", "DABUR.NS", "GRASIM.NS", "SBIN.NS", "TITAN.NS",
    "DRREDDY.NS", "SHRIRAMFIN.NS", "TECHM.NS", "BRITANNIA.NS", "EICHERMOT.NS",
    "TATACONSUM.NS", "DIVISLAB.NS", "TRENT.NS", "VEDL.NS", "IRFC.NS",
    "COLPAL.NS", "BRITANNIA.NS", "EIDPARRY.NS", "UBL.NS", "ADANIGREEN.NS",
    "LTI.NS", "PIDILITIND.NS", "HDFC.NS", "UPL.NS", "COALINDIA.NS",
    "ADANIPOWER.NS", "BPCL.NS", "JSWSTEEL.NS", "DMART.NS", "LTIMINDTREE.NS",
    "ICICIPRULI.NS", "HAVELLS.NS", "TVSMOTOR.NS", "LSHORTBANK?", # fix
    "MINDTREE.NS", "PNB.NS", "BANKBARODA.NS", "GAIL.NS", "GRASIM.NS",
    "IRCTC.NS", "NHPC.NS", "POWERGRID.NS", "RIL.NS", "SBILIFE.NS",
    "SRF.NS", "BEL.NS", "IDFCFIRSTB.NS", "TRENT.NS", "TORNTPHARM.NS",
    "VEDANTA.NS", "ZEEL.NS", "ASHOKLEY.NS", "BANDHANBNK.NS", "CIPLA.NS",
    "HINDZINC.NS", "SHREECEM.NS", "TATAMOTORS.NS", "BHARATFORG.NS", "JIOFIN.NS",
    "PFC.NS", "RBLBANK.NS", "MUTHOOTFIN.NS", "CANBK.NS", "SRTRANSFIN.NS",
    "BALKRISIND.NS", "PAGEIND.NS", "CADILAHC.NS", "LICI.NS", "AMBUJACEM.NS",
    "BOSCHLTD.NS", "COLPAL.NS", "AUROPHARMA.NS"
]


data_dir = 'backend/data/raw'
os.makedirs(data_dir, exist_ok=True)

required_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

for symbol in symbols:
    try:
        print(f"Downloading: {symbol}")
        df = yf.download(symbol, period="10y")
        if not df.empty:
            df.reset_index(inplace=True)
            for col in required_columns:
                if col not in df.columns:
                    df[col] = pd.NA
            df = df[required_columns]
            output_file = os.path.join(data_dir, f"{symbol}.csv")
            df.to_csv(output_file, index=False)
            print(f"✅ Saved: {output_file}")
        else:
            print(f"⚠️ No data for: {symbol} (skipped)")
    except Exception as e:
        print(f"❌ Error downloading {symbol}: {e}")
