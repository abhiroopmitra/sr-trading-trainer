import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import yfinance as yf

st.set_page_config(layout="wide")
st.title("📈 Support & Resistance Drawing Trainer")

# --- SIDEBAR ---
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()

timeframe = st.sidebar.selectbox(
    "Timeframe",
    options=["1m", "5m", "15m", "1h", "1d"],
    index=4
)

# Important Note for User
if timeframe == "1m":
    st.sidebar.warning("1m data only available for last 7 days.")
    days = st.sidebar.slider("Lookback Period", 1, 7, 5)
elif timeframe in ["5m", "15m", "1h"]:
    st.sidebar.warning("Intraday data limited to last 60 days.")
    days = st.sidebar.slider("Lookback Period", 1, 60, 30)
else:
    days = st.sidebar.slider("Lookback Period (days)", 30, 365, 180)

show_levels = st.sidebar.checkbox("✅ Show Algorithm Support & Resistance", value=False)
order = st.sidebar.slider("Sensitivity (Higher = Fewer Lines)", 5, 50, 20)

# --- DATA FETCHING ---
@st.cache_data(ttl=600)
def fetch_data(symbol, period_days, interval):
    try:
        # We use a slightly longer period for the API call to ensure we get enough data
        period_str = f"{period_days}d"
        data = yf.download(symbol, period=period_str, interval=interval, auto_adjust=True, progress=False)
        
        if data.empty:
            return None
        
        # 1. Fix MultiIndex columns if they exist
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # 2. Move Date/Datetime index into a column
        df = data.reset_index()
        
        # 3. Standardize column names to lowercase
        df.columns = [str(col).lower() for col in df.columns]
        
        # 4. Identify the date/time column (it's usually the first one after reset_index)
        # We rename it to 'timestamp' for consistency
        df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

df = fetch_data(ticker, days, timeframe)

# --- CHARTING & LOGIC ---
if df is not None and not df.empty:

    # Algorithmic Detection
    # Finding local peaks and valleys
    support_idx = argrelextrema(df['low'].values, np.less_equal, order=order)[0]
    resistance_idx = argrelextrema(df['high'].values, np.greater_equal, order=order)[0]

    supports = df.iloc[support_idx]['low'].tolist()
    resistances = df.iloc[resistance_idx]['high'].tolist()

    # Create Chart
    fig = go.Figure(data=[go.Candlestick(
        x=df['timestamp'], 
        open=df['open'], 
        high=df['high'],
        low=df['low'], 
        close=df['close'], 
        name=ticker
    )])

    # Draw Algo Lines
    if show_levels:
        for s in supports:
            fig.add_shape(type="line", x0=df['timestamp'].iloc[0], x1=df['timestamp'].iloc[-1],
                           y0=s, y1=s, line=dict(color="lime", width=1, dash="dot"))
        for r in resistances:
            fig.add_shape(type="line", x0=df['timestamp'].iloc[0], x1=df['timestamp'].iloc[-1],
                           y0=r, y1=r, line=dict(color="red", width=1, dash="dot"))

    fig.update_layout(
        newshape=dict(line_color="cyan", line_width=2),
        dragmode="drawline",
        uirevision=ticker + timeframe, # Preserves zoom/lines when switching settings
        template="plotly_dark",
        height=700,
        xaxis_rangeslider_visible=False,
        title=f"{ticker} ({timeframe}) - Draw your lines now"
    )

    config = {"modeBarButtonsToAdd": ["drawline", "eraseshape"], "displaylogo": False}
    st.plotly_chart(fig, use_container_width=True, config=config)

    # Explanation of Outcome
    st.subheader("How to interpret the lines")
    col1, col2 = st.columns(2)
    with col1:
        st.write("🟢 **Green Dotted Lines (Support):**")
        st.write("These are 'floors' where the price hit a low and bounced back up. In trading, this suggests a lot of buyers were waiting at that price.")
    with col2:
        st.write("🔴 **Red Dotted Lines (Resistance):**")
        st.write("These are 'ceilings' where the price hit a peak and fell back down. In trading, this suggests sellers were waiting to take profit at that price.")

else:
    st.warning("No data found. Check your ticker or reduce the lookback period (especially for 1m data).")
