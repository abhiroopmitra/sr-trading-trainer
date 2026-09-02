import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import yfinance as yf

st.set_page_config(layout="wide")
st.title("📈 Support & Resistance Drawing Trainer")

st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()

# Timeframe Selection
timeframe = st.sidebar.selectbox(
    "Timeframe",
    options=["1m", "5m", "15m", "1h", "1d"],
    index=4  # default to daily
)

days = st.sidebar.slider("Lookback Period (days)", min_value=30, max_value=365, value=180)
show_levels = st.sidebar.checkbox("✅ Show Algorithm Support & Resistance", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How to use:**
- Draw your own lines using the **pencil** tool in the chart toolbar.
- Hold **Shift** while drawing to make perfect horizontal lines.
- Toggle the checkbox to reveal the algorithm's levels.
""")

# Fetch data with chosen timeframe
@st.cache_data(ttl=3600)
def fetch_data(symbol, period_days, interval):
    try:
        data = yf.download(symbol, period=f"{period_days}d", interval=interval, progress=False)
        if data.empty:
            return None
        df = data.reset_index()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

df = fetch_data(ticker, days, timeframe)

if df is not None and not df.empty:

    # === IMPROVED S/R DETECTION ===
    # Higher order = fewer but stronger levels
    order = 15  

    support_idx = argrelextrema(df['low'].values, np.less_equal, order=order)[0]
    resistance_idx = argrelextrema(df['high'].values, np.greater_equal, order=order)[0]

    supports = df.iloc[support_idx]['low'].tolist()
    resistances = df.iloc[resistance_idx]['high'].tolist()

    # Build chart
    fig = go.Figure(data=[go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name=ticker
    )])

    if show_levels:
        # Support lines (Green)
        for s in supports:
            fig.add_shape(type="line", x0=df['date'].iloc[0], x1=df['date'].iloc[-1],
                           y0=s, y1=s, line=dict(color="lime", width=2, dash="dot"))

        # Resistance lines (Red)
        for r in resistances:
            fig.add_shape(type="line", x0=df['date'].iloc[0], x1=df['date'].iloc[-1],
                           y0=r, y1=r, line=dict(color="red", width=2, dash="dot"))

    fig.update_layout(
        newshape=dict(line_color="cyan", line_width=2),
        dragmode="drawline",
        uirevision=ticker,
        template="plotly_dark",
        height=700,
        xaxis_rangeslider_visible=False,
        title=f"{ticker} | {timeframe} | Last {days} days"
    )

    config = {
        "modeBarButtonsToAdd": ["drawline", "eraseshape"],
        "displaylogo": False,
    }

    st.plotly_chart(fig, use_container_width=True, config=config)

    # Explanation
    st.markdown("""
    ### How the Algorithm Works
    - The bot finds **local highs** (resistance) and **local lows** (support).
    - It only keeps levels where the price reversed **strongly** (controlled by the `order` parameter).
    - This is why you see fewer lines than before — we are now showing only the more meaningful levels.
    - **Green dotted lines** = Support  
    - **Red dotted lines** = Resistance
    """)

else:
    st.warning("No data available for this combination. Try a larger timeframe or shorter lookback period.")
