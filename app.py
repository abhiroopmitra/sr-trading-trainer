import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from openbb import obb

# Set up web app interface
st.set_page_config(layout="wide")
st.title("📈 Support & Resistance Practice Tool")
st.markdown("Practice identifying Support & Resistance levels on historical data, then reveal the algorithm's calculation to check your work.")

# Sidebar Controls
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
days = st.sidebar.slider("Historical Days", min_value=30, max_value=365, value=120)

# 1. Fetch Historical Data using OpenBB
@st.cache_data(ttl=3600)
def fetch_data(symbol, period_days):
    try:
        # Fetch daily data using OpenBB Platform
        res = obb.equity.price.historical(symbol, provider="yfinance")
        df = res.to_df()
        df = df.tail(period_days)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

df = fetch_data(ticker, days)

if df is not None and not df.empty:
    # Handle index if date is in index
    if 'date' not in df.columns:
        df = df.reset_index()

    # 2. Algorithmic S/R Detection (Local Minima/Maxima)
    # Order=5 means looking 5 candles left and right
    order = 5
    
    # Find Support (Local Minima)
    df['is_support'] = False
    support_idx = argrelextrema(df['low'].values, np.less_equal, order=order)[0]
    df.iloc[support_idx, df.columns.get_loc('is_support')] = True

    # Find Resistance (Local Maxima)
    df['is_resistance'] = False
    resistance_idx = argrelextrema(df['high'].values, np.greater_equal, order=order)[0]
    df.iloc[resistance_idx, df.columns.get_loc('is_resistance')] = True

    # Extract price values for lines
    supports = df[df['is_support']]['low'].tolist()
    resistances = df[df['is_resistance']]['high'].tolist()

    # Practice Mode Toggle
    show_levels = st.sidebar.checkbox("Show Algorithmic Support & Resistance", value=False)

    # 3. Create Interactive Candlestick Chart
    fig = go.Figure(data=[go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=ticker
    )])

    # Add S/R Lines if user checked the box
    if show_levels:
        # Draw Support Lines (Green)
        for s in supports:
            fig.add_shape(
                type="line", x0=df['date'].iloc[0], x1=df['date'].iloc[-1],
                y0=s, y1=s,
                line=dict(color="Green", width=1.5, dash="dash"),
                name="Support"
            )
        # Draw Resistance Lines (Red)
        for r in resistances:
            fig.add_shape(
                type="line", x0=df['date'].iloc[0], x1=df['date'].iloc[-1],
                y0=r, y1=r,
                line=dict(color="Red", width=1.5, dash="dash"),
                name="Resistance"
            )

    fig.update_layout(
        title=f"{ticker} Price Chart",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=650
    )

    st.plotly_chart(fig, use_container_width=True)

    # Instructions for User
    st.info("""
    **How to practice:**
    1. Look at the raw candlestick chart above.
    2. Write down or mentally locate the top 2-3 resistance levels (peaks) and support levels (floors).
    3. Check the **"Show Algorithmic Support & Resistance"** box in the sidebar to see how your eyes compare to the local minima/maxima calculation!
    """)
