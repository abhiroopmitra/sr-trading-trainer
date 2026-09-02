import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import yfinance as yf  # Changed from openbb for cloud compatibility

st.set_page_config(layout="wide")
st.title("📈 Support & Resistance Drawing Trainer")
st.markdown("""
Draw your own support and resistance lines directly on the chart, then reveal the algorithm's answer to check yourself.
""")

# --- Sidebar Controls ---
st.sidebar.header("Settings")
ticker_input = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
days = st.sidebar.slider("Historical Days", min_value=30, max_value=365, value=120)
show_levels = st.sidebar.checkbox("✅ Reveal Algorithm's Support & Resistance", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How to draw:**
1. Click the **line icon** (✏️) in the chart's top toolbar.
2. Click and drag on the chart to draw a horizontal line.
3. Hold **Shift** while dragging to keep it perfectly horizontal.
4. Use the eraser icon to remove a line.
""")

# --- Fetch Data ---
@st.cache_data(ttl=3600)
def fetch_data(symbol, period_days):
    try:
        # Fetching directly from yfinance to avoid PermissionErrors on Cloud
        data = yf.download(symbol, period=f"{period_days+20}d", interval="1d")
        if data.empty:
            return None
        df = data.tail(period_days).reset_index()
        # Clean up column names (yfinance sometimes returns MultiIndex)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

df = fetch_data(ticker_input, days)

if df is not None and not df.empty:

    # --- Algorithm Detection (The Answer Key) ---
    # We look for local peaks/valleys
    order = 5
    support_idx = argrelextrema(df['low'].values, np.less_equal, order=order)[0]
    resistance_idx = argrelextrema(df['high'].values, np.greater_equal, order=order)[0]
    
    supports = df.iloc[support_idx]['low'].tolist()
    resistances = df.iloc[resistance_idx]['high'].tolist()

    # --- Build Chart ---
    fig = go.Figure(data=[go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name=ticker_input
    )])

    # Overlay algorithm's answer lines (only if toggled on)
    if show_levels:
        for s in supports:
            fig.add_shape(type="line", x0=df['date'].iloc[0], x1=df['date'].iloc[-1],
                           y0=s, y1=s, line=dict(color="lime", width=1.5, dash="dot"))
        for r in resistances:
            fig.add_shape(type="line", x0=df['date'].iloc[0], x1=df['date'].iloc[-1],
                           y0=r, y1=r, line=dict(color="red", width=1.5, dash="dot"))

    # Chart Layout & Drawing Tools
    fig.update_layout(
        newshape=dict(line_color="cyan", line_width=2),
        dragmode="drawline",              
        uirevision=ticker_input, # Keeps drawings if you change settings but keep same stock
        template="plotly_dark",
        height=700,
        xaxis_rangeslider_visible=False,
    )

    config = {
        "modeBarButtonsToAdd": ["drawline", "eraseshape"],
        "displaylogo": False,
    }

    st.plotly_chart(fig, use_container_width=True, config=config)

    st.info("🟦 **Cyan lines** = your guesses. Draw them with the pencil tool.\n\n"
            "🟢 **Green dotted** = Algorithm Support | 🔴 **Red dotted** = Algorithm Resistance")
else:
    st.warning("No data found. Please check the ticker symbol.")
