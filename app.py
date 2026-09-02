import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="S&R Master Challenge")

# --- GAME STYLING ---
st.title("🎮 S&R Challenge: Intraday Edition")

# --- SIDEBAR / GAME CONTROLS ---
st.sidebar.header("🕹️ Game Setup")
ticker = st.sidebar.text_input("Stock Ticker", value="SPY").upper()

timeframe = st.sidebar.selectbox("Timeframe (Candle Size)", ["1m", "5m", "15m", "1h", "1d"], index=0)

# Dynamic Windowing: Allow user to look at very short periods
if timeframe in ["1m", "5m"]:
    window_type = st.sidebar.radio("View Window", ["Last few Hours", "Full Day"])
    if window_type == "Last few Hours":
        hours_to_show = st.sidebar.slider("Hours to look back", 1, 8, 2)
        days_to_fetch = 1
    else:
        hours_to_show = None
        days_to_fetch = st.sidebar.slider("Days to fetch", 1, 7, 1)
else:
    hours_to_show = None
    days_to_fetch = st.sidebar.slider("Days to look back", 5, 365, 30)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Submit Your Guesses")
user_input = st.sidebar.text_input("Enter prices (e.g. 450.2, 451.0)", placeholder="0.0, 0.0")
submit_button = st.sidebar.button("🏆 Grade My Guesses")

# Bot Sensitivity (Higher order = ignores noise)
order = st.sidebar.slider("Bot Sensitivity (Noise Filter)", 5, 100, 30)

# --- DATA FETCHING ---
@st.cache_data(ttl=60) # Short cache for intraday
def fetch_data(symbol, period_days, interval, hours):
    try:
        data = yf.download(symbol, period=f"{period_days}d", interval=interval, auto_adjust=True, progress=False)
        if data.empty: return None
        
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        df = data.reset_index()
        df.columns = [str(col).lower() for col in df.columns]
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
        df = df.dropna().drop_duplicates(subset="timestamp")

        # --- THE ZOOM LOGIC ---
        if hours:
            cutoff = df["timestamp"].max() - pd.Timedelta(hours=hours)
            df = df[df["timestamp"] > cutoff]
        
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

df = fetch_data(ticker, days_to_fetch, timeframe, hours_to_show)

if df is not None and not df.empty:
    # Gap Fix labels
    label_fmt = "%H:%M" if timeframe in ["1m", "5m"] else "%b %d %H:%M"
    df["label"] = df["timestamp"].dt.strftime(label_fmt)

    # --- BOT LOGIC (Noise Filtered) ---
    # Using 'order' to ensure a peak is higher than 'order' number of candles around it
    support_idx = argrelextrema(df["low"].values, np.less_equal, order=order)[0]
    resistance_idx = argrelextrema(df["high"].values, np.greater_equal, order=order)[0]

    bot_supports = sorted(list(set(df.iloc[support_idx]["low"].round(2))))
    bot_resistances = sorted(list(set(df.iloc[resistance_idx]["high"].round(2))))
    all_bot_levels = bot_supports + bot_resistances

    # --- GRADING ENGINE ---
    if submit_button and user_input:
        try:
            user_guesses = [float(x.strip()) for x in user_input.split(",")]
            score = 0
            feedback = []
            
            # Use a tighter tolerance for 1m charts (0.1% instead of 0.5%)
            tolerance = 0.001 if timeframe == "1m" else 0.003

            for guess in user_guesses:
                found_match = False
                for bot_level in all_bot_levels:
                    if abs(guess - bot_level) / bot_level <= tolerance:
                        found_match = True
                        break
                if found_match:
                    score += 1
                    feedback.append(f"✅ **{guess}** is a HIT!")
                else:
                    feedback.append(f"❌ **{guess}** is a MISS.")
            
            st.sidebar.success(f"Score: {score} / {len(user_guesses)}")
            for f in feedback: st.sidebar.write(f)
        except:
            st.sidebar.error("Enter valid numbers.")

    # --- CHARTING ---
    fig = go.Figure(data=[go.Candlestick(
        x=df["label"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name=ticker
    )])

    if submit_button:
        x_span = [df["label"].iloc[0], df["label"].iloc[-1]]
        for s in bot_supports:
            fig.add_trace(go.Scatter(x=x_span, y=[s, s], mode="lines", line=dict(color="lime", width=2, dash="dot"), name="Support"))
        for r in bot_resistances:
            fig.add_trace(go.Scatter(x=x_span, y=[r, r], mode="lines", line=dict(color="red", width=2, dash="dot"), name="Resistance"))

    fig.update_layout(
        template="plotly_dark", height=600, xaxis_rangeslider_visible=False,
        title=f"{ticker} {timeframe} - Last {hours_to_show if hours_to_show else days_to_fetch} hours/days",
        dragmode="pan", margin=dict(l=10, r=10, t=40, b=10)
    )
    fig.update_xaxes(type="category", nticks=10)
    st.plotly_chart(fig, use_container_width=True)

    st.info("💡 **Tip:** If the bot shows too many lines, increase the **Noise Filter** slider in the sidebar.")

else:
    st.warning("Market might be closed or ticker invalid. Try SPY or AAPL during market hours.")
