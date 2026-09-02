import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import yfinance as yf

st.set_page_config(layout="wide", page_title="S&R Paper Trading Sim")

# ==========================================
# 1. INITIALIZE SESSION STATE (APP MEMORY)
# ==========================================
if "sim_active" not in st.session_state:
    st.session_state.sim_active = False
if "balance" not in st.session_state:
    st.session_state.balance = 1000.00
if "shares" not in st.session_state:
    st.session_state.shares = 0.0
if "step" not in st.session_state:
    st.session_state.step = 0
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "trade_log" not in st.session_state:
    st.session_state.trade_log = []

# ==========================================
# 2. DATA FETCHING HELPER (CLOUD SAFE)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_1m_data(ticker):
    try:
        # Fetch last 7 days of 1m data
        data = yf.download(ticker, period="7d", interval="1m", auto_adjust=True, progress=False)
        if data.empty: return None
        
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        
        df = data.reset_index()
        df.columns = [str(col).lower() for col in df.columns]
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
        
        # Clean timestamp without requiring system tzdata files
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is not None:
            # Strip timezone offset to avoid cloud ZoneInfo errors
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
            
        # Clean formatting
        df["label"] = df["timestamp"].dt.strftime("%H:%M")
        df["date_only"] = df["timestamp"].dt.date
        
        # Drop non-trading rows
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None

# ==========================================
# 3. SIDEBAR: SIMULATOR SETUP
# ==========================================
st.sidebar.header("⚙️ 1. Setup Simulation")
ticker = st.sidebar.text_input("Ticker", value="SPY").upper()

raw_df = fetch_1m_data(ticker)

if raw_df is not None and not raw_df.empty:
    available_dates = raw_df["date_only"].unique()
    selected_date = st.sidebar.selectbox("Select Day to Trade", available_dates)
    
    # Filter data to selected day
    day_df = raw_df[raw_df["date_only"] == selected_date].reset_index(drop=True)
    
    times = day_df["label"].tolist()
    # Default to around 11:00 AM or midway
    default_idx = min(90, len(times) - 1) if len(times) > 90 else 0
    start_time = st.sidebar.selectbox("Start Time", times, index=default_idx)

    if st.sidebar.button("🚀 Start / Reset Simulation"):
        st.session_state.df = day_df
        st.session_state.step = day_df[day_df["label"] == start_time].index[0]
        st.session_state.balance = 1000.00
        st.session_state.shares = 0.0
        st.session_state.trade_log = []
        st.session_state.sim_active = True
        st.rerun()

st.sidebar.markdown("---")
show_bot_lines = st.sidebar.checkbox("Show Bot S&R Lines", value=False)
order = st.sidebar.slider("Bot Sensitivity", 5, 50, 20)

# ==========================================
# 4. MAIN TRADING DASHBOARD
# ==========================================
st.title("💹 Market Replay Simulator")

if st.session_state.sim_active:
    current_step = st.session_state.step
    df_sim = st.session_state.df.iloc[:current_step + 1]
    current_price = df_sim.iloc[-1]["close"]
    current_time = df_sim.iloc[-1]["label"]

    # Calculate portfolio values
    position_value = st.session_state.shares * current_price
    total_equity = st.session_state.balance + position_value
    pnl = total_equity - 1000.00
    pnl_color = "normal" if pnl == 0 else ("inverse" if pnl > 0 else "off")

    # Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("Total Equity", f"${total_equity:.2f}", f"${pnl:.2f}", delta_color=pnl_color)
    col3.metric("Cash Available", f"${st.session_state.balance:.2f}")
    col4.metric("Open Position", f"{st.session_state.shares:.4f} shares", f"${position_value:.2f}")

    # Candlestick Chart
    fig = go.Figure(data=[go.Candlestick(
        x=df_sim["label"], open=df_sim["open"], high=df_sim["high"],
        low=df_sim["low"], close=df_sim["close"], name=ticker
    )])

    # Bot S&R lines calculated only up to current step (No Future Peeking)
    if show_bot_lines and len(df_sim) > order * 2:
        support_idx = argrelextrema(df_sim["low"].values, np.less_equal, order=order)[0]
        resistance_idx = argrelextrema(df_sim["high"].values, np.greater_equal, order=order)[0]
        
        for s in df_sim.iloc[support_idx]["low"].unique():
            fig.add_shape(type="line", x0=df_sim["label"].iloc[0], x1=df_sim["label"].iloc[-1], y0=s, y1=s, line=dict(color="lime", dash="dot"))
        for r in df_sim.iloc[resistance_idx]["high"].unique():
            fig.add_shape(type="line", x0=df_sim["label"].iloc[0], x1=df_sim["label"].iloc[-1], y0=r, y1=r, line=dict(color="red", dash="dot"))

    fig.update_layout(
        template="plotly_dark", height=500, xaxis_rangeslider_visible=False,
        title=f"Market Time: {current_time}", margin=dict(l=10, r=10, t=40, b=10),
        dragmode="pan"
    )
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)

    # Trading Execution Controls
    st.markdown("### 🎮 Trade Execution")
    tcol1, tcol2, tcol3, tcol4 = st.columns([1, 1, 1, 2])
    
    with tcol1:
        bet_size = st.number_input("Trade Amount ($)", min_value=10.0, max_value=max(10.0, st.session_state.balance), value=min(100.0, st.session_state.balance), step=10.0)
    
    with tcol2:
        st.write("")
        st.write("")
        if st.button("🟢 BUY", use_container_width=True):
            if bet_size <= st.session_state.balance:
                shares_bought = bet_size / current_price
                st.session_state.balance -= bet_size
                st.session_state.shares += shares_bought
                st.session_state.trade_log.append(f"{current_time}: BOUGHT {shares_bought:.2f} shares at ${current_price:.2f}")
                st.rerun()
            else:
                st.error("Not enough cash!")

    with tcol3:
        st.write("")
        st.write("")
        if st.button("🔴 SELL ALL", use_container_width=True):
            if st.session_state.shares > 0:
                proceeds = st.session_state.shares * current_price
                st.session_state.balance += proceeds
                st.session_state.trade_log.append(f"{current_time}: SOLD all shares at ${current_price:.2f}. Proceeds: ${proceeds:.2f}")
                st.session_state.shares = 0.0
                st.rerun()
            else:
                st.warning("No shares to sell!")

    with tcol4:
        st.write("")
        st.write("")
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            if st.button("▶️ Next 1 Min", use_container_width=True):
                if st.session_state.step < len(st.session_state.df) - 1:
                    st.session_state.step += 1
                    st.rerun()
                else:
                    st.error("Market closed for the day!")
        with subcol2:
            if st.button("⏭️ Fast Forward 5 Min", use_container_width=True):
                if st.session_state.step < len(st.session_state.df) - 5:
                    st.session_state.step += 5
                    st.rerun()

    # Trade Log
    if st.session_state.trade_log:
        with st.expander("📝 Trade History"):
            for log in reversed(st.session_state.trade_log):
                st.text(log)

else:
    st.info("👈 Select a Date and Start Time in the sidebar, then click **Start Simulation**.")
