import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import yfinance as yf

st.set_page_config(layout="wide", page_title="S&R Paper Trading Sim")

# ==========================================
# 1. INITIALIZE SESSION STATE
# ==========================================
if "sim_active" not in st.session_state:
    st.session_state.sim_active = False
if "balance" not in st.session_state:
    st.session_state.balance = 1000.00
if "shares" not in st.session_state:
    st.session_state.shares = 0.0
if "stop_loss" not in st.session_state:
    st.session_state.stop_loss = None
if "step" not in st.session_state:
    st.session_state.step = 0
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "trade_log" not in st.session_state:
    st.session_state.trade_log = []

# ==========================================
# 2. DATA FETCHING HELPER
# ==========================================
@st.cache_data(ttl=3600)
def fetch_1m_data(ticker):
    try:
        data = yf.download(ticker, period="7d", interval="1m", auto_adjust=True, progress=False)
        if data.empty: return None
        
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        
        df = data.reset_index()
        df.columns = [str(col).lower() for col in df.columns]
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
            
        df["label"] = df["timestamp"].dt.strftime("%H:%M")
        df["date_only"] = df["timestamp"].dt.date
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None

# ==========================================
# 3. HELPER: STEP FORWARD & CHECK STOP LOSS
# ==========================================
def advance_time(steps_to_move):
    """Advances time candle by candle, checking for Stop-Loss triggers."""
    max_steps = len(st.session_state.df) - 1
    
    for _ in range(steps_to_move):
        if st.session_state.step >= max_steps:
            st.toast("⚠️ Market closed for the day!", icon="🔔")
            break
            
        st.session_state.step += 1
        current_candle = st.session_state.df.iloc[st.session_state.step]
        candle_low = current_candle["low"]
        candle_time = current_candle["label"]
        
        # Check Stop Loss Trigger
        if st.session_state.shares > 0 and st.session_state.stop_loss is not None:
            if candle_low <= st.session_state.stop_loss:
                # Execution Price (Stop-Loss price or candle open if gap down)
                exec_price = min(st.session_state.stop_loss, current_candle["open"])
                proceeds = st.session_state.shares * exec_price
                st.session_state.balance += proceeds
                
                st.session_state.trade_log.append(
                    f"🛑 STOP-LOSS TRIGGERED at {candle_time}! Sold {st.session_state.shares:.2f} shares at ${exec_price:.2f}. Proceeds: ${proceeds:.2f}"
                )
                st.toast(f"🛑 STOP-LOSS TRIGGERED at ${exec_price:.2f}!", icon="💥")
                
                st.session_state.shares = 0.0
                st.session_state.stop_loss = None
                break  # Stop advancing time on trigger so user can inspect chart

# ==========================================
# 4. SIDEBAR: SIMULATOR SETUP
# ==========================================
st.sidebar.header("⚙️ 1. Setup Simulation")
ticker = st.sidebar.text_input("Ticker", value="SPY").upper()

raw_df = fetch_1m_data(ticker)

if raw_df is not None and not raw_df.empty:
    available_dates = raw_df["date_only"].unique()
    selected_date = st.sidebar.selectbox("Select Day to Trade", available_dates)
    
    day_df = raw_df[raw_df["date_only"] == selected_date].reset_index(drop=True)
    times = day_df["label"].tolist()
    default_idx = min(90, len(times) - 1) if len(times) > 90 else 0
    start_time = st.sidebar.selectbox("Start Time", times, index=default_idx)

    if st.sidebar.button("🚀 Start / Reset Simulation"):
        st.session_state.df = day_df
        st.session_state.step = day_df[day_df["label"] == start_time].index[0]
        st.session_state.balance = 1000.00
        st.session_state.shares = 0.0
        st.session_state.stop_loss = None
        st.session_state.trade_log = []
        st.session_state.sim_active = True
        st.rerun()

st.sidebar.markdown("---")
show_bot_lines = st.sidebar.checkbox("Show Bot S&R Lines", value=False)
order = st.sidebar.slider("Bot Sensitivity", 5, 50, 20)

# ==========================================
# 5. MAIN TRADING DASHBOARD
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
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("Total Equity", f"${total_equity:.2f}", f"${pnl:.2f}", delta_color=pnl_color)
    col3.metric("Cash Available", f"${st.session_state.balance:.2f}")
    col4.metric("Open Position", f"{st.session_state.shares:.4f} shares", f"${position_value:.2f}")
    col5.metric("Active Stop-Loss", f"${st.session_state.stop_loss:.2f}" if st.session_state.stop_loss else "None")

    # Candlestick Chart
    fig = go.Figure(data=[go.Candlestick(
        x=df_sim["label"], open=df_sim["open"], high=df_sim["high"],
        low=df_sim["low"], close=df_sim["close"], name=ticker
    )])

    # Bot S&R lines
    if show_bot_lines and len(df_sim) > order * 2:
        support_idx = argrelextrema(df_sim["low"].values, np.less_equal, order=order)[0]
        resistance_idx = argrelextrema(df_sim["high"].values, np.greater_equal, order=order)[0]
        
        for s in df_sim.iloc[support_idx]["low"].unique():
            fig.add_shape(type="line", x0=df_sim["label"].iloc[0], x1=df_sim["label"].iloc[-1], y0=s, y1=s, line=dict(color="lime", dash="dot"))
        for r in df_sim.iloc[resistance_idx]["high"].unique():
            fig.add_shape(type="line", x0=df_sim["label"].iloc[0], x1=df_sim["label"].iloc[-1], y0=r, y1=r, line=dict(color="red", dash="dot"))

    # Active Stop-Loss Line (Orange Dashed Line)
    if st.session_state.shares > 0 and st.session_state.stop_loss:
        fig.add_shape(
            type="line", x0=df_sim["label"].iloc[0], x1=df_sim["label"].iloc[-1],
            y0=st.session_state.stop_loss, y1=st.session_state.stop_loss,
            line=dict(color="orange", width=2, dash="dash")
        )

    fig.update_layout(
        template="plotly_dark", height=500, xaxis_rangeslider_visible=False,
        title=f"Market Time: {current_time}", margin=dict(l=10, r=10, t=40, b=10),
        dragmode="pan"
    )
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)

    # Trading Execution Controls
    st.markdown("### 🎮 Trade Execution")
    tcol1, tcol2, tcol3, tcol4, tcol5 = st.columns([1.2, 1.2, 1, 1, 2])
    
    with tcol1:
        bet_size = st.number_input("Trade Amount ($)", min_value=10.0, max_value=max(10.0, st.session_state.balance), value=min(100.0, st.session_state.balance), step=10.0)
    
    with tcol2:
        # Default stop-loss is set 0.5% below current price
        default_sl = round(current_price * 0.995, 2)
        sl_input = st.number_input("Stop-Loss Price ($)", min_value=0.0, max_value=current_price, value=default_sl, step=0.10)

    with tcol3:
        st.write("")
        st.write("")
        if st.button("🟢 BUY", use_container_width=True):
            if bet_size <= st.session_state.balance:
                if sl_input >= current_price:
                    st.error("Stop-loss must be below current price!")
                else:
                    shares_bought = bet_size / current_price
                    st.session_state.balance -= bet_size
                    st.session_state.shares += shares_bought
                    st.session_state.stop_loss = sl_input
                    st.session_state.trade_log.append(
                        f"{current_time}: 
