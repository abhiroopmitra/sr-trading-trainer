import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="S&R Paper Trading Sim")

# ==========================================
# 1. SESSION STATE
# ==========================================
defaults = {
    "sim_active": False, "balance": 1000.00, "shares": 0.0,
    "stop_loss": None, "target": None, "entry_price": None,
    "step": 0, "df": pd.DataFrame(), "trade_log": [], "sim_start_idx": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================
# 2. DATA FETCHING (multi-day 1m data)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_data(ticker, target_date):
    try:
        start = target_date - timedelta(days=7)
        end = target_date + timedelta(days=1)
        data = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                           end=end.strftime("%Y-%m-%d"), interval="1m",
                           auto_adjust=True, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        df = data.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        df = df.dropna(subset=["close"]).drop_duplicates(subset="timestamp")
        if "volume" not in df.columns:
            df["volume"] = 0.0
        df["volume"] = df["volume"].fillna(0.0)
        df["date_only"] = df["timestamp"].dt.date
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None

# ==========================================
# 3. RESAMPLING (1m -> 5m / 15m view)
# ==========================================
def resample_view(df, timeframe):
    if timeframe == "1m":
        out = df.copy()
    else:
        rule = {"5m": "5min", "15m": "15min"}[timeframe]
        out = (df.set_index("timestamp").resample(rule)
                 .agg({"open": "first", "high": "max",
                       "low": "min", "close": "last",
                       "volume": "sum"})
                 .dropna(subset=["close"]).reset_index())
    out["label"] = out["timestamp"].dt.strftime("%m-%d %H:%M")
    return out

# ==========================================
# 4. POSITION CLOSE HELPER
# ==========================================
def close_position(qty, price, time_str, reason=""):
    pos = st.session_state.shares
    if pos > 0:
        qty = min(qty, pos)
        st.session_state.balance += qty * price
        st.session_state.shares -= qty
        st.session_state.trade_log.append(
            f"{time_str}: SOLD {qty:.4f} sh at ${price:.2f} {reason}")
    elif pos < 0:
        qty = min(qty, abs(pos))
        st.session_state.balance -= qty * price
        st.session_state.shares += qty
        st.session_state.trade_log.append(
            f"{time_str}: COVERED {qty:.4f} sh at ${price:.2f} {reason}")
    if st.session_state.shares == 0:
        st.session_state.stop_loss = None
        st.session_state.target = None
        st.session_state.entry_price = None

# ==========================================
# 5. TIME ADVANCE + SL/TARGET CHECKS (1m engine)
# ==========================================
def advance_time(steps_to_move):
    max_steps = len(st.session_state.df) - 1
    for _ in range(steps_to_move):
        if st.session_state.step >= max_steps:
            st.toast("Market closed for the day!", icon="🔔")
            break
        st.session_state.step += 1
        candle = st.session_state.df.iloc[st.session_state.step]
        t = candle["timestamp"].strftime("%H:%M")
        pos = st.session_state.shares
        sl, tp = st.session_state.stop_loss, st.session_state.target

        if pos > 0:
            if sl is not None and candle["low"] <= sl:
                exec_p = min(sl, candle["open"])
                close_position(pos, exec_p, t, "(🛑 STOP-LOSS)")
                st.toast(f"🛑 Long stopped out at ${exec_p:.2f}", icon="💥")
                break
            if tp is not None and candle["high"] >= tp:
                exec_p = max(tp, candle["open"])
                close_position(pos, exec_p, t, "(🎯 TARGET HIT)")
                st.toast(f"🎯 Target hit! Sold at ${exec_p:.2f}", icon="🎉")
                break
        elif pos < 0:
            if sl is not None and candle["high"] >= sl:
                exec_p = max(sl, candle["open"])
                close_position(abs(pos), exec_p, t, "(🛑 STOP-LOSS)")
                st.toast(f"🛑 Short stopped out at ${exec_p:.2f}", icon="💥")
                break
            if tp is not None and candle["low"] <= tp:
                exec_p = min(tp, candle["open"])
                close_position(abs(pos), exec_p, t, "(🎯 TARGET HIT)")
                st.toast(f"🎯 Target hit! Covered at ${exec_p:.2f}", icon="🎉")
                break

# ==========================================
# 6. SIDEBAR SETUP
# ==========================================
st.sidebar.header("⚙️ Setup")
ticker = st.sidebar.text_input("Ticker", value="SPY").upper()
default_date = datetime.now().date() - timedelta(days=2)
selected_date = st.sidebar.date_input("Trading Date", value=default_date,
                                      max_value=datetime.now().date())

raw_df = fetch_data(ticker, selected_date)

if raw_df is not None and selected_date in raw_df["date_only"].values:
    day_mask = raw_df["date_only"] == selected_date
    day_labels = raw_df.loc[day_mask, "timestamp"].dt.strftime("%H:%M").tolist()
    default_idx = min(90, len(day_labels) - 1)
    start_time = st.sidebar.selectbox("Start Time", day_labels, index=default_idx)

    if st.sidebar.button("🚀 Start / Reset Simulation"):
        st.session_state.df = raw_df
        day_indices = raw_df.index[day_mask].tolist()
        st.session_state.sim_start_idx = day_indices[0]
        st.session_state.step = day_indices[day_labels.index(start_time)]
        st.session_state.balance = 1000.00
        st.session_state.shares = 0.0
        st.session_state.stop_loss = None
        st.session_state.target = None
        st.session_state.entry_price = None
        st.session_state.trade_log = []
        st.session_state.sim_active = True
        st.rerun()
else:
    st.sidebar.error("No 1m data for this date (market closed or too old).")

st.sidebar.markdown("---")
chart_tf = st.sidebar.radio("📊 Chart View", ["1m", "5m", "15m"], horizontal=True)
show_context = st.sidebar.checkbox("Show previous days (context)", value=False)
show_vol_ma = st.sidebar.checkbox("Show volume moving average", value=True)
st.sidebar.markdown("**✏️ Draw:** line tool in chart toolbar. Scroll = zoom, drag = pan.")
st.sidebar.caption("Each time step **re-zooms to today** (open → now). Pan left for older days if context is on.")

# ==========================================
# 7. MAIN DASHBOARD
# ==========================================
st.title("💹 Market Replay Simulator")

if st.session_state.sim_active:
    step = st.session_state.step
    full_df = st.session_state.df
    sim0 = st.session_state.sim_start_idx

    today_1m = full_df.iloc[sim0: step + 1]
    if show_context:
        visible_1m = full_df.iloc[: step + 1]
    else:
        visible_1m = today_1m

    current_price = full_df.iloc[step]["close"]
    current_time = full_df.iloc[step]["timestamp"].strftime("%H:%M")
    pos = st.session_state.shares

    position_value = pos * current_price
    total_equity = st.session_state.balance + position_value
    pnl = total_equity - 1000.00
    pnl_color = "normal" if pnl == 0 else ("inverse" if pnl > 0 else "off")

    if pos > 0:
        pos_text, pos_type = f"{pos:.4f} sh", "🟢 LONG"
    elif pos < 0:
        pos_text, pos_type = f"{abs(pos):.4f} sh", "🔴 SHORT"
    else:
        pos_text, pos_type = "—", "FLAT"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Price", f"${current_price:.2f}")
    c2.metric("Equity", f"${total_equity:.2f}", f"${pnl:.2f}", delta_color=pnl_color)
    c3.metric("Cash", f"${st.session_state.balance:.2f}")
    c4.metric(f"Position ({pos_type})", pos_text,
              f"Entry ${st.session_state.entry_price:.2f}" if st.session_state.entry_price else "")
    c5.metric("Stop-Loss", f"${st.session_state.stop_loss:.2f}" if st.session_state.stop_loss else "None")
    c6.metric("Target", f"${st.session_state.target:.2f}" if st.session_state.target else "None")

    # ----- Chart: plot visible bars, camera = today only -----
    view_df = resample_view(visible_1m, chart_tf)

    # Color volume bars green/red to match candle direction
    vol_colors = np.where(
        view_df["close"] >= view_df["open"],
        "rgba(38,166,154,0.5)",   # green (up candle)
        "rgba(239,83,80,0.5)"     # red (down candle)
    )

    # 2-row layout: 75% price, 25% volume, shared x-axis
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=view_df["label"], open=view_df["open"], high=view_df["high"],
            low=view_df["low"], close=view_df["close"], name=ticker,
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=view_df["label"], y=view_df["volume"],
            marker_color=vol_colors, name="Volume",
        ),
        row=2, col=1,
    )

    # Optional volume moving average line
    if show_vol_ma and len(view_df) >= 20:
        view_df["vol_ma"] = view_df["volume"].rolling(20).mean()
        fig.add_trace(
            go.Scatter(
                x=view_df["label"], y=view_df["vol_ma"],
                line=dict(color="#D500F9", width=2),
                name="Vol MA(20)",
            ),
            row=2, col=1,
        )

    if pos != 0:
        if st.session_state.stop_loss:
            fig.add_hline(y=st.session_state.stop_loss, row=1, col=1,
                          line=dict(color="orange", width=2, dash="dash"))
        if st.session_state.target:
            fig.add_hline(y=st.session_state.target, row=1, col=1,
                          line=dict(color="lime", width=2, dash="dash"))
        if st.session_state.entry_price:
            fig.add_hline(y=st.session_state.entry_price, row=1, col=1,
                          line=dict(color="cyan", width=1, dash="dot"))

    today_mask = view_df["timestamp"].dt.date == selected_date
    today_view = view_df.loc[today_mask]
    labels = view_df["label"].tolist()
    if len(today_view) >= 1:
        x0 = today_view["label"].iloc[0]
        x1 = today_view["label"].iloc[-1]
    else:
        x0, x1 = view_df["label"].iloc[0], view_df["label"].iloc[-1]

    # Category-axis zoom by index (more reliable than label strings)
    try:
        i0 = labels.index(x0)
        i1 = labels.index(x1)
    except ValueError:
        i0, i1 = 0, len(labels) - 1

    fig.update_layout(
        template="plotly_dark", height=680, xaxis_rangeslider_visible=False,
        title=f"{ticker} | {chart_tf} view | Sim Time: {current_time}",
        margin=dict(l=10, r=10, t=40, b=10), dragmode="pan",
        newshape=dict(line_color="cyan", line_width=2),
        uirevision="keep-drawings",
        showlegend=False,
    )

    # Apply category-axis zoom to BOTH rows (shared x)
    fig.update_xaxes(
        type="category",
        range=[i0 - 0.5, i1 + 0.5],
        nticks=12,
        rangeslider_visible=False,
        row=1, col=1,
    )
    fig.update_xaxes(
        type="category",
        range=[i0 - 0.5, i1 + 0.5],
        nticks=12,
        rangeslider_visible=False,
        row=2, col=1,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    # Cap volume axis at 95th percentile so the opening spike
    # doesn't crush all the other bars into invisibility
    if len(view_df) > 0 and view_df["volume"].max() > 0:
        vol_cap = view_df["volume"].quantile(0.95) * 1.15
        # safety: never cap below the median (avoids over-clipping on quiet days)
        vol_cap = max(vol_cap, view_df["volume"].median() * 2)
        fig.update_yaxes(title_text="Volume", range=[0, vol_cap], row=2, col=1)
    else:
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
    st.plotly_chart(fig, use_container_width=True, config={
        "scrollZoom": True,
        "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
        "displaylogo": False,
    })

    # ==========================================
    # ENTRY PANEL (only when flat)
    # ==========================================
    if pos == 0:
        st.markdown("### 🎮 Open a Position")
        e1, e2, e3, e4, e5 = st.columns([1.2, 1.2, 1.2, 1, 1])
        with e1:
            bet_size = st.number_input("Trade Amount ($)", 10.0,
                                       max(10.0, st.session_state.balance),
                                       min(100.0, st.session_state.balance), 10.0)
        with e2:
            sl_input = st.number_input("Stop-Loss ($)", value=round(current_price * 0.995, 2), step=0.05)
        with e3:
            tp_input = st.number_input("Target ($, 0 = none)", value=0.0, step=0.05)
        with e4:
            st.write(""); st.write("")
            if st.button("🟢 BUY (Long)", use_container_width=True):
                if sl_input >= current_price:
                    st.error("Long stop-loss must be BELOW current price!")
                elif tp_input != 0 and tp_input <= current_price:
                    st.error("Long target must be ABOVE current price!")
                else:
                    sh = bet_size / current_price
                    st.session_state.balance -= bet_size
                    st.session_state.shares = sh
                    st.session_state.stop_loss = sl_input
                    st.session_state.target = tp_input if tp_input > 0 else None
                    st.session_state.entry_price = current_price
                    st.session_state.trade_log.append(
                        f"{current_time}: BOUGHT {sh:.4f} sh at ${current_price:.2f} "
                        f"(SL ${sl_input:.2f} / TP {'$'+format(tp_input,'.2f') if tp_input>0 else '—'})")
                    st.rerun()
        with e5:
            st.write(""); st.write("")
            if st.button("🔻 SHORT", use_container_width=True):
                if sl_input <= current_price:
                    st.error("Short stop-loss must be ABOVE current price!")
                elif tp_input != 0 and tp_input >= current_price:
                    st.error("Short target must be BELOW current price!")
                else:
                    sh = bet_size / current_price
                    st.session_state.balance += bet_size
                    st.session_state.shares = -sh
                    st.session_state.stop_loss = sl_input
                    st.session_state.target = tp_input if tp_input > 0 else None
                    st.session_state.entry_price = current_price
                    st.session_state.trade_log.append(
                        f"{current_time}: SHORTED {sh:.4f} sh at ${current_price:.2f} "
                        f"(SL ${sl_input:.2f} / TP {'$'+format(tp_input,'.2f') if tp_input>0 else '—'})")
                    st.rerun()

    # ==========================================
    # MANAGEMENT PANEL (when in a position)
    # ==========================================
    else:
        st.markdown(f"### 🎮 Manage {pos_type} Position")
        m1, m2, m3, m4, m5 = st.columns([1.2, 1.2, 1.2, 1, 1])
        with m1:
            new_sl = st.number_input("Modify Stop-Loss ($)",
                                     value=float(st.session_state.stop_loss or current_price),
                                     step=0.05, key="mod_sl")
        with m2:
            new_tp = st.number_input("Modify Target ($, 0 = none)",
                                     value=float(st.session_state.target or 0.0),
                                     step=0.05, key="mod_tp")
        with m3:
            close_qty = st.number_input("Qty to Close (shares)",
                                        min_value=0.0, max_value=float(abs(pos)),
                                        value=float(abs(pos)), step=0.01)
        with m4:
            st.write(""); st.write("")
            if st.button("💾 Update SL/TP", use_container_width=True):
                valid = True
                if pos > 0 and new_sl >= current_price:
                    st.error("Long SL must be below price!"); valid = False
                if pos < 0 and new_sl <= current_price:
                    st.error("Short SL must be above price!"); valid = False
                if valid:
                    st.session_state.stop_loss = new_sl
                    st.session_state.target = new_tp if new_tp > 0 else None
                    st.session_state.trade_log.append(
                        f"{current_time}: UPDATED SL to ${new_sl:.2f}, "
                        f"TP to {'$'+format(new_tp,'.2f') if new_tp>0 else '—'}")
                    st.rerun()
        with m5:
            st.write(""); st.write("")
            btn_label = "🔴 SELL" if pos > 0 else "🟢 COVER"
            if st.button(f"{btn_label} {close_qty:.2f} sh", use_container_width=True):
                if close_qty > 0:
                    close_position(close_qty, current_price, current_time, "(manual)")
                    st.rerun()

    st.markdown("#### ⏱️ Advance Time")
    a1, a2, a3, _ = st.columns([1, 1, 1, 3])
    with a1:
        if st.button("▶️ +1 Min", use_container_width=True):
            advance_time(1); st.rerun()
    with a2:
        if st.button("⏩ +5 Min", use_container_width=True):
            advance_time(5); st.rerun()
    with a3:
        if st.button("⏭️ +15 Min", use_container_width=True):
            advance_time(15); st.rerun()

    if st.session_state.trade_log:
        with st.expander("📝 Trade History", expanded=True):
            for log in reversed(st.session_state.trade_log):
                st.text(log)
else:
    st.info("👈 Pick a date and start time, then click **Start Simulation**.")
