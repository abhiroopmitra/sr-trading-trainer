import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Market Replay Simulator")

# ==========================================
# STRUCTURE DETECTION CONSTANTS
# ==========================================
SWING_K = 5
ZONE_TOL = 0.0012
MIN_DRAW_TOUCHES = 2
MIN_SWING_PCT = 0.0008
POLARITY_EDGE = 2
CHOCH_ENABLE = True

C_HH = dict(fg="#ffffff", bg="#22c55e")
C_LH = dict(fg="#ffffff", bg="#ef4444")
C_H  = dict(fg="#ffffff", bg="#64748b")
C_BOS_UP = dict(fg="#111111", bg="#fde047")
C_BOS_DN = dict(fg="#ffffff", bg="#dc2626")
C_CH_UP  = dict(fg="#111111", bg="#7dd3fc")
C_CH_DN  = dict(fg="#111111", bg="#fb923c")
C_FLOOR  = "#22c55e"
C_CEIL   = "#ef4444"
C_BROKEN = "#a8a29e"

# ==========================================
# SESSION STATE
# ==========================================
defaults = {
    "sim_active": False, "balance": 1000.00, "shares": 0.0,
    "stop_loss": None, "target": None, "entry_price": None,
    "step": 0, "full_df": pd.DataFrame(), "trade_log": [],
    "sim_start_idx": 0,
    "start_day_idx": 0,              # index of first candle of selected day
    "camera_version": 0,             # used to reset camera when follow is ON
    "pending_draw": None,            # first click for Band/Trend
    "drawings": [],                  # all saved shapes
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================
# DATA FETCHING (native interval, no resampling)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_native(ticker, target_date, interval):
    # Determine lookback
    if interval == "1m":
        lookback_days = 7
    elif interval in ["5m", "15m", "30m", "1h"]:
        lookback_days = 60
    else:
        lookback_days = 60

    start = target_date - timedelta(days=lookback_days)
    end = target_date + timedelta(days=1)

    try:
        data = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                           end=end.strftime("%Y-%m-%d"),
                           interval=interval, auto_adjust=True, progress=False)
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None

    if data is None or data.empty:
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

    df["label"] = df["timestamp"].dt.strftime("%m-%d %H:%M")
    df["date_only"] = df["timestamp"].dt.date
    return df.reset_index(drop=True)

# ==========================================
# MOVING AVERAGE HELPER
# ==========================================
def moving_avg(series, length, kind="EMA"):
    if kind == "EMA":
        return series.ewm(span=length, adjust=False).mean()
    return series.rolling(length).mean()

# ==========================================
# STRUCTURE DETECTION (ported from bot)
# ==========================================
def _raw_swings(df):
    highs, lows = df["high"].values, df["low"].values
    n = len(df)
    if n < SWING_K * 2 + 1:
        return []
    ref = float(df["close"].iloc[-1])
    min_size = max(ref * MIN_SWING_PCT, 1e-6)
    raw = []
    for i in range(SWING_K, n - SWING_K):
        wh = highs[i - SWING_K: i + SWING_K + 1]
        wl = lows[i - SWING_K: i + SWING_K + 1]
        if highs[i] == wh.max() and (highs[i] - wl.min()) >= min_size:
            raw.append((i, float(highs[i]), "H"))
        if lows[i] == wl.min() and (wh.max() - lows[i]) >= min_size:
            raw.append((i, float(lows[i]), "L"))
    raw.sort(key=lambda x: x[0])
    cleaned = []
    for s in raw:
        if cleaned and cleaned[-1][2] == s[2]:
            if (s[2] == "H" and s[1] >= cleaned[-1][1]) or \
               (s[2] == "L" and s[1] <= cleaned[-1][1]):
                cleaned[-1] = s
        else:
            cleaned.append(s)
    labeled, last_h, last_l = [], None, None
    for idx, price, typ in cleaned:
        if typ == "H":
            lab = "H" if last_h is None else ("HH" if price > last_h else "LH")
            last_h = price
        else:
            lab = "L" if last_l is None else ("HL" if price > last_l else "LL")
            last_l = price
        labeled.append({"i": idx, "price": price, "type": typ, "label": lab})
    return labeled

def build_zones(labeled, ref_price):
    zones = []
    for s in labeled:
        placed = False
        for z in zones:
            if abs(s["price"] - z["mid"]) / max(ref_price, 1e-9) < ZONE_TOL:
                z["prices"].append(s["price"])
                z["mid"] = float(np.mean(z["prices"]))
                z["touches"] += 1
                if s["type"] == "L":
                    z["low_touches"] += 1
                else:
                    z["high_touches"] += 1
                placed = True
                break
        if not placed:
            zones.append({"mid": s["price"], "prices": [s["price"]], "touches": 1,
                          "low_touches": 1 if s["type"] == "L" else 0,
                          "high_touches": 1 if s["type"] == "H" else 0})
    out = [z for z in zones if z["touches"] >= MIN_DRAW_TOUCHES]
    for z in out:
        z["min_px"] = min(z["prices"])
        z["max_px"] = max(z["prices"])
    return out

def detect_bos_choch(df, labeled):
    n = len(df)
    bos, choch = [], []
    last_hh = last_hl = last_lh = last_ll = None
    bias = "NEUTRAL"
    ptr = 0
    for i in range(n):
        while ptr < len(labeled) and labeled[ptr]["i"] + SWING_K <= i:
            s = labeled[ptr]
            if s["label"] == "HH": last_hh = s["price"]
            elif s["label"] == "HL": last_hl = s["price"]
            elif s["label"] == "LH": last_lh = s["price"]
            elif s["label"] == "LL": last_ll = s["price"]
            ptr += 1
        c = float(df["close"].iloc[i])
        if last_hh is not None and c > last_hh:
            if bias != "BULL":
                bos.append({"i": i, "price": last_hh, "dir": "up"}); bias = "BULL"
            last_hh = None; continue
        if last_ll is not None and c < last_ll:
            if bias != "BEAR":
                bos.append({"i": i, "price": last_ll, "dir": "down"}); bias = "BEAR"
            last_ll = None; continue
        if CHOCH_ENABLE:
            if bias == "BULL" and last_hl is not None and c < last_hl:
                choch.append({"i": i, "price": last_hl, "dir": "down"}); last_hl = None; bias = "NEUTRAL"
            elif bias == "BEAR" and last_lh is not None and c > last_lh:
                choch.append({"i": i, "price": last_lh, "dir": "up"}); last_lh = None; bias = "NEUTRAL"
    return bos, choch

def zone_role(z, close, noise):
    if z["low_touches"] >= z["high_touches"] + POLARITY_EDGE:
        role = "floor"
    elif z["high_touches"] >= z["low_touches"] + POLARITY_EDGE:
        role = "ceiling"
    else:
        role = "both"
    if role == "floor" and close < z["min_px"] - noise: return "broken_floor"
    if role == "ceiling" and close > z["max_px"] + noise: return "broken_ceiling"
    return role

def badge(fig, x, y, text, pal, yshift=0, arrow=False):
    fig.add_annotation(x=x, y=y, text=f"<b>{text}</b>", row=1, col=1,
        showarrow=arrow, arrowhead=2, arrowsize=1, arrowwidth=1.4,
        arrowcolor="#e5e7eb", yshift=yshift,
        font=dict(size=11, color=pal["fg"], family="Arial"),
        bgcolor=pal["bg"], bordercolor="#e5e7eb", borderwidth=1, borderpad=3, opacity=1)

# ==========================================
# POSITION CLOSE HELPER
# ==========================================
def close_position(qty, price, time_str, reason=""):
    pos = st.session_state.shares
    if pos > 0:
        qty = min(qty, pos)
        st.session_state.balance += qty * price
        st.session_state.shares -= qty
        st.session_state.trade_log.append(f"{time_str}: SOLD {qty:.4f} sh at ${price:.2f} {reason}")
    elif pos < 0:
        qty = min(qty, abs(pos))
        st.session_state.balance -= qty * price
        st.session_state.shares += qty
        st.session_state.trade_log.append(f"{time_str}: COVERED {qty:.4f} sh at ${price:.2f} {reason}")
    if st.session_state.shares == 0:
        st.session_state.stop_loss = None
        st.session_state.target = None
        st.session_state.entry_price = None

# ==========================================
# TIME ADVANCE + SL/TARGET CHECKS
# ==========================================
def advance_time(steps_to_move):
    max_steps = len(st.session_state.full_df) - 1
    for _ in range(steps_to_move):
        if st.session_state.step >= max_steps:
            st.toast("End of available data!", icon="🔔")
            break
        st.session_state.step += 1
        candle = st.session_state.full_df.iloc[st.session_state.step]
        t = candle["label"].split(" ")[-1]
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
# DRAWING CLICK HANDLER
# ==========================================
def handle_draw_click(point_x, point_y):
    mode = st.session_state.get("draw_mode", "None")
    if mode == "None":
        return
    full_df = st.session_state.full_df
    step = st.session_state.step

    # Parse timestamp from click x (could be ms or string)
    if isinstance(point_x, (int, float)):
        ts = pd.to_datetime(point_x, unit="ms")
    else:
        ts = pd.to_datetime(point_x)

    # Find the nearest candle (within the current visible range, nearest by timestamp)
    start_idx = st.session_state.start_day_idx
    visible = full_df.iloc[start_idx: step + 1]
    if visible.empty:
        return
    closest_idx = (visible["timestamp"] - ts).abs().idxmin()
    candle = full_df.loc[closest_idx]

    high = float(candle["high"])
    close = float(candle["close"])
    low = float(candle["low"])
    ts_str = candle["timestamp"]

    if mode == "High":
        st.session_state.drawings.append({"type": "hline", "y": high, "name": f"High {ts_str}"})
        st.session_state.pending_draw = None
    elif mode == "Close":
        st.session_state.drawings.append({"type": "hline", "y": close, "name": f"Close {ts_str}"})
        st.session_state.pending_draw = None
    elif mode == "Low":
        st.session_state.drawings.append({"type": "hline", "y": low, "name": f"Low {ts_str}"})
        st.session_state.pending_draw = None
    elif mode == "Band":
        if st.session_state.pending_draw is None:
            st.session_state.pending_draw = {"x": ts_str, "y": point_y, "type": "band"}
            st.toast("Now click the second level for the band", icon="✏️")
        else:
            p1 = st.session_state.pending_draw
            st.session_state.drawings.append({
                "type": "band",
                "y0": p1["y"],
                "y1": point_y,
                "name": f"Band {p1['y']:.2f} - {point_y:.2f}",
            })
            st.session_state.pending_draw = None
    elif mode == "Trend":
        if st.session_state.pending_draw is None:
            st.session_state.pending_draw = {"x": ts_str, "y": point_y, "type": "trend"}
            st.toast("Now click the second point for the trend line", icon="✏️")
        else:
            p1 = st.session_state.pending_draw
            st.session_state.drawings.append({
                "type": "trend",
                "x0": p1["x"], "y0": p1["y"],
                "x1": ts_str, "y1": point_y,
                "name": "Trend",
            })
            st.session_state.pending_draw = None
    st.rerun()

# ==========================================
# SIDEBAR SETUP
# ==========================================
st.sidebar.header("⚙️ Setup")
ticker = st.sidebar.text_input("Ticker", value="TQQQ").upper()

tf_choices = ["1m", "5m", "15m", "30m", "1h"]
chart_tf = st.sidebar.radio("📊 Timeframe", tf_choices, index=1, horizontal=True)

spacer = st.sidebar.empty()

# Data fetching
raw_df = None
today = datetime.now().date()
default_date = today - timedelta(days=2)

# Fetch data to determine valid date range
if ticker:
    raw_df = fetch_native(ticker, default_date, chart_tf)

if raw_df is not None:
    available_dates = sorted(raw_df["date_only"].unique())
    min_date = min(available_dates)
    max_date = max(available_dates)
    selected_date = st.sidebar.date_input(
        "Trading Date",
        value=default_date,
        min_value=min_date,
        max_value=max_date,
    )
else:
    st.sidebar.error("No data fetched. Try another ticker/timeframe.")
    st.stop()

# Re-fetch if selected_date is outside range (we need exact date)
if selected_date not in available_dates:
    st.sidebar.error("No data for that date. Pick an available date.")
    st.stop()

# Now fetch exactly with proper lookback for selected_date
full_df = fetch_native(ticker, selected_date, chart_tf)
if full_df is None or full_df.empty:
    st.sidebar.error("No data for that date.")
    st.stop()

# Get the day mask for the selected date
day_mask = full_df["date_only"] == selected_date
day_indices = full_df.index[day_mask].tolist()
if not day_indices:
    st.sidebar.error("No intraday candles for that date (weekend/holiday?).")
    st.stop()

day_labels = full_df.loc[day_mask, "label"].tolist()
default_idx = min(15, len(day_labels) - 1)
start_time = st.sidebar.selectbox("Start Time", day_labels, index=default_idx)

# Start / Reset button
if st.sidebar.button("🚀 Start / Reset Simulation"):
    for k, v in defaults.items():
        st.session_state[k] = v
    st.session_state.full_df = full_df
    st.session_state.start_day_idx = day_indices[0]
    st.session_state.step = day_indices[day_labels.index(start_time)]
    st.session_state.balance = 1000.00
    st.session_state.shares = 0.0
    st.session_state.stop_loss = None
    st.session_state.target = None
    st.session_state.entry_price = None
    st.session_state.trade_log = []
    st.session_state.sim_active = True
    st.session_state.camera_version += 1
    st.rerun()

# ==========================================
# OVERLAYS SETTINGS
# ==========================================
st.sidebar.markdown("---")
show_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
ma_type = st.sidebar.radio("MA type", ["EMA", "SMA"], horizontal=True)
st.sidebar.markdown(f"**MA lengths for {chart_tf}**")
fast_len = st.sidebar.number_input(f"Fast {ma_type} ({chart_tf})", 3, 300, 9, 1)
slow_len = st.sidebar.number_input(f"Slow {ma_type} ({chart_tf})", 5, 500, 21, 1)
st.sidebar.caption(f"Showing {ma_type}{fast_len} and {ma_type}{slow_len}")

show_struct = st.sidebar.checkbox("Show structure labels (HH/HL/BOS/CHoCH)", value=True)
show_zones = st.sidebar.checkbox("Show S/R zones", value=True)
show_vol_ma = st.sidebar.checkbox("Show volume MA", value=True)

# ==========================================
# CHART CONTEXT + FOLLOW REPLAY
# ==========================================
st.sidebar.markdown("---")
context_option = st.sidebar.selectbox(
    "Chart Context",
    ["Today", "1 Previous Day", "5 Previous Days", "20 Previous Days", "All Available History"],
    index=0,
)
follow_replay = st.sidebar.checkbox("Follow replay candle ON", value=True)
st.sidebar.caption("OFF lets you pan to old levels while time advances.")

# ==========================================
# DRAWING CONTROLS
# ==========================================
st.sidebar.markdown("---")
draw_mode = st.sidebar.selectbox(
    "✏️ Drawing Mode",
    ["None", "High", "Close", "Low", "Band", "Trend"],
    index=0,
)
st.session_state["draw_mode"] = draw_mode

if st.session_state.drawings:
    st.sidebar.markdown("**Saved Drawings**")
    for i, d in enumerate(st.session_state.drawings):
        if st.sidebar.button(f"❌ {d.get('name', d['type'])}", key=f"del_{i}"):
            st.session_state.drawings.pop(i)
            st.rerun()

# ==========================================
# MAIN APP
# ==========================================
st.title("💹 Market Replay Simulator")

if st.session_state.sim_active:
    step = st.session_state.step
    full_df = st.session_state.full_df
    start_day_idx = st.session_state.start_day_idx

    # Visible data = context days (full) + current day up to current step ONLY
    context_data = full_df[full_df["date_only"] < selected_date]
    current_visible = full_df.iloc[start_day_idx: step + 1]
    visible_df = pd.concat([context_data, current_visible]).sort_values("timestamp").reset_index(drop=True)

    current_candle = full_df.iloc[step]
    current_price = float(current_candle["close"])
    current_time = current_candle["label"]
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

    # ==========================================
    # BUILD FIGURE
    # ==========================================
    vol_colors = np.where(
        visible_df["close"] >= visible_df["open"],
        "rgba(38,166,154,0.6)", "rgba(239,83,80,0.6)"
    )
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.78, 0.22])

    fig.add_trace(go.Candlestick(
        x=visible_df["timestamp"], open=visible_df["open"], high=visible_df["high"],
        low=visible_df["low"], close=visible_df["close"], name=ticker,
        increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"),
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=visible_df["timestamp"], y=visible_df["volume"],
        marker_color=vol_colors, name="Volume",
    ), row=2, col=1)

    # EMAs (computed on full history up to current step, filtered to visible)
    if show_ma and len(visible_df) >= 2:
        full_hist = full_df.iloc[: step + 1]
        # Compute EMA on full history to avoid warm-up issues
        ema_fast = moving_avg(full_hist["close"], fast_len, ma_type)
        ema_slow = moving_avg(full_hist["close"], slow_len, ma_type)
        # Filter to visible timestamps
        vis_ts = visible_df["timestamp"]
        ema_fast_vis = full_hist.loc[full_hist["timestamp"].isin(vis_ts), "timestamp"].map(
            dict(zip(full_hist["timestamp"], ema_fast)))
        ema_slow_vis = full_hist.loc[full_hist["timestamp"].isin(vis_ts), "timestamp"].map(
            dict(zip(full_hist["timestamp"], ema_slow)))
        fig.add_trace(go.Scatter(x=visible_df["timestamp"], y=ema_fast_vis,
                                 line=dict(color="#3b82f6", width=1.6), name=f"{ma_type}{fast_len}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=visible_df["timestamp"], y=ema_slow_vis,
                                 line=dict(color="#f59e0b", width=1.6), name=f"{ma_type}{slow_len}"), row=1, col=1)

    # Volume MA
    if show_vol_ma and len(visible_df) >= 20:
        vol_ma = visible_df["volume"].rolling(20).mean()
        fig.add_trace(go.Scatter(x=visible_df["timestamp"], y=vol_ma,
                                 line=dict(color="#f59e0b", width=1.5), name="VolMA"), row=2, col=1)

    # Structure labels + zones
    noise = float((visible_df["high"] - visible_df["low"]).tail(20).mean() or 0.3)
    if show_struct or show_zones:
        labeled = _raw_swings(visible_df)
        conf = [s for s in labeled if s["i"] + SWING_K <= len(visible_df) - 1]

    if show_struct and len(visible_df) > SWING_K * 2 + 1:
        bos, choch = detect_bos_choch(visible_df, labeled)
        for s in conf:
            x = visible_df["timestamp"].iloc[s["i"]]
            if s["label"] in ("HH", "HL"): pal, ys = C_HH, (18 if s["type"] == "H" else -18)
            elif s["label"] in ("LH", "LL"): pal, ys = C_LH, (18 if s["type"] == "H" else -18)
            else: pal, ys = C_H, (18 if s["type"] == "H" else -18)
            badge(fig, x, s["price"], s["label"], pal, yshift=ys)
        for b in bos:
            if b["i"] < len(visible_df):
                x = visible_df["timestamp"].iloc[b["i"]]
                if b["dir"] == "up": badge(fig, x, float(visible_df["high"].iloc[b["i"]]), "BOS↑", C_BOS_UP, yshift=22, arrow=True)
                else: badge(fig, x, float(visible_df["low"].iloc[b["i"]]), "BOS↓", C_BOS_DN, yshift=-22, arrow=True)
        for h in choch:
            if h["i"] < len(visible_df):
                x = visible_df["timestamp"].iloc[h["i"]]
                if h["dir"] == "up": badge(fig, x, float(visible_df["high"].iloc[h["i"]]), "CHoCH↑", C_CH_UP, yshift=22, arrow=True)
                else: badge(fig, x, float(visible_df["low"].iloc[h["i"]]), "CHoCH↓", C_CH_DN, yshift=-22, arrow=True)

    if show_zones and len(visible_df) > SWING_K * 2 + 1:
        zones = build_zones(conf, current_price)
        last_x = visible_df["timestamp"].iloc[-1]
        for z in zones:
            role = zone_role(z, current_price, noise)
            if role == "floor": col, tag = C_FLOOR, "FLOOR"
            elif role == "ceiling": col, tag = C_CEIL, "CEIL"
            elif role == "both": col, tag = "#eab308", "BOTH"
            else: col, tag = C_BROKEN, role.replace("_", " ").upper()
            fig.add_hline(y=z["mid"], row=1, col=1, line=dict(color=col, width=min(1 + z["touches"], 4), dash="dot"))
            fig.add_annotation(x=last_x, y=z["mid"], xanchor="left", xref="x", row=1, col=1,
                text=f"<b> {z['mid']:.2f} {tag} {z['low_touches']}L/{z['high_touches']}H</b>",
                showarrow=False, font=dict(size=10, color="#ffffff", family="Arial"),
                bgcolor=col, bordercolor="#e5e7eb", borderwidth=1, borderpad=3)

    # Position lines
    if pos != 0:
        if st.session_state.stop_loss:
            fig.add_hline(y=st.session_state.stop_loss, row=1, col=1, line=dict(color="#fb923c", width=2, dash="dash"))
        if st.session_state.target:
            fig.add_hline(y=st.session_state.target, row=1, col=1, line=dict(color="#4ade80", width=2, dash="dash"))
        if st.session_state.entry_price:
            fig.add_hline(y=st.session_state.entry_price, row=1, col=1, line=dict(color="#22d3ee", width=1, dash="dot"))

    # Saved drawings
    for d in st.session_state.drawings:
        if d["type"] == "hline":
            fig.add_hline(y=d["y"], row=1, col=1, line=dict(color="#ffffff", width=1, dash="solid"))
        elif d["type"] == "band":
            fig.add_hrect(y0=d["y0"], y1=d["y1"], row=1, col=1,
                          fillcolor="rgba(255,255,255,0.1)", line=dict(color="#ffffff", width=1))
        elif d["type"] == "trend":
            fig.add_shape(type="line", x0=d["x0"], x1=d["x1"], y0=d["y0"], y1=d["y1"],
                          row=1, col=1, line=dict(color="#ffffff", width=1.5, dash="dot"))

    # ==========================================
    # CAMERA + CONTEXT LOGIC
    # ==========================================
    if follow_replay:
        st.session_state.camera_version += 1
        # Determine start of camera
        today_start = full_df.loc[start_day_idx, "timestamp"]
        step_time = full_df.loc[step, "timestamp"]
        if context_option == "Today":
            cam_start = today_start
        elif context_option == "1 Previous Day":
            cam_start = today_start - timedelta(days=1)
        elif context_option == "5 Previous Days":
            cam_start = today_start - timedelta(days=5)
        elif context_option == "20 Previous Days":
            cam_start = today_start - timedelta(days=20)
        else:
            cam_start = full_df["timestamp"].iloc[0]

        # Add small padding
        cam_end = step_time + timedelta(minutes=5 if chart_tf in ["1m", "5m"] else 30)
        fig.update_xaxes(range=[cam_start, cam_end], row=1, col=1)
        fig.update_xaxes(range=[cam_start, cam_end], row=2, col=1)

    # Chart layout
    fig.update_layout(
        template="plotly_dark", height=600, dragmode="pan",
        paper_bgcolor="#000000", plot_bgcolor="#000000",
        font=dict(color="#e5e7eb"),
        xaxis_rangeslider_visible=False,
        title=f"{ticker} | {chart_tf} | {current_time}",
        margin=dict(l=10, r=140, t=40, b=10),
        uirevision="manual" if not follow_replay else str(st.session_state.camera_version),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="#e5e7eb")),
    )
    fig.update_xaxes(type="date", gridcolor="#1f2937", linecolor="#4b5563", row=1, col=1)
    fig.update_xaxes(type="date", gridcolor="#1f2937", linecolor="#4b5563", row=2, col=1)
    fig.update_yaxes(title_text="Price", gridcolor="#1f2937", linecolor="#4b5563", row=1, col=1)
    fig.update_yaxes(title_text="Volume", gridcolor="#1f2937", linecolor="#4b5563", row=2, col=1)

    if len(visible_df) and visible_df["volume"].max() > 0:
        cap = max(visible_df["volume"].quantile(0.95) * 1.15, visible_df["volume"].median() * 2)
        fig.update_yaxes(range=[0, cap], row=2, col=1)

    # ==========================================
    # PLOT CHART + CLICK EVENT
    # ==========================================
    chart_col, ctrl_col = st.columns([7, 1])

    with chart_col:
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            key="practice_chart",
            on_select="rerun",
            selection_mode="points",
            config={
                "scrollZoom": True,
                "displaylogo": False,
            }
        )
        # Handle click for drawing
        if event and event.selection and event.selection.points:
            point = event.selection.points[0]
            handle_draw_click(point["x"], point["y"])
            st.rerun()

    with ctrl_col:
        st.markdown("#### ⏱️")
        st.caption(f"**{current_time.split(' ')[-1]}**")
        if st.button("▶️ +1", use_container_width=True, key="adv1"):
            advance_time(1); st.rerun()
        if st.button("⏩ +5", use_container_width=True, key="adv5"):
            advance_time(5); st.rerun()
        if st.button("⏭️ +15", use_container_width=True, key="adv15"):
            advance_time(15); st.rerun()
        st.markdown("---")
        st.caption(f"Context:\n{context_option}")
        if draw_mode != "None":
            st.info(f"Draw: {draw_mode}")
        if st.session_state.pending_draw is not None:
            st.warning("Click 2nd point")
        if pos > 0:
            st.success("🟢 LONG")
        elif pos < 0:
            st.error("🔴 SHORT")
        else:
            st.info("FLAT")

    # ==========================================
    # ENTRY / MANAGEMENT PANEL
    # ==========================================
    if pos == 0:
        st.markdown("### 🎮 Open a Position")
        e1, e2, e3, e4, e5 = st.columns([1.2, 1.2, 1.2, 1, 1])
        with e1:
            bet_size = st.number_input("Trade Amount ($)", 10.0, max(10.0, st.session_state.balance), min(100.0, st.session_state.balance), 10.0)
        with e2:
            sl_input = st.number_input("Stop-Loss ($)", value=round(current_price * 0.995, 2), step=0.01)
        with e3:
            tp_input = st.number_input("Target ($, 0 = none)", value=0.0, step=0.01)
        with e4:
            st.write(""); st.write("")
            if st.button("🟢 BUY (Long)", use_container_width=True):
                if sl_input >= current_price:
                    st.error("Long stop must be BELOW price!")
                else:
                    sh = bet_size / current_price
                    st.session_state.balance -= bet_size
                    st.session_state.shares = sh
                    st.session_state.stop_loss = sl_input
                    st.session_state.target = tp_input if tp_input > 0 else None
                    st.session_state.entry_price = current_price
                    st.session_state.trade_log.append(f"{current_time}: BOUGHT {sh:.4f} sh @ ${current_price:.2f}")
                    st.rerun()
        with e5:
            st.write(""); st.write("")
            if st.button("🔻 SHORT", use_container_width=True):
                if sl_input >= current_price:
                    st.error("Short stop must be ABOVE price!")
                else:
                    sh = bet_size / current_price
                    st.session_state.balance += bet_size
                    st.session_state.shares = -sh
                    st.session_state.stop_loss = sl_input
                    st.session_state.target = tp_input if tp_input > 0 else None
                    st.session_state.entry_price = current_price
                    st.session_state.trade_log.append(f"{current_time}: SHORTED {sh:.4f} sh @ ${current_price:.2f}")
                    st.rerun()
    else:
        st.markdown(f"### 🎮 Manage {pos_type} Position")
        m1, m2, m3, m4, m5 = st.columns([1.2, 1.2, 1.2, 1, 1])
        with m1:
            new_sl = st.number_input("Modify Stop-Loss", value=float(st.session_state.stop_loss or current_price), step=0.01, key="mod_sl")
        with m2:
            new_tp = st.number_input("Modify Target (0=none)", value=float(st.session_state.target or 0.0), step=0.01, key="mod_tp")
        with m3:
            close_qty = st.number_input("Qty to Close", min_value=0.0, max_value=float(abs(pos)), value=float(abs(pos)), step=0.0001)
        with m4:
            st.write(""); st.write("")
            if st.button("💾 Update SL/TP", use_container_width=True):
                if pos > 0 and new_sl >= current_price: st.error("Long SL must be below price!")
                elif pos < 0 and new_sl <= current_price: st.error("Short SL must be above price!")
                else:
                    st.session_state.stop_loss = new_sl
                    st.session_state.target = new_tp if new_tp > 0 else None
                    st.rerun()
        with m5:
            st.write(""); st.write("")
            btn_label = "🔴 SELL" if pos > 0 else "🟢 COVER"
            if st.button(f"{btn_label} {close_qty:.4f} sh", use_container_width=True):
                close_position(close_qty, current_price, current_time, "(manual)")
                st.rerun()

    if st.session_state.trade_log:
        with st.expander("📝 Trade History", expanded=True):
            for log in reversed(st.session_state.trade_log):
                st.text(log)

else:
    st.info("👈 Pick a date and start time, then click **Start Simulation**.")
    st.info("Note: 1m data has ~7 days history. 5m/15m/30m/1h have ~60 days.")
