# ============================================================
# HUMAN MARKET REPLAY SIMULATOR
# - Fixed: Range Slider "ghost candle" bug on volume chart
# - Fixed: Y-axis now stretches to fit full context when zoomed out
# - Restored: Full MA controls (EMA/SMA, Fast/Slow) & Structure/Zone toggles
# - Persistent SQLite Database
# ============================================================

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3

st.set_page_config(layout="wide", page_title="Market Replay Simulator")

# ============================================================
# DATABASE SETUP (PERSISTENT WALLET)
# ============================================================
DB_FILE = "paper_trading.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS account
                 (id INTEGER PRIMARY KEY, balance REAL, realized_pnl REAL, cycle INTEGER, status TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS positions
                 (id INTEGER PRIMARY KEY, ticker TEXT, qty REAL, entry_price REAL, sl REAL, tp REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY, cycle INTEGER, log_text TEXT)''')
    
    c.execute("SELECT * FROM account WHERE status='ACTIVE'")
    if not c.fetchone():
        c.execute("INSERT INTO account (balance, realized_pnl, cycle, status) VALUES (1000.0, 0.0, 1, 'ACTIVE')")
        
    conn.commit()
    conn.close()

def get_active_account():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT balance, realized_pnl, cycle FROM account WHERE status='ACTIVE'")
    acc = c.fetchone()
    
    c.execute("SELECT qty, entry_price, sl, tp FROM positions LIMIT 1")
    pos = c.fetchone()
    
    c.execute("SELECT log_text FROM history WHERE cycle=? ORDER BY id ASC", (acc[2],))
    logs = [row[0] for row in c.fetchall()]
    conn.close()
    
    return acc, pos, logs

def save_state_to_db(balance, realized_pnl, qty, entry, sl, tp, new_log=None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE account SET balance=?, realized_pnl=? WHERE status='ACTIVE'", (balance, realized_pnl))
    
    c.execute("DELETE FROM positions")
    if qty != 0:
        c.execute("INSERT INTO positions (ticker, qty, entry_price, sl, tp) VALUES ('MIXED', ?, ?, ?, ?)", 
                  (qty, entry, sl, tp))
        
    if new_log:
        c.execute("SELECT cycle FROM account WHERE status='ACTIVE'")
        cycle = c.fetchone()[0]
        c.execute("INSERT INTO history (cycle, log_text) VALUES (?, ?)", (cycle, new_log))
        
    conn.commit()
    conn.close()

def blow_up_account():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE account SET status='DEPLETED' WHERE status='ACTIVE'")
    c.execute("SELECT MAX(cycle) FROM account")
    last_cycle = c.fetchone()[0]
    
    new_cycle = last_cycle + 1
    c.execute("INSERT INTO account (balance, realized_pnl, cycle, status) VALUES (1000.0, 0.0, ?, 'ACTIVE')", (new_cycle,))
    c.execute("DELETE FROM positions")
    c.execute("INSERT INTO history (cycle, log_text) VALUES (?, ?)", (new_cycle, "⚠️ Account depleted. New $1,000 cycle started."))
    conn.commit()
    conn.close()

init_db()

# ============================================================
# TIMEFRAME & SESSION SETTINGS
# ============================================================
TIMEFRAME_CONFIG = {
    "1m": {"yf_interval": "1m", "period": "7d", "fallback_period": "6d", "ema": (9, 21), "minutes": 1, "intraday": True},
    "2m": {"yf_interval": "2m", "period": "60d", "fallback_period": "59d", "ema": (9, 21), "minutes": 2, "intraday": True},
    "5m": {"yf_interval": "5m", "period": "60d", "fallback_period": "59d", "ema": (20, 50), "minutes": 5, "intraday": True},
    "15m": {"yf_interval": "15m", "period": "60d", "fallback_period": "59d", "ema": (50, 200), "minutes": 15, "intraday": True},
    "30m": {"yf_interval": "30m", "period": "60d", "fallback_period": "59d", "ema": (50, 200), "minutes": 30, "intraday": True},
    "90m": {"yf_interval": "90m", "period": "60d", "fallback_period": "59d", "ema": (50, 100), "minutes": 90, "intraday": True},
    "1h": {"yf_interval": "60m", "period": "2y", "fallback_period": "1y", "ema": (50, 200), "minutes": 60, "intraday": True},
    "1d": {"yf_interval": "1d", "period": "10y", "fallback_period": "5y", "ema": (50, 200), "minutes": 1440, "intraday": False},
}

TIMEFRAME_OPTIONS = list(TIMEFRAME_CONFIG.keys())
CONTEXT_PRESETS = ["Today", "5 Previous Days", "20 Previous Days", "All Available History"]
DRAW_MODES = ["None", "Line at High", "Line at Close", "Line at Low", "Band", "Trend line"]

# PERFORMANCE / STRUCTURE
MAX_STRUCTURE_BARS = 700
MAX_SWING_LABELS = 90
MAX_BOS_LABELS = 30
MAX_CHOCH_LABELS = 20
MAX_ZONE_SWINGS = 180
MAX_ZONES_DRAWN = 10
SWING_K = 5
ZONE_TOL = 0.0012
MIN_DRAW_TOUCHES = 2
MIN_SWING_PCT = 0.0008
POLARITY_EDGE = 2
CHOCH_ENABLE = True

# COLORS
C_HH, C_LH, C_H = dict(fg="#ffffff", bg="#16a34a"), dict(fg="#ffffff", bg="#dc2626"), dict(fg="#ffffff", bg="#475569")
C_BOS_UP, C_BOS_DN = dict(fg="#111111", bg="#fde047"), dict(fg="#ffffff", bg="#dc2626")
C_CH_UP, C_CH_DN = dict(fg="#111111", bg="#7dd3fc"), dict(fg="#111111", bg="#fb923c")
C_FLOOR, C_CEIL, C_BROKEN, C_BOTH = "#22c55e", "#ef4444", "#a8a29e", "#eab308"
C_FAST_EMA, C_SLOW_EMA, C_VOL_MA = "#3b82f6", "#f59e0b", "#ff6d00"

# ============================================================
# SESSION STATE & DB SYNC
# ============================================================
defaults = {
    "sim_active": False,
    "df": pd.DataFrame(),
    "step": 0,
    "active_ticker": None,
    "active_interval": None,
    "active_practice_date": None,
    "session_mode": "US Stocks",
    "carry_overnight": True,
    "markers": [],
    "drawings": [],
    "draw_clicks": [],
    "last_draw_signature": None,
    "last_draw_mode": "None",
    "camera_signature": None,
    "camera_revision": 0,
    "force_camera": True,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

acc, db_pos, logs = get_active_account()
st.session_state.balance = acc[0]
st.session_state.realized_pnl = acc[1]
st.session_state.cycle = acc[2]
st.session_state.trade_log = logs

if db_pos:
    st.session_state.shares, st.session_state.entry_price, st.session_state.stop_loss, st.session_state.target = db_pos
else:
    st.session_state.shares = 0.0
    st.session_state.entry_price = st.session_state.stop_loss = st.session_state.target = None

# ============================================================
# DATA FUNCTIONS
# ============================================================
def find_column(df, possible_names):
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in lower_map: return lower_map[name.lower()]
    return None

def normalize_ohlcv(raw, intraday=True):
    if raw is None or raw.empty: return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex): df.columns = [str(col[0]) for col in df.columns]
    df = df.reset_index()

    out = pd.DataFrame({
        "timestamp": df[df.columns[0]],
        "open": df[find_column(df, ["Open"])],
        "high": df[find_column(df, ["High"])],
        "low": df[find_column(df, ["Low"])],
        "close": df[find_column(df, ["Close", "Adj Close"])],
        "volume": df[find_column(df, ["Volume"])] if find_column(df, ["Volume"]) else 0.0,
    })

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    try:
        if out["timestamp"].dt.tz is not None:
            out["timestamp"] = out["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    except: pass

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    valid = (out["high"] >= out["low"]) & (out["open"] > 0)
    out = out[valid].copy()
    
    out["x_str"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    out["date_only"] = out["timestamp"].dt.date
    
    return out.reset_index(drop=True)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_native_history(ticker, timeframe):
    cfg = TIMEFRAME_CONFIG[timeframe]
    for period in [cfg["period"], cfg["fallback_period"]]:
        try:
            raw = yf.download(ticker, period=period, interval=cfg["yf_interval"], progress=False)
            df = normalize_ohlcv(raw, intraday=cfg["intraday"])
            if not df.empty: return df
        except: continue
    return pd.DataFrame()

def moving_avg(series, length, kind="EMA"):
    if kind == "EMA": return series.ewm(span=int(length), adjust=False).mean()
    return series.rolling(int(length)).mean()

def get_context_render_df(revealed_df, practice_date, preset):
    if revealed_df.empty: return revealed_df.copy()
    unique_dates = sorted(revealed_df["date_only"].unique())
    if practice_date not in unique_dates: return revealed_df.copy()

    idx = unique_dates.index(practice_date)
    days_map = {"Today": 0, "5 Previous Days": 5, "20 Previous Days": 20, "All Available History": 99999}
    days_back = days_map.get(preset, 5)
    
    start_date = unique_dates[max(0, idx - days_back)]
    return revealed_df[revealed_df["date_only"] >= start_date].copy()

def get_price_axis_range(df):
    if df.empty: return None
    values = []
    for col in ["low", "high", "fast_ma", "slow_ma"]:
        if col in df.columns: values.extend(df[col].dropna().tolist())
    if not values: return None
    low, high = float(min(values)), float(max(values))
    padding = max((high - low) * 0.12, high * 0.001)
    return [low - padding, high + padding]

def get_volume_axis_range(df):
    if df.empty or df["volume"].max() <= 0: return None
    cap = max(float(df["volume"].quantile(0.95)) * 1.15, float(df["volume"].median()) * 2.0, 1.0)
    return [0, cap]

# ============================================================
# STRUCTURE / ZONES LOGIC
# ============================================================
def _raw_swings(df):
    if df is None or len(df) < SWING_K * 2 + 1: return []
    highs, lows = df["high"].values, df["low"].values
    n = len(df)
    min_size = max(float(df["close"].iloc[-1]) * MIN_SWING_PCT, 1e-9)

    raw = []
    for i in range(SWING_K, n - SWING_K):
        hw = highs[i - SWING_K:i + SWING_K + 1]
        lw = lows[i - SWING_K:i + SWING_K + 1]
        if highs[i] == hw.max() and highs[i] - lw.min() >= min_size: raw.append((i, float(highs[i]), "H"))
        if lows[i] == lw.min() and hw.max() - lows[i] >= min_size: raw.append((i, float(lows[i]), "L"))

    raw.sort(key=lambda x: x[0])
    cleaned = []
    for s in raw:
        if cleaned and cleaned[-1][2] == s[2]:
            prev = cleaned[-1]
            if (s[2] == "H" and s[1] >= prev[1]) or (s[2] == "L" and s[1] <= prev[1]):
                cleaned[-1] = s
        else: cleaned.append(s)

    labeled, prev_h, prev_l = [], None, None
    for idx, price, stype in cleaned:
        if stype == "H":
            label = "H" if prev_h is None else ("HH" if price > prev_h else "LH")
            prev_h = price
        else:
            label = "L" if prev_l is None else ("HL" if price > prev_l else "LL")
            prev_l = price
        labeled.append({"i": idx, "price": price, "type": stype, "label": label})
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
                z["low_touches"] += 1 if s["type"] == "L" else 0
                z["high_touches"] += 1 if s["type"] == "H" else 0
                placed = True; break
        if not placed:
            zones.append({"mid": s["price"], "prices": [s["price"]], "touches": 1,
                          "low_touches": 1 if s["type"]=="L" else 0, "high_touches": 1 if s["type"]=="H" else 0})
    
    out = [z for z in zones if z["touches"] >= MIN_DRAW_TOUCHES]
    for z in out: z["min_px"] = min(z["prices"]); z["max_px"] = max(z["prices"])
    return out

def zone_role(zone, close, noise):
    if zone["low_touches"] >= zone["high_touches"] + POLARITY_EDGE: role = "floor"
    elif zone["high_touches"] >= zone["low_touches"] + POLARITY_EDGE: role = "ceiling"
    else: role = "both"
    if role == "floor" and close < zone["min_px"] - noise: return "broken_floor"
    if role == "ceiling" and close > zone["max_px"] + noise: return "broken_ceiling"
    return role

def detect_bos_choch(df, labeled):
    bos, choch = [], []
    last_hh, last_hl, last_lh, last_ll = None, None, None, None
    bias, ptr = "NEUTRAL", 0

    for i in range(len(df)):
        while ptr < len(labeled) and labeled[ptr]["i"] + SWING_K <= i:
            lbl, px = labeled[ptr]["label"], labeled[ptr]["price"]
            if lbl == "HH": last_hh = px
            elif lbl == "HL": last_hl = px
            elif lbl == "LH": last_lh = px
            elif lbl == "LL": last_ll = px
            ptr += 1

        close = float(df["close"].iloc[i])
        if last_hh and close > last_hh:
            if bias != "BULL": bos.append({"i": i, "price": last_hh, "dir": "up"}); bias = "BULL"
            last_hh = None; continue
        if last_ll and close < last_ll:
            if bias != "BEAR": bos.append({"i": i, "price": last_ll, "dir": "down"}); bias = "BEAR"
            last_ll = None; continue
        if CHOCH_ENABLE:
            if bias == "BULL" and last_hl and close < last_hl:
                choch.append({"i": i, "price": last_hl, "dir": "down"}); last_hl = None; bias = "NEUTRAL"
            elif bias == "BEAR" and last_lh and close > last_lh:
                choch.append({"i": i, "price": last_lh, "dir": "up"}); last_lh = None; bias = "NEUTRAL"
    return bos, choch

def badge(fig, x_str, y, text, palette, yshift=0, arrow=False):
    fig.add_annotation(
        x=x_str, y=y, row=1, col=1, text=f"<b>{text}</b>",
        showarrow=arrow, arrowhead=2, arrowsize=1, arrowwidth=1.2, arrowcolor="#e5e7eb",
        yshift=yshift, font=dict(size=10, color=palette["fg"]),
        bgcolor=palette["bg"], bordercolor="#e5e7eb", borderwidth=1, borderpad=3
    )

# ============================================================
# TRADING FUNCTIONS
# ============================================================
def add_marker(x_str, price, kind):
    st.session_state.markers.append({"x_str": x_str, "price": float(price), "kind": kind})

def log_and_save(log_text):
    st.session_state.trade_log.append(log_text)
    save_state_to_db(
        st.session_state.balance, st.session_state.realized_pnl,
        st.session_state.shares, st.session_state.entry_price,
        st.session_state.stop_loss, st.session_state.target,
        new_log=log_text
    )

def close_position(qty, price, x_str, reason=""):
    pos = float(st.session_state.shares)
    if pos > 0:
        qty = min(float(qty), pos)
        proceeds = qty * float(price)
        cost = qty * st.session_state.entry_price
        pnl = proceeds - cost
        st.session_state.balance += proceeds
        st.session_state.realized_pnl += pnl
        st.session_state.shares -= qty
        log_and_save(f"{x_str}: SOLD {qty:.4f} sh at ${price:.2f} {reason} | P&L: ${pnl:.2f}")
        add_marker(x_str, price, "SELL")
    elif pos < 0:
        qty = min(float(qty), abs(pos))
        cost_to_cover = qty * float(price)
        short_proceeds = qty * st.session_state.entry_price
        pnl = short_proceeds - cost_to_cover
        st.session_state.balance -= cost_to_cover
        st.session_state.realized_pnl += pnl
        st.session_state.shares += qty
        log_and_save(f"{x_str}: COVERED {qty:.4f} sh at ${price:.2f} {reason} | P&L: ${pnl:.2f}")
        add_marker(x_str, price, "COVER")

    if abs(float(st.session_state.shares)) < 1e-10:
        st.session_state.shares = 0.0
        st.session_state.stop_loss = None
        st.session_state.target = None
        st.session_state.entry_price = None
        save_state_to_db(st.session_state.balance, st.session_state.realized_pnl, 0, 0, None, None)

def check_account_depletion(current_price):
    pos = st.session_state.shares
    pos_val = (pos * current_price) if pos > 0 else (-(abs(pos) * current_price) if pos < 0 else 0)
    equity = st.session_state.balance + pos_val
    if equity <= 0:
        if pos != 0:
            close_position(abs(pos), current_price, "SYSTEM", "(Liquidation)")
        if st.session_state.balance <= 0:
            blow_up_account()
            st.toast("⚠️ Account depleted! Starting new $1,000 cycle.", icon="💀")
            st.rerun()

def advance_bars(number_of_bars):
    df = st.session_state.df
    max_step = len(df) - 1

    for _ in range(number_of_bars):
        if st.session_state.step >= max_step:
            st.toast("End of available history reached.", icon="🔔")
            break

        prev_date = df.iloc[st.session_state.step]["date_only"]
        st.session_state.step += 1
        
        row = df.iloc[st.session_state.step]
        x_str = row["x_str"]
        pos = float(st.session_state.shares)
        sl, tp = st.session_state.stop_loss, st.session_state.target

        if not st.session_state.carry_overnight and pos != 0 and prev_date != row["date_only"]:
            close_position(abs(pos), float(df.iloc[st.session_state.step-1]["close"]), df.iloc[st.session_state.step-1]["x_str"], "(🛑 Session Force-Close)")
            pos = 0

        if pos > 0:
            if sl is not None and float(row["low"]) <= float(sl):
                close_position(pos, min(float(sl), float(row["open"])), x_str, "(🛑 STOP)")
                break
            if tp is not None and float(row["high"]) >= float(tp):
                close_position(pos, max(float(tp), float(row["open"])), x_str, "(🎯 TARGET)")
                break
        elif pos < 0:
            if sl is not None and float(row["high"]) >= float(sl):
                close_position(abs(pos), max(float(sl), float(row["open"])), x_str, "(🛑 STOP)")
                break
            if tp is not None and float(row["low"]) <= float(tp):
                close_position(abs(pos), min(float(tp), float(row["open"])), x_str, "(🎯 TARGET)")
                break

        check_account_depletion(float(row["close"]))

# ============================================================
# SIDEBAR SETUP
# ============================================================
st.sidebar.header("⚙️ Setup")
ticker = st.sidebar.text_input("Ticker", value="NQ=F").upper().strip()
timeframe = st.sidebar.selectbox("Chart Timeframe", TIMEFRAME_OPTIONS, index=2)

session_mode_ui = st.sidebar.selectbox("Session Mode", ["Auto Detect", "US Stocks (Regular Hours)", "Continuous (Futures/Forex)"], index=0)
if session_mode_ui == "Auto Detect":
    if ticker.endswith("=F") or ticker.endswith("=X") or ticker.endswith("-USD"):
        applied_session = "Continuous (Futures/Forex)"
    else:
        applied_session = "US Stocks (Regular Hours)"
    st.sidebar.info(f"Detected Session: **{applied_session}**")
else:
    applied_session = session_mode_ui

carry_overnight = st.sidebar.checkbox("Carry Positions Overnight", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Moving Averages")

show_ma = st.sidebar.checkbox("Show moving averages", value=True)
ma_type = st.sidebar.radio("MA Type", ["EMA", "SMA"], horizontal=True)

default_fast, default_slow = TIMEFRAME_CONFIG[timeframe]["ema"]
fast_len = st.sidebar.number_input("Fast MA length", min_value=2, max_value=400, value=int(default_fast), step=1)
slow_len = st.sidebar.number_input("Slow MA length", min_value=2, max_value=500, value=int(default_slow), step=1)

show_vol_ma = st.sidebar.checkbox("Show volume MA", value=True)
vol_ma_len = st.sidebar.number_input("Volume MA length", min_value=2, max_value=200, value=20, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🏗️ Structure")
show_struct = st.sidebar.checkbox("Show structure labels", value=True)
show_zones = st.sidebar.checkbox("Show S/R zones", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📷 Chart Context")
context_preset = st.sidebar.selectbox("Context preset", CONTEXT_PRESETS, index=1)
follow_replay = st.sidebar.checkbox("Follow replay candle", value=True)
st.sidebar.caption(
    "Follow ON + continuous (futures/forex/crypto): sliding 220-bar window.\n\n"
    "Follow ON + stocks: current session zoom.\n\n"
    "Follow OFF: chart shows the FULL selected context, y-axis auto-fits everything."
)

st.sidebar.markdown("---")
st.sidebar.subheader("✏️ Drawings")
draw_mode = st.sidebar.selectbox("Drawing Mode", DRAW_MODES, index=0)
if st.sidebar.button("Undo last drawing"):
    if st.session_state.drawings: st.session_state.drawings.pop()
if st.sidebar.button("Clear all drawings"):
    st.session_state.drawings = []

st.sidebar.markdown("---")
source_df = pd.DataFrame()
can_start = False
if ticker:
    with st.spinner(f"Loading data..."):
        source_df = fetch_native_history(ticker, timeframe)

if not source_df.empty:
    available_dates = sorted(source_df["date_only"].unique())
    default_date = available_dates[-3] if len(available_dates) >= 3 else available_dates[-1]
    
    st.sidebar.success(f"{len(source_df):,} bars available")
    practice_date = st.sidebar.date_input("Practice Date", min_value=available_dates[0], max_value=available_dates[-1], value=default_date)
    practice_rows = source_df[source_df["date_only"] == practice_date]
    
    if not practice_rows.empty:
        start_labels = practice_rows["x_str"].tolist()
        selected_start_str = st.sidebar.selectbox("Start Time", start_labels, index=min(10, len(start_labels)-1))
        can_start = True

if st.sidebar.button("🚀 Start / Reset Replay", disabled=not can_start):
    active_df = source_df.copy().reset_index(drop=True)
    idx_list = active_df.index[active_df["x_str"] == selected_start_str].tolist()
    
    st.session_state.df = active_df
    st.session_state.step = idx_list[0]
    st.session_state.active_ticker = ticker
    st.session_state.active_interval = timeframe
    st.session_state.active_practice_date = practice_date
    st.session_state.session_mode = applied_session
    st.session_state.carry_overnight = carry_overnight
    
    st.session_state.camera_revision += 1
    st.session_state.force_camera = True
    st.session_state.sim_active = True
    st.rerun()

if not st.session_state.sim_active or st.session_state.df.empty:
    st.info("Set parameters and click Start / Reset Replay.")
    st.stop()

# ============================================================
# MAIN DATA & CALCULATIONS
# ============================================================
df_all = st.session_state.df
step = st.session_state.step
current_row = df_all.iloc[step]
current_price = float(current_row["close"])
current_x_str = current_row["x_str"]

revealed_df = df_all.iloc[:step + 1].copy()
calc_df = revealed_df.copy()

if show_ma:
    calc_df["fast_ma"] = moving_avg(calc_df["close"], fast_len, ma_type)
    calc_df["slow_ma"] = moving_avg(calc_df["close"], slow_len, ma_type)
if show_vol_ma:
    calc_df["vol_ma"] = calc_df["volume"].rolling(int(vol_ma_len)).mean()

chart_df = get_context_render_df(calc_df, st.session_state.active_practice_date, context_preset).reset_index(drop=True)

# Account Metrics
pos = float(st.session_state.shares)
if pos > 0:
    pos_type = "🟢 LONG"
    pos_val = pos * current_price
    equity = st.session_state.balance + pos_val
    unrealized = pos_val - (pos * st.session_state.entry_price)
elif pos < 0:
    pos_type = "🔴 SHORT"
    pos_val = -(abs(pos) * current_price) 
    equity = st.session_state.balance + pos_val
    unrealized = (abs(pos) * st.session_state.entry_price) - abs(pos) * current_price
else:
    pos_type = "FLAT"
    pos_val = 0.0
    equity = st.session_state.balance
    unrealized = 0.0

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Price", f"${current_price:.2f}")
m2.metric("Cash", f"${st.session_state.balance:.2f}")
m3.metric("Equity", f"${equity:.2f}", f"${st.session_state.realized_pnl + unrealized:.2f} Net")
m4.metric("Realized P&L", f"${st.session_state.realized_pnl:.2f}")
m5.metric("Unrealized P&L", f"${unrealized:.2f}")
m6.metric("Cycle", f"#{st.session_state.cycle}")

st.caption(f"Account cycle {st.session_state.cycle} • Starting $1000.00 • Session: {st.session_state.session_mode}")

# ============================================================
# CHART RENDER (GAPLESS CATEGORICAL X-AXIS)
# ============================================================
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03)

fig.add_trace(go.Candlestick(
    x=chart_df["x_str"], 
    open=chart_df["open"], 
    high=chart_df["high"], 
    low=chart_df["low"], 
    close=chart_df["close"],
    name=st.session_state.active_ticker, 
    increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
    decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350")
), row=1, col=1)

vol_colors = np.where(chart_df["close"] >= chart_df["open"], "rgba(38,166,154,0.58)", "rgba(239,83,80,0.58)")
fig.add_trace(go.Bar(x=chart_df["x_str"], y=chart_df["volume"], marker_color=vol_colors, name="Volume"), row=2, col=1)

if show_ma and "fast_ma" in chart_df.columns:
    fig.add_trace(go.Scatter(x=chart_df["x_str"], y=chart_df["fast_ma"], name=f"{ma_type}{int(fast_len)}", line=dict(color=C_FAST_EMA, width=1.7)), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df["x_str"], y=chart_df["slow_ma"], name=f"{ma_type}{int(slow_len)}", line=dict(color=C_SLOW_EMA, width=1.7)), row=1, col=1)

if show_vol_ma and "vol_ma" in chart_df.columns:
    fig.add_trace(go.Scatter(x=chart_df["x_str"], y=chart_df["vol_ma"], name=f"VolMA{int(vol_ma_len)}", line=dict(color=C_VOL_MA, width=1.7)), row=2, col=1)

# FIX: type=category + explicitly killing rangeslider on BOTH rows prevents "ghost candle" duplication
fig.update_xaxes(type='category', categoryorder='array', categoryarray=chart_df["x_str"], nticks=12, gridcolor="#1f2937", showspikes=True, spikemode="across")
fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
fig.update_yaxes(gridcolor="#1f2937")

# Positions & Markers
if pos != 0:
    if st.session_state.entry_price: fig.add_hline(y=st.session_state.entry_price, line=dict(color="#22d3ee", dash="dot"))
    if st.session_state.stop_loss: fig.add_hline(y=st.session_state.stop_loss, line=dict(color="#fb923c", dash="dash"))
    if st.session_state.target: fig.add_hline(y=st.session_state.target, line=dict(color="#4ade80", dash="dash"))

for m in st.session_state.markers:
    if m["x_str"] in chart_df["x_str"].values:
        sym = "triangle-up" if m["kind"] == "LONG" else "triangle-down" if m["kind"] == "SHORT" else "circle"
        col = "#22c55e" if m["kind"] == "LONG" else "#ef4444" if m["kind"] == "SHORT" else "#f97316"
        fig.add_trace(go.Scatter(x=[m["x_str"]], y=[m["price"]], mode="markers", marker=dict(symbol=sym, size=12, color=col)), row=1, col=1)

# Structure and Zones
if show_struct or show_zones:
    struct_df = chart_df.tail(MAX_STRUCTURE_BARS).copy().reset_index(drop=True)
    labeled = _raw_swings(struct_df)
    conf_swings = [s for s in labeled if s["i"] + SWING_K <= len(struct_df) - 1]
    
    if show_struct:
        bos, choch = detect_bos_choch(struct_df, labeled)
        for s in conf_swings[-MAX_SWING_LABELS:]:
            palette = C_HH if s["label"] in ("HH","HL") else C_LH if s["label"] in ("LH","LL") else C_H
            badge(fig, struct_df["x_str"].iloc[s["i"]], s["price"], s["label"], palette, yshift=18 if s["type"]=="H" else -18)
        for b in bos[-MAX_BOS_LABELS:]:
            if 0 <= b["i"] < len(struct_df):
                badge(fig, struct_df["x_str"].iloc[b["i"]], float(struct_df["high"].iloc[b["i"]] if b["dir"]=="up" else struct_df["low"].iloc[b["i"]]),
                      "BOS↑" if b["dir"]=="up" else "BOS↓", C_BOS_UP if b["dir"]=="up" else C_BOS_DN, yshift=22 if b["dir"]=="up" else -22, arrow=True)
        for ch in choch[-MAX_CHOCH_LABELS:]:
            if 0 <= ch["i"] < len(struct_df):
                badge(fig, struct_df["x_str"].iloc[ch["i"]], float(struct_df["high"].iloc[ch["i"]] if ch["dir"]=="up" else struct_df["low"].iloc[ch["i"]]),
                      "CHoCH↑" if ch["dir"]=="up" else "CHoCH↓", C_CH_UP if ch["dir"]=="up" else C_CH_DN, yshift=22 if ch["dir"]=="up" else -22, arrow=True)

    if show_zones and conf_swings:
        noise = float((struct_df["high"] - struct_df["low"]).tail(20).mean() or 0.01)
        zones = build_zones(conf_swings[-MAX_ZONE_SWINGS:], current_price)
        zones = sorted(zones, key=lambda z: (abs(z["mid"] - current_price), -z["touches"]))[:MAX_ZONES_DRAWN]

        for z in zones:
            role = zone_role(z, current_price, noise)
            color = C_FLOOR if role == "floor" else C_CEIL if role == "ceiling" else C_BOTH if role == "both" else C_BROKEN
            fig.add_hline(y=z["mid"], line=dict(color=color, width=min(1 + z["touches"], 4), dash="dot"))

# Drawings
for d in st.session_state.drawings:
    if d["type"] == "hline": fig.add_hline(y=d["price"], line=dict(color=d.get("color", "#22d3ee")))
    elif d["type"] == "band": fig.add_hrect(y0=d["y0"], y1=d["y1"], fillcolor="rgba(34,197,94,0.18)", line_width=0, layer="below")
    elif d["type"] == "trend": 
        if d["x0"] in chart_df["x_str"].values and d["x1"] in chart_df["x_str"].values:
            fig.add_trace(go.Scatter(x=[d["x0"], d["x1"]], y=[d["y0"], d["y1"]], mode="lines", line=dict(color="#f472b6")))

# ============================================================
# CAMERA CONTROL — FIXED Y-AXIS SCALING
# ============================================================
apply_camera = follow_replay or st.session_state.force_camera

if apply_camera:
    if follow_replay:
        # Tight, windowed "current action" view
        if "Continuous" in st.session_state.session_mode:
            end_idx = len(chart_df) - 1
            start_idx = max(0, end_idx - 220)
        else:
            day_rows = chart_df[chart_df["date_only"] == current_row["date_only"]]
            if not day_rows.empty:
                start_idx = day_rows.index[0]
                end_idx = day_rows.index[-1]
            else:
                start_idx, end_idx = 0, len(chart_df) - 1
        pad = 10
        end_idx = min(len(chart_df) - 1, end_idx + pad)
    else:
        # FOLLOW OFF: fit the ENTIRE selected context (e.g. all 5 previous days)
        # so nothing gets clipped off the top/bottom of the Y-axis.
        start_idx, end_idx = 0, len(chart_df) - 1

    fig.update_xaxes(range=[start_idx - 0.5, end_idx + 0.5], row=1, col=1)
    fig.update_xaxes(range=[start_idx - 0.5, end_idx + 0.5], row=2, col=1)

    vis_df = chart_df.iloc[start_idx: min(end_idx + 1, len(chart_df))]
    pr = get_price_axis_range(vis_df)
    vr = get_volume_axis_range(vis_df)
    if pr: fig.update_yaxes(range=pr, row=1, col=1)
    if vr: fig.update_yaxes(range=vr, row=2, col=1)

    st.session_state.force_camera = False

# uirevision preserves manual pan/zoom when Follow is OFF and camera isn't being forced
uirevision_key = f"follow-{step}" if follow_replay else f"manual-{st.session_state.camera_revision}"

fig.update_layout(
    template="plotly_dark", height=650, paper_bgcolor="#000", plot_bgcolor="#000",
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_rangeslider_visible=False,
    uirevision=uirevision_key,
)

# Rendering UI
c1, c2 = st.columns([6, 1])
with c1:
    if draw_mode == "None":
        st.plotly_chart(fig, key="main_chart")
    else:
        event = st.plotly_chart(fig, key="draw_chart", on_select="rerun", selection_mode="points")
        if event and event.selection.points:
            pt = event.selection.points[-1]
            x_str = pt.get("x")
            clicked_y = pt.get("y")
            
            if (draw_mode, x_str, clicked_y) != st.session_state.last_draw_signature:
                st.session_state.last_draw_signature = (draw_mode, x_str, clicked_y)
                row_match = chart_df[chart_df["x_str"] == x_str]
                if not row_match.empty:
                    row = row_match.iloc[0]
                    if draw_mode == "Line at High": st.session_state.drawings.append({"type":"hline", "price":float(row["high"]), "color":"#ef4444"})
                    elif draw_mode == "Line at Close": st.session_state.drawings.append({"type":"hline", "price":float(row["close"]), "color":"#22d3ee"})
                    elif draw_mode == "Line at Low": st.session_state.drawings.append({"type":"hline", "price":float(row["low"]), "color":"#22c55e"})
                    elif draw_mode in ["Band", "Trend line"]:
                        st.session_state.draw_clicks.append({"x": x_str, "y": float(clicked_y)})
                        if len(st.session_state.draw_clicks) >= 2:
                            p1, p2 = st.session_state.draw_clicks[0], st.session_state.draw_clicks[1]
                            if draw_mode == "Band": st.session_state.drawings.append({"type":"band", "y0":min(p1["y"], p2["y"]), "y1":max(p1["y"], p2["y"])})
                            else: st.session_state.drawings.append({"type":"trend", "x0":p1["x"], "y0":p1["y"], "x1":p2["x"], "y1":p2["y"]})
                            st.session_state.draw_clicks = []
                st.rerun()

with c2:
    st.markdown(f"**{current_x_str}**")
    
    mins = TIMEFRAME_CONFIG[st.session_state.active_interval]["minutes"]
    if st.button("▶️ +5m"): advance_bars(max(1, 5 // mins)); st.rerun()
    if st.button("⏩ +15m"): advance_bars(max(1, 15 // mins)); st.rerun()
    if st.button("⏭️ +30m"): advance_bars(max(1, 30 // mins)); st.rerun()
    if st.button("⏭️ +60m"): advance_bars(max(1, 60 // mins)); st.rerun()

    st.markdown("---")
    st.caption(f"Pos: {pos_type}")
    st.caption(f"Bars: {len(chart_df)}")

# ============================================================
# TRADING PANEL
# ============================================================
st.markdown("### 🎮 Trade Panel")
t1, t2, t3, t4, t5 = st.columns(5)

with t1: bet_size = st.number_input("Trade Amount ($)", min_value=10.0, value=100.0, step=10.0)
with t2: sl_input = st.number_input("Stop Loss ($)", value=round(current_price * 0.99, 2))
with t3: tp_input = st.number_input("Target ($)", value=0.0)

if pos == 0:
    with t4:
        if st.button("🟢 BUY (Long)"):
            sh = bet_size / current_price
            st.session_state.balance -= bet_size
            st.session_state.shares = sh
            st.session_state.entry_price = current_price
            st.session_state.stop_loss = sl_input
            st.session_state.target = tp_input if tp_input > 0 else None
            log_and_save(f"{current_x_str}: BOUGHT {sh:.4f} sh at ${current_price:.2f}")
            add_marker(current_x_str, current_price, "LONG")
            st.rerun()
    with t5:
        if st.button("🔴 SHORT"):
            sh = bet_size / current_price
            st.session_state.balance += bet_size
            st.session_state.shares = -sh
            st.session_state.entry_price = current_price
            st.session_state.stop_loss = sl_input
            st.session_state.target = tp_input if tp_input > 0 else None
            log_and_save(f"{current_x_str}: SHORTED {sh:.4f} sh at ${current_price:.2f}")
            add_marker(current_x_str, current_price, "SHORT")
            st.rerun()
else:
    with t4:
        if st.button("💾 Update SL/TP"):
            st.session_state.stop_loss = sl_input
            st.session_state.target = tp_input if tp_input > 0 else None
            log_and_save(f"{current_x_str}: UPDATED SL ${sl_input} / TP ${tp_input}")
            st.rerun()
    with t5:
        if st.button("EXIT POSITION"):
            close_position(abs(pos), current_price, current_x_str, "(Manual Exit)")
            st.rerun()

with st.expander("📝 Trade History", expanded=True):
    for log in reversed(st.session_state.trade_log): st.text(log)

# ============================================================
# SESSION INFORMATION
# ============================================================
with st.expander("📘 Replay Information"):
    st.markdown(
        f"""
| Item | Value |
|---|---|
| Ticker | `{st.session_state.active_ticker}` |
| Native timeframe | `{st.session_state.active_interval}` |
| Current replay time | `{current_x_str}` |
| Context preset | `{context_preset}` |
| Follow replay | `{"ON" if follow_replay else "OFF"}` |
| Saved drawings | `{len(st.session_state.drawings)}` |

### Database Persistence
This app uses a local SQLite database (`paper_trading.db`) to store your wallet. 
You can switch tickers, refresh the page, or stop the server, and your cash, positions, and history will remain fully intact.
"""
    )
