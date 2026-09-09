# ============================================================
# HUMAN PAPER-TRADING REPLAY APP
# Native multi-timeframe data + context camera + drawings
# ============================================================

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, time as dtime

st.set_page_config(layout="wide", page_title="Market Replay Simulator")

# ============================================================
# CONSTANTS
# ============================================================
INTERVAL_OPTIONS = ["1m", "2m", "5m", "15m", "30m", "1h", "1d"]

# conservative Yahoo lookbacks (calendar days)
INTERVAL_LOOKBACK_DAYS = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "1h": 730,
    "1d": 3650,
}

INTERVAL_MAX_AGE_DAYS = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "1h": 730,
    "1d": 3650,
}

DEFAULT_EMA = {
    "1m":  (9, 21),
    "2m":  (9, 21),
    "5m":  (20, 50),
    "15m": (50, 200),
    "30m": (50, 200),
    "1h":  (50, 200),
    "1d":  (50, 200),
}

# structure constants
SWING_K = 5
ZONE_TOL = 0.0012
MIN_DRAW_TOUCHES = 2
MIN_SWING_PCT = 0.0008
POLARITY_EDGE = 2
CHOCH_ENABLE = True

# colors for black background
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
C_BOTH   = "#eab308"

DRAW_MODES = [
    "None",
    "Line at High",
    "Line at Close",
    "Line at Low",
    "Band",
    "Trend line",
]

CONTEXT_PRESETS = [
    "Today",
    "1 Previous Day",
    "5 Previous Days",
    "20 Previous Days",
    "All Available History",
]

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "sim_active": False,
    "balance": 1000.0,
    "shares": 0.0,
    "stop_loss": None,
    "target": None,
    "entry_price": None,
    "step": 0,
    "df": pd.DataFrame(),          # native bars for selected interval/date window
    "trade_log": [],
    "sim_start_idx": 0,            # first bar of practice day
    "drawings": [],                # persistent drawings
    "draw_clicks": [],             # temp clicks for band/trend
    "selected_interval": "5m",
    "practice_date": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# HELPERS
# ============================================================
def max_practice_date_for_interval(interval: str) -> datetime.date:
    return datetime.now().date()

def min_practice_date_for_interval(interval: str) -> datetime.date:
    age = INTERVAL_MAX_AGE_DAYS.get(interval, 60)
    # leave a little buffer for weekends/holidays/Yahoo quirks
    return datetime.now().date() - timedelta(days=max(age - 1, 1))


def interval_step_minutes(interval: str) -> int:
    return {
        "1m": 1,
        "2m": 2,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "1d": 24 * 60,
    }[interval]


def moving_avg(series: pd.Series, length: int, kind: str = "EMA") -> pd.Series:
    if kind == "EMA":
        return series.ewm(span=length, adjust=False).mean()
    return series.rolling(length).mean()


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    out = out.reset_index()
    out.columns = [str(c).lower() for c in out.columns]

    # timestamp column
    ts_col = None
    for c in out.columns:
        if "date" in c or "time" in c:
            ts_col = c
            break
    if ts_col is None:
        ts_col = out.columns[0]

    out.rename(columns={ts_col: "timestamp"}, inplace=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if out["timestamp"].dt.tz is not None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(None)

    # standardize names
    rename_map = {}
    for c in out.columns:
        lc = c.lower()
        if lc.startswith("open"):
            rename_map[c] = "open"
        elif lc.startswith("high"):
            rename_map[c] = "high"
        elif lc.startswith("low"):
            rename_map[c] = "low"
        elif lc.startswith("close") or lc == "adj close":
            rename_map[c] = "close"
        elif lc.startswith("volume"):
            rename_map[c] = "volume"
    out.rename(columns=rename_map, inplace=True)

    need = ["timestamp", "open", "high", "low", "close"]
    for c in need:
        if c not in out.columns:
            return pd.DataFrame()

    if "volume" not in out.columns:
        out["volume"] = 0.0
    out["volume"] = out["volume"].fillna(0.0)

    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    out["date_only"] = out["timestamp"].dt.date
    out["label"] = out["timestamp"].dt.strftime("%m-%d %H:%M")
    return out.reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_native_bars(ticker: str, interval: str, practice_date) -> pd.DataFrame:
    """
    Fetch native bars for the selected interval.
    Loads prior history + the practice day.
    """
    lookback = INTERVAL_LOOKBACK_DAYS.get(interval, 60)
    start = practice_date - timedelta(days=lookback)
    end = practice_date + timedelta(days=2)  # end is exclusive-ish in yfinance

    try:
        raw = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
            progress=False,
            prepost=False,
        )
    except Exception:
        return pd.DataFrame()

    df = normalize_ohlcv(raw)
    if df.empty:
        return df

    # keep only up to practice_date (inclusive)
    df = df[df["date_only"] <= practice_date].copy()
    return df.reset_index(drop=True)


def get_practice_day_indices(df: pd.DataFrame, practice_date):
    idxs = df.index[df["date_only"] == practice_date].tolist()
    return idxs


def context_cutoff_date(practice_date, preset: str):
    if preset == "Today":
        return practice_date
    if preset == "1 Previous Day":
        return practice_date - timedelta(days=1)
    if preset == "5 Previous Days":
        return practice_date - timedelta(days=7)   # calendar buffer
    if preset == "20 Previous Days":
        return practice_date - timedelta(days=30)  # calendar buffer
    return None  # all available


def badge(fig, x, y, text, pal, yshift=0, arrow=False):
    fig.add_annotation(
        x=x, y=y, text=f"<b>{text}</b>", row=1, col=1,
        showarrow=arrow, arrowhead=2, arrowsize=1, arrowwidth=1.4,
        arrowcolor="#e5e7eb", yshift=yshift,
        font=dict(size=11, color=pal["fg"], family="Arial"),
        bgcolor=pal["bg"], bordercolor="#e5e7eb", borderwidth=1, borderpad=3,
        opacity=1, align="center",
    )


# ============================================================
# STRUCTURE DETECTION
# ============================================================
def _raw_swings(df: pd.DataFrame):
    if df is None or len(df) < SWING_K * 2 + 1:
        return []
    highs, lows = df["high"].values, df["low"].values
    n = len(df)
    ref = float(df["close"].iloc[-1])
    min_size = max(ref * MIN_SWING_PCT, 1e-9)
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
            if (s[2] == "H" and s[1] >= cleaned[-1][1]) or (s[2] == "L" and s[1] <= cleaned[-1][1]):
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
            zones.append({
                "mid": s["price"], "prices": [s["price"]], "touches": 1,
                "low_touches": 1 if s["type"] == "L" else 0,
                "high_touches": 1 if s["type"] == "H" else 0,
            })
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
            if s["label"] == "HH":
                last_hh = s["price"]
            elif s["label"] == "HL":
                last_hl = s["price"]
            elif s["label"] == "LH":
                last_lh = s["price"]
            elif s["label"] == "LL":
                last_ll = s["price"]
            ptr += 1
        c = float(df["close"].iloc[i])
        if last_hh is not None and c > last_hh:
            if bias != "BULL":
                bos.append({"i": i, "price": last_hh, "dir": "up"})
                bias = "BULL"
            last_hh = None
            continue
        if last_ll is not None and c < last_ll:
            if bias != "BEAR":
                bos.append({"i": i, "price": last_ll, "dir": "down"})
                bias = "BEAR"
            last_ll = None
            continue
        if CHOCH_ENABLE:
            if bias == "BULL" and last_hl is not None and c < last_hl:
                choch.append({"i": i, "price": last_hl, "dir": "down"})
                last_hl = None
                bias = "NEUTRAL"
            elif bias == "BEAR" and last_lh is not None and c > last_lh:
                choch.append({"i": i, "price": last_lh, "dir": "up"})
                last_lh = None
                bias = "NEUTRAL"
    return bos, choch


def zone_role(z, close, noise):
    if z["low_touches"] >= z["high_touches"] + POLARITY_EDGE:
        role = "floor"
    elif z["high_touches"] >= z["low_touches"] + POLARITY_EDGE:
        role = "ceiling"
    else:
        role = "both"
    if role == "floor" and close < z["min_px"] - noise:
        return "broken_floor"
    if role == "ceiling" and close > z["max_px"] + noise:
        return "broken_ceiling"
    return role


# ============================================================
# TRADING HELPERS
# ============================================================
def close_position(qty, price, time_str, reason=""):
    pos = st.session_state.shares
    if pos > 0:
        qty = min(qty, pos)
        st.session_state.balance += qty * price
        st.session_state.shares -= qty
        st.session_state.trade_log.append(
            f"{time_str}: SOLD {qty:.4f} sh at ${price:.2f} {reason}"
        )
    elif pos < 0:
        qty = min(qty, abs(pos))
        st.session_state.balance -= qty * price
        st.session_state.shares += qty
        st.session_state.trade_log.append(
            f"{time_str}: COVERED {qty:.4f} sh at ${price:.2f} {reason}"
        )
    if abs(st.session_state.shares) < 1e-12:
        st.session_state.shares = 0.0
        st.session_state.stop_loss = None
        st.session_state.target = None
        st.session_state.entry_price = None


def advance_bars(n_bars: int):
    df = st.session_state.df
    max_step = len(df) - 1
    for _ in range(n_bars):
        if st.session_state.step >= max_step:
            st.toast("No more bars in this dataset / day.", icon="🔔")
            break

        st.session_state.step += 1
        row = df.iloc[st.session_state.step]
        t = row["timestamp"].strftime("%Y-%m-%d %H:%M")
        pos = st.session_state.shares
        sl, tp = st.session_state.stop_loss, st.session_state.target

        # stop/target checks using this bar's high/low
        if pos > 0:
            if sl is not None and row["low"] <= sl:
                exec_p = min(sl, row["open"])
                close_position(pos, float(exec_p), t, "(🛑 STOP-LOSS)")
                st.toast(f"🛑 Long stopped at ${exec_p:.2f}", icon="💥")
                break
            if tp is not None and row["high"] >= tp:
                exec_p = max(tp, row["open"])
                close_position(pos, float(exec_p), t, "(🎯 TARGET HIT)")
                st.toast(f"🎯 Target hit at ${exec_p:.2f}", icon="🎉")
                break
        elif pos < 0:
            if sl is not None and row["high"] >= sl:
                exec_p = max(sl, row["open"])
                close_position(abs(pos), float(exec_p), t, "(🛑 STOP-LOSS)")
                st.toast(f"🛑 Short stopped at ${exec_p:.2f}", icon="💥")
                break
            if tp is not None and row["low"] <= tp:
                exec_p = min(tp, row["open"])
                close_position(abs(pos), float(exec_p), t, "(🎯 TARGET HIT)")
                st.toast(f"🎯 Target hit at ${exec_p:.2f}", icon="🎉")
                break


# ============================================================
# DRAWING HELPERS
# ============================================================
def add_hline_drawing(price: float, label: str = "", color: str = "#22d3ee"):
    st.session_state.drawings.append({
        "type": "hline",
        "price": float(price),
        "label": label,
        "color": color,
    })


def add_band_drawing(p1: float, p2: float, color: str = "rgba(34,197,94,0.18)"):
    lo, hi = min(p1, p2), max(p1, p2)
    st.session_state.drawings.append({
        "type": "band",
        "y0": float(lo),
        "y1": float(hi),
        "color": color,
        "label": f"Band {lo:.2f}-{hi:.2f}",
    })


def add_trend_drawing(x0, y0, x1, y1, color: str = "#f472b6"):
    st.session_state.drawings.append({
        "type": "trend",
        "x0": x0, "y0": float(y0),
        "x1": x1, "y1": float(y1),
        "color": color,
        "label": "Trend",
    })


def nearest_row_from_click(df: pd.DataFrame, x_val, y_val=None):
    """
    Map a click to nearest candle.
    x_val may be label string or timestamp-like value.
    """
    if df.empty:
        return None

    # try label match first
    if isinstance(x_val, str):
        hits = df.index[df["label"] == x_val].tolist()
        if hits:
            return df.iloc[hits[0]]
        # fuzzy: contains
        m = df["label"].astype(str) == str(x_val)
        if m.any():
            return df[m].iloc[0]

    # fallback nearest by y only if needed
    if y_val is not None:
        # choose candle whose body/range contains y, else nearest close
        y = float(y_val)
        containing = df[(df["low"] <= y) & (df["high"] >= y)]
        if not containing.empty:
            # nearest in time to end
            return containing.iloc[-1]
        idx = (df["close"] - y).abs().idxmin()
        return df.loc[idx]
    return df.iloc[-1]


def handle_drawing_click(mode: str, row: pd.Series, x_label: str):
    if row is None or mode == "None":
        return

    if mode == "Line at High":
        add_hline_drawing(float(row["high"]), label=f"H {float(row['high']):.2f}", color="#ef4444")
        st.session_state.draw_clicks = []
        st.toast(f"Line at High {float(row['high']):.2f}", icon="📌")

    elif mode == "Line at Close":
        add_hline_drawing(float(row["close"]), label=f"C {float(row['close']):.2f}", color="#22d3ee")
        st.session_state.draw_clicks = []
        st.toast(f"Line at Close {float(row['close']):.2f}", icon="📌")

    elif mode == "Line at Low":
        add_hline_drawing(float(row["low"]), label=f"L {float(row['low']):.2f}", color="#22c55e")
        st.session_state.draw_clicks = []
        st.toast(f"Line at Low {float(row['low']):.2f}", icon="📌")

    elif mode == "Band":
        clicks = st.session_state.draw_clicks
        clicks.append({"x": x_label, "y": float(row["close"]), "row": row})
        st.session_state.draw_clicks = clicks
        if len(clicks) == 1:
            st.toast("Band: first point set. Click second point.", icon="🖱️")
        elif len(clicks) >= 2:
            y1 = float(clicks[0]["y"])
            y2 = float(clicks[1]["y"])
            # if user clicked wicks intentionally, prefer high/low by proximity
            # simple: use close/close; advanced selectors available below too
            add_band_drawing(y1, y2)
            st.session_state.draw_clicks = []
            st.toast(f"Band {min(y1,y2):.2f} - {max(y1,y2):.2f}", icon="🟩")

    elif mode == "Trend line":
        clicks = st.session_state.draw_clicks
        # use close as default anchor; good enough and stable
        clicks.append({"x": x_label, "y": float(row["close"])})
        st.session_state.draw_clicks = clicks
        if len(clicks) == 1:
            st.toast("Trend: first point set. Click second point.", icon="🖱️")
        elif len(clicks) >= 2:
            a, b = clicks[0], clicks[1]
            add_trend_drawing(a["x"], a["y"], b["x"], b["y"])
            st.session_state.draw_clicks = []
            st.toast("Trend line added", icon="📈")


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Setup")

ticker = st.sidebar.text_input("Ticker", value="TQQQ").upper().strip()
interval = st.sidebar.selectbox("Chart Timeframe (native)", INTERVAL_OPTIONS, index=2)

min_d = min_practice_date_for_interval(interval)
max_d = max_practice_date_for_interval(interval)
default_d = max(min_d, max_d - timedelta(days=2))
practice_date = st.sidebar.date_input(
    "Practice Date",
    value=default_d,
    min_value=min_d,
    max_value=max_d,
)

st.sidebar.caption(
    f"Native **{interval}** history window ≈ last {INTERVAL_LOOKBACK_DAYS[interval]} days. "
    f"1m is short; 5m/15m support older practice dates."
)

# EMA settings for active interval
st.sidebar.markdown("---")
show_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
ma_type = st.sidebar.radio("MA type", ["EMA", "SMA"], horizontal=True)
d_fast, d_slow = DEFAULT_EMA.get(interval, (20, 50))
fast_len = st.sidebar.number_input(f"{interval} Fast MA", 2, 400, int(d_fast), 1)
slow_len = st.sidebar.number_input(f"{interval} Slow MA", 2, 500, int(d_slow), 1)

st.sidebar.markdown("---")
show_struct = st.sidebar.checkbox("Show structure labels", value=True)
show_zones = st.sidebar.checkbox("Show S/R zones", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📷 Chart Context")
context_preset = st.sidebar.selectbox("Context preset", CONTEXT_PRESETS, index=0)
follow_replay = st.sidebar.checkbox("Follow replay candle", value=True)
st.sidebar.caption("Follow ON = after each advance, camera returns to practice day.")

st.sidebar.markdown("---")
st.sidebar.subheader("✏️ Drawing Mode")
draw_mode = st.sidebar.selectbox("Mode", DRAW_MODES, index=0)
if st.sidebar.button("Undo last drawing"):
    if st.session_state.drawings:
        st.session_state.drawings.pop()
        st.toast("Removed last drawing", icon="↩️")
if st.sidebar.button("Clear all drawings"):
    st.session_state.drawings = []
    st.session_state.draw_clicks = []
    st.toast("All drawings cleared", icon="🧹")
if st.session_state.draw_clicks:
    st.sidebar.info(f"Pending clicks: {len(st.session_state.draw_clicks)} / 2")

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Start / Reset Simulation", use_container_width=True):
    df = fetch_native_bars(ticker, interval, practice_date)
    if df.empty:
        st.sidebar.error("No data returned. Try another ticker/date/timeframe.")
    else:
        day_idxs = get_practice_day_indices(df, practice_date)
        if not day_idxs:
            st.sidebar.error(
                f"No {interval} bars for {practice_date}. "
                f"Market closed, or date too old for this timeframe."
            )
        else:
            # reset state
            st.session_state.df = df
            st.session_state.sim_start_idx = day_idxs[0]
            # start a bit into the day if many bars, else first bar
            st.session_state.step = day_idxs[min(10, len(day_idxs) - 1)]
            st.session_state.balance = 1000.0
            st.session_state.shares = 0.0
            st.session_state.stop_loss = None
            st.session_state.target = None
            st.session_state.entry_price = None
            st.session_state.trade_log = []
            st.session_state.drawings = []
            st.session_state.draw_clicks = []
            st.session_state.selected_interval = interval
            st.session_state.practice_date = practice_date
            st.session_state.sim_active = True
            st.rerun()

# ============================================================
# MAIN
# ============================================================
st.title("💹 Market Replay Simulator (Human)")
st.caption(
    "Native timeframe data • multi-day context • follow-camera • persistent drawings • paper trading"
)

if not st.session_state.sim_active or st.session_state.df.empty:
    st.info(
        "Choose ticker, **native timeframe**, and practice date in the sidebar, then click "
        "**Start / Reset Simulation**.\n\n"
        "- `1m`: recent days only\n"
        "- `5m/15m/30m`: practice dates further back (~60 days)\n"
        "- Chart starts on the practice day; zoom/pan left for history"
    )
    st.stop()

# If user changed interval in sidebar after start, ask them to restart
if st.session_state.selected_interval != interval or st.session_state.practice_date != practice_date:
    st.warning(
        f"Sidebar timeframe/date differs from active session "
        f"({st.session_state.selected_interval} / {st.session_state.practice_date}). "
        f"Click **Start / Reset Simulation** to apply."
    )

df_all = st.session_state.df
step = st.session_state.step
sim0 = st.session_state.sim_start_idx
practice_day = st.session_state.practice_date
active_interval = st.session_state.selected_interval

# revealed data: all prior days fully + practice day up to current step
revealed = df_all.iloc[: step + 1].copy()
if revealed.empty:
    st.error("No revealed bars.")
    st.stop()

# context filter for camera defaults (data still remains in revealed unless preset trims display)
cutoff = context_cutoff_date(practice_day, context_preset)
if cutoff is not None:
    display_df = revealed[revealed["date_only"] >= cutoff].copy()
else:
    display_df = revealed.copy()

if display_df.empty:
    display_df = revealed.copy()

display_df = display_df.reset_index(drop=True)

current = df_all.iloc[step]
current_price = float(current["close"])
current_time = current["timestamp"]
current_time_str = current_time.strftime("%Y-%m-%d %H:%M")

pos = st.session_state.shares
position_value = pos * current_price
equity = st.session_state.balance + position_value
pnl = equity - 1000.0
pnl_color = "normal" if pnl == 0 else ("inverse" if pnl > 0 else "off")

if pos > 0:
    pos_text, pos_type = f"{pos:.4f} sh", "🟢 LONG"
elif pos < 0:
    pos_text, pos_type = f"{abs(pos):.4f} sh", "🔴 SHORT"
else:
    pos_text, pos_type = "—", "FLAT"

# metrics
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Price", f"${current_price:.2f}")
c2.metric("Equity", f"${equity:.2f}", f"${pnl:.2f}", delta_color=pnl_color)
c3.metric("Cash", f"${st.session_state.balance:.2f}")
c4.metric(
    f"Position ({pos_type})",
    pos_text,
    f"Entry ${st.session_state.entry_price:.2f}" if st.session_state.entry_price else "",
)
c5.metric("Stop-Loss", f"${st.session_state.stop_loss:.2f}" if st.session_state.stop_loss else "None")
c6.metric("Target", f"${st.session_state.target:.2f}" if st.session_state.target else "None")

# ============================================================
# CHART
# ============================================================
vol_colors = np.where(
    display_df["close"] >= display_df["open"],
    "rgba(38,166,154,0.55)",
    "rgba(239,83,80,0.55)",
)

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.03, row_heights=[0.78, 0.22],
)

fig.add_trace(
    go.Candlestick(
        x=display_df["label"],
        open=display_df["open"], high=display_df["high"],
        low=display_df["low"], close=display_df["close"],
        name=ticker,
        increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"),
        customdata=np.stack([
            display_df["timestamp"].astype(str).values,
            display_df["open"].values,
            display_df["high"].values,
            display_df["low"].values,
            display_df["close"].values,
        ], axis=-1),
        hovertemplate=(
            "Time=%{customdata[0]}<br>O=%{customdata[1]:.2f}<br>H=%{customdata[2]:.2f}"
            "<br>L=%{customdata[3]:.2f}<br>C=%{customdata[4]:.2f}<extra>%{fullData.name}</extra>"
        ),
    ),
    row=1, col=1,
)

fig.add_trace(
    go.Bar(
        x=display_df["label"], y=display_df["volume"],
        marker_color=vol_colors, name="Volume",
    ),
    row=2, col=1,
)

# EMAs (computed on full revealed history, then filtered to display labels)
if show_ma and len(revealed) >= 2:
    tmp = revealed.copy()
    tmp["fast_ma"] = moving_avg(tmp["close"], int(fast_len), ma_type)
    tmp["slow_ma"] = moving_avg(tmp["close"], int(slow_len), ma_type)
    ema_plot = tmp[tmp["label"].isin(display_df["label"])]
    fig.add_trace(
        go.Scatter(
            x=ema_plot["label"], y=ema_plot["fast_ma"],
            line=dict(color="#3b82f6", width=1.6),
            name=f"{ma_type}{int(fast_len)} ({active_interval})",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ema_plot["label"], y=ema_plot["slow_ma"],
            line=dict(color="#f59e0b", width=1.6),
            name=f"{ma_type}{int(slow_len)} ({active_interval})",
        ),
        row=1, col=1,
    )

# structure / zones on display_df
noise = float((display_df["high"] - display_df["low"]).tail(20).mean() or 0.05)
labeled = _raw_swings(display_df) if (show_struct or show_zones) else []
conf = [s for s in labeled if s["i"] + SWING_K <= len(display_df) - 1]

if show_struct and len(display_df) > SWING_K * 2 + 1:
    bos, choch = detect_bos_choch(display_df, labeled)
    for s in conf:
        x = display_df["label"].iloc[s["i"]]
        if s["label"] in ("HH", "HL"):
            pal, ys = C_HH, (18 if s["type"] == "H" else -18)
        elif s["label"] in ("LH", "LL"):
            pal, ys = C_LH, (18 if s["type"] == "H" else -18)
        else:
            pal, ys = C_H, (18 if s["type"] == "H" else -18)
        badge(fig, x, s["price"], s["label"], pal, yshift=ys)

    for b in bos:
        if 0 <= b["i"] < len(display_df):
            x = display_df["label"].iloc[b["i"]]
            if b["dir"] == "up":
                badge(fig, x, float(display_df["high"].iloc[b["i"]]), "BOS↑", C_BOS_UP, yshift=22, arrow=True)
            else:
                badge(fig, x, float(display_df["low"].iloc[b["i"]]), "BOS↓", C_BOS_DN, yshift=-22, arrow=True)

    for h in choch:
        if 0 <= h["i"] < len(display_df):
            x = display_df["label"].iloc[h["i"]]
            if h["dir"] == "up":
                badge(fig, x, float(display_df["high"].iloc[h["i"]]), "CHoCH↑", C_CH_UP, yshift=22, arrow=True)
            else:
                badge(fig, x, float(display_df["low"].iloc[h["i"]]), "CHoCH↓", C_CH_DN, yshift=-22, arrow=True)

if show_zones and conf:
    zones = build_zones(conf, current_price)
    last_x = display_df["label"].iloc[-1]
    for z in zones:
        role = zone_role(z, current_price, noise)
        if role == "floor":
            col, tag = C_FLOOR, "FLOOR"
        elif role == "ceiling":
            col, tag = C_CEIL, "CEIL"
        elif role == "both":
            col, tag = C_BOTH, "BOTH"
        else:
            col, tag = C_BROKEN, role.replace("_", " ").upper()
        fig.add_hline(
            y=z["mid"], row=1, col=1,
            line=dict(color=col, width=min(1 + z["touches"], 4), dash="dot"),
        )
        fig.add_annotation(
            x=last_x, y=z["mid"], xanchor="left", xref="x", row=1, col=1,
            text=f"<b> {z['mid']:.2f} {tag} {z['low_touches']}L/{z['high_touches']}H</b>",
            showarrow=False,
            font=dict(size=10, color="#ffffff", family="Arial"),
            bgcolor=col, bordercolor="#e5e7eb", borderwidth=1, borderpad=3,
        )

# position lines
if pos != 0:
    if st.session_state.stop_loss is not None:
        fig.add_hline(y=float(st.session_state.stop_loss), row=1, col=1,
                      line=dict(color="#fb923c", width=2, dash="dash"))
    if st.session_state.target is not None:
        fig.add_hline(y=float(st.session_state.target), row=1, col=1,
                      line=dict(color="#4ade80", width=2, dash="dash"))
    if st.session_state.entry_price is not None:
        fig.add_hline(y=float(st.session_state.entry_price), row=1, col=1,
                      line=dict(color="#22d3ee", width=1, dash="dot"))

# persistent drawings
for d in st.session_state.drawings:
    if d["type"] == "hline":
        fig.add_hline(
            y=d["price"], row=1, col=1,
            line=dict(color=d.get("color", "#22d3ee"), width=1.8),
        )
        fig.add_annotation(
            x=display_df["label"].iloc[-1], y=d["price"], row=1, col=1,
            text=f"{d.get('label', '')}", showarrow=False, xanchor="left",
            font=dict(size=10, color=d.get("color", "#22d3ee")),
        )
    elif d["type"] == "band":
        fig.add_hrect(
            y0=d["y0"], y1=d["y1"], row=1, col=1,
            fillcolor=d.get("color", "rgba(34,197,94,0.18)"),
            line_width=0, layer="below",
        )
    elif d["type"] == "trend":
        fig.add_trace(
            go.Scatter(
                x=[d["x0"], d["x1"]], y=[d["y0"], d["y1"]],
                mode="lines",
                line=dict(color=d.get("color", "#f472b6"), width=2),
                name=d.get("label", "Trend"),
                showlegend=False,
            ),
            row=1, col=1,
        )

# camera range
labels = display_df["label"].tolist()
today_mask = display_df["date_only"] == practice_day
today_view = display_df.loc[today_mask]

if follow_replay and len(today_view) >= 1:
    x0 = today_view["label"].iloc[0]
    x1 = today_view["label"].iloc[-1]
elif context_preset == "Today" and len(today_view) >= 1:
    x0 = today_view["label"].iloc[0]
    x1 = today_view["label"].iloc[-1]
else:
    # show whatever display_df currently contains
    x0 = display_df["label"].iloc[0]
    x1 = display_df["label"].iloc[-1]

try:
    i0 = labels.index(x0)
    i1 = labels.index(x1)
except ValueError:
    i0, i1 = 0, max(0, len(labels) - 1)

# right padding so latest candle isn't glued to edge
pad = 3
i1_pad = min(len(labels) - 1, i1 + pad)

fig.update_layout(
    template="plotly_dark",
    height=640,
    dragmode="pan",
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    font=dict(color="#e5e7eb"),
    title=f"{ticker} | {active_interval} native | {current_time_str} | Follow={'ON' if follow_replay else 'OFF'}",
    margin=dict(l=10, r=150, t=40, b=10),
    xaxis_rangeslider_visible=False,
    showlegend=True,
    legend=dict(bgcolor="rgba(0,0,0,0.45)", font=dict(color="#e5e7eb")),
    uirevision=None if follow_replay else "keep-zoom",
    clickmode="event+select",
)

for r in (1, 2):
    fig.update_xaxes(
        type="category",
        range=[i0 - 0.5, i1_pad + 0.5],
        nticks=14,
        rangeslider_visible=False,
        gridcolor="#1f2937",
        linecolor="#4b5563",
        row=r, col=1,
    )
fig.update_yaxes(title_text="Price", gridcolor="#1f2937", linecolor="#4b5563", row=1, col=1)

if len(display_df) and display_df["volume"].max() > 0:
    cap = max(display_df["volume"].quantile(0.95) * 1.15, display_df["volume"].median() * 2)
    fig.update_yaxes(range=[0, cap], title_text="Volume", gridcolor="#1f2937",
                     linecolor="#4b5563", row=2, col=1)
else:
    fig.update_yaxes(title_text="Volume", gridcolor="#1f2937", linecolor="#4b5563", row=2, col=1)

# layout: chart left, controls right
chart_col, ctrl_col = st.columns([6, 1])

with chart_col:
    # Plotly selection for drawing clicks
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key=f"chart_{active_interval}_{practice_day}_{step}_{draw_mode}",
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["drawline", "eraseshape"],
        },
    )

    # process selection → drawing
    try:
        pts = event.selection.points if event and event.selection else []
    except Exception:
        pts = []

    if draw_mode != "None" and pts:
        p = pts[-1]
        # x is category label
        x_val = p.get("x")
        y_val = p.get("y")
        row = nearest_row_from_click(display_df, x_val, y_val)
        if row is not None:
            # For High/Low modes, snap using selected y proximity if possible
            if draw_mode == "Line at High":
                add_hline_drawing(float(row["high"]), label=f"H {float(row['high']):.2f}", color="#ef4444")
                st.session_state.draw_clicks = []
                st.rerun()
            elif draw_mode == "Line at Low":
                add_hline_drawing(float(row["low"]), label=f"L {float(row['low']):.2f}", color="#22c55e")
                st.session_state.draw_clicks = []
                st.rerun()
            elif draw_mode == "Line at Close":
                add_hline_drawing(float(row["close"]), label=f"C {float(row['close']):.2f}", color="#22d3ee")
                st.session_state.draw_clicks = []
                st.rerun()
            elif draw_mode in ("Band", "Trend line"):
                handle_drawing_click(draw_mode, row, str(row["label"]))
                st.rerun()

with ctrl_col:
    st.markdown("#### ⏱️")
    st.caption(f"**{current_time.strftime('%H:%M')}**")
    st.caption(f"{active_interval} bars")

    # advance controls depend on timeframe
    if active_interval == "1m":
        b1, b2, b3 = "+1m", "+5m", "+15m"
        n1, n2, n3 = 1, 5, 15
    elif active_interval == "2m":
        b1, b2, b3 = "+1 bar", "+3 bars", "+8 bars"
        n1, n2, n3 = 1, 3, 8
    elif active_interval == "5m":
        b1, b2, b3 = "+1 bar", "+3 bars", "+6 bars"
        n1, n2, n3 = 1, 3, 6
    elif active_interval == "15m":
        b1, b2, b3 = "+1 bar", "+2 bars", "+4 bars"
        n1, n2, n3 = 1, 2, 4
    elif active_interval == "30m":
        b1, b2, b3 = "+1 bar", "+2 bars", "+4 bars"
        n1, n2, n3 = 1, 2, 4
    elif active_interval == "1h":
        b1, b2, b3 = "+1 bar", "+2 bars", "+4 bars"
        n1, n2, n3 = 1, 2, 4
    else:  # 1d
        b1, b2, b3 = "+1 day", "+3 days", "+5 days"
        n1, n2, n3 = 1, 3, 5

    if st.button(f"▶️ {b1}", use_container_width=True, key="adv1"):
        advance_bars(n1)
        st.rerun()
    if st.button(f"⏩ {b2}", use_container_width=True, key="adv2"):
        advance_bars(n2)
        st.rerun()
    if st.button(f"⏭️ {b3}", use_container_width=True, key="adv3"):
        advance_bars(n3)
        st.rerun()

    st.markdown("---")
    if pos > 0:
        st.success("🟢 LONG")
    elif pos < 0:
        st.error("🔴 SHORT")
    else:
        st.info("FLAT")

    st.caption(f"Context:\n{context_preset}")
    st.caption(f"Follow:\n{'ON' if follow_replay else 'OFF'}")
    st.caption(f"Drawings:\n{len(st.session_state.drawings)}")

# ============================================================
# MANUAL DRAW FALLBACK (if click selection is awkward)
# ============================================================
with st.expander("✏️ Manual draw helpers (fallback)", expanded=False):
    st.caption("If chart click-select is inconvenient, use this.")
    opts = list(display_df["label"].astype(str).values)
    if opts:
        chosen = st.selectbox("Candle", opts[::-1])  # newest first
        row = display_df[display_df["label"].astype(str) == chosen].iloc[0]
        h1, h2, h3, h4, h5 = st.columns(5)
        if h1.button("Add High line"):
            add_hline_drawing(float(row["high"]), label=f"H {float(row['high']):.2f}", color="#ef4444")
            st.rerun()
        if h2.button("Add Close line"):
            add_hline_drawing(float(row["close"]), label=f"C {float(row['close']):.2f}", color="#22d3ee")
            st.rerun()
        if h3.button("Add Low line"):
            add_hline_drawing(float(row["low"]), label=f"L {float(row['low']):.2f}", color="#22c55e")
            st.rerun()
        if h4.button("Use in Band/Trend click queue"):
            handle_drawing_click(draw_mode if draw_mode in ("Band", "Trend line") else "Band", row, str(row["label"]))
            st.rerun()
        if h5.button("Clear click queue"):
            st.session_state.draw_clicks = []
            st.rerun()

# ============================================================
# TRADE PANEL
# ============================================================
if pos == 0:
    st.markdown("### 🎮 Open a Position")
    e1, e2, e3, e4, e5 = st.columns([1.2, 1.2, 1.2, 1, 1])
    with e1:
        bet_size = st.number_input(
            "Trade Amount ($)",
            min_value=10.0,
            max_value=max(10.0, float(st.session_state.balance)),
            value=min(100.0, float(st.session_state.balance)),
            step=10.0,
        )
    with e2:
        sl_input = st.number_input("Stop-Loss ($)", value=round(current_price * 0.99, 2), step=0.01)
    with e3:
        tp_input = st.number_input("Target ($, 0=none)", value=0.0, step=0.01)
    with e4:
        st.write(""); st.write("")
        if st.button("🟢 BUY (Long)", use_container_width=True):
            if sl_input >= current_price:
                st.error("Long stop must be below price")
            elif tp_input != 0 and tp_input <= current_price:
                st.error("Long target must be above price")
            else:
                sh = bet_size / current_price
                st.session_state.balance -= bet_size
                st.session_state.shares = sh
                st.session_state.stop_loss = sl_input
                st.session_state.target = tp_input if tp_input > 0 else None
                st.session_state.entry_price = current_price
                st.session_state.trade_log.append(
                    f"{current_time_str}: BOUGHT {sh:.4f} @ ${current_price:.2f} "
                    f"(SL {sl_input:.2f} / TP {tp_input if tp_input>0 else '—'})"
                )
                st.rerun()
    with e5:
        st.write(""); st.write("")
        if st.button("🔻 SHORT", use_container_width=True):
            if sl_input <= current_price:
                st.error("Short stop must be above price")
            elif tp_input != 0 and tp_input >= current_price:
                st.error("Short target must be below price")
            else:
                sh = bet_size / current_price
                st.session_state.balance += bet_size
                st.session_state.shares = -sh
                st.session_state.stop_loss = sl_input
                st.session_state.target = tp_input if tp_input > 0 else None
                st.session_state.entry_price = current_price
                st.session_state.trade_log.append(
                    f"{current_time_str}: SHORTED {sh:.4f} @ ${current_price:.2f} "
                    f"(SL {sl_input:.2f} / TP {tp_input if tp_input>0 else '—'})"
                )
                st.rerun()
else:
    st.markdown(f"### 🎮 Manage {pos_type}")
    m1, m2, m3, m4, m5 = st.columns([1.2, 1.2, 1.2, 1, 1])
    with m1:
        new_sl = st.number_input(
            "Modify Stop ($)",
            value=float(st.session_state.stop_loss or current_price),
            step=0.01, key="mod_sl",
        )
    with m2:
        new_tp = st.number_input(
            "Modify Target (0=none)",
            value=float(st.session_state.target or 0.0),
            step=0.01, key="mod_tp",
        )
    with m3:
        close_qty = st.number_input(
            "Qty to close",
            min_value=0.0,
            max_value=float(abs(pos)),
            value=float(abs(pos)),
            step=0.0001,
        )
    with m4:
        st.write(""); st.write("")
        if st.button("💾 Update SL/TP", use_container_width=True):
            ok = True
            if pos > 0 and new_sl >= current_price:
                st.error("Long SL must be below price"); ok = False
            if pos < 0 and new_sl <= current_price:
                st.error("Short SL must be above price"); ok = False
            if ok:
                st.session_state.stop_loss = new_sl
                st.session_state.target = new_tp if new_tp > 0 else None
                st.session_state.trade_log.append(
                    f"{current_time_str}: UPDATED SL {new_sl:.2f} / TP {new_tp if new_tp>0 else '—'}"
                )
                st.rerun()
    with m5:
        st.write(""); st.write("")
        lab = "🔴 SELL" if pos > 0 else "🟢 COVER"
        if st.button(f"{lab}", use_container_width=True):
            if close_qty > 0:
                close_position(close_qty, current_price, current_time_str, "(manual)")
                st.rerun()

if st.session_state.trade_log:
    with st.expander("📝 Trade History", expanded=True):
        for line in reversed(st.session_state.trade_log):
            st.text(line)

with st.expander("📘 Session notes"):
    st.markdown(
        f"""
**Active:** `{ticker}` • `{active_interval}` native • practice date `{practice_day}`  
**Revealed bars:** `{len(revealed)}` • **Displayed bars:** `{len(display_df)}`  
**Follow replay:** `{"ON" if follow_replay else "OFF"}`  
**Drawings:** `{len(st.session_state.drawings)}`

### Behavior
- Chart data is **native** for the selected timeframe (not rebuilt from 1m).
- Practice day only reveals bars up to the current replay step.
- Previous days are available for context.
- Initial/follow camera focuses on the practice day so candles stay readable.
- Drawing modes persist across advances.
"""
    )
