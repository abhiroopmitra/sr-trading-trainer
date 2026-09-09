# ============================================================
# HUMAN MARKET REPLAY SIMULATOR
#
# Native Yahoo timeframe data:
#   1m  -> ~7 days
#   2m  -> ~60 days
#   5m  -> ~60 days
#   15m -> ~60 days
#   30m -> ~60 days
#   90m -> ~60 days
#   1h  -> ~2 years
#   1d  -> ~10 years
#
# Important:
# - 5m/15m candles are fetched as NATIVE Yahoo candles.
# - They are NOT resampled from 1m candles.
# - Historical prior days are fully available.
# - The selected replay day is revealed bar-by-bar.
# ============================================================

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Market Replay Simulator")

# ============================================================
# TIMEFRAME CONFIGURATION
# ============================================================
TIMEFRAME_CONFIG = {
    "1m": {
        "yf_interval": "1m",
        "period": "7d",
        "fallback_period": "6d",
        "ema": (9, 21),
        "step_minutes": 1,
        "intraday": True,
    },
    "2m": {
        "yf_interval": "2m",
        "period": "60d",
        "fallback_period": "59d",
        "ema": (9, 21),
        "step_minutes": 2,
        "intraday": True,
    },
    "5m": {
        "yf_interval": "5m",
        "period": "60d",
        "fallback_period": "59d",
        "ema": (20, 50),
        "step_minutes": 5,
        "intraday": True,
    },
    "15m": {
        "yf_interval": "15m",
        "period": "60d",
        "fallback_period": "59d",
        "ema": (50, 200),
        "step_minutes": 15,
        "intraday": True,
    },
    "30m": {
        "yf_interval": "30m",
        "period": "60d",
        "fallback_period": "59d",
        "ema": (50, 200),
        "step_minutes": 30,
        "intraday": True,
    },
    "90m": {
        "yf_interval": "90m",
        "period": "60d",
        "fallback_period": "59d",
        "ema": (50, 100),
        "step_minutes": 90,
        "intraday": True,
    },
    "1h": {
        "yf_interval": "60m",
        "period": "2y",
        "fallback_period": "1y",
        "ema": (50, 200),
        "step_minutes": 60,
        "intraday": True,
    },
    "1d": {
        "yf_interval": "1d",
        "period": "10y",
        "fallback_period": "5y",
        "ema": (50, 200),
        "step_minutes": 24 * 60,
        "intraday": False,
    },
}

TIMEFRAME_OPTIONS = list(TIMEFRAME_CONFIG.keys())

CONTEXT_PRESETS = [
    "Today",
    "1 Previous Day",
    "5 Previous Days",
    "20 Previous Days",
    "All Available History",
]

DRAW_MODES = [
    "None",
    "Line at High",
    "Line at Close",
    "Line at Low",
    "Band",
    "Trend line",
]

# ============================================================
# STRUCTURE SETTINGS
# ============================================================
SWING_K = 5
ZONE_TOL = 0.0012
MIN_DRAW_TOUCHES = 2
MIN_SWING_PCT = 0.0008
POLARITY_EDGE = 2
CHOCH_ENABLE = True

# ============================================================
# BLACK THEME COLORS
# ============================================================
C_HH = dict(fg="#ffffff", bg="#16a34a")
C_LH = dict(fg="#ffffff", bg="#dc2626")
C_H = dict(fg="#ffffff", bg="#475569")

C_BOS_UP = dict(fg="#111111", bg="#fde047")
C_BOS_DN = dict(fg="#ffffff", bg="#dc2626")

C_CH_UP = dict(fg="#111111", bg="#7dd3fc")
C_CH_DN = dict(fg="#111111", bg="#fb923c")

C_FLOOR = "#22c55e"
C_CEIL = "#ef4444"
C_BROKEN = "#a8a29e"
C_BOTH = "#eab308"

C_FAST_EMA = "#3b82f6"
C_SLOW_EMA = "#f59e0b"
C_VOL_MA = "#ff6d00"

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
    "df": pd.DataFrame(),
    "sim_start_idx": 0,
    "active_ticker": None,
    "active_interval": None,
    "active_practice_date": None,
    "trade_log": [],
    "markers": [],
    "drawings": [],
    "draw_clicks": [],
    "last_draw_event": None,
    "last_draw_mode": "None",
    "camera_revision": 0,
    "camera_signature": None,
    "force_camera": True,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# DATA HELPERS
# ============================================================
def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Yahoo/yfinance data to:
    timestamp, open, high, low, close, volume, date_only, label
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]

    timestamp_col = None
    for col in df.columns:
        if col in ("datetime", "date", "index"):
            timestamp_col = col
            break

    if timestamp_col is None:
        timestamp_col = df.columns[0]

    df.rename(columns={timestamp_col: "timestamp"}, inplace=True)

    rename_map = {}
    for col in df.columns:
        col_lower = str(col).lower()

        if col_lower.startswith("open"):
            rename_map[col] = "open"
        elif col_lower.startswith("high"):
            rename_map[col] = "high"
        elif col_lower.startswith("low"):
            rename_map[col] = "low"
        elif col_lower == "close" or col_lower.startswith("close"):
            rename_map[col] = "close"
        elif col_lower.startswith("volume"):
            rename_map[col] = "volume"

    df.rename(columns=rename_map, inplace=True)

    required = ["timestamp", "open", "high", "low", "close"]
    if any(col not in df.columns for col in required):
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Convert Yahoo intraday timestamps to US market time.
    if getattr(df["timestamp"].dt, "tz", None) is not None:
        try:
            df["timestamp"] = (
                df["timestamp"]
                .dt.tz_convert("America/New_York")
                .dt.tz_localize(None)
            )
        except Exception:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["date_only"] = df["timestamp"].dt.date
    df["label"] = df["timestamp"].dt.strftime("%m-%d %H:%M")

    return df.reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_native_history(ticker: str, timeframe: str) -> pd.DataFrame:
    """
    Fetch all currently available native Yahoo history for a timeframe.

    This intentionally uses Yahoo's period-based request, rather than asking
    for a date range that may start too far before Yahoo's intraday cutoff.

    For example:
      5m -> period='60d'
      1m -> period='7d'

    This fixes the issue where a selected practice day 20 days ago caused the
    app to request 60 days BEFORE that date, which Yahoo can reject entirely.
    """
    if timeframe not in TIMEFRAME_CONFIG:
        return pd.DataFrame()

    cfg = TIMEFRAME_CONFIG[timeframe]

    periods_to_try = [
        cfg["period"],
        cfg["fallback_period"],
    ]

    for period in periods_to_try:
        try:
            raw = yf.download(
                ticker,
                period=period,
                interval=cfg["yf_interval"],
                auto_adjust=True,
                progress=False,
                prepost=False,
                threads=False,
            )

            df = normalize_ohlcv(raw)

            if not df.empty:
                return df

        except Exception:
            continue

    return pd.DataFrame()


def moving_avg(series: pd.Series, length: int, kind: str = "EMA") -> pd.Series:
    if kind == "EMA":
        return series.ewm(span=length, adjust=False).mean()

    return series.rolling(length).mean()


def get_day_rows(df: pd.DataFrame, practice_date):
    if df is None or df.empty:
        return pd.DataFrame()

    return df[df["date_only"] == practice_date].copy()


def get_context_start_timestamp(
    revealed_df: pd.DataFrame,
    practice_date,
    preset: str,
):
    """
    Context preset affects camera/viewport only.
    Historical data remains loaded in the chart.
    """
    if revealed_df.empty:
        return None

    unique_dates = sorted(revealed_df["date_only"].unique())

    if practice_date not in unique_dates:
        return revealed_df["timestamp"].iloc[0]

    current_idx = unique_dates.index(practice_date)

    previous_map = {
        "Today": 0,
        "1 Previous Day": 1,
        "5 Previous Days": 5,
        "20 Previous Days": 20,
        "All Available History": 10_000,
    }

    n_previous = previous_map.get(preset, 0)
    start_idx = max(0, current_idx - n_previous)
    start_date = unique_dates[start_idx]

    subset = revealed_df[revealed_df["date_only"] >= start_date]
    if subset.empty:
        return revealed_df["timestamp"].iloc[0]

    return subset["timestamp"].iloc[0]


def right_padding_timestamp(timestamp, timeframe: str):
    cfg = TIMEFRAME_CONFIG[timeframe]

    if cfg["intraday"]:
        return timestamp + pd.Timedelta(minutes=cfg["step_minutes"] * 3)

    return timestamp + pd.Timedelta(days=5)


# ============================================================
# STRUCTURE FUNCTIONS
# ============================================================
def _raw_swings(df: pd.DataFrame):
    if df is None or len(df) < SWING_K * 2 + 1:
        return []

    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    ref_price = float(df["close"].iloc[-1])
    min_size = max(ref_price * MIN_SWING_PCT, 1e-9)

    raw = []

    for i in range(SWING_K, n - SWING_K):
        high_window = highs[i - SWING_K:i + SWING_K + 1]
        low_window = lows[i - SWING_K:i + SWING_K + 1]

        if highs[i] == high_window.max():
            if highs[i] - low_window.min() >= min_size:
                raw.append((i, float(highs[i]), "H"))

        if lows[i] == low_window.min():
            if high_window.max() - lows[i] >= min_size:
                raw.append((i, float(lows[i]), "L"))

    raw.sort(key=lambda x: x[0])

    cleaned = []

    for swing in raw:
        if cleaned and cleaned[-1][2] == swing[2]:
            last = cleaned[-1]

            if swing[2] == "H" and swing[1] >= last[1]:
                cleaned[-1] = swing

            elif swing[2] == "L" and swing[1] <= last[1]:
                cleaned[-1] = swing
        else:
            cleaned.append(swing)

    labeled = []
    last_high = None
    last_low = None

    for idx, price, swing_type in cleaned:
        if swing_type == "H":
            label = "H" if last_high is None else ("HH" if price > last_high else "LH")
            last_high = price
        else:
            label = "L" if last_low is None else ("HL" if price > last_low else "LL")
            last_low = price

        labeled.append({
            "i": idx,
            "price": price,
            "type": swing_type,
            "label": label,
        })

    return labeled


def build_zones(labeled, reference_price):
    zones = []

    for swing in labeled:
        placed = False

        for zone in zones:
            if abs(swing["price"] - zone["mid"]) / max(reference_price, 1e-9) < ZONE_TOL:
                zone["prices"].append(swing["price"])
                zone["mid"] = float(np.mean(zone["prices"]))
                zone["touches"] += 1

                if swing["type"] == "L":
                    zone["low_touches"] += 1
                else:
                    zone["high_touches"] += 1

                placed = True
                break

        if not placed:
            zones.append({
                "mid": swing["price"],
                "prices": [swing["price"]],
                "touches": 1,
                "low_touches": 1 if swing["type"] == "L" else 0,
                "high_touches": 1 if swing["type"] == "H" else 0,
            })

    output = [zone for zone in zones if zone["touches"] >= MIN_DRAW_TOUCHES]

    for zone in output:
        zone["min_px"] = min(zone["prices"])
        zone["max_px"] = max(zone["prices"])

    return output


def detect_bos_choch(df: pd.DataFrame, labeled):
    """
    BOS = continuation in the existing direction.
    CHoCH = break against existing direction / change warning.
    """
    n = len(df)

    bos = []
    choch = []

    last_hh = None
    last_hl = None
    last_lh = None
    last_ll = None

    bias = "NEUTRAL"
    pointer = 0

    for i in range(n):
        while pointer < len(labeled) and labeled[pointer]["i"] + SWING_K <= i:
            swing = labeled[pointer]

            if swing["label"] == "HH":
                last_hh = swing["price"]
            elif swing["label"] == "HL":
                last_hl = swing["price"]
            elif swing["label"] == "LH":
                last_lh = swing["price"]
            elif swing["label"] == "LL":
                last_ll = swing["price"]

            pointer += 1

        close = float(df["close"].iloc[i])

        # Bullish BOS: break above confirmed HH.
        if last_hh is not None and close > last_hh:
            if bias != "BULL":
                bos.append({
                    "i": i,
                    "price": last_hh,
                    "dir": "up",
                })
                bias = "BULL"

            last_hh = None
            continue

        # Bearish BOS: break below confirmed LL.
        if last_ll is not None and close < last_ll:
            if bias != "BEAR":
                bos.append({
                    "i": i,
                    "price": last_ll,
                    "dir": "down",
                })
                bias = "BEAR"

            last_ll = None
            continue

        if CHOCH_ENABLE:
            # Bull trend loses HL -> CHoCH down.
            if bias == "BULL" and last_hl is not None and close < last_hl:
                choch.append({
                    "i": i,
                    "price": last_hl,
                    "dir": "down",
                })
                last_hl = None
                bias = "NEUTRAL"

            # Bear trend breaks LH -> CHoCH up.
            elif bias == "BEAR" and last_lh is not None and close > last_lh:
                choch.append({
                    "i": i,
                    "price": last_lh,
                    "dir": "up",
                })
                last_lh = None
                bias = "NEUTRAL"

    return bos, choch


def zone_role(zone, close, noise):
    if zone["low_touches"] >= zone["high_touches"] + POLARITY_EDGE:
        role = "floor"

    elif zone["high_touches"] >= zone["low_touches"] + POLARITY_EDGE:
        role = "ceiling"

    else:
        role = "both"

    if role == "floor" and close < zone["min_px"] - noise:
        return "broken_floor"

    if role == "ceiling" and close > zone["max_px"] + noise:
        return "broken_ceiling"

    return role


def badge(fig, x, y, text, palette, yshift=0, arrow=False):
    fig.add_annotation(
        x=x,
        y=y,
        row=1,
        col=1,
        text=f"<b>{text}</b>",
        showarrow=arrow,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.2,
        arrowcolor="#e5e7eb",
        yshift=yshift,
        font=dict(
            size=10,
            color=palette["fg"],
            family="Arial",
        ),
        bgcolor=palette["bg"],
        bordercolor="#e5e7eb",
        borderwidth=1,
        borderpad=3,
        opacity=1,
        align="center",
    )


# ============================================================
# PAPER TRADING FUNCTIONS
# ============================================================
def add_marker(timestamp, price, kind):
    st.session_state.markers.append({
        "timestamp": pd.Timestamp(timestamp),
        "price": float(price),
        "kind": kind,
    })


def close_position(qty, price, timestamp, reason=""):
    pos = st.session_state.shares
    timestamp = pd.Timestamp(timestamp)
    time_text = timestamp.strftime("%Y-%m-%d %H:%M")

    if pos > 0:
        qty = min(float(qty), float(pos))

        st.session_state.balance += qty * price
        st.session_state.shares -= qty

        st.session_state.trade_log.append(
            f"{time_text}: SOLD {qty:.4f} sh at ${price:.2f} {reason}"
        )

        add_marker(timestamp, price, "SELL")

    elif pos < 0:
        qty = min(float(qty), float(abs(pos)))

        st.session_state.balance -= qty * price
        st.session_state.shares += qty

        st.session_state.trade_log.append(
            f"{time_text}: COVERED {qty:.4f} sh at ${price:.2f} {reason}"
        )

        add_marker(timestamp, price, "COVER")

    if abs(st.session_state.shares) < 1e-10:
        st.session_state.shares = 0.0
        st.session_state.stop_loss = None
        st.session_state.target = None
        st.session_state.entry_price = None


def advance_bars(number_of_bars: int):
    """
    Advance by native timeframe bars.

    Note:
    For native 5m/15m/etc bars, if both SL and TP occur inside the same
    OHLC candle, this simulator assumes stop-loss happens first.
    That is conservative.
    """
    df = st.session_state.df
    max_step = len(df) - 1

    for _ in range(number_of_bars):
        if st.session_state.step >= max_step:
            st.toast("Replay session complete.", icon="🔔")
            break

        st.session_state.step += 1

        row = df.iloc[st.session_state.step]
        timestamp = row["timestamp"]

        pos = st.session_state.shares
        sl = st.session_state.stop_loss
        tp = st.session_state.target

        if pos > 0:
            if sl is not None and float(row["low"]) <= float(sl):
                execution_price = min(float(sl), float(row["open"]))

                close_position(
                    pos,
                    execution_price,
                    timestamp,
                    "(🛑 STOP-LOSS)",
                )

                st.toast(
                    f"🛑 Long stopped at ${execution_price:.2f}",
                    icon="💥",
                )
                break

            if tp is not None and float(row["high"]) >= float(tp):
                execution_price = max(float(tp), float(row["open"]))

                close_position(
                    pos,
                    execution_price,
                    timestamp,
                    "(🎯 TARGET HIT)",
                )

                st.toast(
                    f"🎯 Long target at ${execution_price:.2f}",
                    icon="🎉",
                )
                break

        elif pos < 0:
            if sl is not None and float(row["high"]) >= float(sl):
                execution_price = max(float(sl), float(row["open"]))

                close_position(
                    abs(pos),
                    execution_price,
                    timestamp,
                    "(🛑 STOP-LOSS)",
                )

                st.toast(
                    f"🛑 Short stopped at ${execution_price:.2f}",
                    icon="💥",
                )
                break

            if tp is not None and float(row["low"]) <= float(tp):
                execution_price = min(float(tp), float(row["open"]))

                close_position(
                    abs(pos),
                    execution_price,
                    timestamp,
                    "(🎯 TARGET HIT)",
                )

                st.toast(
                    f"🎯 Short target at ${execution_price:.2f}",
                    icon="🎉",
                )
                break


# ============================================================
# DRAWING FUNCTIONS
# ============================================================
def add_horizontal_line(price, label="", color="#22d3ee"):
    st.session_state.drawings.append({
        "type": "hline",
        "price": float(price),
        "label": label,
        "color": color,
    })


def add_band(price_1, price_2, color="rgba(34,197,94,0.18)"):
    y0 = min(float(price_1), float(price_2))
    y1 = max(float(price_1), float(price_2))

    st.session_state.drawings.append({
        "type": "band",
        "y0": y0,
        "y1": y1,
        "color": color,
        "label": f"Band {y0:.2f}-{y1:.2f}",
    })


def add_trend_line(x0, y0, x1, y1, color="#f472b6"):
    st.session_state.drawings.append({
        "type": "trend",
        "x0": pd.Timestamp(x0),
        "y0": float(y0),
        "x1": pd.Timestamp(x1),
        "y1": float(y1),
        "color": color,
        "label": "Trend line",
    })


def extract_selected_points(chart_event):
    """
    Supports current Streamlit Plotly selection object and dict fallback.
    """
    if chart_event is None:
        return []

    try:
        selection = chart_event.selection
        if selection is not None:
            return list(selection.points)
    except Exception:
        pass

    try:
        selection = chart_event.get("selection", {})
        return selection.get("points", [])
    except Exception:
        return []


def nearest_row_from_event(df: pd.DataFrame, point: dict):
    """
    Resolve a Plotly selected point to a candle row.
    """
    if df.empty:
        return None

    point_index = point.get("point_index", point.get("pointIndex"))

    if point_index is not None:
        try:
            point_index = int(point_index)

            if 0 <= point_index < len(df):
                return df.iloc[point_index]
        except Exception:
            pass

    x_value = point.get("x")

    if x_value is not None:
        try:
            clicked_time = pd.to_datetime(x_value)
            index = (df["timestamp"] - clicked_time).abs().idxmin()
            return df.loc[index]
        except Exception:
            pass

    return df.iloc[-1]


def process_drawing_click(mode: str, row: pd.Series, clicked_y=None):
    if row is None or mode == "None":
        return

    timestamp = row["timestamp"]

    # For Band / Trend line, prefer click y-value if available.
    if clicked_y is None:
        anchor_price = float(row["close"])
    else:
        try:
            anchor_price = float(clicked_y)
        except Exception:
            anchor_price = float(row["close"])

    if mode == "Line at High":
        price = float(row["high"])
        add_horizontal_line(
            price,
            label=f"H {price:.2f}",
            color="#ef4444",
        )
        st.toast(f"Resistance line added: {price:.2f}", icon="📌")

    elif mode == "Line at Close":
        price = float(row["close"])
        add_horizontal_line(
            price,
            label=f"C {price:.2f}",
            color="#22d3ee",
        )
        st.toast(f"Close line added: {price:.2f}", icon="📌")

    elif mode == "Line at Low":
        price = float(row["low"])
        add_horizontal_line(
            price,
            label=f"L {price:.2f}",
            color="#22c55e",
        )
        st.toast(f"Support line added: {price:.2f}", icon="📌")

    elif mode == "Band":
        st.session_state.draw_clicks.append({
            "timestamp": timestamp,
            "price": anchor_price,
        })

        if len(st.session_state.draw_clicks) == 1:
            st.toast("Band: first point saved. Click a second point.", icon="🖱️")

        elif len(st.session_state.draw_clicks) >= 2:
            first = st.session_state.draw_clicks[0]
            second = st.session_state.draw_clicks[1]

            add_band(first["price"], second["price"])
            st.session_state.draw_clicks = []

            st.toast("Price band added.", icon="🟩")

    elif mode == "Trend line":
        st.session_state.draw_clicks.append({
            "timestamp": timestamp,
            "price": anchor_price,
        })

        if len(st.session_state.draw_clicks) == 1:
            st.toast("Trend line: first point saved. Click second point.", icon="🖱️")

        elif len(st.session_state.draw_clicks) >= 2:
            first = st.session_state.draw_clicks[0]
            second = st.session_state.draw_clicks[1]

            add_trend_line(
                first["timestamp"],
                first["price"],
                second["timestamp"],
                second["price"],
            )

            st.session_state.draw_clicks = []

            st.toast("Trend line added.", icon="📈")


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Setup")

ticker = st.sidebar.text_input("Ticker", value="TQQQ").upper().strip()

timeframe = st.sidebar.selectbox(
    "Chart Timeframe (native)",
    TIMEFRAME_OPTIONS,
    index=TIMEFRAME_OPTIONS.index("5m"),
)

timeframe_cfg = TIMEFRAME_CONFIG[timeframe]

st.sidebar.caption(
    f"Yahoo native `{timeframe}` data. "
    f"Fetching rolling period: `{timeframe_cfg['period']}`."
)

# ------------------------------------------------------------
# Fetch preview data FIRST.
#
# This is the key fix:
# The date picker is driven by the actual dates Yahoo returns.
# ------------------------------------------------------------
source_df = pd.DataFrame()
practice_date = None
selected_start_timestamp = None
can_start = False

if ticker:
    with st.sidebar:
        with st.spinner(f"Loading native {timeframe} data for {ticker}..."):
            source_df = fetch_native_history(ticker, timeframe)

if source_df.empty:
    st.sidebar.error(
        f"No native {timeframe} data returned for {ticker}.\n\n"
        "Possible causes:\n"
        "- Yahoo temporary rate limit\n"
        "- invalid ticker\n"
        "- market/provider outage\n"
        "- timeframe history unavailable"
    )
else:
    available_dates = sorted(source_df["date_only"].unique())

    first_available = available_dates[0]
    last_available = available_dates[-1]

    st.sidebar.success(
        f"{len(source_df):,} native bars\n"
        f"{first_available} → {last_available}"
    )

    st.sidebar.caption(
        "The practice-date picker below uses dates actually returned "
        "by Yahoo for this timeframe."
    )

    # Default to third-most recent trading day where possible.
    if len(available_dates) >= 3:
        default_practice_date = available_dates[-3]
    else:
        default_practice_date = available_dates[-1]

    practice_date_key = f"practice_date_{ticker}_{timeframe}"

    if (
        practice_date_key not in st.session_state
        or st.session_state[practice_date_key] not in available_dates
    ):
        st.session_state[practice_date_key] = default_practice_date

    practice_date = st.sidebar.date_input(
        "Practice Date",
        min_value=first_available,
        max_value=last_available,
        key=practice_date_key,
    )

    practice_day_rows = get_day_rows(source_df, practice_date)

    if practice_day_rows.empty:
        st.sidebar.warning(
            "No market bars for this selected date. "
            "It may be a weekend, holiday, or unavailable session."
        )
    else:
        if timeframe_cfg["intraday"]:
            start_labels = practice_day_rows["timestamp"].dt.strftime("%H:%M").tolist()

            default_start_idx = min(10, len(start_labels) - 1)

            start_time_key = f"start_time_{ticker}_{timeframe}_{practice_date}"

            selected_start_label = st.sidebar.selectbox(
                "Replay Start Time",
                start_labels,
                index=default_start_idx,
                key=start_time_key,
            )

            selected_start_timestamp = practice_day_rows[
                practice_day_rows["timestamp"].dt.strftime("%H:%M") == selected_start_label
            ]["timestamp"].iloc[0]

        else:
            selected_start_timestamp = practice_day_rows["timestamp"].iloc[0]
            st.sidebar.caption(
                f"Daily replay begins at: {selected_start_timestamp.strftime('%Y-%m-%d')}"
            )

        can_start = True

# ------------------------------------------------------------
# EMA SETTINGS
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Moving Averages")

show_ma = st.sidebar.checkbox("Show Moving Averages", value=True)

ma_type = st.sidebar.radio(
    "MA Type",
    ["EMA", "SMA"],
    horizontal=True,
)

default_fast, default_slow = timeframe_cfg["ema"]

fast_len = st.sidebar.number_input(
    f"{timeframe} Fast MA",
    min_value=2,
    max_value=400,
    value=int(default_fast),
    step=1,
    key=f"fast_ma_{timeframe}",
)

slow_len = st.sidebar.number_input(
    f"{timeframe} Slow MA",
    min_value=2,
    max_value=500,
    value=int(default_slow),
    step=1,
    key=f"slow_ma_{timeframe}",
)

show_vol_ma = st.sidebar.checkbox("Show Volume MA", value=True)

vol_ma_len = st.sidebar.number_input(
    "Volume MA Length",
    min_value=2,
    max_value=200,
    value=20,
    step=1,
)

st.sidebar.caption(
    f"Active: {ma_type}{int(fast_len)} and {ma_type}{int(slow_len)} "
    f"on native {timeframe} candles."
)

# ------------------------------------------------------------
# STRUCTURE SETTINGS
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🏗️ Structure")

show_struct = st.sidebar.checkbox(
    "Show structure labels",
    value=True,
)

show_zones = st.sidebar.checkbox(
    "Show S/R zones",
    value=True,
)

# ------------------------------------------------------------
# CONTEXT / CAMERA SETTINGS
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📷 Chart Context")

context_preset = st.sidebar.selectbox(
    "Context preset",
    CONTEXT_PRESETS,
    index=0,
)

follow_replay = st.sidebar.checkbox(
    "Follow replay candle",
    value=True,
)

st.sidebar.caption(
    "Follow ON: every advance returns the camera to the replay day.\n\n"
    "Follow OFF: you can remain zoomed/panned on old levels."
)

# ------------------------------------------------------------
# DRAWING SETTINGS
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("✏️ Drawing Mode")

draw_mode = st.sidebar.selectbox(
    "Mode",
    DRAW_MODES,
    index=0,
)

if st.session_state.last_draw_mode != draw_mode:
    st.session_state.last_draw_mode = draw_mode
    st.session_state.last_draw_event = None
    st.session_state.draw_clicks = []

if st.session_state.draw_clicks:
    st.sidebar.info(
        f"Pending drawing clicks: {len(st.session_state.draw_clicks)} / 2"
    )

if st.sidebar.button("Undo last drawing"):
    if st.session_state.drawings:
        st.session_state.drawings.pop()
        st.toast("Removed last drawing.", icon="↩️")

if st.sidebar.button("Clear all drawings"):
    st.session_state.drawings = []
    st.session_state.draw_clicks = []
    st.session_state.last_draw_event = None
    st.toast("All drawings cleared.", icon="🧹")

# ------------------------------------------------------------
# START / RESET
# ------------------------------------------------------------
st.sidebar.markdown("---")

if st.sidebar.button(
    "🚀 Start / Reset Simulation",
    use_container_width=True,
    disabled=not can_start,
):
    active_df = source_df[
        source_df["date_only"] <= practice_date
    ].copy().reset_index(drop=True)

    selected_indices = active_df.index[
        active_df["timestamp"] == selected_start_timestamp
    ].tolist()

    day_indices = active_df.index[
        active_df["date_only"] == practice_date
    ].tolist()

    if not day_indices or not selected_indices:
        st.sidebar.error(
            "Could not initialize replay for this date/time."
        )
    else:
        st.session_state.df = active_df
        st.session_state.sim_start_idx = day_indices[0]
        st.session_state.step = selected_indices[0]

        st.session_state.balance = 1000.0
        st.session_state.shares = 0.0
        st.session_state.stop_loss = None
        st.session_state.target = None
        st.session_state.entry_price = None

        st.session_state.trade_log = []
        st.session_state.markers = []

        st.session_state.drawings = []
        st.session_state.draw_clicks = []
        st.session_state.last_draw_event = None

        st.session_state.active_ticker = ticker
        st.session_state.active_interval = timeframe
        st.session_state.active_practice_date = practice_date

        st.session_state.camera_revision += 1
        st.session_state.camera_signature = None
        st.session_state.force_camera = True

        st.session_state.sim_active = True

        st.rerun()


# ============================================================
# MAIN APP
# ============================================================
st.title("💹 Market Replay Simulator (Human)")
st.caption(
    "Native timeframe replay • historical context • EMAs • volume • structure • paper trading"
)

if not st.session_state.sim_active or st.session_state.df.empty:
    st.info(
        "Pick a ticker, native timeframe, valid practice date, and replay "
        "start time. Then click **Start / Reset Simulation**."
    )
    st.stop()

# Warn if user changed sidebar values while replay is active.
if (
    st.session_state.active_ticker != ticker
    or st.session_state.active_interval != timeframe
    or st.session_state.active_practice_date != practice_date
):
    st.warning(
        "Sidebar ticker/timeframe/date differs from the active replay. "
        "Click **Start / Reset Simulation** to apply the new selection."
    )

df_all = st.session_state.df
step = st.session_state.step
practice_day = st.session_state.active_practice_date
active_interval = st.session_state.active_interval
active_ticker = st.session_state.active_ticker

# Historical days are fully available.
# Current replay day is visible only up to current step.
revealed_df = df_all.iloc[:step + 1].copy()

if revealed_df.empty:
    st.error("No revealed data available.")
    st.stop()

current_row = df_all.iloc[step]
current_price = float(current_row["close"])
current_timestamp = pd.Timestamp(current_row["timestamp"])
current_time_text = current_timestamp.strftime("%Y-%m-%d %H:%M")

pos = float(st.session_state.shares)
position_value = pos * current_price
equity = st.session_state.balance + position_value
pnl = equity - 1000.0

pnl_color = "normal" if pnl == 0 else ("inverse" if pnl > 0 else "off")

if pos > 0:
    pos_text = f"{pos:.4f} sh"
    pos_type = "🟢 LONG"
elif pos < 0:
    pos_text = f"{abs(pos):.4f} sh"
    pos_type = "🔴 SHORT"
else:
    pos_text = "—"
    pos_type = "FLAT"

# ============================================================
# CAMERA STATE
# ============================================================
camera_signature = (
    context_preset,
    follow_replay,
    active_interval,
    practice_day,
)

if st.session_state.camera_signature != camera_signature:
    st.session_state.camera_signature = camera_signature
    st.session_state.camera_revision += 1
    st.session_state.force_camera = True

# Follow ON always resets camera toward current replay day.
apply_camera = follow_replay or st.session_state.force_camera

# ============================================================
# METRICS
# ============================================================
m1, m2, m3, m4, m5, m6 = st.columns(6)

m1.metric("Price", f"${current_price:.2f}")
m2.metric("Equity", f"${equity:.2f}", f"${pnl:.2f}", delta_color=pnl_color)
m3.metric("Cash", f"${st.session_state.balance:.2f}")

m4.metric(
    f"Position ({pos_type})",
    pos_text,
    f"Entry ${st.session_state.entry_price:.2f}"
    if st.session_state.entry_price is not None
    else "",
)

m5.metric(
    "Stop-Loss",
    f"${st.session_state.stop_loss:.2f}"
    if st.session_state.stop_loss is not None
    else "None",
)

m6.metric(
    "Target",
    f"${st.session_state.target:.2f}"
    if st.session_state.target is not None
    else "None",
)

# ============================================================
# BUILD CHART
# ============================================================
chart_df = revealed_df.copy()

vol_colors = np.where(
    chart_df["close"] >= chart_df["open"],
    "rgba(38,166,154,0.58)",
    "rgba(239,83,80,0.58)",
)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.78, 0.22],
)

# Candles
fig.add_trace(
    go.Candlestick(
        x=chart_df["timestamp"],
        open=chart_df["open"],
        high=chart_df["high"],
        low=chart_df["low"],
        close=chart_df["close"],
        name=active_ticker,
        increasing=dict(
            line=dict(color="#26a69a"),
            fillcolor="#26a69a",
        ),
        decreasing=dict(
            line=dict(color="#ef5350"),
            fillcolor="#ef5350",
        ),
    ),
    row=1,
    col=1,
)

# Volume
fig.add_trace(
    go.Bar(
        x=chart_df["timestamp"],
        y=chart_df["volume"],
        marker_color=vol_colors,
        name="Volume",
    ),
    row=2,
    col=1,
)

# ------------------------------------------------------------
# EMA / SMA
#
# Important:
# Uses ALL revealed history:
# previous available days + current day only up to replay bar.
# ------------------------------------------------------------
if show_ma and len(chart_df) >= 2:
    chart_df["fast_ma"] = moving_avg(
        chart_df["close"],
        int(fast_len),
        ma_type,
    )

    chart_df["slow_ma"] = moving_avg(
        chart_df["close"],
        int(slow_len),
        ma_type,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df["timestamp"],
            y=chart_df["fast_ma"],
            name=f"{ma_type}{int(fast_len)} ({active_interval})",
            line=dict(color=C_FAST_EMA, width=1.7),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df["timestamp"],
            y=chart_df["slow_ma"],
            name=f"{ma_type}{int(slow_len)} ({active_interval})",
            line=dict(color=C_SLOW_EMA, width=1.7),
        ),
        row=1,
        col=1,
    )

# Volume MA
if show_vol_ma and len(chart_df) >= 2:
    chart_df["vol_ma"] = chart_df["volume"].rolling(
        int(vol_ma_len)
    ).mean()

    fig.add_trace(
        go.Scatter(
            x=chart_df["timestamp"],
            y=chart_df["vol_ma"],
            name=f"VolMA({int(vol_ma_len)})",
            line=dict(color=C_VOL_MA, width=1.7),
        ),
        row=2,
        col=1,
    )

# ============================================================
# STRUCTURE AND ZONES
# ============================================================
noise = float(
    (chart_df["high"] - chart_df["low"]).tail(20).mean() or 0.01
)

labeled = []
confirmed_swings = []

if show_struct or show_zones:
    labeled = _raw_swings(chart_df)

    confirmed_swings = [
        swing
        for swing in labeled
        if swing["i"] + SWING_K <= len(chart_df) - 1
    ]

if show_struct and len(chart_df) > SWING_K * 2 + 1:
    bos, choch = detect_bos_choch(chart_df, labeled)

    for swing in confirmed_swings:
        x = chart_df["timestamp"].iloc[swing["i"]]

        if swing["label"] in ("HH", "HL"):
            palette = C_HH
            yshift = 18 if swing["type"] == "H" else -18

        elif swing["label"] in ("LH", "LL"):
            palette = C_LH
            yshift = 18 if swing["type"] == "H" else -18

        else:
            palette = C_H
            yshift = 18 if swing["type"] == "H" else -18

        badge(
            fig,
            x,
            swing["price"],
            swing["label"],
            palette,
            yshift=yshift,
        )

    for event in bos:
        i = event["i"]

        if 0 <= i < len(chart_df):
            x = chart_df["timestamp"].iloc[i]

            if event["dir"] == "up":
                badge(
                    fig,
                    x,
                    float(chart_df["high"].iloc[i]),
                    "BOS↑",
                    C_BOS_UP,
                    yshift=22,
                    arrow=True,
                )
            else:
                badge(
                    fig,
                    x,
                    float(chart_df["low"].iloc[i]),
                    "BOS↓",
                    C_BOS_DN,
                    yshift=-22,
                    arrow=True,
                )

    for event in choch:
        i = event["i"]

        if 0 <= i < len(chart_df):
            x = chart_df["timestamp"].iloc[i]

            if event["dir"] == "up":
                badge(
                    fig,
                    x,
                    float(chart_df["high"].iloc[i]),
                    "CHoCH↑",
                    C_CH_UP,
                    yshift=22,
                    arrow=True,
                )
            else:
                badge(
                    fig,
                    x,
                    float(chart_df["low"].iloc[i]),
                    "CHoCH↓",
                    C_CH_DN,
                    yshift=-22,
                    arrow=True,
                )

if show_zones and confirmed_swings:
    zones = build_zones(confirmed_swings, current_price)
    last_time = chart_df["timestamp"].iloc[-1]

    for zone in zones:
        role = zone_role(zone, current_price, noise)

        if role == "floor":
            color = C_FLOOR
            tag = "FLOOR"

        elif role == "ceiling":
            color = C_CEIL
            tag = "CEIL"

        elif role == "both":
            color = C_BOTH
            tag = "BOTH"

        else:
            color = C_BROKEN
            tag = role.replace("_", " ").upper()

        fig.add_hline(
            y=zone["mid"],
            row=1,
            col=1,
            line=dict(
                color=color,
                width=min(1 + zone["touches"], 4),
                dash="dot",
            ),
        )

        fig.add_annotation(
            x=last_time,
            y=zone["mid"],
            row=1,
            col=1,
            xanchor="left",
            text=(
                f"<b> {zone['mid']:.2f} {tag} "
                f"{zone['low_touches']}L/{zone['high_touches']}H</b>"
            ),
            showarrow=False,
            font=dict(
                size=10,
                color="#ffffff",
                family="Arial",
            ),
            bgcolor=color,
            bordercolor="#e5e7eb",
            borderwidth=1,
            borderpad=3,
        )

# ============================================================
# POSITION LINES
# ============================================================
if pos != 0:
    if st.session_state.stop_loss is not None:
        fig.add_hline(
            y=float(st.session_state.stop_loss),
            row=1,
            col=1,
            line=dict(
                color="#fb923c",
                width=2,
                dash="dash",
            ),
        )

    if st.session_state.target is not None:
        fig.add_hline(
            y=float(st.session_state.target),
            row=1,
            col=1,
            line=dict(
                color="#4ade80",
                width=2,
                dash="dash",
            ),
        )

    if st.session_state.entry_price is not None:
        fig.add_hline(
            y=float(st.session_state.entry_price),
            row=1,
            col=1,
            line=dict(
                color="#22d3ee",
                width=1.4,
                dash="dot",
            ),
        )

# ============================================================
# TRADE MARKERS
# ============================================================
marker_map = {
    "LONG": {"symbol": "triangle-up", "color": "#22c55e"},
    "SHORT": {"symbol": "triangle-down", "color": "#ef4444"},
    "SELL": {"symbol": "circle", "color": "#f97316"},
    "COVER": {"symbol": "circle", "color": "#38bdf8"},
}

for marker in st.session_state.markers:
    if marker["timestamp"] > current_timestamp:
        continue

    style = marker_map.get(
        marker["kind"],
        {"symbol": "circle", "color": "#ffffff"},
    )

    fig.add_trace(
        go.Scatter(
            x=[marker["timestamp"]],
            y=[marker["price"]],
            mode="markers",
            showlegend=False,
            marker=dict(
                symbol=style["symbol"],
                size=12,
                color=style["color"],
                line=dict(color="#ffffff", width=1),
            ),
        ),
        row=1,
        col=1,
    )

# ============================================================
# PERSISTENT DRAWINGS
# ============================================================
for drawing in st.session_state.drawings:
    if drawing["type"] == "hline":
        fig.add_hline(
            y=drawing["price"],
            row=1,
            col=1,
            line=dict(
                color=drawing.get("color", "#22d3ee"),
                width=1.8,
            ),
        )

        fig.add_annotation(
            x=chart_df["timestamp"].iloc[-1],
            y=drawing["price"],
            row=1,
            col=1,
            xanchor="left",
            text=drawing.get("label", ""),
            showarrow=False,
            font=dict(
                size=10,
                color=drawing.get("color", "#22d3ee"),
            ),
        )

    elif drawing["type"] == "band":
        fig.add_hrect(
            y0=drawing["y0"],
            y1=drawing["y1"],
            row=1,
            col=1,
            fillcolor=drawing.get(
                "color",
                "rgba(34,197,94,0.18)",
            ),
            line_width=1,
            line_color="#22c55e",
            layer="below",
        )

    elif drawing["type"] == "trend":
        fig.add_trace(
            go.Scatter(
                x=[drawing["x0"], drawing["x1"]],
                y=[drawing["y0"], drawing["y1"]],
                mode="lines",
                name=drawing.get("label", "Trend line"),
                showlegend=False,
                line=dict(
                    color=drawing.get("color", "#f472b6"),
                    width=2,
                ),
            ),
            row=1,
            col=1,
        )

# ============================================================
# CAMERA RANGE
# ============================================================
today_rows = chart_df[
    chart_df["date_only"] == practice_day
]

if today_rows.empty:
    today_start = chart_df["timestamp"].iloc[0]
else:
    today_start = today_rows["timestamp"].iloc[0]

if follow_replay:
    # Follow ON: always go back to current replay day.
    camera_start = today_start
    camera_end = right_padding_timestamp(
        current_timestamp,
        active_interval,
    )

else:
    # Follow OFF: context preset determines initial range,
    # then Plotly uirevision preserves manual zoom/pan.
    camera_start = get_context_start_timestamp(
        chart_df,
        practice_day,
        context_preset,
    )

    camera_end = right_padding_timestamp(
        current_timestamp,
        active_interval,
    )

# ============================================================
# LAYOUT
# ============================================================
if follow_replay:
    uirevision_value = (
        f"follow-{st.session_state.camera_revision}-{step}"
    )
else:
    uirevision_value = (
        f"manual-{st.session_state.camera_revision}"
    )

fig.update_layout(
    template="plotly_dark",
    height=640,
    dragmode="pan",
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    font=dict(color="#e5e7eb"),
    title=(
        f"{active_ticker} | Native {active_interval} | "
        f"Replay: {current_time_text} | "
        f"Follow: {'ON' if follow_replay else 'OFF'}"
    ),
    margin=dict(l=10, r=190, t=42, b=10),
    xaxis_rangeslider_visible=False,
    showlegend=True,
    legend=dict(
        bgcolor="rgba(0,0,0,0.45)",
        font=dict(color="#e5e7eb"),
    ),
    hovermode="x",
    uirevision=uirevision_value,
)

# Apply chart range only when:
# - Follow ON
# - First load/reset
# - Context setting changes
if apply_camera:
    fig.update_xaxes(
        range=[camera_start, camera_end],
        row=1,
        col=1,
    )

    fig.update_xaxes(
        range=[camera_start, camera_end],
        row=2,
        col=1,
    )

    # After initial camera placement, allow manual pan/zoom if Follow OFF.
    if not follow_replay:
        st.session_state.force_camera = False

time_tick_format = "%m-%d\n%H:%M"
if not TIMEFRAME_CONFIG[active_interval]["intraday"]:
    time_tick_format = "%Y-%m-%d"

for row in (1, 2):
    fig.update_xaxes(
        type="date",
        tickformat=time_tick_format,
        nticks=12,
        gridcolor="#1f2937",
        linecolor="#4b5563",
        zerolinecolor="#374151",
        showspikes=True,
        spikemode="across",
        spikecolor="#94a3b8",
        spikethickness=1,
        row=row,
        col=1,
    )

fig.update_yaxes(
    title_text="Price",
    gridcolor="#1f2937",
    linecolor="#4b5563",
    row=1,
    col=1,
)

# Volume axis cap to prevent one massive opening volume bar
# from making all other bars invisible.
if len(chart_df) and chart_df["volume"].max() > 0:
    volume_cap = max(
        chart_df["volume"].quantile(0.95) * 1.15,
        chart_df["volume"].median() * 2,
    )

    fig.update_yaxes(
        range=[0, volume_cap],
        title_text="Volume",
        gridcolor="#1f2937",
        linecolor="#4b5563",
        row=2,
        col=1,
    )
else:
    fig.update_yaxes(
        title_text="Volume",
        gridcolor="#1f2937",
        linecolor="#4b5563",
        row=2,
        col=1,
    )

# ============================================================
# CHART + TIME CONTROLS
# ============================================================
chart_col, control_col = st.columns([6, 1])

with chart_col:
    chart_event = None

    # Newer Streamlit versions support on_select.
    # Manual fallback drawing controls appear below if unavailable.
    try:
        chart_event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="human_replay_chart",
            config={
                "scrollZoom": True,
                "displaylogo": False,
            },
        )
    except TypeError:
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="human_replay_chart_fallback",
            config={
                "scrollZoom": True,
                "displaylogo": False,
            },
        )

    # --------------------------------------------------------
    # Process click-based drawing selection
    # --------------------------------------------------------
    selected_points = extract_selected_points(chart_event)

    if draw_mode != "None" and selected_points:
        point = selected_points[-1]
        row = nearest_row_from_event(chart_df, point)

        clicked_y = point.get("y")
        point_index = point.get(
            "point_index",
            point.get("pointIndex", -1),
        )

        event_signature = (
            draw_mode,
            str(row["timestamp"]) if row is not None else "",
            str(point_index),
            str(clicked_y),
            len(st.session_state.draw_clicks),
        )

        if event_signature != st.session_state.last_draw_event:
            st.session_state.last_draw_event = event_signature
            process_drawing_click(
                draw_mode,
                row,
                clicked_y,
            )
            st.rerun()

with control_col:
    st.markdown("#### ⏱️")
    st.caption(f"**{current_timestamp.strftime('%H:%M')}**")
    st.caption(f"Native {active_interval}")

    if active_interval == "1m":
        button_1, button_2, button_3 = "+1m", "+5m", "+15m"
        step_1, step_2, step_3 = 1, 5, 15

    elif active_interval == "2m":
        button_1, button_2, button_3 = "+1 bar", "+3 bars", "+8 bars"
        step_1, step_2, step_3 = 1, 3, 8

    elif active_interval == "5m":
        button_1, button_2, button_3 = "+1 bar", "+3 bars", "+6 bars"
        step_1, step_2, step_3 = 1, 3, 6

    elif active_interval == "15m":
        button_1, button_2, button_3 = "+1 bar", "+2 bars", "+4 bars"
        step_1, step_2, step_3 = 1, 2, 4

    elif active_interval in ("30m", "90m", "1h"):
        button_1, button_2, button_3 = "+1 bar", "+2 bars", "+4 bars"
        step_1, step_2, step_3 = 1, 2, 4

    else:
        button_1, button_2, button_3 = "+1 day", "+3 days", "+5 days"
        step_1, step_2, step_3 = 1, 3, 5

    if st.button(f"▶️ {button_1}", use_container_width=True):
        advance_bars(step_1)
        st.rerun()

    if st.button(f"⏩ {button_2}", use_container_width=True):
        advance_bars(step_2)
        st.rerun()

    if st.button(f"⏭️ {button_3}", use_container_width=True):
        advance_bars(step_3)
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
# MANUAL DRAWING FALLBACK
# ============================================================
with st.expander("✏️ Manual drawing helpers", expanded=False):
    st.caption(
        "Use this if click-to-draw does not work reliably in your Streamlit version."
    )

    candle_options = chart_df.copy()
    candle_options["choice"] = candle_options["timestamp"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    selected_choice = st.selectbox(
        "Select candle",
        candle_options["choice"].tolist()[::-1],
    )

    selected_row = candle_options[
        candle_options["choice"] == selected_choice
    ].iloc[0]

    d1, d2, d3, d4, d5 = st.columns(5)

    with d1:
        if st.button("High line"):
            price = float(selected_row["high"])
            add_horizontal_line(
                price,
                label=f"H {price:.2f}",
                color="#ef4444",
            )
            st.rerun()

    with d2:
        if st.button("Close line"):
            price = float(selected_row["close"])
            add_horizontal_line(
                price,
                label=f"C {price:.2f}",
                color="#22d3ee",
            )
            st.rerun()

    with d3:
        if st.button("Low line"):
            price = float(selected_row["low"])
            add_horizontal_line(
                price,
                label=f"L {price:.2f}",
                color="#22c55e",
            )
            st.rerun()

    with d4:
        if st.button("Band / Trend point"):
            mode_to_use = draw_mode

            if mode_to_use not in ("Band", "Trend line"):
                mode_to_use = "Band"

            process_drawing_click(
                mode_to_use,
                selected_row,
                float(selected_row["close"]),
            )

            st.rerun()

    with d5:
        if st.button("Clear pending"):
            st.session_state.draw_clicks = []
            st.session_state.last_draw_event = None
            st.rerun()


# ============================================================
# TRADE PANEL
# ============================================================
if pos == 0:
    st.markdown("### 🎮 Open a Position")

    e1, e2, e3, e4, e5 = st.columns([1.2, 1.2, 1.2, 1, 1])

    with e1:
        max_trade = max(10.0, float(st.session_state.balance))

        bet_size = st.number_input(
            "Trade Amount ($)",
            min_value=10.0,
            max_value=max_trade,
            value=min(100.0, max_trade),
            step=10.0,
        )

    with e2:
        sl_input = st.number_input(
            "Stop-Loss ($)",
            value=round(current_price * 0.99, 2),
            step=0.01,
        )

    with e3:
        tp_input = st.number_input(
            "Target ($, 0 = none)",
            value=0.0,
            step=0.01,
        )

    with e4:
        st.write("")
        st.write("")

        if st.button("🟢 BUY (Long)", use_container_width=True):
            if sl_input >= current_price:
                st.error("Long stop-loss must be below current price.")

            elif tp_input != 0 and tp_input <= current_price:
                st.error("Long target must be above current price.")

            else:
                shares = bet_size / current_price

                st.session_state.balance -= bet_size
                st.session_state.shares = shares
                st.session_state.stop_loss = float(sl_input)
                st.session_state.target = float(tp_input) if tp_input > 0 else None
                st.session_state.entry_price = current_price

                st.session_state.trade_log.append(
                    f"{current_time_text}: BOUGHT {shares:.4f} sh at "
                    f"${current_price:.2f} "
                    f"(SL ${sl_input:.2f} / "
                    f"TP {'$' + format(tp_input, '.2f') if tp_input > 0 else '—'})"
                )

                add_marker(
                    current_timestamp,
                    current_price,
                    "LONG",
                )

                st.rerun()

    with e5:
        st.write("")
        st.write("")

        if st.button("🔻 SHORT", use_container_width=True):
            if sl_input <= current_price:
                st.error("Short stop-loss must be above current price.")

            elif tp_input != 0 and tp_input >= current_price:
                st.error("Short target must be below current price.")

            else:
                shares = bet_size / current_price

                st.session_state.balance += bet_size
                st.session_state.shares = -shares
                st.session_state.stop_loss = float(sl_input)
                st.session_state.target = float(tp_input) if tp_input > 0 else None
                st.session_state.entry_price = current_price

                st.session_state.trade_log.append(
                    f"{current_time_text}: SHORTED {shares:.4f} sh at "
                    f"${current_price:.2f} "
                    f"(SL ${sl_input:.2f} / "
                    f"TP {'$' + format(tp_input, '.2f') if tp_input > 0 else '—'})"
                )

                add_marker(
                    current_timestamp,
                    current_price,
                    "SHORT",
                )

                st.rerun()

else:
    st.markdown(f"### 🎮 Manage {pos_type} Position")

    m1, m2, m3, m4, m5 = st.columns([1.2, 1.2, 1.2, 1, 1])

    with m1:
        new_sl = st.number_input(
            "Modify Stop-Loss ($)",
            value=float(
                st.session_state.stop_loss
                if st.session_state.stop_loss is not None
                else current_price
            ),
            step=0.01,
            key="modify_stop",
        )

    with m2:
        new_tp = st.number_input(
            "Modify Target ($, 0 = none)",
            value=float(
                st.session_state.target
                if st.session_state.target is not None
                else 0.0
            ),
            step=0.01,
            key="modify_target",
        )

    with m3:
        close_qty = st.number_input(
            "Quantity to close",
            min_value=0.0,
            max_value=float(abs(pos)),
            value=float(abs(pos)),
            step=0.0001,
        )

    with m4:
        st.write("")
        st.write("")

        if st.button("💾 Update SL/TP", use_container_width=True):
            valid = True

            if pos > 0 and new_sl >= current_price:
                st.error("Long stop-loss must be below current price.")
                valid = False

            if pos < 0 and new_sl <= current_price:
                st.error("Short stop-loss must be above current price.")
                valid = False

            if valid:
                st.session_state.stop_loss = float(new_sl)
                st.session_state.target = float(new_tp) if new_tp > 0 else None

                st.session_state.trade_log.append(
                    f"{current_time_text}: UPDATED "
                    f"SL ${new_sl:.2f} / "
                    f"TP {'$' + format(new_tp, '.2f') if new_tp > 0 else '—'}"
                )

                st.rerun()

    with m5:
        st.write("")
        st.write("")

        button_label = "🔴 SELL" if pos > 0 else "🟢 COVER"

        if st.button(button_label, use_container_width=True):
            if close_qty > 0:
                close_position(
                    close_qty,
                    current_price,
                    current_timestamp,
                    "(manual)",
                )

                st.rerun()


# ============================================================
# TRADE HISTORY
# ============================================================
if st.session_state.trade_log:
    with st.expander("📝 Trade History", expanded=True):
        for log in reversed(st.session_state.trade_log):
            st.text(log)


# ============================================================
# SESSION INFO
# ============================================================
with st.expander("📘 Replay Information"):
    st.markdown(
        f"""
### Active Replay

| Item | Value |
|---|---|
| Ticker | `{active_ticker}` |
| Native timeframe | `{active_interval}` |
| Practice date | `{practice_day}` |
| Current replay time | `{current_time_text}` |
| Revealed bars | `{len(revealed_df):,}` |
| Current price | `${current_price:.2f}` |
| Follow replay | `{"ON" if follow_replay else "OFF"}` |
| Context preset | `{context_preset}` |
| Saved drawings | `{len(st.session_state.drawings)}` |

### Important behavior

- Native `{active_interval}` data is used directly from Yahoo.
- Historical bars before the practice day are fully available.
- Current replay-day bars are hidden until you advance.
- EMAs use all revealed historical bars plus revealed current-day bars.
- With **Follow ON**, each advance returns camera focus to the current replay day.
- With **Follow OFF**, you can inspect historical support/resistance without being forced back.
- Native 5m/15m stop-loss simulation uses OHLC bars. If both target and stop occur inside one higher-timeframe candle, this version assumes stop occurs first.
"""
    )
