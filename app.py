# ============================================================
# HUMAN MARKET REPLAY SIMULATOR
#
# FIXED VERSION:
# - Native Yahoo timeframe data: 1m, 2m, 5m, 15m, 30m, 1h, 1d
# - Native 5m/15m data is NOT built from 1m bars
# - Yahoo rolling-period fetch avoids old-date request failure
# - Validates malformed OHLC bars before charting
# - Uses full revealed history for EMA calculations
# - Renders only chosen context bars for speed
# - Limits structure labels/zones for performance
# - Follow replay mode focuses BOTH x-axis and y-axis on current day
# - Right-side advance buttons
# - Persistent manual drawings
# ============================================================

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(layout="wide", page_title="Market Replay Simulator")

# ============================================================
# TIMEFRAME SETTINGS
# ============================================================
TIMEFRAME_CONFIG = {
    "1m": {
        "yf_interval": "1m",
        "period": "7d",
        "fallback_period": "6d",
        "ema": (9, 21),
        "minutes": 1,
        "intraday": True,
    },
    "2m": {
        "yf_interval": "2m",
        "period": "60d",
        "fallback_period": "59d",
        "ema": (9, 21),
        "minutes": 2,
        "intraday": True,
    },
    "5m": {
        "yf_interval": "5m",
        "period": "60d",
        "fallback_period": "59d",
        "ema": (20, 50),
        "minutes": 5,
        "intraday": True,
    },
    "15m": {
        "yf_interval": "15m",
        "period": "60d",
        "fallback_period": "59d",
        "ema": (50, 200),
        "minutes": 15,
        "intraday": True,
    },
    "30m": {
        "yf_interval": "30m",
        "period": "60d",
        "fallback_period": "59d",
        "ema": (50, 200),
        "minutes": 30,
        "intraday": True,
    },
    "90m": {
        "yf_interval": "90m",
        "period": "60d",
        "fallback_period": "59d",
        "ema": (50, 100),
        "minutes": 90,
        "intraday": True,
    },
    "1h": {
        "yf_interval": "60m",
        "period": "2y",
        "fallback_period": "1y",
        "ema": (50, 200),
        "minutes": 60,
        "intraday": True,
    },
    "1d": {
        "yf_interval": "1d",
        "period": "10y",
        "fallback_period": "5y",
        "ema": (50, 200),
        "minutes": 1440,
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
# PERFORMANCE SETTINGS
# ============================================================
# Structure calculation is intentionally limited.
# EMAs still use ALL revealed history.
MAX_STRUCTURE_BARS = 700
MAX_SWING_LABELS = 90
MAX_BOS_LABELS = 30
MAX_CHOCH_LABELS = 20
MAX_ZONE_SWINGS = 180
MAX_ZONES_DRAWN = 10

# ============================================================
# MARKET STRUCTURE SETTINGS
# ============================================================
SWING_K = 5
ZONE_TOL = 0.0012
MIN_DRAW_TOUCHES = 2
MIN_SWING_PCT = 0.0008
POLARITY_EDGE = 2
CHOCH_ENABLE = True

# ============================================================
# COLORS - BLACK CHART
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
    "last_draw_signature": None,
    "last_draw_mode": "None",
    "camera_signature": None,
    "camera_revision": 0,
    "force_camera": True,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# DATA FUNCTIONS
# ============================================================
def find_column(df, possible_names):
    """Find the first matching column name, case-insensitive."""
    lower_map = {str(col).strip().lower(): col for col in df.columns}

    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    return None


def normalize_ohlcv(raw, intraday=True):
    """
    Converts Yahoo/yfinance output into:
    timestamp, open, high, low, close, volume, date_only
    """

    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()

    # Yahoo frequently returns MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]) for col in df.columns]

    df = df.reset_index()

    timestamp_col = df.columns[0]

    open_col = find_column(df, ["Open"])
    high_col = find_column(df, ["High"])
    low_col = find_column(df, ["Low"])
    close_col = find_column(df, ["Close", "Adj Close"])
    volume_col = find_column(df, ["Volume"])

    if any(col is None for col in [open_col, high_col, low_col, close_col]):
        return pd.DataFrame()

    out = pd.DataFrame({
        "timestamp": df[timestamp_col],
        "open": df[open_col],
        "high": df[high_col],
        "low": df[low_col],
        "close": df[close_col],
        "volume": df[volume_col] if volume_col is not None else 0.0,
    })

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    # Convert Yahoo intraday timestamps to New York time if timezone exists.
    try:
        if out["timestamp"].dt.tz is not None:
            out["timestamp"] = (
                out["timestamp"]
                .dt.tz_convert("America/New_York")
                .dt.tz_localize(None)
            )
    except Exception:
        pass

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["volume"] = out["volume"].fillna(0.0)

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    out = out.drop_duplicates(subset=["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    # --------------------------------------------------------
    # OHLC VALIDATION
    #
    # Removes malformed Yahoo bars that can destroy Plotly axes.
    # Example bad bar:
    # normal TQQQ price = $65
    # malformed Yahoo bar = $80,000
    # --------------------------------------------------------
    valid = (
        (out["open"] > 0) &
        (out["high"] > 0) &
        (out["low"] > 0) &
        (out["close"] > 0) &
        (out["high"] >= out["low"]) &
        (out["high"] >= out["open"]) &
        (out["high"] >= out["close"]) &
        (out["low"] <= out["open"]) &
        (out["low"] <= out["close"])
    )

    out = out[valid].copy()

    if out.empty:
        return pd.DataFrame()

    # Remove absurd single-candle ranges.
    out["bar_ratio"] = out["high"] / out["low"]
    out = out[out["bar_ratio"] <= 3.0].copy()

    if out.empty:
        return pd.DataFrame()

    # Robust rolling-median outlier filter.
    # Intraday price should not suddenly be 10x normal price.
    rolling_med = (
        out["close"]
        .rolling(80, min_periods=10)
        .median()
        .shift(1)
    )

    expanding_med = out["close"].expanding(min_periods=1).median()

    reference = rolling_med.fillna(expanding_med)
    reference = reference.replace(0, np.nan)

    price_ratio = out["close"] / reference

    if intraday:
        out = out[
            (price_ratio >= 0.10) &
            (price_ratio <= 10.0)
        ].copy()
    else:
        out = out[
            (price_ratio >= 0.02) &
            (price_ratio <= 50.0)
        ].copy()

    if out.empty:
        return pd.DataFrame()

    out = out.drop(columns=["bar_ratio"], errors="ignore")

    out["date_only"] = out["timestamp"].dt.date
    out["label"] = out["timestamp"].dt.strftime("%m-%d %H:%M")

    return out.reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_native_history(ticker, timeframe):
    """
    Fetches Yahoo's currently retained native history.

    Important:
    Uses period-based Yahoo requests.

    Example:
    5m = period='60d', interval='5m'

    This avoids requesting:
    selected date - 60 days → selected date

    ...which can fail if the beginning of that requested range is
    outside Yahoo's current rolling intraday retention window.
    """
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

            df = normalize_ohlcv(
                raw,
                intraday=cfg["intraday"],
            )

            if not df.empty:
                return df

        except Exception:
            continue

    return pd.DataFrame()


def moving_avg(series, length, kind="EMA"):
    if kind == "EMA":
        return series.ewm(span=int(length), adjust=False).mean()

    return series.rolling(int(length)).mean()


def get_day_rows(df, practice_date):
    return df[df["date_only"] == practice_date].copy()


def get_context_render_df(revealed_df, practice_date, preset):
    """
    Returns only the bars needed for chart rendering.

    EMA calculations are still done separately on all revealed history.
    This is the major performance improvement.
    """

    if revealed_df.empty:
        return revealed_df.copy()

    unique_dates = sorted(revealed_df["date_only"].unique())

    if practice_date not in unique_dates:
        return revealed_df.copy()

    index = unique_dates.index(practice_date)

    previous_days_map = {
        "Today": 0,
        "1 Previous Day": 1,
        "5 Previous Days": 5,
        "20 Previous Days": 20,
        "All Available History": 99999,
    }

    previous_days = previous_days_map.get(preset, 0)
    start_index = max(0, index - previous_days)
    start_date = unique_dates[start_index]

    return revealed_df[
        revealed_df["date_only"] >= start_date
    ].copy()


def get_price_axis_range(df):
    """
    Creates a readable y-axis range based only on currently visible data.

    This fixes the problem where candles were tiny/invisible because the
    y-axis used all historical data instead of the active replay window.
    """

    if df.empty:
        return None

    values = []

    for col in ["low", "high", "fast_ma", "slow_ma"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
            values.extend(vals)

    if not values:
        return None

    low = float(min(values))
    high = float(max(values))

    median_price = float(
        pd.to_numeric(df["close"], errors="coerce").median()
    )

    spread = high - low

    # Prevent a single narrow candle from having a zero-height chart.
    min_spread = max(median_price * 0.003, 0.10)
    spread = max(spread, min_spread)

    padding = max(spread * 0.12, median_price * 0.001)

    return [
        low - padding,
        high + padding,
    ]


def get_volume_axis_range(df):
    """Volume axis based only on visible context/current window."""

    if df.empty or df["volume"].max() <= 0:
        return None

    volume_values = df["volume"].dropna()

    cap = max(
        float(volume_values.quantile(0.95)) * 1.15,
        float(volume_values.median()) * 2.0,
        1.0,
    )

    return [0, cap]


def right_padding_timestamp(timestamp, timeframe):
    cfg = TIMEFRAME_CONFIG[timeframe]

    if cfg["intraday"]:
        return pd.Timestamp(timestamp) + pd.Timedelta(
            minutes=cfg["minutes"] * 3
        )

    return pd.Timestamp(timestamp) + pd.Timedelta(days=5)


# ============================================================
# MARKET STRUCTURE FUNCTIONS
# ============================================================
def _raw_swings(df):
    if df is None or len(df) < SWING_K * 2 + 1:
        return []

    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    reference = float(df["close"].iloc[-1])
    min_size = max(reference * MIN_SWING_PCT, 1e-9)

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
            previous = cleaned[-1]

            if swing[2] == "H" and swing[1] >= previous[1]:
                cleaned[-1] = swing

            elif swing[2] == "L" and swing[1] <= previous[1]:
                cleaned[-1] = swing

        else:
            cleaned.append(swing)

    labeled = []
    previous_high = None
    previous_low = None

    for idx, price, swing_type in cleaned:
        if swing_type == "H":
            label = (
                "H"
                if previous_high is None
                else ("HH" if price > previous_high else "LH")
            )
            previous_high = price

        else:
            label = (
                "L"
                if previous_low is None
                else ("HL" if price > previous_low else "LL")
            )
            previous_low = price

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
            distance = abs(swing["price"] - zone["mid"])
            normalized_distance = distance / max(reference_price, 1e-9)

            if normalized_distance < ZONE_TOL:
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

    output = [
        zone for zone in zones
        if zone["touches"] >= MIN_DRAW_TOUCHES
    ]

    for zone in output:
        zone["min_px"] = min(zone["prices"])
        zone["max_px"] = max(zone["prices"])

    return output


def detect_bos_choch(df, labeled):
    """
    BOS = continuation in active direction.
    CHoCH = break against active direction / potential reversal.
    """

    bos = []
    choch = []

    last_hh = None
    last_hl = None
    last_lh = None
    last_ll = None

    bias = "NEUTRAL"
    pointer = 0

    for i in range(len(df)):
        while (
            pointer < len(labeled) and
            labeled[pointer]["i"] + SWING_K <= i
        ):
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
            if bias == "BULL" and last_hl is not None and close < last_hl:
                choch.append({
                    "i": i,
                    "price": last_hl,
                    "dir": "down",
                })

                last_hl = None
                bias = "NEUTRAL"

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


def select_relevant_zones(zones, current_price):
    """
    Shows only nearby/important zones.
    Prevents hundreds of horizontal lines from being drawn.
    """

    if not zones:
        return []

    ranked = sorted(
        zones,
        key=lambda zone: (
            abs(zone["mid"] - current_price),
            -zone["touches"],
        ),
    )

    return ranked[:MAX_ZONES_DRAWN]


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
# TRADING FUNCTIONS
# ============================================================
def add_marker(timestamp, price, kind):
    st.session_state.markers.append({
        "timestamp": pd.Timestamp(timestamp),
        "price": float(price),
        "kind": kind,
    })


def close_position(qty, price, timestamp, reason=""):
    pos = float(st.session_state.shares)
    timestamp = pd.Timestamp(timestamp)
    time_text = timestamp.strftime("%Y-%m-%d %H:%M")

    if pos > 0:
        qty = min(float(qty), pos)

        st.session_state.balance += qty * float(price)
        st.session_state.shares -= qty

        st.session_state.trade_log.append(
            f"{time_text}: SOLD {qty:.4f} sh at ${price:.2f} {reason}"
        )

        add_marker(timestamp, price, "SELL")

    elif pos < 0:
        qty = min(float(qty), abs(pos))

        st.session_state.balance -= qty * float(price)
        st.session_state.shares += qty

        st.session_state.trade_log.append(
            f"{time_text}: COVERED {qty:.4f} sh at ${price:.2f} {reason}"
        )

        add_marker(timestamp, price, "COVER")

    if abs(float(st.session_state.shares)) < 1e-10:
        st.session_state.shares = 0.0
        st.session_state.stop_loss = None
        st.session_state.target = None
        st.session_state.entry_price = None


def advance_bars(number_of_bars):
    """
    Advances native bars.

    Example:
    On 5m:
      +5m = one native 5-minute candle.
      +15m = three native 5-minute candles.

    Stop/target checks use OHLC of the selected native timeframe.
    If stop and target are touched inside the same larger candle,
    this simulator conservatively assumes stop happens first.
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

        pos = float(st.session_state.shares)
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


def add_band(price_1, price_2):
    y0 = min(float(price_1), float(price_2))
    y1 = max(float(price_1), float(price_2))

    st.session_state.drawings.append({
        "type": "band",
        "y0": y0,
        "y1": y1,
        "color": "rgba(34,197,94,0.18)",
        "label": f"Band {y0:.2f}-{y1:.2f}",
    })


def add_trend_line(x0, y0, x1, y1):
    st.session_state.drawings.append({
        "type": "trend",
        "x0": pd.Timestamp(x0),
        "y0": float(y0),
        "x1": pd.Timestamp(x1),
        "y1": float(y1),
        "color": "#f472b6",
        "label": "Trend line",
    })


def extract_selected_points(chart_event):
    if chart_event is None:
        return []

    try:
        return [dict(point) for point in chart_event.selection.points]
    except Exception:
        pass

    try:
        return chart_event.get("selection", {}).get("points", [])
    except Exception:
        return []


def nearest_row_from_event(df, point):
    if df.empty:
        return None

    x_value = point.get("x")

    if x_value is not None:
        try:
            clicked_time = pd.to_datetime(x_value)
            nearest_idx = (df["timestamp"] - clicked_time).abs().idxmin()
            return df.loc[nearest_idx]
        except Exception:
            pass

    point_index = point.get("point_index", point.get("pointIndex"))

    if point_index is not None:
        try:
            idx = int(point_index)

            if 0 <= idx < len(df):
                return df.iloc[idx]
        except Exception:
            pass

    return df.iloc[-1]


def process_drawing_click(mode, row, clicked_y=None):
    if row is None or mode == "None":
        return

    timestamp = row["timestamp"]

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

        st.toast(f"High line added at {price:.2f}", icon="📌")

    elif mode == "Line at Close":
        price = float(row["close"])

        add_horizontal_line(
            price,
            label=f"C {price:.2f}",
            color="#22d3ee",
        )

        st.toast(f"Close line added at {price:.2f}", icon="📌")

    elif mode == "Line at Low":
        price = float(row["low"])

        add_horizontal_line(
            price,
            label=f"L {price:.2f}",
            color="#22c55e",
        )

        st.toast(f"Low line added at {price:.2f}", icon="📌")

    elif mode == "Band":
        st.session_state.draw_clicks.append({
            "timestamp": timestamp,
            "price": anchor_price,
        })

        if len(st.session_state.draw_clicks) == 1:
            st.toast("Band point one saved. Click second point.", icon="🖱️")

        elif len(st.session_state.draw_clicks) >= 2:
            p1 = st.session_state.draw_clicks[0]["price"]
            p2 = st.session_state.draw_clicks[1]["price"]

            add_band(p1, p2)

            st.session_state.draw_clicks = []

            st.toast("Band added.", icon="🟩")

    elif mode == "Trend line":
        st.session_state.draw_clicks.append({
            "timestamp": timestamp,
            "price": anchor_price,
        })

        if len(st.session_state.draw_clicks) == 1:
            st.toast("Trend point one saved. Click second point.", icon="🖱️")

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

ticker = st.sidebar.text_input(
    "Ticker",
    value="TQQQ",
).upper().strip()

timeframe = st.sidebar.selectbox(
    "Chart Timeframe (native)",
    TIMEFRAME_OPTIONS,
    index=TIMEFRAME_OPTIONS.index("5m"),
)

timeframe_cfg = TIMEFRAME_CONFIG[timeframe]

st.sidebar.caption(
    f"Yahoo native `{timeframe}` data. "
    f"Rolling request: `{timeframe_cfg['period']}`."
)

# ------------------------------------------------------------
# DATA PREVIEW
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
        "- Yahoo outage\n"
        "- timeframe unavailable"
    )

else:
    available_dates = sorted(source_df["date_only"].unique())

    first_date = available_dates[0]
    last_date = available_dates[-1]

    st.sidebar.success(
        f"{len(source_df):,} native bars\n"
        f"{first_date} → {last_date}"
    )

    st.sidebar.caption(
        "Practice dates below are based on dates Yahoo actually returned."
    )

    if len(available_dates) >= 3:
        default_practice_date = available_dates[-3]
    else:
        default_practice_date = available_dates[-1]

    practice_date_key = f"practice_date_{ticker}_{timeframe}"

    if (
        practice_date_key not in st.session_state or
        st.session_state[practice_date_key] not in available_dates
    ):
        st.session_state[practice_date_key] = default_practice_date

    practice_date = st.sidebar.date_input(
        "Practice Date",
        min_value=first_date,
        max_value=last_date,
        key=practice_date_key,
    )

    practice_rows = get_day_rows(source_df, practice_date)

    if practice_rows.empty:
        st.sidebar.warning(
            "No market bars on this date. It may be a holiday/weekend."
        )

    else:
        if timeframe_cfg["intraday"]:
            start_labels = practice_rows["timestamp"].dt.strftime(
                "%H:%M"
            ).tolist()

            default_index = min(10, len(start_labels) - 1)

            start_time_key = (
                f"start_time_{ticker}_{timeframe}_{practice_date}"
            )

            selected_start_label = st.sidebar.selectbox(
                "Replay Start Time",
                start_labels,
                index=default_index,
                key=start_time_key,
            )

            selected_start_timestamp = practice_rows[
                practice_rows["timestamp"].dt.strftime("%H:%M") ==
                selected_start_label
            ]["timestamp"].iloc[0]

        else:
            selected_start_timestamp = practice_rows["timestamp"].iloc[0]

        can_start = True

# ------------------------------------------------------------
# EMA SETTINGS
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Moving Averages")

show_ma = st.sidebar.checkbox(
    "Show Moving Averages",
    value=True,
)

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

show_vol_ma = st.sidebar.checkbox(
    "Show Volume MA",
    value=True,
)

vol_ma_len = st.sidebar.number_input(
    "Volume MA Length",
    min_value=2,
    max_value=200,
    value=20,
    step=1,
)

# ------------------------------------------------------------
# STRUCTURE SETTINGS
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🏗️ Structure")

show_struct = st.sidebar.checkbox(
    "Show structure labels",
    value=False,  # Default OFF for performance/readability.
)

show_zones = st.sidebar.checkbox(
    "Show S/R zones",
    value=False,  # Default OFF for performance/readability.
)

# ------------------------------------------------------------
# CONTEXT SETTINGS
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
    "Follow ON: after advancing, the chart returns to current replay candles.\n\n"
    "Follow OFF: zoom/pan stays where you left it."
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

if draw_mode != st.session_state.last_draw_mode:
    st.session_state.last_draw_mode = draw_mode
    st.session_state.last_draw_signature = None
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
    st.session_state.last_draw_signature = None
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
    # Keep only history before or on selected practice day.
    # This prevents future-day leakage.
    active_df = source_df[
        source_df["date_only"] <= practice_date
    ].copy().reset_index(drop=True)

    selected_index_list = active_df.index[
        active_df["timestamp"] == selected_start_timestamp
    ].tolist()

    day_indices = active_df.index[
        active_df["date_only"] == practice_date
    ].tolist()

    if not selected_index_list or not day_indices:
        st.sidebar.error("Could not initialize the replay.")
    else:
        st.session_state.df = active_df
        st.session_state.sim_start_idx = day_indices[0]
        st.session_state.step = selected_index_list[0]

        st.session_state.balance = 1000.0
        st.session_state.shares = 0.0
        st.session_state.stop_loss = None
        st.session_state.target = None
        st.session_state.entry_price = None

        st.session_state.trade_log = []
        st.session_state.markers = []

        st.session_state.drawings = []
        st.session_state.draw_clicks = []
        st.session_state.last_draw_signature = None

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
        "Choose ticker, native timeframe, practice date, and start time. "
        "Then click **Start / Reset Simulation**."
    )
    st.stop()

# Alert if user changes settings without pressing reset.
if (
    st.session_state.active_ticker != ticker or
    st.session_state.active_interval != timeframe or
    st.session_state.active_practice_date != practice_date
):
    st.warning(
        "Sidebar ticker/timeframe/date differs from active replay. "
        "Click **Start / Reset Simulation** to apply changes."
    )

df_all = st.session_state.df
step = st.session_state.step
practice_day = st.session_state.active_practice_date
active_ticker = st.session_state.active_ticker
active_interval = st.session_state.active_interval

# ------------------------------------------------------------
# Revealed data:
# - all previous dates fully available
# - current replay day only up to current bar
# ------------------------------------------------------------
revealed_df = df_all.iloc[:step + 1].copy()

if revealed_df.empty:
    st.error("No revealed bars available.")
    st.stop()

current_row = df_all.iloc[step]
current_price = float(current_row["close"])
current_timestamp = pd.Timestamp(current_row["timestamp"])
current_time_text = current_timestamp.strftime("%Y-%m-%d %H:%M")

# ------------------------------------------------------------
# EMA CALCULATION DATA:
# Uses ALL revealed historical bars.
# ------------------------------------------------------------
calc_df = revealed_df.copy()

if show_ma:
    calc_df["fast_ma"] = moving_avg(
        calc_df["close"],
        fast_len,
        ma_type,
    )

    calc_df["slow_ma"] = moving_avg(
        calc_df["close"],
        slow_len,
        ma_type,
    )

if show_vol_ma:
    calc_df["vol_ma"] = calc_df["volume"].rolling(
        int(vol_ma_len)
    ).mean()

# ------------------------------------------------------------
# RENDERED DATA:
# Only selected context bars are sent to Plotly.
# This fixes speed/performance.
# ------------------------------------------------------------
chart_df = get_context_render_df(
    calc_df,
    practice_day,
    context_preset,
).reset_index(drop=True)

if chart_df.empty:
    chart_df = calc_df.copy().reset_index(drop=True)

# ------------------------------------------------------------
# POSITION / ACCOUNT
# ------------------------------------------------------------
pos = float(st.session_state.shares)
position_value = pos * current_price
equity = float(st.session_state.balance) + position_value
pnl = equity - 1000.0

pnl_color = "normal" if pnl == 0 else (
    "inverse" if pnl > 0 else "off"
)

if pos > 0:
    pos_type = "🟢 LONG"
    pos_text = f"{pos:.4f} sh"
elif pos < 0:
    pos_type = "🔴 SHORT"
    pos_text = f"{abs(pos):.4f} sh"
else:
    pos_type = "FLAT"
    pos_text = "—"

# ------------------------------------------------------------
# CAMERA
# ------------------------------------------------------------
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

apply_camera = follow_replay or st.session_state.force_camera

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------
m1, m2, m3, m4, m5, m6 = st.columns(6)

m1.metric("Price", f"${current_price:.2f}")
m2.metric("Equity", f"${equity:.2f}", f"${pnl:.2f}", delta_color=pnl_color)
m3.metric("Cash", f"${st.session_state.balance:.2f}")

m4.metric(
    f"Position ({pos_type})",
    pos_text,
    (
        f"Entry ${st.session_state.entry_price:.2f}"
        if st.session_state.entry_price is not None
        else ""
    ),
)

m5.metric(
    "Stop-Loss",
    (
        f"${st.session_state.stop_loss:.2f}"
        if st.session_state.stop_loss is not None
        else "None"
    ),
)

m6.metric(
    "Target",
    (
        f"${st.session_state.target:.2f}"
        if st.session_state.target is not None
        else "None"
    ),
)

# ============================================================
# BUILD CHART
# ============================================================
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

# Candlestick trace
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

# Volume trace
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

# EMA traces
if show_ma and "fast_ma" in chart_df.columns:
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

# Volume MA trace
if show_vol_ma and "vol_ma" in chart_df.columns:
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
# STRUCTURE / ZONES
#
# Uses a LIMITED recent section of rendered bars.
# This prevents lag and annotation overload.
# ============================================================
if show_struct or show_zones:
    structure_df = chart_df.tail(MAX_STRUCTURE_BARS).copy().reset_index(drop=True)

    labeled = _raw_swings(structure_df)

    confirmed_swings = [
        swing
        for swing in labeled
        if swing["i"] + SWING_K <= len(structure_df) - 1
    ]
else:
    structure_df = pd.DataFrame()
    labeled = []
    confirmed_swings = []

if show_struct and not structure_df.empty:
    bos, choch = detect_bos_choch(structure_df, labeled)

    # Only draw latest swing labels.
    swings_to_draw = confirmed_swings[-MAX_SWING_LABELS:]

    for swing in swings_to_draw:
        x = structure_df["timestamp"].iloc[swing["i"]]

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

    for event in bos[-MAX_BOS_LABELS:]:
        i = event["i"]

        if 0 <= i < len(structure_df):
            x = structure_df["timestamp"].iloc[i]

            if event["dir"] == "up":
                badge(
                    fig,
                    x,
                    float(structure_df["high"].iloc[i]),
                    "BOS↑",
                    C_BOS_UP,
                    yshift=22,
                    arrow=True,
                )
            else:
                badge(
                    fig,
                    x,
                    float(structure_df["low"].iloc[i]),
                    "BOS↓",
                    C_BOS_DN,
                    yshift=-22,
                    arrow=True,
                )

    for event in choch[-MAX_CHOCH_LABELS:]:
        i = event["i"]

        if 0 <= i < len(structure_df):
            x = structure_df["timestamp"].iloc[i]

            if event["dir"] == "up":
                badge(
                    fig,
                    x,
                    float(structure_df["high"].iloc[i]),
                    "CHoCH↑",
                    C_CH_UP,
                    yshift=22,
                    arrow=True,
                )
            else:
                badge(
                    fig,
                    x,
                    float(structure_df["low"].iloc[i]),
                    "CHoCH↓",
                    C_CH_DN,
                    yshift=-22,
                    arrow=True,
                )

if show_zones and confirmed_swings:
    noise = float(
        (structure_df["high"] - structure_df["low"])
        .tail(20)
        .mean()
        or 0.01
    )

    zones = build_zones(
        confirmed_swings[-MAX_ZONE_SWINGS:],
        current_price,
    )

    zones = select_relevant_zones(zones, current_price)

    right_x = chart_df["timestamp"].iloc[-1]

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
            x=right_x,
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
                width=1.3,
                dash="dot",
            ),
        )

# ============================================================
# TRADE MARKERS
# ============================================================
marker_styles = {
    "LONG": {"symbol": "triangle-up", "color": "#22c55e"},
    "SHORT": {"symbol": "triangle-down", "color": "#ef4444"},
    "SELL": {"symbol": "circle", "color": "#f97316"},
    "COVER": {"symbol": "circle", "color": "#38bdf8"},
}

for marker in st.session_state.markers:
    if marker["timestamp"] > current_timestamp:
        continue

    style = marker_styles.get(
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
# CAMERA / AXIS CONTROL
# ============================================================
today_df = chart_df[
    chart_df["date_only"] == practice_day
].copy()

if today_df.empty:
    today_df = chart_df.copy()

# Follow mode means:
# x-axis = current practice day
# y-axis = current practice day highs/lows/EMAs
#
# This fixes tiny/invisible candles.
if follow_replay:
    focus_df = today_df

    camera_start = today_df["timestamp"].iloc[0]
    camera_end = right_padding_timestamp(
        current_timestamp,
        active_interval,
    )

else:
    # With Follow OFF, context preset determines starting camera.
    focus_df = chart_df

    camera_start = chart_df["timestamp"].iloc[0]
    camera_end = right_padding_timestamp(
        current_timestamp,
        active_interval,
    )

focus_price_range = get_price_axis_range(focus_df)
focus_volume_range = get_volume_axis_range(focus_df)

# ============================================================
# CHART LAYOUT
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

tick_format = "%m-%d\n%H:%M"

if not TIMEFRAME_CONFIG[active_interval]["intraday"]:
    tick_format = "%Y-%m-%d"

for row_num in (1, 2):
    fig.update_xaxes(
        type="date",
        tickformat=tick_format,
        nticks=12,
        gridcolor="#1f2937",
        linecolor="#4b5563",
        zerolinecolor="#374151",
        showspikes=True,
        spikemode="across",
        spikecolor="#94a3b8",
        spikethickness=1,
        row=row_num,
        col=1,
    )

fig.update_yaxes(
    title_text="Price",
    gridcolor="#1f2937",
    linecolor="#4b5563",
    row=1,
    col=1,
)

fig.update_yaxes(
    title_text="Volume",
    gridcolor="#1f2937",
    linecolor="#4b5563",
    row=2,
    col=1,
)

# Apply camera only when:
# - Follow ON, or
# - initial/reset/context change with Follow OFF.
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

    if focus_price_range is not None:
        fig.update_yaxes(
            range=focus_price_range,
            row=1,
            col=1,
        )

    if focus_volume_range is not None:
        fig.update_yaxes(
            range=focus_volume_range,
            row=2,
            col=1,
        )

    # After first camera setup, manual mode may preserve zoom.
    if not follow_replay:
        st.session_state.force_camera = False

# ============================================================
# CHART + RIGHT-SIDE CONTROLS
# ============================================================
chart_col, control_col = st.columns([6, 1])

with chart_col:
    chart_event = None

    # Important performance improvement:
    # Only activate chart selection/reruns when drawing mode is active.
    if draw_mode == "None":
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="market_chart_view",
            config={
                "scrollZoom": True,
                "displaylogo": False,
            },
        )

    else:
        try:
            chart_event = st.plotly_chart(
                fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="market_chart_draw",
                config={
                    "scrollZoom": True,
                    "displaylogo": False,
                },
            )

        except TypeError:
            # Older Streamlit fallback.
            st.plotly_chart(
                fig,
                use_container_width=True,
                key="market_chart_draw_fallback",
                config={
                    "scrollZoom": True,
                    "displaylogo": False,
                },
            )

    # --------------------------------------------------------
    # Drawing click processing.
    # --------------------------------------------------------
    if draw_mode != "None":
        selected_points = extract_selected_points(chart_event)

        if selected_points:
            point = selected_points[-1]
            selected_row = nearest_row_from_event(chart_df, point)

            clicked_y = point.get("y")
            curve = point.get("curve_number", point.get("curveNumber", ""))

            signature = (
                draw_mode,
                str(selected_row["timestamp"]) if selected_row is not None else "",
                str(clicked_y),
                str(curve),
            )

            if signature != st.session_state.last_draw_signature:
                st.session_state.last_draw_signature = signature

                process_drawing_click(
                    draw_mode,
                    selected_row,
                    clicked_y,
                )

                st.rerun()

with control_col:
    st.markdown("#### ⏱️")
    st.caption(f"**{current_timestamp.strftime('%H:%M')}**")
    st.caption(f"Native {active_interval}")

    # Buttons are based on selected native interval.
    if active_interval == "1m":
        button_1, button_2, button_3 = "+1m", "+5m", "+15m"
        step_1, step_2, step_3 = 1, 5, 15

    elif active_interval == "2m":
        button_1, button_2, button_3 = "+2m", "+10m", "+30m"
        step_1, step_2, step_3 = 1, 5, 15

    elif active_interval == "5m":
        button_1, button_2, button_3 = "+5m", "+15m", "+30m"
        step_1, step_2, step_3 = 1, 3, 6

    elif active_interval == "15m":
        button_1, button_2, button_3 = "+15m", "+30m", "+60m"
        step_1, step_2, step_3 = 1, 2, 4

    elif active_interval == "30m":
        button_1, button_2, button_3 = "+30m", "+60m", "+120m"
        step_1, step_2, step_3 = 1, 2, 4

    elif active_interval == "90m":
        button_1, button_2, button_3 = "+90m", "+180m", "+360m"
        step_1, step_2, step_3 = 1, 2, 4

    elif active_interval == "1h":
        button_1, button_2, button_3 = "+1h", "+2h", "+4h"
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
    st.caption(f"Rendered bars:\n{len(chart_df):,}")
    st.caption(f"Drawings:\n{len(st.session_state.drawings)}")

# ============================================================
# MANUAL DRAWING FALLBACK
# ============================================================
with st.expander("✏️ Manual drawing helpers", expanded=False):
    st.caption(
        "Use this if click-to-draw is unreliable in your browser/Streamlit version."
    )

    helper_df = chart_df.copy()
    helper_df["choice"] = helper_df["timestamp"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    selected_choice = st.selectbox(
        "Select candle",
        helper_df["choice"].tolist()[::-1],
    )

    selected_row = helper_df[
        helper_df["choice"] == selected_choice
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
            use_mode = draw_mode

            if use_mode not in ("Band", "Trend line"):
                use_mode = "Band"

            process_drawing_click(
                use_mode,
                selected_row,
                float(selected_row["close"]),
            )

            st.rerun()

    with d5:
        if st.button("Clear pending"):
            st.session_state.draw_clicks = []
            st.session_state.last_draw_signature = None
            st.rerun()

# ============================================================
# TRADE ENTRY PANEL
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
                st.session_state.target = (
                    float(tp_input)
                    if tp_input > 0
                    else None
                )
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
                st.session_state.target = (
                    float(tp_input)
                    if tp_input > 0
                    else None
                )
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

# ============================================================
# TRADE MANAGEMENT PANEL
# ============================================================
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
                st.session_state.target = (
                    float(new_tp)
                    if new_tp > 0
                    else None
                )

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
# SESSION INFORMATION
# ============================================================
with st.expander("📘 Replay Information"):
    st.markdown(
        f"""
| Item | Value |
|---|---|
| Ticker | `{active_ticker}` |
| Native timeframe | `{active_interval}` |
| Practice date | `{practice_day}` |
| Current replay time | `{current_time_text}` |
| All revealed bars used for EMA | `{len(calc_df):,}` |
| Bars rendered in chart | `{len(chart_df):,}` |
| Follow replay | `{"ON" if follow_replay else "OFF"}` |
| Context preset | `{context_preset}` |
| Saved drawings | `{len(st.session_state.drawings)}` |

### Important behavior

- Native `{active_interval}` candles come directly from Yahoo.
- Current replay day reveals bars only up to the current simulation time.
- EMA calculations use all revealed historical data.
- Chart rendering uses only the selected context to remain fast.
- Follow ON zooms both x-axis and y-axis onto current replay candles.
- Follow OFF preserves manual zoom/pan after the initial camera setup.
- Structure labels and S/R zones are intentionally capped for speed.
"""
    )
