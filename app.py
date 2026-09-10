# ============================================================
# HUMAN MARKET REPLAY SIMULATOR
# ============================================================

import os
import sqlite3
from datetime import date, datetime, timedelta, time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

st.set_page_config(layout="wide", page_title="Market Replay Simulator")

DB_PATH = "paper_trading.db"
ACCOUNT_ID = 1
STARTING_BALANCE = 1000.0

TIMEFRAME_CONFIG = {
    "1m":  {"yf_interval": "1m",  "period": "7d",  "fallback_period": "6d",  "ema": (9, 21),   "minutes": 1,    "intraday": True},
    "2m":  {"yf_interval": "2m",  "period": "60d", "fallback_period": "59d", "ema": (9, 21),   "minutes": 2,    "intraday": True},
    "5m":  {"yf_interval": "5m",  "period": "60d", "fallback_period": "59d", "ema": (20, 50),  "minutes": 5,    "intraday": True},
    "15m": {"yf_interval": "15m", "period": "60d", "fallback_period": "59d", "ema": (50, 200), "minutes": 15,   "intraday": True},
    "30m": {"yf_interval": "30m", "period": "60d", "fallback_period": "59d", "ema": (50, 200), "minutes": 30,   "intraday": True},
    "90m": {"yf_interval": "90m", "period": "60d", "fallback_period": "59d", "ema": (50, 100), "minutes": 90,   "intraday": True},
    "1h":  {"yf_interval": "60m", "period": "2y",  "fallback_period": "1y",  "ema": (50, 200), "minutes": 60,   "intraday": True},
    "1d":  {"yf_interval": "1d",  "period": "10y", "fallback_period": "5y",  "ema": (50, 200), "minutes": 1440, "intraday": False},
}
TIMEFRAME_OPTIONS = list(TIMEFRAME_CONFIG.keys())
CONTEXT_PRESETS = ["Today", "1 Previous Day", "5 Previous Days", "20 Previous Days", "All Available History"]
FOLLOW_WINDOW_OPTIONS = [100, 200]
SESSION_OPTIONS = [
    "Auto Detect",
    "US Stocks / ETFs - Regular Hours",
    "US Stocks / ETFs - Extended Hours",
    "CME Futures",
    "Forex 24/5",
    "Crypto 24/7",
    "Custom Session",
]
DRAW_MODES = ["None", "Line at High", "Line at Close", "Line at Low", "Band", "Trend line"]

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

C_HH = dict(fg="#ffffff", bg="#16a34a")
C_LH = dict(fg="#ffffff", bg="#dc2626")
C_H = dict(fg="#ffffff", bg="#475569")
C_BOS_UP = dict(fg="#111111", bg="#fde047")
C_BOS_DN = dict(fg="#ffffff", bg="#dc2626")
C_CH_UP = dict(fg="#111111", bg="#7dd3fc")
C_CH_DN = dict(fg="#111111", bg="#fb923c")
C_FLOOR, C_CEIL, C_BROKEN, C_BOTH = "#22c55e", "#ef4444", "#a8a29e", "#eab308"
C_FAST_EMA, C_SLOW_EMA, C_VOL_MA = "#3b82f6", "#f59e0b", "#ff6d00"

DEFAULTS = {
    "sim_active": False, "df": pd.DataFrame(), "step": 0,
    "active_ticker": None, "active_interval": None, "active_session_mode": None,
    "active_practice_date": None, "active_carry_mode": True,
    "markers": [], "drawings": [], "draw_clicks": [],
    "last_draw_signature": None, "last_draw_mode": "None",
    "camera_signature": None, "camera_revision": 0, "force_camera": True,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def to_pydate(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    return ts.date()


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn, name):
    return conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def table_columns(conn, name):
    if not table_exists(conn, name):
        return set()
    return {r[1] for r in conn.execute(f"PRAGMA table_info({name})")}


def initialize_database():
    conn = get_connection()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute("""CREATE TABLE IF NOT EXISTS accounts (
        account_id INTEGER PRIMARY KEY, starting_balance REAL NOT NULL, cash_balance REAL NOT NULL,
        realized_pnl REAL NOT NULL DEFAULT 0, cycle_number INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS positions (
        account_id INTEGER PRIMARY KEY, ticker TEXT NOT NULL, asset_class TEXT NOT NULL, side TEXT NOT NULL,
        quantity REAL NOT NULL, entry_price REAL NOT NULL, entry_time TEXT NOT NULL,
        stop_loss REAL, target REAL, invested_amount REAL NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, account_id INTEGER NOT NULL, cycle_number INTEGER NOT NULL,
        ticker TEXT NOT NULL, asset_class TEXT NOT NULL, side TEXT NOT NULL, quantity REAL NOT NULL,
        entry_price REAL NOT NULL, exit_price REAL, entry_time TEXT NOT NULL, exit_time TEXT,
        realized_pnl REAL, fees REAL DEFAULT 0, reason TEXT DEFAULT '')""")
    c.execute("""CREATE TABLE IF NOT EXISTS account_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, account_id INTEGER NOT NULL, cycle_number INTEGER NOT NULL,
        event_type TEXT NOT NULL, message TEXT NOT NULL, created_at TEXT NOT NULL)""")

    if table_exists(conn, "account") and "balance" in table_columns(conn, "account"):
        row = conn.execute("SELECT balance, realized_pnl, cycle FROM account WHERE status='ACTIVE' ORDER BY id DESC LIMIT 1").fetchone()
        if row is not None and conn.execute("SELECT account_id FROM accounts WHERE account_id=?", (ACCOUNT_ID,)).fetchone() is None:
            c.execute("""INSERT INTO accounts (account_id, starting_balance, cash_balance, realized_pnl, cycle_number, created_at, updated_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?)""",
                      (ACCOUNT_ID, STARTING_BALANCE, float(row[0]), float(row[1] or 0), int(row[2] or 1), now, now))

    pos_cols = table_columns(conn, "positions")
    if "qty" in pos_cols and "quantity" not in pos_cols:
        c.execute("ALTER TABLE positions RENAME TO positions_legacy")
        c.execute("""CREATE TABLE positions (
            account_id INTEGER PRIMARY KEY, ticker TEXT NOT NULL, asset_class TEXT NOT NULL, side TEXT NOT NULL,
            quantity REAL NOT NULL, entry_price REAL NOT NULL, entry_time TEXT NOT NULL,
            stop_loss REAL, target REAL, invested_amount REAL NOT NULL)""")
        old = conn.execute("SELECT ticker, qty, entry_price, sl, tp FROM positions_legacy LIMIT 1").fetchone()
        if old is not None and float(old[1] or 0) != 0:
            qty = float(old[1])
            sl = old[3] if old[3] not in (None, -1) else None
            tp = old[4] if old[4] not in (None, -1) else None
            c.execute("""INSERT INTO positions (account_id, ticker, asset_class, side, quantity, entry_price, entry_time, stop_loss, target, invested_amount)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                      (ACCOUNT_ID, old[0] or "MIXED", "stocks", "long" if qty > 0 else "short", abs(qty),
                       float(old[2] or 0), now, sl, tp, abs(qty) * float(old[2] or 0)))
        c.execute("DROP TABLE positions_legacy")

    if table_exists(conn, "history"):
        for cycle, text in conn.execute("SELECT cycle, log_text FROM history"):
            c.execute("INSERT INTO account_events (account_id, cycle_number, event_type, message, created_at) VALUES (?, ?, ?, ?, ?)",
                      (ACCOUNT_ID, int(cycle or 1), "LEGACY_LOG", str(text), now))
        c.execute("DROP TABLE history")

    if conn.execute("SELECT account_id FROM accounts WHERE account_id=?", (ACCOUNT_ID,)).fetchone() is None:
        c.execute("""INSERT INTO accounts (account_id, starting_balance, cash_balance, realized_pnl, cycle_number, created_at, updated_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (ACCOUNT_ID, STARTING_BALANCE, STARTING_BALANCE, 0.0, 1, now, now))
    conn.commit()
    conn.close()


def load_account():
    conn = get_connection()
    row = conn.execute("SELECT * FROM accounts WHERE account_id=?", (ACCOUNT_ID,)).fetchone()
    conn.close()
    return dict(row)


def load_position():
    conn = get_connection()
    row = conn.execute("SELECT * FROM positions WHERE account_id=?", (ACCOUNT_ID,)).fetchone()
    conn.close()
    return dict(row) if row else None


def save_account(cash_balance, realized_pnl):
    conn = get_connection()
    conn.execute("UPDATE accounts SET cash_balance=?, realized_pnl=?, updated_at=? WHERE account_id=?",
                 (float(cash_balance), float(realized_pnl), datetime.utcnow().isoformat(), ACCOUNT_ID))
    conn.commit()
    conn.close()


def save_position(position):
    conn = get_connection()
    if position is None:
        conn.execute("DELETE FROM positions WHERE account_id=?", (ACCOUNT_ID,))
    else:
        conn.execute("""INSERT INTO positions
            (account_id, ticker, asset_class, side, quantity, entry_price, entry_time, stop_loss, target, invested_amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_id) DO UPDATE SET
                ticker=excluded.ticker, asset_class=excluded.asset_class, side=excluded.side,
                quantity=excluded.quantity, entry_price=excluded.entry_price, entry_time=excluded.entry_time,
                stop_loss=excluded.stop_loss, target=excluded.target, invested_amount=excluded.invested_amount""",
            (ACCOUNT_ID, position["ticker"], position["asset_class"], position["side"], float(position["quantity"]),
             float(position["entry_price"]), str(position["entry_time"]), position.get("stop_loss"),
             position.get("target"), float(position["invested_amount"])))
    conn.commit()
    conn.close()


def save_trade(ticker, asset_class, side, quantity, entry_price, exit_price, entry_time, exit_time, realized_pnl, reason):
    account = load_account()
    conn = get_connection()
    conn.execute("""INSERT INTO trades
        (account_id, cycle_number, ticker, asset_class, side, quantity, entry_price, exit_price, entry_time, exit_time, realized_pnl, fees, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ACCOUNT_ID, account["cycle_number"], ticker, asset_class, side, float(quantity), float(entry_price),
         float(exit_price), str(entry_time), str(exit_time), float(realized_pnl), 0.0, reason))
    conn.commit()
    conn.close()


def save_account_event(event_type, message):
    account = load_account()
    conn = get_connection()
    conn.execute("INSERT INTO account_events (account_id, cycle_number, event_type, message, created_at) VALUES (?, ?, ?, ?, ?)",
                 (ACCOUNT_ID, account["cycle_number"], event_type, message, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def reset_account_cycle():
    account = load_account()
    conn = get_connection()
    next_cycle = int(account["cycle_number"]) + 1
    conn.execute("""UPDATE accounts SET starting_balance=?, cash_balance=?, realized_pnl=?, cycle_number=?, updated_at=? WHERE account_id=?""",
                 (STARTING_BALANCE, STARTING_BALANCE, 0.0, next_cycle, datetime.utcnow().isoformat(), ACCOUNT_ID))
    conn.execute("DELETE FROM positions WHERE account_id=?", (ACCOUNT_ID,))
    conn.commit()
    conn.close()
    save_account_event("ACCOUNT_RESET", f"Cycle {account['cycle_number']} depleted. New cycle funded with ${STARTING_BALANCE:.2f}.")


initialize_database()


def detect_asset_class(ticker):
    t = ticker.upper().strip()
    if t.endswith("=F"):
        return "futures"
    if t.endswith("=X"):
        return "forex"
    if t.endswith("-USD"):
        return "crypto"
    return "stocks"


def detect_session_mode(ticker):
    return {"futures": "CME Futures", "forex": "Forex 24/5", "crypto": "Crypto 24/7"}.get(
        detect_asset_class(ticker), "US Stocks / ETFs - Regular Hours"
    )


def normalize_session_mode(ticker, selected):
    return detect_session_mode(ticker) if selected == "Auto Detect" else selected


def is_continuous_mode(mode):
    return mode in {"CME Futures", "Forex 24/5", "Crypto 24/7"}


def nth_weekday(year, month, weekday, n):
    first = date(year, month, 1)
    return first + timedelta(days=((weekday - first.weekday()) % 7) + (n - 1) * 7)


def last_weekday(year, month, weekday):
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    current = nxt - timedelta(days=1)
    return current - timedelta(days=(current.weekday() - weekday) % 7)


def observed_fixed_holiday(year, month, day):
    actual = date(year, month, day)
    if actual.weekday() == 5:
        return actual - timedelta(days=1)
    if actual.weekday() == 6:
        return actual + timedelta(days=1)
    return actual


def calculate_easter(year):
    a, b, c = year % 19, year // 100, year % 100
    d, e, f = b // 4, b % 4, (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def us_market_holidays(year):
    h = {
        observed_fixed_holiday(year, 1, 1), nth_weekday(year, 1, 0, 3), nth_weekday(year, 2, 0, 3),
        calculate_easter(year) - timedelta(days=2), last_weekday(year, 5, 0), observed_fixed_holiday(year, 7, 4),
        nth_weekday(year, 9, 0, 1), nth_weekday(year, 11, 3, 4), observed_fixed_holiday(year, 12, 25),
    }
    if year >= 2022:
        h.add(observed_fixed_holiday(year, 6, 19))
    return h


def is_us_market_holiday(d):
    d = to_pydate(d)
    return d in us_market_holidays(d.year)


def is_early_close(d):
    d = to_pydate(d)
    y = d.year
    return d in {date(y, 7, 3), nth_weekday(y, 11, 3, 4) + timedelta(days=1), date(y, 12, 24)}


def futures_session_date(ts):
    d = pd.Timestamp(ts).date()
    return d + timedelta(days=1) if pd.Timestamp(ts).time() >= time(18, 0) else d


def find_column(df, names):
    m = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in m:
            return m[n.lower()]
    return None


def normalize_ohlcv(raw, intraday=True):
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]
    df = df.reset_index()
    o = find_column(df, ["Open"])
    h = find_column(df, ["High"])
    l = find_column(df, ["Low"])
    c = find_column(df, ["Close", "Adj Close"])
    v = find_column(df, ["Volume"])
    if None in (o, h, l, c):
        return pd.DataFrame()
    out = pd.DataFrame({"timestamp": df[df.columns[0]], "open": df[o], "high": df[h], "low": df[l], "close": df[c], "volume": df[v] if v else 0.0})
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    try:
        if out["timestamp"].dt.tz is not None:
            out["timestamp"] = out["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    except Exception:
        pass
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["volume"] = out["volume"].fillna(0.0)
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    valid = (out["open"] > 0) & (out["high"] > 0) & (out["low"] > 0) & (out["close"] > 0) & (out["high"] >= out["low"]) & (out["high"] >= out["open"]) & (out["high"] >= out["close"]) & (out["low"] <= out["open"]) & (out["low"] <= out["close"])
    out = out[valid].copy()
    if out.empty:
        return pd.DataFrame()
    out["bar_ratio"] = out["high"] / out["low"]
    out = out[out["bar_ratio"] <= 3].copy()
    if out.empty:
        return pd.DataFrame()
    ref = out["close"].rolling(80, min_periods=10).median().shift(1).fillna(out["close"].expanding(min_periods=1).median()).replace(0, np.nan)
    ratio = out["close"] / ref
    out = out[(ratio >= (0.10 if intraday else 0.02)) & (ratio <= (10 if intraday else 50))].copy()
    if out.empty:
        return pd.DataFrame()
    out = out.drop(columns=["bar_ratio"], errors="ignore")
    out["calendar_date"] = out["timestamp"].dt.date
    return out.reset_index(drop=True)


def apply_session_filter(df, mode):
    if df.empty:
        return df
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if mode == "US Stocks / ETFs - Regular Hours":
        out = out[out["timestamp"].dt.weekday < 5]
        out = out[~out["calendar_date"].apply(is_us_market_holiday)]
        minutes = out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute
        close_m = np.where(out["calendar_date"].apply(is_early_close), 13 * 60, 16 * 60)
        out = out[(minutes >= 9 * 60 + 30) & (minutes <= close_m)].copy()
        out["session_date"] = out["calendar_date"].map(to_pydate)
    elif mode == "US Stocks / ETFs - Extended Hours":
        out = out[out["timestamp"].dt.weekday < 5]
        out = out[~out["calendar_date"].apply(is_us_market_holiday)].copy()
        out["session_date"] = out["calendar_date"].map(to_pydate)
    elif mode == "CME Futures":
        wd, t = out["timestamp"].dt.weekday, out["timestamp"].dt.time
        out = out[(wd < 5) | ((wd == 6) & (t >= time(18, 0)))].copy()
        t = out["timestamp"].dt.time
        out = out[~((t >= time(17, 0)) & (t < time(18, 0)))].copy()
        out["session_date"] = out["timestamp"].apply(futures_session_date)
    elif mode == "Forex 24/5":
        wd, t = out["timestamp"].dt.weekday, out["timestamp"].dt.time
        out = out[((wd < 5) | ((wd == 6) & (t >= time(17, 0)))) & ~((wd == 4) & (t >= time(17, 0)))].copy()
        out["session_date"] = out["calendar_date"].map(to_pydate)
    else:
        out["session_date"] = out["calendar_date"].map(to_pydate)
    out["session_date"] = out["session_date"].map(to_pydate)
    return out.sort_values("timestamp").reset_index(drop=True)


def apply_custom_session(df, start_t, end_t, weekends):
    if df.empty:
        return df
    out = df.copy()
    if not weekends:
        out = out[out["timestamp"].dt.weekday < 5].copy()
    minutes = out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute
    s, e = start_t.hour * 60 + start_t.minute, end_t.hour * 60 + end_t.minute
    out = out[(minutes >= s) & (minutes <= e)].copy() if s <= e else out[(minutes >= s) | (minutes <= e)].copy()
    out["session_date"] = out["calendar_date"].map(to_pydate)
    return out.reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_history(ticker, timeframe):
    cfg = TIMEFRAME_CONFIG[timeframe]
    for period in [cfg["period"], cfg["fallback_period"]]:
        try:
            raw = yf.download(ticker, period=period, interval=cfg["yf_interval"], auto_adjust=True, progress=False, prepost=True, threads=False)
            df = normalize_ohlcv(raw, cfg["intraday"])
            if not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def moving_average(series, length, kind):
    return series.ewm(span=int(length), adjust=False).mean() if kind == "EMA" else series.rolling(int(length)).mean()


def get_context_dataframe(revealed, session_date, preset):
    if revealed.empty:
        return revealed.copy()
    dates = sorted([d for d in revealed["session_date"].map(to_pydate).unique() if d is not None])
    session_date = to_pydate(session_date)
    if session_date not in dates:
        return revealed.copy()
    prev = {"Today": 0, "1 Previous Day": 1, "5 Previous Days": 5, "20 Previous Days": 20, "All Available History": 999999}
    start = dates[max(0, dates.index(session_date) - prev.get(preset, 0))]
    return revealed[revealed["session_date"].map(to_pydate) >= start].copy()


def build_chart_dataframe(revealed, session_date, preset, continuous, follow_on, follow_bars):
    if revealed.empty:
        return revealed.copy().reset_index(drop=True)
    if continuous:
        # Never cut at 18:00. Keep trailing trading bars (maintenance already removed).
        n = max(int(follow_bars), 50) + 40
        if follow_on:
            return revealed.tail(n).copy().reset_index(drop=True)
        ctx = get_context_dataframe(revealed, session_date, preset)
        tail = revealed.tail(n)
        combined = pd.concat([ctx, tail], ignore_index=True)
        return combined.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return get_context_dataframe(revealed, session_date, preset).reset_index(drop=True)


def get_price_axis_range(df):
    if df.empty:
        return None
    vals = []
    for col in ["low", "high", "fast_ma", "slow_ma"]:
        if col in df.columns:
            vals.extend(pd.to_numeric(df[col], errors="coerce").dropna().tolist())
    if not vals:
        return None
    low, high = float(min(vals)), float(max(vals))
    med = float(pd.to_numeric(df["close"], errors="coerce").median())
    spread = max(high - low, max(med * 0.003, 0.10))
    pad = max(spread * 0.12, med * 0.001)
    return [low - pad, high + pad]


def get_volume_axis_range(df):
    if df.empty or df["volume"].max() <= 0:
        return None
    v = df["volume"].dropna()
    return [0, max(float(v.quantile(0.95)) * 1.15, float(v.median()) * 2, 1.0)]


def disable_rangesliders(fig):
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=2, col=1)


def plotly_chart_safe(fig, **kwargs):
    try:
        return st.plotly_chart(fig, width="stretch", **kwargs)
    except TypeError:
        return st.plotly_chart(fig, use_container_width=True, **kwargs)


def _raw_swings(df):
    if df is None or len(df) < SWING_K * 2 + 1:
        return []
    highs, lows, n = df["high"].values, df["low"].values, len(df)
    min_size = max(float(df["close"].iloc[-1]) * MIN_SWING_PCT, 1e-9)
    raw = []
    for i in range(SWING_K, n - SWING_K):
        hw, lw = highs[i - SWING_K:i + SWING_K + 1], lows[i - SWING_K:i + SWING_K + 1]
        if highs[i] == hw.max() and highs[i] - lw.min() >= min_size:
            raw.append((i, float(highs[i]), "H"))
        if lows[i] == lw.min() and hw.max() - lows[i] >= min_size:
            raw.append((i, float(lows[i]), "L"))
    raw.sort(key=lambda x: x[0])
    cleaned = []
    for s in raw:
        if cleaned and cleaned[-1][2] == s[2]:
            p = cleaned[-1]
            if (s[2] == "H" and s[1] >= p[1]) or (s[2] == "L" and s[1] <= p[1]):
                cleaned[-1] = s
        else:
            cleaned.append(s)
    labeled, ph, pl = [], None, None
    for i, px, t in cleaned:
        if t == "H":
            lab = "H" if ph is None else ("HH" if px > ph else "LH")
            ph = px
        else:
            lab = "L" if pl is None else ("HL" if px > pl else "LL")
            pl = px
        labeled.append({"i": i, "price": px, "type": t, "label": lab})
    return labeled


def build_zones(labeled, ref):
    zones = []
    for s in labeled:
        placed = False
        for z in zones:
            if abs(s["price"] - z["mid"]) / max(ref, 1e-9) < ZONE_TOL:
                z["prices"].append(s["price"])
                z["mid"] = float(np.mean(z["prices"]))
                z["touches"] += 1
                z["low_touches"] += 1 if s["type"] == "L" else 0
                z["high_touches"] += 1 if s["type"] == "H" else 0
                placed = True
                break
        if not placed:
            zones.append({"mid": s["price"], "prices": [s["price"]], "touches": 1, "low_touches": int(s["type"] == "L"), "high_touches": int(s["type"] == "H")})
    out = [z for z in zones if z["touches"] >= MIN_DRAW_TOUCHES]
    for z in out:
        z["min_px"], z["max_px"] = min(z["prices"]), max(z["prices"])
    return out


def detect_bos_choch(df, labeled):
    bos, choch = [], []
    last_hh = last_hl = last_lh = last_ll = None
    bias, ptr = "NEUTRAL", 0
    for i in range(len(df)):
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
        close = float(df["close"].iloc[i])
        if last_hh is not None and close > last_hh:
            if bias != "BULL":
                bos.append({"i": i, "dir": "up"})
                bias = "BULL"
            last_hh = None
            continue
        if last_ll is not None and close < last_ll:
            if bias != "BEAR":
                bos.append({"i": i, "dir": "down"})
                bias = "BEAR"
            last_ll = None
            continue
        if CHOCH_ENABLE:
            if bias == "BULL" and last_hl is not None and close < last_hl:
                choch.append({"i": i, "dir": "down"})
                last_hl = None
                bias = "NEUTRAL"
            elif bias == "BEAR" and last_lh is not None and close > last_lh:
                choch.append({"i": i, "dir": "up"})
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


def badge(fig, x, y, text, pal, yshift=0, arrow=False):
    fig.add_annotation(
        x=x, y=y, row=1, col=1, text=f"<b>{text}</b>", showarrow=arrow, arrowhead=2, arrowsize=1,
        arrowwidth=1.2, arrowcolor="#e5e7eb", yshift=yshift, font=dict(size=10, color=pal["fg"]),
        bgcolor=pal["bg"], bordercolor="#e5e7eb", borderwidth=1, borderpad=3,
    )


def add_marker(ts, price, kind):
    st.session_state.markers.append({"timestamp": pd.Timestamp(ts), "price": float(price), "kind": kind})


def maybe_reset_depleted_account():
    if load_position() is not None:
        return
    if float(load_account()["cash_balance"]) <= 0:
        reset_account_cycle()
        st.toast("Account depleted. New $1,000 cycle started.", icon="⚠️")


def open_position(ticker, price, ts, side, amount, sl, tp):
    acc = load_account()
    qty = float(amount) / float(price)
    cash = float(acc["cash_balance"]) - float(amount) if side == "long" else float(acc["cash_balance"]) + float(amount)
    save_account(cash, float(acc["realized_pnl"]))
    save_position({
        "ticker": ticker, "asset_class": detect_asset_class(ticker), "side": side, "quantity": qty,
        "entry_price": float(price), "entry_time": str(ts), "stop_loss": sl, "target": tp, "invested_amount": float(amount),
    })


def close_position(ticker, price, ts, reason="", qty_close=None):
    pos = load_position()
    if pos is None:
        return
    acc = load_account()
    total = float(pos["quantity"])
    qty = total if qty_close is None else min(float(qty_close), total)
    entry, price = float(pos["entry_price"]), float(price)
    if pos["side"] == "long":
        realized = (price - entry) * qty
        cash = float(acc["cash_balance"]) + price * qty
    else:
        realized = (entry - price) * qty
        cash = float(acc["cash_balance"]) - price * qty
    save_account(cash, float(acc["realized_pnl"]) + realized)
    save_trade(ticker, pos["asset_class"], pos["side"], qty, entry, price, pos["entry_time"], str(ts), realized, reason)
    rem = total - qty
    if rem <= 1e-10:
        save_position(None)
    else:
        pos["quantity"] = rem
        save_position(pos)


def add_horizontal_line(price, label, color):
    st.session_state.drawings.append({"type": "hline", "price": float(price), "label": label, "color": color})


def add_band(a, b):
    st.session_state.drawings.append({"type": "band", "y0": min(float(a), float(b)), "y1": max(float(a), float(b)), "color": "rgba(34,197,94,0.18)"})


def add_trend_line(x0, y0, x1, y1):
    st.session_state.drawings.append({"type": "trend", "x0": pd.Timestamp(x0), "y0": float(y0), "x1": pd.Timestamp(x1), "y1": float(y1), "color": "#f472b6"})


def extract_selected_points(ev):
    if ev is None:
        return []
    try:
        return [dict(p) for p in ev.selection.points]
    except Exception:
        try:
            return ev.get("selection", {}).get("points", [])
        except Exception:
            return []


def nearest_row_from_event(df, point):
    if df.empty:
        return None
    try:
        idx = int(round(float(point.get("x"))))
        if 0 <= idx < len(df):
            return df.iloc[idx]
    except Exception:
        pass
    return df.iloc[-1]


def process_drawing_click(mode, row, clicked_y=None):
    if row is None or mode == "None":
        return
    ts = row["timestamp"]
    try:
        anchor = float(clicked_y)
    except Exception:
        anchor = float(row["close"])
    if mode == "Line at High":
        add_horizontal_line(float(row["high"]), f"H {float(row['high']):.2f}", "#ef4444")
    elif mode == "Line at Close":
        add_horizontal_line(float(row["close"]), f"C {float(row['close']):.2f}", "#22d3ee")
    elif mode == "Line at Low":
        add_horizontal_line(float(row["low"]), f"L {float(row['low']):.2f}", "#22c55e")
    elif mode == "Band":
        st.session_state.draw_clicks.append({"timestamp": ts, "price": anchor})
        if len(st.session_state.draw_clicks) >= 2:
            add_band(st.session_state.draw_clicks[0]["price"], st.session_state.draw_clicks[1]["price"])
            st.session_state.draw_clicks = []
    elif mode == "Trend line":
        st.session_state.draw_clicks.append({"timestamp": ts, "price": anchor})
        if len(st.session_state.draw_clicks) >= 2:
            a, b = st.session_state.draw_clicks[0], st.session_state.draw_clicks[1]
            add_trend_line(a["timestamp"], a["price"], b["timestamp"], b["price"])
            st.session_state.draw_clicks = []


def advance_bars(n):
    df = st.session_state.df
    if df.empty:
        return
    ticker = st.session_state.active_ticker
    max_step = len(df) - 1
    for _ in range(int(n)):
        if st.session_state.step >= max_step:
            st.toast("End of available history reached.", icon="🔔")
            break
        prev = df.iloc[st.session_state.step]
        st.session_state.step += 1
        row = df.iloc[st.session_state.step]
        pos = load_position()
        if pos is not None and (not st.session_state.active_carry_mode) and prev["session_date"] != row["session_date"]:
            close_position(ticker, float(prev["close"]), prev["timestamp"], "(SESSION CLOSE)")
            add_marker(prev["timestamp"], float(prev["close"]), "SELL" if pos["side"] == "long" else "COVER")
            pos = None
        if pos is not None:
            sl, tp, side = pos["stop_loss"], pos["target"], pos["side"]
            if side == "long":
                if sl is not None and float(row["low"]) <= float(sl):
                    px = min(float(sl), float(row["open"]))
                    close_position(ticker, px, row["timestamp"], "(STOP-LOSS)")
                    add_marker(row["timestamp"], px, "SELL")
                    break
                if tp is not None and float(row["high"]) >= float(tp):
                    px = max(float(tp), float(row["open"]))
                    close_position(ticker, px, row["timestamp"], "(TARGET HIT)")
                    add_marker(row["timestamp"], px, "SELL")
                    break
            else:
                if sl is not None and float(row["high"]) >= float(sl):
                    px = max(float(sl), float(row["open"]))
                    close_position(ticker, px, row["timestamp"], "(STOP-LOSS)")
                    add_marker(row["timestamp"], px, "COVER")
                    break
                if tp is not None and float(row["low"]) <= float(tp):
                    px = min(float(tp), float(row["open"]))
                    close_position(ticker, px, row["timestamp"], "(TARGET HIT)")
                    add_marker(row["timestamp"], px, "COVER")
                    break
    maybe_reset_depleted_account()


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Setup")
ticker = st.sidebar.text_input("Ticker", value="NQ=F").upper().strip()
timeframe = st.sidebar.selectbox("Chart Timeframe", TIMEFRAME_OPTIONS, index=TIMEFRAME_OPTIONS.index("5m"))
selected_session_mode = st.sidebar.selectbox("Session Mode", SESSION_OPTIONS, index=0)
effective_session_mode = normalize_session_mode(ticker, selected_session_mode)
st.sidebar.info(f"Detected/applied session: **{effective_session_mode}**")

custom_start, custom_end, custom_weekends = time(9, 30), time(16, 0), False
if effective_session_mode == "Custom Session":
    custom_start = st.sidebar.time_input("Custom session start", value=time(9, 30))
    custom_end = st.sidebar.time_input("Custom session end", value=time(16, 0))
    custom_weekends = st.sidebar.checkbox("Include weekends", value=False)

st.sidebar.caption(f"Yahoo native interval: `{TIMEFRAME_CONFIG[timeframe]['yf_interval']}`")

source_df = pd.DataFrame()
filtered_df = pd.DataFrame()
can_start = False
practice_date = None
selected_start_timestamp = None

if ticker:
    with st.sidebar:
        with st.spinner(f"Loading {timeframe} data for {ticker}..."):
            source_df = fetch_history(ticker, timeframe)

st.sidebar.subheader("📅 Practice Date & Time")

if source_df.empty:
    st.sidebar.error("No data returned. Date/time selectors appear after Yahoo data loads.")
    st.sidebar.date_input("Practice Session", value=date.today(), disabled=True)
    st.sidebar.selectbox("Replay Start Time", ["--"], disabled=True)
else:
    filtered_df = apply_session_filter(source_df, effective_session_mode)
    if effective_session_mode == "Custom Session":
        filtered_df = apply_custom_session(filtered_df, custom_start, custom_end, custom_weekends)
    if filtered_df.empty:
        st.sidebar.error("No bars matched this session. Try another session mode.")
        st.sidebar.date_input("Practice Session", value=date.today(), disabled=True)
        st.sidebar.selectbox("Replay Start Time", ["--"], disabled=True)
    else:
        available_dates = sorted({to_pydate(d) for d in filtered_df["session_date"].unique() if to_pydate(d) is not None})
        first_date, last_date = available_dates[0], available_dates[-1]
        default_date = available_dates[-3] if len(available_dates) >= 3 else available_dates[-1]
        st.sidebar.success(f"{len(filtered_df):,} session bars\n{first_date} → {last_date}")
        date_key = f"practice_date_{ticker}_{timeframe}_{effective_session_mode}"
        if to_pydate(st.session_state.get(date_key)) not in available_dates:
            st.session_state[date_key] = default_date
        practice_date = to_pydate(st.sidebar.date_input("Practice Session", min_value=first_date, max_value=last_date, key=date_key))
        practice_rows = filtered_df[filtered_df["session_date"].map(to_pydate) == practice_date].copy()
        if practice_rows.empty:
            st.sidebar.warning("No bars on this session date.")
            st.sidebar.selectbox("Replay Start Time", ["--"], disabled=True)
        elif TIMEFRAME_CONFIG[timeframe]["intraday"]:
            start_labels = practice_rows["timestamp"].dt.strftime("%H:%M").tolist()
            start_key = f"start_time_{ticker}_{timeframe}_{practice_date}"
            selected_label = st.sidebar.selectbox("Replay Start Time", start_labels, index=min(10, len(start_labels) - 1), key=start_key)
            selected_start_timestamp = practice_rows[practice_rows["timestamp"].dt.strftime("%H:%M") == selected_label]["timestamp"].iloc[0]
            can_start = True
        else:
            selected_start_timestamp = practice_rows["timestamp"].iloc[0]
            st.sidebar.caption("Daily chart — replay starts at the session open.")
            can_start = True

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Position Behavior")
carry_positions = st.sidebar.checkbox("Carry positions across sessions", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Moving Averages")
show_ma = st.sidebar.checkbox("Show moving averages", value=True)
ma_type = st.sidebar.radio("MA Type", ["EMA", "SMA"], horizontal=True)
dfast, dslow = TIMEFRAME_CONFIG[timeframe]["ema"]
fast_length = st.sidebar.number_input("Fast MA length", 2, 500, int(dfast), 1)
slow_length = st.sidebar.number_input("Slow MA length", 2, 500, int(dslow), 1)
show_volume_ma = st.sidebar.checkbox("Show volume MA", value=True)
volume_ma_length = st.sidebar.number_input("Volume MA length", 2, 200, 20, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("🏗️ Structure")
show_struct = st.sidebar.checkbox("Show structure labels", value=False)
show_zones = st.sidebar.checkbox("Show S/R zones", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("📷 Chart Context")
context_preset = st.sidebar.selectbox("Context preset", CONTEXT_PRESETS, index=2)
follow_replay = st.sidebar.checkbox("Follow replay candle", value=True)
follow_bars = st.sidebar.radio("Follow window (futures/forex/crypto)", FOLLOW_WINDOW_OPTIONS, index=1, horizontal=True)
st.sidebar.caption(
    "Continuous markets: Follow shows the last 100 or 200 trading bars, including across 16:59 → 18:00. "
    "The 17:00–18:00 break is skipped and not counted. "
    "Stocks: Follow frames the current regular session. "
    "Follow OFF uses the context preset."
)

st.sidebar.markdown("---")
st.sidebar.subheader("✏️ Drawings")
draw_mode = st.sidebar.selectbox("Drawing mode", DRAW_MODES, 0)
if draw_mode != st.session_state.last_draw_mode:
    st.session_state.last_draw_mode = draw_mode
    st.session_state.last_draw_signature = None
    st.session_state.draw_clicks = []
if st.sidebar.button("Undo last drawing") and st.session_state.drawings:
    st.session_state.drawings.pop()
if st.sidebar.button("Clear all drawings"):
    st.session_state.drawings = []
    st.session_state.draw_clicks = []

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Start / Reset Replay", disabled=not can_start):
    active = filtered_df.copy().reset_index(drop=True)
    matching = active.index[active["timestamp"] == selected_start_timestamp].tolist()
    if matching:
        st.session_state.df = active
        st.session_state.step = matching[0]
        st.session_state.active_ticker = ticker
        st.session_state.active_interval = timeframe
        st.session_state.active_session_mode = effective_session_mode
        st.session_state.active_practice_date = practice_date
        st.session_state.active_carry_mode = carry_positions
        st.session_state.markers = []
        st.session_state.drawings = []
        st.session_state.draw_clicks = []
        st.session_state.last_draw_signature = None
        st.session_state.camera_revision += 1
        st.session_state.force_camera = True
        st.session_state.sim_active = True
        st.rerun()

st.title("💹 Market Replay Simulator")
if not st.session_state.sim_active or st.session_state.df.empty:
    st.info("Use the sidebar: ticker → session → Practice Session → Replay Start Time → Start / Reset Replay.")
    st.stop()

dataframe = st.session_state.df
step = min(st.session_state.step, len(dataframe) - 1)
st.session_state.step = step
revealed = dataframe.iloc[:step + 1].copy()
cur = dataframe.iloc[step]
current_price = float(cur["close"])
current_ts = pd.Timestamp(cur["timestamp"])
current_session = to_pydate(cur["session_date"])
active_mode = st.session_state.active_session_mode
continuous = is_continuous_mode(active_mode)

calc = revealed.copy()
if show_ma:
    calc["fast_ma"] = moving_average(calc["close"], fast_length, ma_type)
    calc["slow_ma"] = moving_average(calc["close"], slow_length, ma_type)
if show_volume_ma:
    calc["volume_ma"] = calc["volume"].rolling(int(volume_ma_length)).mean()

chart_df = build_chart_dataframe(calc, current_session, context_preset, continuous, follow_replay, follow_bars)
if chart_df.empty:
    chart_df = calc.reset_index(drop=True)
chart_df["x_index"] = np.arange(len(chart_df))
hits = chart_df.index[chart_df["timestamp"] == current_ts].tolist()
current_x = hits[-1] if hits else len(chart_df) - 1

acc = load_account()
pos = load_position()
cash = float(acc["cash_balance"])
realized = float(acc["realized_pnl"])
signed = 0.0
if pos is None:
    pos_val = unrl = 0.0
    pos_text, pos_type = "—", "FLAT"
else:
    q = float(pos["quantity"])
    if pos["side"] == "long":
        signed, pos_val, unrl, pos_type = q, q * current_price, (current_price - float(pos["entry_price"])) * q, "🟢 LONG"
    else:
        signed, pos_val, unrl, pos_type = -q, q * current_price, (float(pos["entry_price"]) - current_price) * q, "🔴 SHORT"
    pos_text = f"{abs(signed):.4f} units"
equity = cash + signed * current_price

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Price", f"${current_price:.2f}")
m2.metric("Cash", f"${cash:.2f}")
m3.metric("Position Value", f"${pos_val:.2f}")
m4.metric("Equity", f"${equity:.2f}", f"${equity - float(acc['starting_balance']):.2f}")
m5.metric("Realized P&L", f"${realized:.2f}")
m6.metric("Unrealized P&L", f"${unrl:.2f}")
st.caption(f"Position ({pos_type}): {pos_text} • Cycle {acc['cycle_number']} • {active_mode}")

vol_colors = np.where(chart_df["close"] >= chart_df["open"], "rgba(38,166,154,0.58)", "rgba(239,83,80,0.58)")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.78, 0.22])
fig.add_trace(go.Candlestick(
    x=chart_df["x_index"], open=chart_df["open"], high=chart_df["high"], low=chart_df["low"], close=chart_df["close"],
    name=st.session_state.active_ticker,
    increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
    decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"),
    customdata=np.column_stack([chart_df["timestamp"].astype(str), chart_df["session_date"].astype(str)]),
    hovertemplate="Time: %{customdata[0]}<br>Session: %{customdata[1]}<br>O:%{open:.2f} H:%{high:.2f} L:%{low:.2f} C:%{close:.2f}<extra></extra>",
), row=1, col=1)
fig.add_trace(go.Bar(x=chart_df["x_index"], y=chart_df["volume"], marker_color=vol_colors, name="Volume"), row=2, col=1)
if show_ma and "fast_ma" in chart_df.columns:
    fig.add_trace(go.Scatter(x=chart_df["x_index"], y=chart_df["fast_ma"], name=f"{ma_type}{int(fast_length)}", line=dict(color=C_FAST_EMA, width=1.7)), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df["x_index"], y=chart_df["slow_ma"], name=f"{ma_type}{int(slow_length)}", line=dict(color=C_SLOW_EMA, width=1.7)), row=1, col=1)
if show_volume_ma and "volume_ma" in chart_df.columns:
    fig.add_trace(go.Scatter(x=chart_df["x_index"], y=chart_df["volume_ma"], name=f"VolMA {int(volume_ma_length)}", line=dict(color=C_VOL_MA, width=1.7)), row=2, col=1)

if show_struct or show_zones:
    sdf = chart_df.tail(MAX_STRUCTURE_BARS).copy().reset_index(drop=True)
    labeled = _raw_swings(sdf)
    confirmed = [s for s in labeled if s["i"] + SWING_K <= len(sdf) - 1]
else:
    sdf, labeled, confirmed = pd.DataFrame(), [], []

if show_struct and not sdf.empty:
    bos, choch = detect_bos_choch(sdf, labeled)
    for s in confirmed[-MAX_SWING_LABELS:]:
        pal = C_HH if s["label"] in ("HH", "HL") else C_LH if s["label"] in ("LH", "LL") else C_H
        badge(fig, sdf["x_index"].iloc[s["i"]], s["price"], s["label"], pal, 18 if s["type"] == "H" else -18)
    for e in bos[-MAX_BOS_LABELS:]:
        if 0 <= e["i"] < len(sdf):
            x = sdf["x_index"].iloc[e["i"]]
            badge(fig, x, float(sdf["high" if e["dir"] == "up" else "low"].iloc[e["i"]]),
                  "BOS↑" if e["dir"] == "up" else "BOS↓",
                  C_BOS_UP if e["dir"] == "up" else C_BOS_DN, 22 if e["dir"] == "up" else -22, True)
    for e in choch[-MAX_CHOCH_LABELS:]:
        if 0 <= e["i"] < len(sdf):
            x = sdf["x_index"].iloc[e["i"]]
            badge(fig, x, float(sdf["high" if e["dir"] == "up" else "low"].iloc[e["i"]]),
                  "CHoCH↑" if e["dir"] == "up" else "CHoCH↓",
                  C_CH_UP if e["dir"] == "up" else C_CH_DN, 22 if e["dir"] == "up" else -22, True)

if show_zones and confirmed:
    noise = float((sdf["high"] - sdf["low"]).tail(20).mean() or 0.01)
    zones = sorted(build_zones(confirmed[-MAX_ZONE_SWINGS:], current_price), key=lambda z: (abs(z["mid"] - current_price), -z["touches"]))[:MAX_ZONES_DRAWN]
    rx = chart_df["x_index"].iloc[-1]
    for z in zones:
        role = zone_role(z, current_price, noise)
        color = C_FLOOR if role == "floor" else C_CEIL if role == "ceiling" else C_BOTH if role == "both" else C_BROKEN
        tag = {"floor": "FLOOR", "ceiling": "CEIL", "both": "BOTH"}.get(role, role.replace("_", " ").upper())
        fig.add_hline(y=z["mid"], row=1, col=1, line=dict(color=color, width=min(1 + z["touches"], 4), dash="dot"))
        fig.add_annotation(x=rx, y=z["mid"], row=1, col=1, xanchor="left",
                           text=f"<b> {z['mid']:.2f} {tag} {z['low_touches']}L/{z['high_touches']}H</b>",
                           showarrow=False, font=dict(size=10, color="#fff"), bgcolor=color)

if pos is not None:
    if pos["stop_loss"] is not None:
        fig.add_hline(y=float(pos["stop_loss"]), row=1, col=1, line=dict(color="#fb923c", width=2, dash="dash"))
    if pos["target"] is not None:
        fig.add_hline(y=float(pos["target"]), row=1, col=1, line=dict(color="#4ade80", width=2, dash="dash"))
    fig.add_hline(y=float(pos["entry_price"]), row=1, col=1, line=dict(color="#22d3ee", width=1.3, dash="dot"))

styles = {"LONG": ("triangle-up", "#22c55e"), "SHORT": ("triangle-down", "#ef4444"), "SELL": ("circle", "#f97316"), "COVER": ("circle", "#38bdf8")}
for m in st.session_state.markers:
    ix = chart_df.index[chart_df["timestamp"] == m["timestamp"]].tolist()
    if not ix:
        continue
    sym, col = styles.get(m["kind"], ("circle", "#fff"))
    fig.add_trace(go.Scatter(x=[ix[-1]], y=[m["price"]], mode="markers", showlegend=False,
                             marker=dict(symbol=sym, size=12, color=col, line=dict(color="#fff", width=1))), row=1, col=1)

for d in st.session_state.drawings:
    if d["type"] == "hline":
        fig.add_hline(y=d["price"], row=1, col=1, line=dict(color=d.get("color", "#22d3ee"), width=1.8))
    elif d["type"] == "band":
        fig.add_hrect(y0=d["y0"], y1=d["y1"], row=1, col=1, fillcolor=d.get("color", "rgba(34,197,94,0.18)"), line_width=1, line_color="#22c55e", layer="below")
    elif d["type"] == "trend":
        x0 = chart_df.index[chart_df["timestamp"] == d["x0"]].tolist()
        x1 = chart_df.index[chart_df["timestamp"] == d["x1"]].tolist()
        if x0 and x1:
            fig.add_trace(go.Scatter(x=[x0[-1], x1[-1]], y=[d["y0"], d["y1"]], mode="lines", showlegend=False, line=dict(color="#f472b6", width=2)), row=1, col=1)

if follow_replay and continuous:
    start_idx = max(0, current_x - (int(follow_bars) - 1))
    end_idx = min(len(chart_df) - 1, current_x + 6)
    follow_title = f"Follow: {int(follow_bars)} bars"
elif follow_replay:
    day = chart_df[chart_df["session_date"].map(to_pydate) == current_session]
    start_idx = int(day["x_index"].iloc[0]) if not day.empty else 0
    end_idx = min(len(chart_df) - 1, int(day["x_index"].iloc[-1]) + 8) if not day.empty else len(chart_df) - 1
    follow_title = "Follow: current session"
else:
    start_idx, end_idx = 0, max(0, len(chart_df) - 1)
    follow_title = "Follow: OFF"

visible = chart_df.iloc[start_idx:end_idx + 1]
pr, vr = get_price_axis_range(visible), get_volume_axis_range(visible)
ticks = chart_df["x_index"].tolist()
if len(ticks) > 12:
    ticks = ticks[:: max(1, len(ticks) // 12)]
tick_text = [pd.Timestamp(chart_df.iloc[int(i)]["timestamp"]).strftime("%m-%d\n%H:%M") for i in ticks]

cam_sig = (context_preset, follow_replay, follow_bars, timeframe, current_session, active_mode, len(chart_df))
if st.session_state.camera_signature != cam_sig:
    st.session_state.camera_signature = cam_sig
    st.session_state.camera_revision += 1
    st.session_state.force_camera = True
apply_camera = follow_replay or st.session_state.force_camera

fig.update_layout(
    template="plotly_dark", height=640, dragmode="pan", paper_bgcolor="#000", plot_bgcolor="#000",
    font=dict(color="#e5e7eb"),
    title=f"{st.session_state.active_ticker} | {st.session_state.active_interval} | {current_ts.strftime('%Y-%m-%d %H:%M')} | {follow_title}",
    margin=dict(l=10, r=190, t=42, b=10), showlegend=True, hovermode="x",
    uirevision=f"follow-{st.session_state.camera_revision}-{step}" if follow_replay else f"manual-{st.session_state.camera_revision}",
    xaxis_rangeslider_visible=False,
)
disable_rangesliders(fig)
for r in (1, 2):
    fig.update_xaxes(type="linear", tickmode="array", tickvals=ticks, ticktext=tick_text, gridcolor="#1f2937",
                     rangeslider_visible=False, showspikes=True, spikemode="across", row=r, col=1)
fig.update_yaxes(title_text="Price", gridcolor="#1f2937", row=1, col=1)
fig.update_yaxes(title_text="Volume", gridcolor="#1f2937", row=2, col=1)
if apply_camera:
    xr = [start_idx - 0.5, max(start_idx + 1, end_idx) + 0.5]
    fig.update_xaxes(range=xr, row=1, col=1)
    fig.update_xaxes(range=xr, row=2, col=1)
    if pr:
        fig.update_yaxes(range=pr, autorange=False, row=1, col=1)
    if vr:
        fig.update_yaxes(range=vr, autorange=False, row=2, col=1)
    if not follow_replay:
        st.session_state.force_camera = False
else:
    fig.update_yaxes(autorange=True, row=1, col=1)
    fig.update_yaxes(autorange=True, row=2, col=1)
disable_rangesliders(fig)

left, right = st.columns([6, 1])
with left:
    ev = None
    if draw_mode == "None":
        plotly_chart_safe(fig, key="chart_view", config={"scrollZoom": True, "displaylogo": False})
    else:
        try:
            ev = plotly_chart_safe(fig, key="chart_draw", on_select="rerun", selection_mode="points", config={"scrollZoom": True, "displaylogo": False})
        except TypeError:
            plotly_chart_safe(fig, key="chart_draw_fb", config={"scrollZoom": True, "displaylogo": False})
        pts = extract_selected_points(ev)
        if pts:
            pt = pts[-1]
            row = nearest_row_from_event(chart_df, pt)
            sig = (draw_mode, str(row["timestamp"]) if row is not None else "", str(pt.get("y")))
            if sig != st.session_state.last_draw_signature:
                st.session_state.last_draw_signature = sig
                process_drawing_click(draw_mode, row, pt.get("y"))
                st.rerun()

with right:
    st.markdown("#### ⏱️")
    st.caption(current_ts.strftime("%H:%M"))
    mins = TIMEFRAME_CONFIG[st.session_state.active_interval]["minutes"]
    btns = [("+1d", 1), ("+3d", 3), ("+5d", 5), ("+10d", 10)] if mins >= 1440 else [
        ("▶️ +5m", max(1, 5 // mins)), ("▶️ +15m", max(1, 15 // mins)),
        ("⏭️ +30m", max(1, 30 // mins)), ("⏭️ +60m", max(1, 60 // mins)),
    ]
    for lab, n in btns:
        if st.button(lab):
            advance_bars(n)
            st.rerun()
    st.markdown("---")
    if signed > 0:
        st.success("🟢 LONG")
    elif signed < 0:
        st.error("🔴 SHORT")
    else:
        st.info("FLAT")
    st.caption(f"Session: {current_session}")
    st.caption(f"Visible: {len(visible):,}")
    st.caption(f"Rendered: {len(chart_df):,}")
    st.caption(f"Window: {int(follow_bars)} bars" if continuous else "Window: session")

if pos is None:
    st.markdown("### 🎮 Open a Position")
    a, b, c, d, e = st.columns([1.2, 1.2, 1.2, 1, 1])
    with a:
        amt = st.number_input("Trade Amount ($)", 10.0, max(10.0, cash), min(100.0, max(10.0, cash)), 10.0)
    with b:
        sl_in = st.number_input("Stop-Loss ($)", value=round(current_price * 0.99, 2), step=0.01)
    with c:
        tp_in = st.number_input("Target ($, 0 = none)", value=0.0, step=0.01)
    with d:
        st.write(""); st.write("")
        if st.button("🟢 BUY (Long)"):
            if sl_in >= current_price:
                st.error("Long stop must be below price.")
            elif tp_in != 0 and tp_in <= current_price:
                st.error("Long target must be above price.")
            else:
                open_position(st.session_state.active_ticker, current_price, current_ts, "long", amt, float(sl_in), float(tp_in) if tp_in > 0 else None)
                add_marker(current_ts, current_price, "LONG")
                st.rerun()
    with e:
        st.write(""); st.write("")
        if st.button("🔻 SHORT"):
            if sl_in <= current_price:
                st.error("Short stop must be above price.")
            elif tp_in != 0 and tp_in >= current_price:
                st.error("Short target must be below price.")
            else:
                open_position(st.session_state.active_ticker, current_price, current_ts, "short", amt, float(sl_in), float(tp_in) if tp_in > 0 else None)
                add_marker(current_ts, current_price, "SHORT")
                st.rerun()
else:
    st.markdown(f"### 🎮 Manage {'LONG' if pos['side']=='long' else 'SHORT'} Position")
    a, b, c, d, e = st.columns([1.2, 1.2, 1.2, 1, 1])
    with a:
        nsl = st.number_input("Modify Stop-Loss", value=float(pos["stop_loss"] if pos["stop_loss"] is not None else current_price), step=0.01, key="msl")
    with b:
        ntp = st.number_input("Modify Target", value=float(pos["target"] if pos["target"] is not None else 0.0), step=0.01, key="mtp")
    with c:
        cq = st.number_input("Quantity to close", 0.0, float(pos["quantity"]), float(pos["quantity"]), 0.0001)
    with d:
        st.write(""); st.write("")
        if st.button("💾 Update SL/TP"):
            ok = True
            if pos["side"] == "long" and nsl >= current_price:
                st.error("Long stop must be below price.")
                ok = False
            if pos["side"] == "short" and nsl <= current_price:
                st.error("Short stop must be above price.")
                ok = False
            if ok:
                pos["stop_loss"] = float(nsl)
                pos["target"] = float(ntp) if ntp > 0 else None
                save_position(pos)
                st.rerun()
    with e:
        st.write(""); st.write("")
        if st.button("🔴 SELL" if pos["side"] == "long" else "🟢 COVER") and cq > 0:
            close_position(st.session_state.active_ticker, current_price, current_ts, "Manual close", cq)
            add_marker(current_ts, current_price, "SELL" if pos["side"] == "long" else "COVER")
            maybe_reset_depleted_account()
            st.rerun()

with st.expander("📝 Persistent Trade History"):
    conn = get_connection()
    trades = pd.read_sql_query(
        "SELECT cycle_number, ticker, side, quantity, entry_price, exit_price, entry_time, exit_time, realized_pnl, reason FROM trades WHERE account_id=? ORDER BY id DESC",
        conn, params=(ACCOUNT_ID,),
    )
    conn.close()
    st.info("No closed trades.") if trades.empty else st.dataframe(trades, hide_index=True)

with st.expander("🔁 Account Cycles and Events"):
    conn = get_connection()
    events = pd.read_sql_query(
        "SELECT cycle_number, event_type, message, created_at FROM account_events WHERE account_id=? ORDER BY id DESC",
        conn, params=(ACCOUNT_ID,),
    )
    conn.close()
    st.info("No events.") if events.empty else st.dataframe(events, hide_index=True)

with st.expander("📘 Replay Information"):
    st.markdown(
        f"| Item | Value |\n|---|---|\n"
        f"| Ticker | `{st.session_state.active_ticker}` |\n"
        f"| Timeframe | `{st.session_state.active_interval}` |\n"
        f"| Session | `{active_mode}` |\n"
        f"| Time | `{current_ts}` |\n"
        f"| Follow | `{follow_title}` |\n"
        f"| Rendered bars | `{len(chart_df):,}` |\n"
        f"| Visible bars | `{len(visible):,}` |\n"
        f"| DB | `{os.path.abspath(DB_PATH)}` |\n"
    )
