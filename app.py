# ============================================================
# HUMAN MARKET REPLAY SIMULATOR
#
# - Native Yahoo timeframe data: 1m, 2m, 5m, 15m, 30m, 1h, 1d
# - Session-compressed x-axis (hides closed time, keeps price gaps)
# - Instrument session calendars + Auto Detect
# - Position carry / force-close at session end
# - Persistent SQLite paper account across restarts/tickers
# - Universal Notional Paper Account (fractional-share P&L)
# ============================================================

import sqlite3
from datetime import date, datetime, time as dt_time, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="Market Replay Simulator")

DB_PATH = Path(__file__).resolve().parent / "paper_trading.db"
STARTING_BALANCE = 1000.0

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

SESSION_MODE_OPTIONS = [
    "Auto Detect",
    "US Stocks / ETFs — Regular Hours",
    "US Stocks / ETFs — Extended Hours",
    "CME Equity Futures",
    "Forex 24/5",
    "Crypto 24/7",
    "Custom Session",
]

STOCKS_RTH = "US Stocks / ETFs — Regular Hours"
STOCKS_EXT = "US Stocks / ETFs — Extended Hours"
FUTURES = "CME Equity Futures"
FOREX = "Forex 24/5"
CRYPTO = "Crypto 24/7"
CUSTOM = "Custom Session"

# ============================================================
# PERFORMANCE SETTINGS
# ============================================================
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
    "balance": STARTING_BALANCE,
    "shares": 0.0,
    "stop_loss": None,
    "target": None,
    "entry_price": None,
    "entry_time": None,
    "position_ticker": None,
    "position_asset_class": None,
    "position_mark": None,
    "realized_pnl": 0.0,
    "cycle_number": 1,
    "account_created_at": None,
    "depletion_notice": None,
    "step": 0,
    "df": pd.DataFrame(),
    "sim_start_idx": 0,
    "active_ticker": None,
    "active_interval": None,
    "active_practice_date": None,
    "active_session_mode": None,
    "trade_log": [],
    "markers": [],
    "drawings": [],
    "draw_clicks": [],
    "last_draw_signature": None,
    "last_draw_mode": "None",
    "camera_signature": None,
    "camera_revision": 0,
    "force_camera": True,
    "account_loaded": False,
    "carry_positions": True,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# NYSE / SESSION CALENDAR
# ============================================================
def _nth_weekday(year, month, weekday, n):
    first = date(year, month, 1)
    adj = (weekday - first.weekday()) % 7
    return first + timedelta(days=adj + 7 * (n - 1))


def _last_weekday(year, month, weekday):
    if month == 12:
        last = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d):
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _easter(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


@lru_cache(maxsize=32)
def nyse_holidays(year):
    holidays = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    }
    if year >= 2021:
        holidays.add(_observed(date(year, 6, 19)))
    return holidays


@lru_cache(maxsize=32)
def nyse_early_closes(year):
    early = set()
    july3 = date(year, 7, 3)
    if july3.weekday() < 5:
        early.add(july3)
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    friday = thanksgiving + timedelta(days=1)
    if friday.weekday() < 5:
        early.add(friday)
    christmas_eve = date(year, 12, 24)
    if christmas_eve.weekday() < 5:
        early.add(christmas_eve)
    nye = date(year, 12, 31)
    if nye.weekday() < 5:
        early.add(nye)
    return early


def detect_session_mode(ticker):
    t = (ticker or "").upper().strip()
    if t.endswith("=F"):
        return FUTURES
    if t.endswith("=X"):
        return FOREX
    if (
        t.endswith("-USD")
        or t.endswith("-USDT")
        or t.endswith("-BTC")
        or t in {"BTC", "ETH", "BTCUSD", "ETHUSD", "SOL", "DOGE"}
    ):
        return CRYPTO
    return STOCKS_RTH


def resolve_session_mode(choice, ticker):
    if choice == "Auto Detect":
        return detect_session_mode(ticker)
    return choice


def asset_class_from_mode(mode):
    if mode in (STOCKS_RTH, STOCKS_EXT):
        return "stock"
    if mode == FUTURES:
        return "futures"
    if mode == FOREX:
        return "forex"
    if mode == CRYPTO:
        return "crypto"
    return "custom"


def _holiday_sets(df):
    years = sorted(set(pd.to_datetime(df["timestamp"]).dt.year.dropna().astype(int)))
    holidays = set()
    early = {}
    for year in years:
        holidays |= nyse_holidays(int(year))
        for d in nyse_early_closes(int(year)):
            early[d] = 13 * 60
    return holidays, early


def filter_session_bars(
    df,
    mode,
    is_intraday,
    custom_start=None,
    custom_end=None,
    custom_weekdays_only=True,
):
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    wd = ts.dt.weekday
    minutes = ts.dt.hour * 60 + ts.dt.minute
    cal = ts.dt.date
    holidays, early = _holiday_sets(out)
    holiday_mask = cal.isin(holidays)

    if not is_intraday:
        if mode == CRYPTO:
            return out.reset_index(drop=True)
        if mode in (STOCKS_RTH, STOCKS_EXT, CUSTOM):
            keep = (wd < 5) & (~holiday_mask)
            return out.loc[keep].reset_index(drop=True)
        if mode == FUTURES:
            keep = ~((wd == 5) | holiday_mask)
            return out.loc[keep].reset_index(drop=True)
        if mode == FOREX:
            keep = wd < 5
            return out.loc[keep].reset_index(drop=True)
        return out.reset_index(drop=True)

    if mode == CRYPTO:
        return out.reset_index(drop=True)

    if mode == STOCKS_RTH:
        start_m = 9 * 60 + 30
        end_m = pd.Series(16 * 60, index=out.index)
        for d, em in early.items():
            end_m.loc[cal == d] = em
        keep = (wd < 5) & (~holiday_mask) & (minutes >= start_m) & (minutes <= end_m)
        return out.loc[keep].reset_index(drop=True)

    if mode == STOCKS_EXT:
        start_m = 4 * 60
        end_m = pd.Series(20 * 60, index=out.index)
        for d, em in early.items():
            end_m.loc[cal == d] = 17 * 60
        keep = (wd < 5) & (~holiday_mask) & (minutes >= start_m) & (minutes <= end_m)
        return out.loc[keep].reset_index(drop=True)

    if mode == FUTURES:
        hm = ts.dt.hour + ts.dt.minute / 60.0
        closed = (
            (wd == 5)
            | ((wd == 6) & (hm < 18.0))
            | ((wd == 4) & (hm >= 17.0))
            | ((wd.isin([0, 1, 2, 3])) & (hm >= 17.0) & (hm < 18.0))
        )
        return out.loc[~closed].reset_index(drop=True)

    if mode == FOREX:
        hm = ts.dt.hour + ts.dt.minute / 60.0
        closed = (
            (wd == 5)
            | ((wd == 4) & (hm >= 17.0))
            | ((wd == 6) & (hm < 17.0))
        )
        return out.loc[~closed].reset_index(drop=True)

    if mode == CUSTOM:
        if custom_start is None or custom_end is None:
            return out.reset_index(drop=True)
        start_m = custom_start.hour * 60 + custom_start.minute
        end_m = custom_end.hour * 60 + custom_end.minute
        if end_m >= start_m:
            keep = (minutes >= start_m) & (minutes <= end_m)
        else:
            keep = (minutes >= start_m) | (minutes <= end_m)
        if custom_weekdays_only:
            keep = keep & (wd < 5) & (~holiday_mask)
        return out.loc[keep].reset_index(drop=True)

    return out.reset_index(drop=True)


def add_session_date(df, mode):
    if df is None or df.empty:
        out = df.copy() if df is not None else pd.DataFrame()
        if not out.empty:
            out["session_date"] = out["timestamp"].dt.date
        return out

    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])

    if mode == FUTURES:
        offset = (ts.dt.hour >= 18).astype(int)
        out["session_date"] = (ts + pd.to_timedelta(offset, unit="D")).dt.date
    elif mode == FOREX:
        offset = (ts.dt.hour >= 17).astype(int)
        out["session_date"] = (ts + pd.to_timedelta(offset, unit="D")).dt.date
    else:
        out["session_date"] = ts.dt.date

    return out


def apply_session_calendar(
    df,
    mode,
    is_intraday,
    custom_start=None,
    custom_end=None,
    custom_weekdays_only=True,
):
    out = filter_session_bars(
        df,
        mode,
        is_intraday,
        custom_start=custom_start,
        custom_end=custom_end,
        custom_weekdays_only=custom_weekdays_only,
    )
    out = add_session_date(out, mode)
    if out.empty:
        return out
    out["date_only"] = out["timestamp"].dt.date
    out["label"] = out["timestamp"].dt.strftime("%m-%d %H:%M")
    return out.reset_index(drop=True)


# ============================================================
# SQLITE ACCOUNT
# ============================================================
def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db_connect()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS account (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            cycle_number INTEGER NOT NULL,
            starting_balance REAL NOT NULL,
            cash_balance REAL NOT NULL,
            realized_pnl REAL NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS account_cycles (
            cycle_number INTEGER PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            starting_balance REAL NOT NULL,
            ending_equity REAL,
            realized_pnl REAL,
            reason TEXT
        );

        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_number INTEGER NOT NULL,
            ticker TEXT,
            asset_class TEXT,
            side TEXT,
            entry_time TEXT,
            exit_time TEXT,
            entry_price REAL,
            exit_price REAL,
            quantity REAL,
            realized_pnl REAL,
            fees REAL,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS open_position (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            ticker TEXT,
            asset_class TEXT,
            side TEXT,
            quantity REAL,
            entry_price REAL,
            stop_loss REAL,
            target REAL,
            entry_time TEXT,
            mark_price REAL
        );

        CREATE TABLE IF NOT EXISTS trade_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_number INTEGER NOT NULL,
            message TEXT NOT NULL
        );
        """
    )
    cur.execute("SELECT id FROM account WHERE id = 1")
    if cur.fetchone() is None:
        now = datetime.now().isoformat(timespec="seconds")
        cur.execute(
            """
            INSERT INTO account (
                id, cycle_number, starting_balance, cash_balance,
                realized_pnl, created_at, updated_at
            ) VALUES (1, 1, ?, ?, 0, ?, ?)
            """,
            (STARTING_BALANCE, STARTING_BALANCE, now, now),
        )
        cur.execute(
            """
            INSERT INTO account_cycles (
                cycle_number, started_at, starting_balance
            ) VALUES (1, ?, ?)
            """,
            (now, STARTING_BALANCE),
        )
    conn.commit()
    conn.close()


def persist_account():
    init_db()
    conn = db_connect()
    cur = conn.cursor()
    now = datetime.now().isoformat(timespec="seconds")
    created = st.session_state.account_created_at or now
    st.session_state.account_created_at = created
    cur.execute(
        """
        UPDATE account SET
            cycle_number = ?,
            starting_balance = ?,
            cash_balance = ?,
            realized_pnl = ?,
            created_at = ?,
            updated_at = ?
        WHERE id = 1
        """,
        (
            int(st.session_state.cycle_number),
            STARTING_BALANCE,
            float(st.session_state.balance),
            float(st.session_state.realized_pnl),
            created,
            now,
        ),
    )

    cur.execute("DELETE FROM open_position")
    pos = float(st.session_state.shares)
    if abs(pos) > 1e-10:
        side = "LONG" if pos > 0 else "SHORT"
        cur.execute(
            """
            INSERT INTO open_position (
                id, ticker, asset_class, side, quantity, entry_price,
                stop_loss, target, entry_time, mark_price
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                st.session_state.position_ticker,
                st.session_state.position_asset_class,
                side,
                abs(pos),
                st.session_state.entry_price,
                st.session_state.stop_loss,
                st.session_state.target,
                None
                if st.session_state.entry_time is None
                else str(pd.Timestamp(st.session_state.entry_time)),
                st.session_state.position_mark,
            ),
        )
    conn.commit()
    conn.close()


def persist_trade_log_message(message):
    init_db()
    conn = db_connect()
    conn.execute(
        "INSERT INTO trade_log (cycle_number, message) VALUES (?, ?)",
        (int(st.session_state.cycle_number), message),
    )
    conn.commit()
    conn.close()


def persist_closed_trade(side, qty, entry, exit_price, ts, reason, realized):
    init_db()
    conn = db_connect()
    conn.execute(
        """
        INSERT INTO trades (
            cycle_number, ticker, asset_class, side, entry_time, exit_time,
            entry_price, exit_price, quantity, realized_pnl, fees, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """,
        (
            int(st.session_state.cycle_number),
            st.session_state.position_ticker,
            st.session_state.position_asset_class,
            side,
            None
            if st.session_state.entry_time is None
            else str(pd.Timestamp(st.session_state.entry_time)),
            str(pd.Timestamp(ts)),
            None if entry is None else float(entry),
            float(exit_price),
            float(qty),
            float(realized),
            reason,
        ),
    )
    conn.commit()
    conn.close()


def load_account_into_state():
    init_db()
    conn = db_connect()
    acc = conn.execute("SELECT * FROM account WHERE id = 1").fetchone()
    pos = conn.execute("SELECT * FROM open_position WHERE id = 1").fetchone()
    logs = conn.execute(
        """
        SELECT message FROM trade_log
        WHERE cycle_number = ?
        ORDER BY id
        """,
        (acc["cycle_number"],),
    ).fetchall()
    conn.close()

    st.session_state.cycle_number = int(acc["cycle_number"])
    st.session_state.balance = float(acc["cash_balance"])
    st.session_state.realized_pnl = float(acc["realized_pnl"])
    st.session_state.account_created_at = acc["created_at"]
    st.session_state.trade_log = [row["message"] for row in logs]

    if pos is None:
        st.session_state.shares = 0.0
        st.session_state.stop_loss = None
        st.session_state.target = None
        st.session_state.entry_price = None
        st.session_state.entry_time = None
        st.session_state.position_ticker = None
        st.session_state.position_asset_class = None
        st.session_state.position_mark = None
    else:
        qty = float(pos["quantity"] or 0)
        if pos["side"] == "SHORT":
            qty = -qty
        st.session_state.shares = qty
        st.session_state.stop_loss = pos["stop_loss"]
        st.session_state.target = pos["target"]
        st.session_state.entry_price = pos["entry_price"]
        st.session_state.entry_time = (
            None if not pos["entry_time"] else pd.Timestamp(pos["entry_time"])
        )
        st.session_state.position_ticker = pos["ticker"]
        st.session_state.position_asset_class = pos["asset_class"]
        st.session_state.position_mark = pos["mark_price"]

    st.session_state.account_loaded = True


def load_cycle_history():
    init_db()
    conn = db_connect()
    rows = conn.execute(
        """
        SELECT * FROM account_cycles
        ORDER BY cycle_number DESC
        LIMIT 12
        """
    ).fetchall()
    conn.close()
    return rows


def log_trade(message):
    st.session_state.trade_log.append(message)
    persist_trade_log_message(message)


def start_new_account_cycle(reason, ending_equity):
    init_db()
    now = datetime.now().isoformat(timespec="seconds")
    old_cycle = int(st.session_state.cycle_number)
    conn = db_connect()
    conn.execute(
        """
        UPDATE account_cycles
        SET ended_at = ?, ending_equity = ?, realized_pnl = ?, reason = ?
        WHERE cycle_number = ?
        """,
        (
            now,
            float(ending_equity),
            float(st.session_state.realized_pnl),
            reason,
            old_cycle,
        ),
    )
    new_cycle = old_cycle + 1
    conn.execute(
        """
        INSERT INTO account_cycles (cycle_number, started_at, starting_balance)
        VALUES (?, ?, ?)
        """,
        (new_cycle, now, STARTING_BALANCE),
    )
    conn.commit()
    conn.close()

    st.session_state.cycle_number = new_cycle
    st.session_state.balance = STARTING_BALANCE
    st.session_state.shares = 0.0
    st.session_state.stop_loss = None
    st.session_state.target = None
    st.session_state.entry_price = None
    st.session_state.entry_time = None
    st.session_state.position_ticker = None
    st.session_state.position_asset_class = None
    st.session_state.position_mark = None
    st.session_state.realized_pnl = 0.0
    st.session_state.account_created_at = now
    notice = (
        f"⚠️ Account depleted. Cycle {old_cycle} ended at ${ending_equity:.2f}. "
        f"New paper account funded with ${STARTING_BALANCE:.0f}."
    )
    st.session_state.depletion_notice = notice
    log_trade(notice)
    persist_account()


def mark_price_for_position(fallback_price):
    pos = float(st.session_state.shares)
    if abs(pos) < 1e-10:
        return float(fallback_price)
    if (
        st.session_state.position_ticker
        and st.session_state.active_ticker
        and st.session_state.position_ticker == st.session_state.active_ticker
    ):
        return float(fallback_price)
    if st.session_state.position_mark is not None:
        return float(st.session_state.position_mark)
    if st.session_state.entry_price is not None:
        return float(st.session_state.entry_price)
    return float(fallback_price)


def compute_equity(mark):
    return float(st.session_state.balance) + float(st.session_state.shares) * float(mark)


def check_depletion(mark, timestamp):
    pos = float(st.session_state.shares)
    equity = compute_equity(mark)
    if equity <= 0 and abs(pos) > 1e-10:
        close_position(
            abs(pos),
            max(float(mark), 0.0),
            timestamp,
            "(account liquidation)",
            skip_depletion=True,
        )
        equity = float(st.session_state.balance)
    if abs(float(st.session_state.shares)) < 1e-10 and equity <= 0:
        start_new_account_cycle("depleted", equity)


init_db()
if not st.session_state.account_loaded:
    load_account_into_state()


# ============================================================
# DATA FUNCTIONS
# ============================================================
def find_column(df, possible_names):
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def normalize_ohlcv(raw, intraday=True):
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()

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

    valid = (
        (out["open"] > 0)
        & (out["high"] > 0)
        & (out["low"] > 0)
        & (out["close"] > 0)
        & (out["high"] >= out["low"])
        & (out["high"] >= out["open"])
        & (out["high"] >= out["close"])
        & (out["low"] <= out["open"])
        & (out["low"] <= out["close"])
    )
    out = out[valid].copy()
    if out.empty:
        return pd.DataFrame()

    out["bar_ratio"] = out["high"] / out["low"]
    out = out[out["bar_ratio"] <= 3.0].copy()
    if out.empty:
        return pd.DataFrame()

    rolling_med = out["close"].rolling(80, min_periods=10).median().shift(1)
    expanding_med = out["close"].expanding(min_periods=1).median()
    reference = rolling_med.fillna(expanding_med).replace(0, np.nan)
    price_ratio = out["close"] / reference

    if intraday:
        out = out[(price_ratio >= 0.10) & (price_ratio <= 10.0)].copy()
    else:
        out = out[(price_ratio >= 0.02) & (price_ratio <= 50.0)].copy()

    if out.empty:
        return pd.DataFrame()

    out = out.drop(columns=["bar_ratio"], errors="ignore")
    out["date_only"] = out["timestamp"].dt.date
    out["label"] = out["timestamp"].dt.strftime("%m-%d %H:%M")
    return out.reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_native_history(ticker, timeframe, prepost):
    cfg = TIMEFRAME_CONFIG[timeframe]
    periods_to_try = [cfg["period"], cfg["fallback_period"]]

    for period in periods_to_try:
        try:
            raw = yf.download(
                ticker,
                period=period,
                interval=cfg["yf_interval"],
                auto_adjust=True,
                progress=False,
                prepost=bool(prepost),
                threads=False,
            )
            df = normalize_ohlcv(raw, intraday=cfg["intraday"])
            if not df.empty:
                return df
        except Exception:
            continue

    return pd.DataFrame()


def moving_avg(series, length, kind="EMA"):
    if kind == "EMA":
        return series.ewm(span=int(length), adjust=False).mean()
    return series.rolling(int(length)).mean()


def get_session_rows(df, session_date):
    return df[df["session_date"] == session_date].copy()


def get_context_render_df(revealed_df, session_date, preset):
    if revealed_df.empty:
        return revealed_df.copy()

    unique_dates = sorted(revealed_df["session_date"].unique())
    if session_date not in unique_dates:
        return revealed_df.copy()

    index = unique_dates.index(session_date)
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
    return revealed_df[revealed_df["session_date"] >= start_date].copy()


def get_price_axis_range(df):
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
    median_price = float(pd.to_numeric(df["close"], errors="coerce").median())
    spread = high - low
    min_spread = max(median_price * 0.003, 0.10)
    spread = max(spread, min_spread)
    padding = max(spread * 0.12, median_price * 0.001)
    return [low - padding, high + padding]


def get_volume_axis_range(df):
    if df.empty or df["volume"].max() <= 0:
        return None
    volume_values = df["volume"].dropna()
    cap = max(
        float(volume_values.quantile(0.95)) * 1.15,
        float(volume_values.median()) * 2.0,
        1.0,
    )
    return [0, cap]


def make_time_ticks(chart_df, intraday, n=12):
    if chart_df.empty:
        return [], []
    n = min(max(n, 2), len(chart_df))
    idxs = np.unique(np.linspace(0, len(chart_df) - 1, n, dtype=int))
    vals = chart_df["x"].iloc[idxs].tolist()
    if intraday:
        texts = [
            pd.Timestamp(ts).strftime("%m-%d\n%H:%M")
            for ts in chart_df["timestamp"].iloc[idxs]
        ]
    else:
        texts = [
            pd.Timestamp(ts).strftime("%Y-%m-%d")
            for ts in chart_df["timestamp"].iloc[idxs]
        ]
    return vals, texts


def timestamp_to_x(chart_df, timestamp):
    if chart_df.empty:
        return None
    clicked_time = pd.Timestamp(timestamp)
    nearest_idx = (chart_df["timestamp"] - clicked_time).abs().idxmin()
    return int(chart_df.loc[nearest_idx, "x"])


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
            label = "H" if previous_high is None else ("HH" if price > previous_high else "LH")
            previous_high = price
        else:
            label = "L" if previous_low is None else ("HL" if price > previous_low else "LL")
            previous_low = price
        labeled.append({"i": idx, "price": price, "type": swing_type, "label": label})
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

    output = [zone for zone in zones if zone["touches"] >= MIN_DRAW_TOUCHES]
    for zone in output:
        zone["min_px"] = min(zone["prices"])
        zone["max_px"] = max(zone["prices"])
    return output


def detect_bos_choch(df, labeled):
    bos = []
    choch = []
    last_hh = None
    last_hl = None
    last_lh = None
    last_ll = None
    bias = "NEUTRAL"
    pointer = 0

    for i in range(len(df)):
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

        if last_hh is not None and close > last_hh:
            if bias != "BULL":
                bos.append({"i": i, "price": last_hh, "dir": "up"})
                bias = "BULL"
            last_hh = None
            continue

        if last_ll is not None and close < last_ll:
            if bias != "BEAR":
                bos.append({"i": i, "price": last_ll, "dir": "down"})
                bias = "BEAR"
            last_ll = None
            continue

        if CHOCH_ENABLE:
            if bias == "BULL" and last_hl is not None and close < last_hl:
                choch.append({"i": i, "price": last_hl, "dir": "down"})
                last_hl = None
                bias = "NEUTRAL"
            elif bias == "BEAR" and last_lh is not None and close > last_lh:
                choch.append({"i": i, "price": last_lh, "dir": "up"})
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
    if not zones:
        return []
    ranked = sorted(
        zones,
        key=lambda zone: (abs(zone["mid"] - current_price), -zone["touches"]),
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
        font=dict(size=10, color=palette["fg"], family="Arial"),
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


def close_position(qty, price, timestamp, reason="", skip_depletion=False):
    pos = float(st.session_state.shares)
    timestamp = pd.Timestamp(timestamp)
    time_text = timestamp.strftime("%Y-%m-%d %H:%M")
    entry = st.session_state.entry_price
    realized = 0.0
    side = "LONG" if pos > 0 else "SHORT"

    if pos > 0:
        qty = min(float(qty), pos)
        if entry is not None:
            realized = qty * (float(price) - float(entry))
        st.session_state.realized_pnl += realized
        st.session_state.balance += qty * float(price)
        st.session_state.shares -= qty
        log_trade(
            f"{time_text}: SOLD {qty:.4f} sh at ${price:.2f} "
            f"{reason}  P&L ${realized:.2f}"
        )
        add_marker(timestamp, price, "SELL")
        persist_closed_trade(side, qty, entry, price, timestamp, reason, realized)

    elif pos < 0:
        qty = min(float(qty), abs(pos))
        if entry is not None:
            realized = qty * (float(entry) - float(price))
        st.session_state.realized_pnl += realized
        st.session_state.balance -= qty * float(price)
        st.session_state.shares += qty
        log_trade(
            f"{time_text}: COVERED {qty:.4f} sh at ${price:.2f} "
            f"{reason}  P&L ${realized:.2f}"
        )
        add_marker(timestamp, price, "COVER")
        persist_closed_trade(side, qty, entry, price, timestamp, reason, realized)

    if abs(float(st.session_state.shares)) < 1e-10:
        st.session_state.shares = 0.0
        st.session_state.stop_loss = None
        st.session_state.target = None
        st.session_state.entry_price = None
        st.session_state.entry_time = None
        st.session_state.position_ticker = None
        st.session_state.position_asset_class = None
        st.session_state.position_mark = None

    persist_account()
    if not skip_depletion:
        check_depletion(price, timestamp)


def position_is_on_active_ticker():
    pos = float(st.session_state.shares)
    if abs(pos) < 1e-10:
        return False
    return st.session_state.position_ticker == st.session_state.active_ticker


def advance_bars(number_of_bars):
    df = st.session_state.df
    max_step = len(df) - 1
    carry = bool(st.session_state.carry_positions)

    for _ in range(number_of_bars):
        if st.session_state.step >= max_step:
            st.toast("Replay session complete.", icon="🔔")
            break

        current_idx = st.session_state.step
        next_idx = current_idx + 1
        current_session = df.iloc[current_idx]["session_date"]
        next_session = df.iloc[next_idx]["session_date"]

        if (
            (not carry)
            and position_is_on_active_ticker()
            and current_session != next_session
        ):
            row = df.iloc[current_idx]
            close_position(
                abs(float(st.session_state.shares)),
                float(row["close"]),
                row["timestamp"],
                "(session end flatten)",
            )
            st.toast("Session ended — position closed.", icon="🔔")

        st.session_state.step += 1
        row = df.iloc[st.session_state.step]
        timestamp = row["timestamp"]

        if not position_is_on_active_ticker():
            continue

        pos = float(st.session_state.shares)
        sl = st.session_state.stop_loss
        tp = st.session_state.target

        if pos > 0:
            if sl is not None and float(row["low"]) <= float(sl):
                execution_price = min(float(sl), float(row["open"]))
                close_position(pos, execution_price, timestamp, "(🛑 STOP-LOSS)")
                st.toast(f"🛑 Long stopped at ${execution_price:.2f}", icon="💥")
                break
            if tp is not None and float(row["high"]) >= float(tp):
                execution_price = max(float(tp), float(row["open"]))
                close_position(pos, execution_price, timestamp, "(🎯 TARGET HIT)")
                st.toast(f"🎯 Long target at ${execution_price:.2f}", icon="🎉")
                break

        elif pos < 0:
            if sl is not None and float(row["high"]) >= float(sl):
                execution_price = max(float(sl), float(row["open"]))
                close_position(abs(pos), execution_price, timestamp, "(🛑 STOP-LOSS)")
                st.toast(f"🛑 Short stopped at ${execution_price:.2f}", icon="💥")
                break
            if tp is not None and float(row["low"]) <= float(tp):
                execution_price = min(float(tp), float(row["open"]))
                close_position(abs(pos), execution_price, timestamp, "(🎯 TARGET HIT)")
                st.toast(f"🎯 Short target at ${execution_price:.2f}", icon="🎉")
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
    if x_value is not None and "x" in df.columns:
        try:
            xi = int(round(float(x_value)))
            matched = df[df["x"] == xi]
            if not matched.empty:
                return matched.iloc[0]
        except Exception:
            pass

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
        add_horizontal_line(price, label=f"H {price:.2f}", color="#ef4444")
        st.toast(f"High line added at {price:.2f}", icon="📌")
    elif mode == "Line at Close":
        price = float(row["close"])
        add_horizontal_line(price, label=f"C {price:.2f}", color="#22d3ee")
        st.toast(f"Close line added at {price:.2f}", icon="📌")
    elif mode == "Line at Low":
        price = float(row["low"])
        add_horizontal_line(price, label=f"L {price:.2f}", color="#22c55e")
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

ticker = st.sidebar.text_input("Ticker", value="TQQQ").upper().strip()

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

st.sidebar.markdown("---")
st.sidebar.subheader("🕒 Session Mode")

session_choice = st.sidebar.selectbox(
    "Session Mode",
    SESSION_MODE_OPTIONS,
    index=0,
)

resolved_session_mode = resolve_session_mode(session_choice, ticker)

if session_choice == "Auto Detect":
    st.sidebar.caption(f"Detected: **{resolved_session_mode}**")

custom_start = dt_time(9, 30)
custom_end = dt_time(16, 0)
custom_weekdays_only = True

if resolved_session_mode == CUSTOM:
    custom_start = st.sidebar.time_input("Custom start (ET)", dt_time(9, 30))
    custom_end = st.sidebar.time_input("Custom end (ET)", dt_time(16, 0))
    custom_weekdays_only = st.sidebar.checkbox("Weekdays only", value=True)

st.sidebar.caption(
    "Chart uses a **session-compressed** axis: closed hours/weekends "
    "are hidden. Price gaps still appear."
)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Account")
st.sidebar.caption("**Universal Notional Paper Account**")
st.sidebar.caption(
    "Every ticker uses fractional-share notional P&L. "
    "Futures/forex are NOT contract/lot realistic."
)

carry_positions = st.sidebar.checkbox(
    "Carry positions across sessions",
    value=bool(st.session_state.carry_positions),
)
st.session_state.carry_positions = carry_positions
if carry_positions:
    st.sidebar.caption("Positions stay open overnight / into the next session.")
else:
    st.sidebar.caption("Force-close at the last bar of each session.")

sidebar_mark = mark_price_for_position(st.session_state.position_mark or 0.0)
sidebar_equity = compute_equity(sidebar_mark if sidebar_mark else 0.0)
st.sidebar.write(f"Cycle **#{st.session_state.cycle_number}**")
st.sidebar.write(f"Cash: `${st.session_state.balance:,.2f}`")
st.sidebar.write(f"Realized P&L: `${st.session_state.realized_pnl:,.2f}`")
if abs(float(st.session_state.shares)) > 1e-10:
    st.sidebar.write(
        f"Open: `{st.session_state.position_ticker}` "
        f"{'LONG' if st.session_state.shares > 0 else 'SHORT'} "
        f"{abs(st.session_state.shares):.4f}"
    )
    st.sidebar.write(f"Equity ≈ `${sidebar_equity:,.2f}`")
else:
    st.sidebar.write(f"Equity: `${st.session_state.balance:,.2f}`")

confirm_wallet_reset = st.sidebar.checkbox("Confirm wallet reset")
if st.sidebar.button(
    "Reset wallet to $1,000",
    disabled=not confirm_wallet_reset,
    use_container_width=True,
):
    start_new_account_cycle("manual_reset", compute_equity(sidebar_mark or 0.0))
    st.rerun()

# ------------------------------------------------------------
# DATA PREVIEW
# ------------------------------------------------------------
source_df = pd.DataFrame()
practice_date = None
selected_start_timestamp = None
can_start = False
prepost = resolved_session_mode == STOCKS_EXT

if ticker:
    with st.sidebar:
        with st.spinner(f"Loading native {timeframe} data for {ticker}..."):
            raw_df = fetch_native_history(ticker, timeframe, prepost)
            source_df = apply_session_calendar(
                raw_df,
                resolved_session_mode,
                timeframe_cfg["intraday"],
                custom_start=custom_start,
                custom_end=custom_end,
                custom_weekdays_only=custom_weekdays_only,
            )

if source_df.empty:
    st.sidebar.error(
        f"No native {timeframe} session bars for {ticker}.\n\n"
        "Possible causes:\n"
        "- Yahoo temporary rate limit\n"
        "- invalid ticker\n"
        "- session filter removed all bars\n"
        "- timeframe unavailable"
    )
else:
    available_dates = sorted(source_df["session_date"].unique())
    first_date = available_dates[0]
    last_date = available_dates[-1]

    st.sidebar.success(
        f"{len(source_df):,} session bars\n"
        f"{first_date} → {last_date}"
    )
    st.sidebar.caption("Practice dates are **trading sessions**, not calendar midnights.")

    if len(available_dates) >= 3:
        default_practice_date = available_dates[-3]
    else:
        default_practice_date = available_dates[-1]

    practice_date_key = f"practice_date_{ticker}_{timeframe}_{resolved_session_mode}"
    if (
        practice_date_key not in st.session_state
        or st.session_state[practice_date_key] not in available_dates
    ):
        st.session_state[practice_date_key] = default_practice_date

    practice_date = st.sidebar.date_input(
        "Practice Session Date",
        min_value=first_date,
        max_value=last_date,
        key=practice_date_key,
    )

    practice_rows = get_session_rows(source_df, practice_date)

    if practice_rows.empty:
        st.sidebar.warning("No session bars on this date.")
    else:
        if timeframe_cfg["intraday"]:
            start_labels = practice_rows["timestamp"].dt.strftime("%m-%d %H:%M").tolist()
            default_index = min(10, len(start_labels) - 1)
            start_time_key = (
                f"start_time_{ticker}_{timeframe}_{practice_date}_{resolved_session_mode}"
            )
            selected_start_label = st.sidebar.selectbox(
                "Replay Start Time",
                start_labels,
                index=default_index,
                key=start_time_key,
            )
            selected_start_timestamp = practice_rows[
                practice_rows["timestamp"].dt.strftime("%m-%d %H:%M")
                == selected_start_label
            ]["timestamp"].iloc[0]
        else:
            selected_start_timestamp = practice_rows["timestamp"].iloc[0]
        can_start = True

# ------------------------------------------------------------
# EMA SETTINGS
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Moving Averages")

show_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
ma_type = st.sidebar.radio("MA Type", ["EMA", "SMA"], horizontal=True)
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

# ------------------------------------------------------------
# STRUCTURE SETTINGS
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🏗️ Structure")
show_struct = st.sidebar.checkbox("Show structure labels", value=False)
show_zones = st.sidebar.checkbox("Show S/R zones", value=False)

# ------------------------------------------------------------
# CONTEXT SETTINGS
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📷 Chart Context")
context_preset = st.sidebar.selectbox("Context preset", CONTEXT_PRESETS, index=0)
follow_replay = st.sidebar.checkbox("Follow replay candle", value=True)
st.sidebar.caption(
    "Follow ON: after advancing, the chart returns to current replay candles.\n\n"
    "Follow OFF: zoom/pan stays where you left it.\n\n"
    "Context days are trading sessions, packed with no overnight blanks."
)

# ------------------------------------------------------------
# DRAWING SETTINGS
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("✏️ Drawing Mode")
draw_mode = st.sidebar.selectbox("Mode", DRAW_MODES, index=0)

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
# START / RESET REPLAY (does not wipe wallet)
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Start / Reset Replay does **not** reset the wallet.")

if st.sidebar.button(
    "🚀 Start / Reset Replay",
    use_container_width=True,
    disabled=not can_start,
):
    active_df = source_df.copy().reset_index(drop=True)
    selected_index_list = active_df.index[
        active_df["timestamp"] == selected_start_timestamp
    ].tolist()
    day_indices = active_df.index[
        active_df["session_date"] == practice_date
    ].tolist()

    if not selected_index_list or not day_indices:
        st.sidebar.error("Could not initialize the replay.")
    else:
        st.session_state.df = active_df
        st.session_state.sim_start_idx = day_indices[0]
        st.session_state.step = selected_index_list[0]
        st.session_state.drawings = []
        st.session_state.draw_clicks = []
        st.session_state.last_draw_signature = None
        st.session_state.active_ticker = ticker
        st.session_state.active_interval = timeframe
        st.session_state.active_practice_date = practice_date
        st.session_state.active_session_mode = resolved_session_mode
        st.session_state.camera_revision += 1
        st.session_state.camera_signature = None
        st.session_state.force_camera = True
        st.session_state.sim_active = True

        st.session_state.markers = []
        if abs(float(st.session_state.shares)) > 1e-10:
            kind = "LONG" if st.session_state.shares > 0 else "SHORT"
            mark_ts = st.session_state.entry_time or selected_start_timestamp
            mark_px = st.session_state.entry_price or float(
                active_df.iloc[selected_index_list[0]]["close"]
            )
            add_marker(mark_ts, mark_px, kind)

        st.rerun()

# ============================================================
# MAIN APP
# ============================================================
st.title("💹 Market Replay Simulator (Human)")
st.caption(
    "Session-compressed replay • persistent notional paper account • "
    "native timeframes • EMAs • structure"
)

if st.session_state.depletion_notice:
    st.warning(st.session_state.depletion_notice)

if not st.session_state.sim_active or st.session_state.df.empty:
    st.info(
        "Choose ticker, session mode, native timeframe, practice session, "
        "and start time. Then click **Start / Reset Replay**."
    )
    with st.expander("📘 Account cycles"):
        for row in load_cycle_history():
            ended = row["ended_at"] or "open"
            ending = (
                "—" if row["ending_equity"] is None
                else f"${row['ending_equity']:.2f}"
            )
            st.text(
                f"Cycle {row['cycle_number']}: started {row['started_at']} "
                f"ended {ended} equity {ending} ({row['reason'] or 'active'})"
            )
    st.stop()

if (
    st.session_state.active_ticker != ticker
    or st.session_state.active_interval != timeframe
    or st.session_state.active_practice_date != practice_date
    or st.session_state.active_session_mode != resolved_session_mode
):
    st.warning(
        "Sidebar ticker/timeframe/date/session differs from active replay. "
        "Click **Start / Reset Replay** to apply changes."
    )

df_all = st.session_state.df
step = st.session_state.step
practice_day = st.session_state.active_practice_date
active_ticker = st.session_state.active_ticker
active_interval = st.session_state.active_interval
active_session_mode = st.session_state.active_session_mode

revealed_df = df_all.iloc[:step + 1].copy()
if revealed_df.empty:
    st.error("No revealed bars available.")
    st.stop()

current_row = df_all.iloc[step]
current_price = float(current_row["close"])
current_timestamp = pd.Timestamp(current_row["timestamp"])
current_time_text = current_timestamp.strftime("%Y-%m-%d %H:%M")
current_session_date = current_row["session_date"]

if position_is_on_active_ticker():
    st.session_state.position_mark = current_price
    persist_account()

calc_df = revealed_df.copy()
if show_ma:
    calc_df["fast_ma"] = moving_avg(calc_df["close"], fast_len, ma_type)
    calc_df["slow_ma"] = moving_avg(calc_df["close"], slow_len, ma_type)
if show_vol_ma:
    calc_df["vol_ma"] = calc_df["volume"].rolling(int(vol_ma_len)).mean()

chart_df = get_context_render_df(
    calc_df,
    current_session_date,
    context_preset,
).reset_index(drop=True)

if chart_df.empty:
    chart_df = calc_df.copy().reset_index(drop=True)

chart_df["x"] = np.arange(len(chart_df), dtype=int)

pos = float(st.session_state.shares)
mark = mark_price_for_position(current_price)
position_value = pos * mark
equity = float(st.session_state.balance) + position_value
unrealized = 0.0
if abs(pos) > 1e-10 and st.session_state.entry_price is not None:
    unrealized = pos * (mark - float(st.session_state.entry_price))

equity_delta = equity - STARTING_BALANCE
pnl_color = "normal" if equity_delta == 0 else ("inverse" if equity_delta > 0 else "off")

if pos > 0:
    pos_type = "🟢 LONG"
    pos_text = f"{pos:.4f} sh"
elif pos < 0:
    pos_type = "🔴 SHORT"
    pos_text = f"{abs(pos):.4f} sh"
else:
    pos_type = "FLAT"
    pos_text = "—"

camera_signature = (
    context_preset,
    follow_replay,
    active_interval,
    current_session_date,
    active_session_mode,
)
if st.session_state.camera_signature != camera_signature:
    st.session_state.camera_signature = camera_signature
    st.session_state.camera_revision += 1
    st.session_state.force_camera = True

apply_camera = follow_replay or st.session_state.force_camera

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Price", f"${current_price:.2f}")
m2.metric("Cash", f"${st.session_state.balance:.2f}")
m3.metric("Position Value", f"${abs(position_value):.2f}")
m4.metric("Equity", f"${equity:.2f}", f"${equity_delta:.2f}", delta_color=pnl_color)
m5.metric("Realized P&L", f"${st.session_state.realized_pnl:.2f}")
m6.metric("Unrealized P&L", f"${unrealized:.2f}")

p1, p2, p3, p4 = st.columns(4)
p1.metric(
    f"Position ({pos_type})",
    pos_text,
    (
        f"{st.session_state.position_ticker}  Entry ${st.session_state.entry_price:.2f}"
        if st.session_state.entry_price is not None
        else ""
    ),
)
p2.metric(
    "Stop-Loss",
    f"${st.session_state.stop_loss:.2f}" if st.session_state.stop_loss is not None else "None",
)
p3.metric(
    "Target",
    f"${st.session_state.target:.2f}" if st.session_state.target is not None else "None",
)
p4.metric("Account Cycle", f"#{st.session_state.cycle_number}")

if abs(pos) > 1e-10 and not position_is_on_active_ticker():
    st.info(
        f"Open position is on **{st.session_state.position_ticker}**, "
        f"not {active_ticker}. This chart will not stop/target that position. "
        "Close it at last mark below, or replay the position's ticker."
    )

# ============================================================
# BUILD CHART
# ============================================================
vol_colors = np.where(
    chart_df["close"] >= chart_df["open"],
    "rgba(38,166,154,0.58)",
    "rgba(239,83,80,0.58)",
)
time_labels = chart_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.78, 0.22],
)

fig.add_trace(
    go.Candlestick(
        x=chart_df["x"],
        open=chart_df["open"],
        high=chart_df["high"],
        low=chart_df["low"],
        close=chart_df["close"],
        name=active_ticker,
        increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"),
        customdata=time_labels,
        hovertemplate=(
            "Time: %{customdata}<br>"
            "Open: %{open:.4f}<br>"
            "High: %{high:.4f}<br>"
            "Low: %{low:.4f}<br>"
            "Close: %{close:.4f}<extra></extra>"
        ),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Bar(
        x=chart_df["x"],
        y=chart_df["volume"],
        marker_color=vol_colors,
        name="Volume",
        customdata=time_labels,
        hovertemplate="Time: %{customdata}<br>Volume: %{y:,.0f}<extra></extra>",
    ),
    row=2,
    col=1,
)

if show_ma and "fast_ma" in chart_df.columns:
    fig.add_trace(
        go.Scatter(
            x=chart_df["x"],
            y=chart_df["fast_ma"],
            name=f"{ma_type}{int(fast_len)} ({active_interval})",
            line=dict(color=C_FAST_EMA, width=1.7),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["x"],
            y=chart_df["slow_ma"],
            name=f"{ma_type}{int(slow_len)} ({active_interval})",
            line=dict(color=C_SLOW_EMA, width=1.7),
        ),
        row=1,
        col=1,
    )

if show_vol_ma and "vol_ma" in chart_df.columns:
    fig.add_trace(
        go.Scatter(
            x=chart_df["x"],
            y=chart_df["vol_ma"],
            name=f"VolMA({int(vol_ma_len)})",
            line=dict(color=C_VOL_MA, width=1.7),
        ),
        row=2,
        col=1,
    )

if show_struct or show_zones:
    structure_df = chart_df.tail(MAX_STRUCTURE_BARS).copy().reset_index(drop=True)
    labeled = _raw_swings(structure_df)
    confirmed_swings = [
        swing for swing in labeled if swing["i"] + SWING_K <= len(structure_df) - 1
    ]
else:
    structure_df = pd.DataFrame()
    labeled = []
    confirmed_swings = []

if show_struct and not structure_df.empty:
    bos, choch = detect_bos_choch(structure_df, labeled)
    swings_to_draw = confirmed_swings[-MAX_SWING_LABELS:]

    for swing in swings_to_draw:
        x = structure_df["x"].iloc[swing["i"]]
        if swing["label"] in ("HH", "HL"):
            palette = C_HH
            yshift = 18 if swing["type"] == "H" else -18
        elif swing["label"] in ("LH", "LL"):
            palette = C_LH
            yshift = 18 if swing["type"] == "H" else -18
        else:
            palette = C_H
            yshift = 18 if swing["type"] == "H" else -18
        badge(fig, x, swing["price"], swing["label"], palette, yshift=yshift)

    for event in bos[-MAX_BOS_LABELS:]:
        i = event["i"]
        if 0 <= i < len(structure_df):
            x = structure_df["x"].iloc[i]
            if event["dir"] == "up":
                badge(fig, x, float(structure_df["high"].iloc[i]), "BOS↑", C_BOS_UP, yshift=22, arrow=True)
            else:
                badge(fig, x, float(structure_df["low"].iloc[i]), "BOS↓", C_BOS_DN, yshift=-22, arrow=True)

    for event in choch[-MAX_CHOCH_LABELS:]:
        i = event["i"]
        if 0 <= i < len(structure_df):
            x = structure_df["x"].iloc[i]
            if event["dir"] == "up":
                badge(fig, x, float(structure_df["high"].iloc[i]), "CHoCH↑", C_CH_UP, yshift=22, arrow=True)
            else:
                badge(fig, x, float(structure_df["low"].iloc[i]), "CHoCH↓", C_CH_DN, yshift=-22, arrow=True)

if show_zones and confirmed_swings:
    noise = float((structure_df["high"] - structure_df["low"]).tail(20).mean() or 0.01)
    zones = build_zones(confirmed_swings[-MAX_ZONE_SWINGS:], current_price)
    zones = select_relevant_zones(zones, current_price)
    right_x = chart_df["x"].iloc[-1]

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
            line=dict(color=color, width=min(1 + zone["touches"], 4), dash="dot"),
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
            font=dict(size=10, color="#ffffff", family="Arial"),
            bgcolor=color,
            bordercolor="#e5e7eb",
            borderwidth=1,
            borderpad=3,
        )

if pos != 0 and position_is_on_active_ticker():
    if st.session_state.stop_loss is not None:
        fig.add_hline(
            y=float(st.session_state.stop_loss),
            row=1,
            col=1,
            line=dict(color="#fb923c", width=2, dash="dash"),
        )
    if st.session_state.target is not None:
        fig.add_hline(
            y=float(st.session_state.target),
            row=1,
            col=1,
            line=dict(color="#4ade80", width=2, dash="dash"),
        )
    if st.session_state.entry_price is not None:
        fig.add_hline(
            y=float(st.session_state.entry_price),
            row=1,
            col=1,
            line=dict(color="#22d3ee", width=1.3, dash="dot"),
        )

marker_styles = {
    "LONG": {"symbol": "triangle-up", "color": "#22c55e"},
    "SHORT": {"symbol": "triangle-down", "color": "#ef4444"},
    "SELL": {"symbol": "circle", "color": "#f97316"},
    "COVER": {"symbol": "circle", "color": "#38bdf8"},
}

for marker in st.session_state.markers:
    if marker["timestamp"] > current_timestamp:
        continue
    x_val = timestamp_to_x(chart_df, marker["timestamp"])
    if x_val is None:
        continue
    style = marker_styles.get(marker["kind"], {"symbol": "circle", "color": "#ffffff"})
    fig.add_trace(
        go.Scatter(
            x=[x_val],
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

for drawing in st.session_state.drawings:
    if drawing["type"] == "hline":
        fig.add_hline(
            y=drawing["price"],
            row=1,
            col=1,
            line=dict(color=drawing.get("color", "#22d3ee"), width=1.8),
        )
        fig.add_annotation(
            x=chart_df["x"].iloc[-1],
            y=drawing["price"],
            row=1,
            col=1,
            xanchor="left",
            text=drawing.get("label", ""),
            showarrow=False,
            font=dict(size=10, color=drawing.get("color", "#22d3ee")),
        )
    elif drawing["type"] == "band":
        fig.add_hrect(
            y0=drawing["y0"],
            y1=drawing["y1"],
            row=1,
            col=1,
            fillcolor=drawing.get("color", "rgba(34,197,94,0.18)"),
            line_width=1,
            line_color="#22c55e",
            layer="below",
        )
    elif drawing["type"] == "trend":
        x0 = timestamp_to_x(chart_df, drawing["x0"])
        x1 = timestamp_to_x(chart_df, drawing["x1"])
        if x0 is None or x1 is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[drawing["y0"], drawing["y1"]],
                mode="lines",
                showlegend=False,
                line=dict(color=drawing.get("color", "#f472b6"), width=2),
            ),
            row=1,
            col=1,
        )

today_df = chart_df[chart_df["session_date"] == current_session_date].copy()
if today_df.empty:
    today_df = chart_df.copy()

current_x = int(chart_df["x"].iloc[-1])
if follow_replay:
    focus_df = today_df
    camera_start = float(today_df["x"].iloc[0]) - 0.5
    camera_end = float(current_x + 3)
else:
    focus_df = chart_df
    camera_start = float(chart_df["x"].iloc[0]) - 0.5
    camera_end = float(current_x + 3)

focus_price_range = get_price_axis_range(focus_df)
focus_volume_range = get_volume_axis_range(focus_df)
tickvals, ticktext = make_time_ticks(
    chart_df,
    TIMEFRAME_CONFIG[active_interval]["intraday"],
)

if follow_replay:
    uirevision_value = f"follow-{st.session_state.camera_revision}-{step}"
else:
    uirevision_value = f"manual-{st.session_state.camera_revision}"

fig.update_layout(
    template="plotly_dark",
    height=640,
    dragmode="pan",
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    font=dict(color="#e5e7eb"),
    title=(
        f"{active_ticker} | Native {active_interval} | "
        f"{active_session_mode} | Replay: {current_time_text} | "
        f"Follow: {'ON' if follow_replay else 'OFF'} | "
        f"Compressed sessions"
    ),
    margin=dict(l=10, r=190, t=42, b=10),
    xaxis_rangeslider_visible=False,
    showlegend=True,
    legend=dict(bgcolor="rgba(0,0,0,0.45)", font=dict(color="#e5e7eb")),
    hovermode="x",
    uirevision=uirevision_value,
)

for row_num in (1, 2):
    fig.update_xaxes(
        type="linear",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
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

fig.update_yaxes(title_text="Price", gridcolor="#1f2937", linecolor="#4b5563", row=1, col=1)
fig.update_yaxes(title_text="Volume", gridcolor="#1f2937", linecolor="#4b5563", row=2, col=1)

if apply_camera:
    fig.update_xaxes(range=[camera_start, camera_end], row=1, col=1)
    fig.update_xaxes(range=[camera_start, camera_end], row=2, col=1)
    if focus_price_range is not None:
        fig.update_yaxes(range=focus_price_range, row=1, col=1)
    if focus_volume_range is not None:
        fig.update_yaxes(range=focus_volume_range, row=2, col=1)
    if not follow_replay:
        st.session_state.force_camera = False

# ============================================================
# CHART + RIGHT-SIDE CONTROLS
# ============================================================
chart_col, control_col = st.columns([6, 1])

with chart_col:
    chart_event = None
    if draw_mode == "None":
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="market_chart_view",
            config={"scrollZoom": True, "displaylogo": False},
        )
    else:
        try:
            chart_event = st.plotly_chart(
                fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="market_chart_draw",
                config={"scrollZoom": True, "displaylogo": False},
            )
        except TypeError:
            st.plotly_chart(
                fig,
                use_container_width=True,
                key="market_chart_draw_fallback",
                config={"scrollZoom": True, "displaylogo": False},
            )

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
                process_drawing_click(draw_mode, selected_row, clicked_y)
                st.rerun()

with control_col:
    st.markdown("#### ⏱️")
    st.caption(f"**{current_timestamp.strftime('%m-%d %H:%M')}**")
    st.caption(f"Native {active_interval}")
    st.caption(f"{active_session_mode}")

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
    st.caption(f"Carry:\n{'ON' if carry_positions else 'OFF'}")
    st.caption(f"Rendered bars:\n{len(chart_df):,}")
    st.caption(f"Drawings:\n{len(st.session_state.drawings)}")

# ============================================================
# MANUAL DRAWING FALLBACK
# ============================================================
with st.expander("✏️ Manual drawing helpers", expanded=False):
    st.caption("Use this if click-to-draw is unreliable in your browser/Streamlit version.")
    helper_df = chart_df.copy()
    helper_df["choice"] = helper_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    selected_choice = st.selectbox("Select candle", helper_df["choice"].tolist()[::-1])
    selected_row = helper_df[helper_df["choice"] == selected_choice].iloc[0]
    d1, d2, d3, d4, d5 = st.columns(5)

    with d1:
        if st.button("High line"):
            price = float(selected_row["high"])
            add_horizontal_line(price, label=f"H {price:.2f}", color="#ef4444")
            st.rerun()
    with d2:
        if st.button("Close line"):
            price = float(selected_row["close"])
            add_horizontal_line(price, label=f"C {price:.2f}", color="#22d3ee")
            st.rerun()
    with d3:
        if st.button("Low line"):
            price = float(selected_row["low"])
            add_horizontal_line(price, label=f"L {price:.2f}", color="#22c55e")
            st.rerun()
    with d4:
        if st.button("Band / Trend point"):
            use_mode = draw_mode if draw_mode in ("Band", "Trend line") else "Band"
            process_drawing_click(use_mode, selected_row, float(selected_row["close"]))
            st.rerun()
    with d5:
        if st.button("Clear pending"):
            st.session_state.draw_clicks = []
            st.session_state.last_draw_signature = None
            st.rerun()

# ============================================================
# TRADE ENTRY / MANAGEMENT
# ============================================================
same_ticker = position_is_on_active_
