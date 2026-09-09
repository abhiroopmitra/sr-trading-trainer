# ============================================================
# HUMAN MARKET REPLAY SIMULATOR
#
# FEATURES
# - Session-compressed x-axis
# - Auto/manual market session modes
# - US stocks/ETFs regular-hours sessions
# - Futures overnight sessions with maintenance breaks
# - Forex 24/5 sessions
# - Crypto 24/7 sessions
# - Session-aware replay dates
# - Persistent SQLite paper-trading account
# - Persistent trades and open positions
# - Cash, equity, realized P&L, unrealized P&L
# - Carry positions across sessions
# - Optional force-close-at-session-end mode
# - Account-cycle reset after true equity depletion
# - Native Yahoo timeframe data
# - EMA/SMA, volume, stops, targets, markers, drawings
#
# Recommended packages:
# pip install streamlit plotly pandas numpy yfinance
# ============================================================

import os
import sqlite3
import hashlib
from datetime import date, datetime, timedelta, time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    layout="wide",
    page_title="Market Replay Simulator",
)


# ============================================================
# CONSTANTS
# ============================================================

DB_PATH = "paper_trading.db"
ACCOUNT_ID = 1
STARTING_BALANCE = 1000.0

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

SESSION_OPTIONS = [
    "Auto Detect",
    "US Stocks / ETFs - Regular Hours",
    "US Stocks / ETFs - Extended Hours",
    "CME Futures",
    "Forex 24/5",
    "Crypto 24/7",
    "Custom Session",
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
# SESSION STATE
# ============================================================

DEFAULTS = {
    "sim_active": False,
    "df": pd.DataFrame(),
    "step": 0,
    "sim_start_idx": 0,
    "active_ticker": None,
    "active_interval": None,
    "active_session_mode": None,
    "active_practice_date": None,
    "active_carry_mode": True,
    "markers": [],
    "drawings": [],
    "draw_clicks": [],
    "last_draw_signature": None,
    "last_draw_mode": "None",
    "camera_signature": None,
    "camera_revision": 0,
    "force_camera": True,
    "account_message": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS accounts (
            account_id INTEGER PRIMARY KEY,
            starting_balance REAL NOT NULL,
            cash_balance REAL NOT NULL,
            realized_pnl REAL NOT NULL DEFAULT 0,
            cycle_number INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS positions (
            account_id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            asset_class TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            entry_price REAL NOT NULL,
            entry_time TEXT NOT NULL,
            stop_loss REAL,
            target REAL,
            invested_amount REAL NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            cycle_number INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            asset_class TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            realized_pnl REAL,
            fees REAL DEFAULT 0,
            reason TEXT DEFAULT ''
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS account_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            cycle_number INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        "SELECT account_id FROM accounts WHERE account_id = ?",
        (ACCOUNT_ID,),
    )

    if cursor.fetchone() is None:
        now = datetime.utcnow().isoformat()
        cursor.execute(
            """
            INSERT INTO accounts
            (account_id, starting_balance, cash_balance,
             realized_pnl, cycle_number, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ACCOUNT_ID,
                STARTING_BALANCE,
                STARTING_BALANCE,
                0.0,
                1,
                now,
                now,
            ),
        )

    connection.commit()
    connection.close()


def load_account():
    connection = get_connection()
    row = connection.execute(
        "SELECT * FROM accounts WHERE account_id = ?",
        (ACCOUNT_ID,),
    ).fetchone()
    connection.close()
    return dict(row)


def load_position():
    connection = get_connection()
    row = connection.execute(
        "SELECT * FROM positions WHERE account_id = ?",
        (ACCOUNT_ID,),
    ).fetchone()
    connection.close()
    return dict(row) if row else None


def save_account(cash_balance, realized_pnl):
    connection = get_connection()
    connection.execute(
        """
        UPDATE accounts
        SET cash_balance = ?,
            realized_pnl = ?,
            updated_at = ?
        WHERE account_id = ?
        """,
        (
            float(cash_balance),
            float(realized_pnl),
            datetime.utcnow().isoformat(),
            ACCOUNT_ID,
        ),
    )
    connection.commit()
    connection.close()


def save_position(position):
    connection = get_connection()

    if position is None:
        connection.execute(
            "DELETE FROM positions WHERE account_id = ?",
            (ACCOUNT_ID,),
        )

    else:
        connection.execute(
            """
            INSERT INTO positions
            (account_id, ticker, asset_class, side, quantity,
             entry_price, entry_time, stop_loss, target, invested_amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_id) DO UPDATE SET
                ticker = excluded.ticker,
                asset_class = excluded.asset_class,
                side = excluded.side,
                quantity = excluded.quantity,
                entry_price = excluded.entry_price,
                entry_time = excluded.entry_time,
                stop_loss = excluded.stop_loss,
                target = excluded.target,
                invested_amount = excluded.invested_amount
            """,
            (
                ACCOUNT_ID,
                position["ticker"],
                position["asset_class"],
                position["side"],
                float(position["quantity"]),
                float(position["entry_price"]),
                str(position["entry_time"]),
                position.get("stop_loss"),
                position.get("target"),
                float(position["invested_amount"]),
            ),
        )

    connection.commit()
    connection.close()


def save_trade(
    ticker,
    asset_class,
    side,
    quantity,
    entry_price,
    exit_price,
    entry_time,
    exit_time,
    realized_pnl,
    reason,
):
    account = load_account()
    connection = get_connection()

    connection.execute(
        """
        INSERT INTO trades
        (account_id, cycle_number, ticker, asset_class, side,
         quantity, entry_price, exit_price, entry_time, exit_time,
         realized_pnl, fees, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ACCOUNT_ID,
            account["cycle_number"],
            ticker,
            asset_class,
            side,
            float(quantity),
            float(entry_price),
            float(exit_price),
            str(entry_time),
            str(exit_time),
            float(realized_pnl),
            0.0,
            reason,
        ),
    )

    connection.commit()
    connection.close()


def save_account_event(event_type, message):
    account = load_account()
    connection = get_connection()

    connection.execute(
        """
        INSERT INTO account_events
        (account_id, cycle_number, event_type, message, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            ACCOUNT_ID,
            account["cycle_number"],
            event_type,
            message,
            datetime.utcnow().isoformat(),
        ),
    )

    connection.commit()
    connection.close()


def reset_account_cycle():
    connection = get_connection()
    account = load_account()
    now = datetime.utcnow().isoformat()
    next_cycle = int(account["cycle_number"]) + 1

    connection.execute(
        """
        UPDATE accounts
        SET starting_balance = ?,
            cash_balance = ?,
            realized_pnl = ?,
            cycle_number = ?,
            updated_at = ?
        WHERE account_id = ?
        """,
        (
            STARTING_BALANCE,
            STARTING_BALANCE,
            0.0,
            next_cycle,
            now,
            ACCOUNT_ID,
        ),
    )

    connection.execute(
        "DELETE FROM positions WHERE account_id = ?",
        (ACCOUNT_ID,),
    )

    connection.commit()
    connection.close()

    save_account_event(
        "ACCOUNT_RESET",
        (
            f"Account cycle {account['cycle_number']} ended after "
            f"equity depletion. New cycle funded with "
            f"${STARTING_BALANCE:.2f}."
        ),
    )


initialize_database()


# ============================================================
# ASSET / SESSION IDENTIFICATION
# ============================================================

def detect_asset_class(ticker):
    ticker = ticker.upper().strip()

    if ticker.endswith("=F"):
        return "futures"

    if ticker.endswith("=X"):
        return "forex"

    crypto_tokens = [
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
        "DOGE-USD",
        "ADA-USD",
        "XRP-USD",
        "LTC-USD",
    ]

    if ticker in crypto_tokens or ticker.endswith("-USD"):
        return "crypto"

    return "stocks"


def detect_session_mode(ticker):
    asset_class = detect_asset_class(ticker)

    if asset_class == "futures":
        return "CME Futures"

    if asset_class == "forex":
        return "Forex 24/5"

    if asset_class == "crypto":
        return "Crypto 24/7"

    return "US Stocks / ETFs - Regular Hours"


def normalize_session_mode(ticker, selected_mode):
    if selected_mode == "Auto Detect":
        return detect_session_mode(ticker)

    return selected_mode


# ============================================================
# MARKET HOLIDAY HELPERS
# ============================================================

def nth_weekday(year, month, weekday, n):
    """
    weekday: Monday=0 ... Sunday=6
    """
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + (n - 1) * 7)


def last_weekday(year, month, weekday):
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)

    current = next_month - timedelta(days=1)
    offset = (current.weekday() - weekday) % 7
    return current - timedelta(days=offset)


def observed_fixed_holiday(year, month, day):
    actual = date(year, month, day)

    if actual.weekday() == 5:
        return actual - timedelta(days=1)

    if actual.weekday() == 6:
        return actual + timedelta(days=1)

    return actual


def us_market_holidays(year):
    holidays = set()

    # New Year's Day
    holidays.add(observed_fixed_holiday(year, 1, 1))

    # Martin Luther King Jr. Day
    holidays.add(nth_weekday(year, 1, 0, 3))

    # Presidents' Day
    holidays.add(nth_weekday(year, 2, 0, 3))

    # Good Friday
    easter = calculate_easter(year)
    holidays.add(easter - timedelta(days=2))

    # Memorial Day
    holidays.add(last_weekday(year, 5, 0))

    # Juneteenth
    if year >= 2022:
        holidays.add(observed_fixed_holiday(year, 6, 19))

    # Independence Day
    holidays.add(observed_fixed_holiday(year, 7, 4))

    # Labor Day
    holidays.add(nth_weekday(year, 9, 0, 1))

    # Thanksgiving
    holidays.add(nth_weekday(year, 11, 3, 4))

    # Christmas
    holidays.add(observed_fixed_holiday(year, 12, 25))

    return holidays


def calculate_easter(year):
    """
    Anonymous Gregorian algorithm.
    """
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


def is_us_market_holiday(day_value):
    if isinstance(day_value, pd.Timestamp):
        day_value = day_value.date()

    return day_value in us_market_holidays(day_value.year)


def is_early_close(day_value):
    """
    Common US equity early closes.
    """
    if isinstance(day_value, pd.Timestamp):
        day_value = day_value.date()

    year = day_value.year

    day_before_independence = date(year, 7, 3)
    friday_after_thanksgiving = nth_weekday(year, 11, 3, 4) + timedelta(days=1)
    christmas_eve = date(year, 12, 24)

    return day_value in {
        day_before_independence,
        friday_after_thanksgiving,
        christmas_eve,
    }


# ============================================================
# DATA NORMALIZATION
# ============================================================

def find_column(dataframe, possible_names):
    lower_map = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    return None


def normalize_ohlcv(raw, intraday=True):
    if raw is None or raw.empty:
        return pd.DataFrame()

    dataframe = raw.copy()

    if isinstance(dataframe.columns, pd.MultiIndex):
        dataframe.columns = [
            str(column[0])
            for column in dataframe.columns
        ]

    dataframe = dataframe.reset_index()

    timestamp_col = dataframe.columns[0]

    open_col = find_column(dataframe, ["Open"])
    high_col = find_column(dataframe, ["High"])
    low_col = find_column(dataframe, ["Low"])
    close_col = find_column(dataframe, ["Close", "Adj Close"])
    volume_col = find_column(dataframe, ["Volume"])

    if any(
        column is None
        for column in [open_col, high_col, low_col, close_col]
    ):
        return pd.DataFrame()

    output = pd.DataFrame(
        {
            "timestamp": dataframe[timestamp_col],
            "open": dataframe[open_col],
            "high": dataframe[high_col],
            "low": dataframe[low_col],
            "close": dataframe[close_col],
            "volume": (
                dataframe[volume_col]
                if volume_col is not None
                else 0.0
            ),
        }
    )

    output["timestamp"] = pd.to_datetime(
        output["timestamp"],
        errors="coerce",
    )

    try:
        if output["timestamp"].dt.tz is not None:
            output["timestamp"] = (
                output["timestamp"]
                .dt.tz_convert("America/New_York")
                .dt.tz_localize(None)
            )
    except Exception:
        pass

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        )

    output["volume"] = output["volume"].fillna(0.0)

    output = output.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    output = output.drop_duplicates(
        subset=["timestamp"]
    )

    output = output.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    valid = (
        (output["open"] > 0)
        & (output["high"] > 0)
        & (output["low"] > 0)
        & (output["close"] > 0)
        & (output["high"] >= output["low"])
        & (output["high"] >= output["open"])
        & (output["high"] >= output["close"])
        & (output["low"] <= output["open"])
        & (output["low"] <= output["close"])
    )

    output = output[valid].copy()

    if output.empty:
        return pd.DataFrame()

    output["bar_ratio"] = (
        output["high"] / output["low"]
    )

    output = output[
        output["bar_ratio"] <= 3.0
    ].copy()

    if output.empty:
        return pd.DataFrame()

    rolling_median = (
        output["close"]
        .rolling(80, min_periods=10)
        .median()
        .shift(1)
    )

    expanding_median = (
        output["close"]
        .expanding(min_periods=1)
        .median()
    )

    reference = rolling_median.fillna(expanding_median)
    reference = reference.replace(0, np.nan)

    price_ratio = output["close"] / reference

    if intraday:
        output = output[
            (price_ratio >= 0.10)
            & (price_ratio <= 10.0)
        ].copy()
    else:
        output = output[
            (price_ratio >= 0.02)
            & (price_ratio <= 50.0)
        ].copy()

    if output.empty:
        return pd.DataFrame()

    output = output.drop(
        columns=["bar_ratio"],
        errors="ignore",
    )

    output["calendar_date"] = (
        output["timestamp"].dt.date
    )

    output["label"] = output["timestamp"].dt.strftime(
        "%m-%d %H:%M"
    )

    return output.reset_index(drop=True)


# ============================================================
# SESSION FILTERING
# ============================================================

def futures_session_date(timestamp):
    """
    CME convention:
    Sunday 18:00 through Monday 17:00 is treated as Monday's
    trading session.

    For timestamps at or after 18:00, assign the next calendar day.
    """
    current_date = timestamp.date()

    if timestamp.time() >= time(18, 0):
        return current_date + timedelta(days=1)

    return current_date


def apply_session_filter(dataframe, ticker, session_mode):
    if dataframe.empty:
        return dataframe

    output = dataframe.copy()
    output["timestamp"] = pd.to_datetime(
        output["timestamp"]
    )

    asset_class = detect_asset_class(ticker)

    if session_mode == "US Stocks / ETFs - Regular Hours":
        output = output[
            output["timestamp"].dt.weekday < 5
        ].copy()

        output = output[
            ~output["calendar_date"].apply(
                is_us_market_holiday
            )
        ].copy()

        minutes_since_midnight = (
            output["timestamp"].dt.hour * 60
            + output["timestamp"].dt.minute
        )

        regular_open = 9 * 60 + 30
        regular_close = np.where(
            output["calendar_date"].apply(is_early_close),
            13 * 60,
            16 * 60,
        )

        output = output[
            (minutes_since_midnight >= regular_open)
            & (minutes_since_midnight <= regular_close)
        ].copy()

        output["session_date"] = output["calendar_date"]

    elif session_mode == "US Stocks / ETFs - Extended Hours":
        output = output[
            output["timestamp"].dt.weekday < 5
        ].copy()

        output = output[
            ~output["calendar_date"].apply(
                is_us_market_holiday
            )
        ].copy()

        output["session_date"] = output["calendar_date"]

    elif session_mode == "CME Futures":
        weekday = output["timestamp"].dt.weekday
        current_time = output["timestamp"].dt.time

        # Remove Saturday and most of Sunday.
        valid_weekday = weekday < 5
        sunday_evening = (
            (weekday == 6)
            & (current_time >= time(18, 0))
        )

        output = output[
            valid_weekday | sunday_evening
        ].copy()

        # Remove the CME daily maintenance break.
        current_time = output["timestamp"].dt.time

        in_break = (
            (current_time >= time(17, 0))
            & (current_time < time(18, 0))
        )

        output = output[~in_break].copy()

        output["session_date"] = output[
            "timestamp"
        ].apply(futures_session_date)

    elif session_mode == "Forex 24/5":
        weekday = output["timestamp"].dt.weekday
        current_time = output["timestamp"].dt.time

        valid_weekday = weekday < 5
        sunday_open = (
            (weekday == 6)
            & (current_time >= time(17, 0))
        )

        friday_after_close = (
            (weekday == 4)
            & (current_time >= time(17, 0))
        )

        output = output[
            (valid_weekday | sunday_open)
            & ~friday_after_close
        ].copy()

        output["session_date"] = output[
            "calendar_date"
        ]

    elif session_mode == "Crypto 24/7":
        output["session_date"] = output[
            "calendar_date"
        ]

    elif session_mode == "Custom Session":
        # The custom-session controls are applied separately below.
        output["session_date"] = output[
            "calendar_date"
        ]

    else:
        output["session_date"] = output[
            "calendar_date"
        ]

    output = output.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return output


def apply_custom_session(
    dataframe,
    start_time,
    end_time,
    include_weekends,
):
    if dataframe.empty:
        return dataframe

    output = dataframe.copy()

    if not include_weekends:
        output = output[
            output["timestamp"].dt.weekday < 5
        ].copy()

    current_minutes = (
        output["timestamp"].dt.hour * 60
        + output["timestamp"].dt.minute
    )

    start_minutes = (
        start_time.hour * 60
        + start_time.minute
    )

    end_minutes = (
        end_time.hour * 60
        + end_time.minute
    )

    if start_minutes <= end_minutes:
        output = output[
            (current_minutes >= start_minutes)
            & (current_minutes <= end_minutes)
        ].copy()
    else:
        output = output[
            (current_minutes >= start_minutes)
            | (current_minutes <= end_minutes)
        ].copy()

    output["session_date"] = output[
        "calendar_date"
    ]

    return output.reset_index(drop=True)


# ============================================================
# DATA FETCHING
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_history(ticker, timeframe):
    config = TIMEFRAME_CONFIG[timeframe]

    periods = [
        config["period"],
        config["fallback_period"],
    ]

    for period in periods:
        try:
            raw = yf.download(
                ticker,
                period=period,
                interval=config["yf_interval"],
                auto_adjust=True,
                progress=False,
                prepost=True,
                threads=False,
            )

            normalized = normalize_ohlcv(
                raw,
                intraday=config["intraday"],
            )

            if not normalized.empty:
                return normalized

        except Exception:
            continue

    return pd.DataFrame()


# ============================================================
# INDICATORS
# ============================================================

def moving_average(series, length, kind):
    if kind == "EMA":
        return series.ewm(
            span=int(length),
            adjust=False,
        ).mean()

    return series.rolling(
        int(length)
    ).mean()


def get_context_dataframe(
    revealed_dataframe,
    practice_date,
    preset,
):
    if revealed_dataframe.empty:
        return revealed_dataframe.copy()

    available_dates = sorted(
        revealed_dataframe["session_date"].unique()
    )

    if practice_date not in available_dates:
        return revealed_dataframe.copy()

    current_index = available_dates.index(
        practice_date
    )

    previous_days_map = {
        "Today": 0,
        "1 Previous Day": 1,
        "5 Previous Days": 5,
        "20 Previous Days": 20,
        "All Available History": 999999,
    }

    previous_days = previous_days_map.get(
        preset,
        0,
    )

    start_index = max(
        0,
        current_index - previous_days,
    )

    start_date = available_dates[start_index]

    return revealed_dataframe[
        revealed_dataframe["session_date"] >= start_date
    ].copy()


def get_price_axis_range(dataframe):
    if dataframe.empty:
        return None

    values = []

    for column in [
        "low",
        "high",
        "fast_ma",
        "slow_ma",
    ]:
        if column in dataframe.columns:
            values.extend(
                pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                ).dropna().tolist()
            )

    if not values:
        return None

    low_value = float(min(values))
    high_value = float(max(values))

    median_price = float(
        dataframe["close"].median()
    )

    spread = max(
        high_value - low_value,
        max(median_price * 0.003, 0.10),
    )

    padding = max(
        spread * 0.12,
        median_price * 0.001,
    )

    return [
        low_value - padding,
        high_value + padding,
    ]


def get_volume_axis_range(dataframe):
    if dataframe.empty:
        return None

    if dataframe["volume"].max() <= 0:
        return None

    values = dataframe["volume"].dropna()

    cap = max(
        float(values.quantile(0.95)) * 1.15,
        float(values.median()) * 2,
        1.0,
    )

    return [0, cap]


# ============================================================
# ACCOUNT HELPERS
# ============================================================

def get_position_quantity():
    position = load_position()

    if position is None:
        return 0.0

    quantity = float(position["quantity"])

    if position["side"] == "short":
        return -quantity

    return quantity


def get_realized_pnl():
    account = load_account()
    return float(account["realized_pnl"])


def calculate_equity(current_price):
    account = load_account()
    position = load_position()

    cash = float(account["cash_balance"])

    if position is None:
        return cash

    quantity = float(position["quantity"])

    if position["side"] == "short":
        return cash - quantity * current_price

    return cash + quantity * current_price


def maybe_reset_depleted_account(current_price):
    account = load_account()
    position = load_position()

    if position is not None:
        return False

    equity = float(account["cash_balance"])

    if equity <= 0:
        reset_account_cycle()
        st.session_state.account_message = (
            "Account depleted. A new $1,000 account cycle was created."
        )
        return True

    return False


def open_position(
    ticker,
    price,
    timestamp,
    side,
    amount,
    stop_loss,
    target,
):
    account = load_account()

    quantity = float(amount) / float(price)
    asset_class = detect_asset_class(ticker)

    if side == "long":
        new_cash = float(account["cash_balance"]) - float(amount)
    else:
        new_cash = float(account["cash_balance"]) + float(amount)

    save_account(
        new_cash,
        float(account["realized_pnl"]),
    )

    save_position(
        {
            "ticker": ticker,
            "asset_class": asset_class,
            "side": side,
            "quantity": quantity,
            "entry_price": float(price),
            "entry_time": str(timestamp),
            "stop_loss": stop_loss,
            "target": target,
            "invested_amount": float(amount),
        }
    )


def close_position(
    ticker,
    price,
    timestamp,
    reason="",
    quantity_to_close=None,
):
    position = load_position()

    if position is None:
        return

    account = load_account()

    total_quantity = float(position["quantity"])

    if quantity_to_close is None:
        quantity_to_close = total_quantity

    quantity_to_close = min(
        float(quantity_to_close),
        total_quantity,
    )

    entry_price = float(position["entry_price"])
    price = float(price)

    if position["side"] == "long":
        realized = (
            price - entry_price
        ) * quantity_to_close

        cash_change = price * quantity_to_close
        new_cash = float(account["cash_balance"]) + cash_change

    else:
        realized = (
            entry_price - price
        ) * quantity_to_close

        cash_change = -price * quantity_to_close
        new_cash = float(account["cash_balance"]) + cash_change

    new_realized = (
        float(account["realized_pnl"])
        + realized
    )

    save_account(
        new_cash,
        new_realized,
    )

    save_trade(
        ticker=ticker,
        asset_class=position["asset_class"],
        side=position["side"],
        quantity=quantity_to_close,
        entry_price=entry_price,
        exit_price=price,
        entry_time=position["entry_time"],
        exit_time=str(timestamp),
        realized_pnl=realized,
        reason=reason,
    )

    remaining = total_quantity - quantity_to_close

    if remaining <= 1e-10:
        save_position(None)
    else:
        position["quantity"] = remaining
        save_position(position)


# ============================================================
# MARKERS AND DRAWINGS
# ============================================================

def add_marker(timestamp, price, kind):
    st.session_state.markers.append(
        {
            "timestamp": pd.Timestamp(timestamp),
            "price": float(price),
            "kind": kind,
        }
    )


def add_horizontal_line(price, label, color):
    st.session_state.drawings.append(
        {
            "type": "hline",
            "price": float(price),
            "label": label,
            "color": color,
        }
    )


def add_band(price_one, price_two):
    st.session_state.drawings.append(
        {
            "type": "band",
            "y0": min(float(price_one), float(price_two)),
            "y1": max(float(price_one), float(price_two)),
            "color": "rgba(34,197,94,0.18)",
        }
    )


def add_trend_line(x0, y0, x1, y1):
    st.session_state.drawings.append(
        {
            "type": "trend",
            "x0": pd.Timestamp(x0),
            "y0": float(y0),
            "x1": pd.Timestamp(x1),
            "y1": float(y1),
            "color": "#f472b6",
        }
    )


# ============================================================
# REPLAY FUNCTIONS
# ============================================================

def advance_bars(number_of_bars):
    dataframe = st.session_state.df

    if dataframe.empty:
        return

    max_step = len(dataframe) - 1

    for _ in range(int(number_of_bars)):
        if st.session_state.step >= max_step:
            st.toast(
                "Replay session complete.",
                icon="🔔",
            )
            break

        previous_row = dataframe.iloc[
            st.session_state.step
        ]

        st.session_state.step += 1

        row = dataframe.iloc[
            st.session_state.step
        ]

        timestamp = row["timestamp"]
        current_ticker = st.session_state.active_ticker
        position = load_position()

        if position is not None:
            side = position["side"]
            stop_loss = position["stop_loss"]
            target = position["target"]

            if side == "long":
                if (
                    stop_loss is not None
                    and float(row["low"]) <= float(stop_loss)
                ):
                    execution_price = min(
                        float(stop_loss),
                        float(row["open"]),
                    )

                    close_position(
                        current_ticker,
                        execution_price,
                        timestamp,
                        "(STOP-LOSS)",
                    )

                    add_marker(
                        timestamp,
                        execution_price,
                        "SELL",
                    )

                    st.toast(
                        f"Long stopped at "
                        f"${execution_price:.2f}",
                        icon="💥",
                    )

                elif (
                    target is not None
                    and float(row["high"]) >= float(target)
                ):
                    execution_price = max(
                        float(target),
                        float(row["open"]),
                    )

                    close_position(
                        current_ticker,
                        execution_price,
                        timestamp,
                        "(TARGET HIT)",
                    )

                    add_marker(
                        timestamp,
                        execution_price,
                        "SELL",
                    )

                    st.toast(
                        f"Long target at "
                        f"${execution_price:.2f}",
                        icon="🎉",
                    )

            else:
                if (
                    stop_loss is not None
                    and float(row["high"]) >= float(stop_loss)
                ):
                    execution_price = max(
                        float(stop_loss),
                        float(row["open"]),
                    )

                    close_position(
                        current_ticker,
                        execution_price,
                        timestamp,
                        "(STOP-LOSS)",
                    )

                    add_marker(
                        timestamp,
                        execution_price,
                        "COVER",
                    )

                    st.toast(
                        f"Short stopped at "
                        f"${execution_price:.2f}",
                        icon="💥",
                    )

                elif (
                    target is not None
                    and float(row["low"]) <= float(target)
                ):
                    execution_price = min(
                        float(target),
                        float(row["open"]),
                    )

                    close_position(
                        current_ticker,
                        execution_price,
                        timestamp,
                        "(TARGET HIT)",
                    )

                    add_marker(
                        timestamp,
                        execution_price,
                        "COVER",
                    )

                    st.toast(
                        f"Short target at "
                        f"${execution_price:.2f}",
                        icon="🎉",
                    )

        # Optional force-close behavior.
        if (
            not st.session_state.active_carry_mode
            and position is not None
        ):
            old_session = previous_row["session_date"]
            new_session = row["session_date"]

            if old_session != new_session:
                close_position(
                    current_ticker,
                    float(previous_row["close"]),
                    previous_row["timestamp"],
                    "(SESSION CLOSE)",
                )

                add_marker(
                    previous_row["timestamp"],
                    float(previous_row["close"]),
                    "SELL"
                    if position["side"] == "long"
                    else "COVER",
                )

    current_position = load_position()

    if current_position is None:
        current_row = dataframe.iloc[
            st.session_state.step
        ]

        maybe_reset_depleted_account(
            float(current_row["close"])
        )


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("⚙️ Setup")

ticker = st.sidebar.text_input(
    "Ticker",
    value="TQQQ",
).upper().strip()

timeframe = st.sidebar.selectbox(
    "Chart Timeframe",
    TIMEFRAME_OPTIONS,
    index=TIMEFRAME_OPTIONS.index("5m"),
)

selected_session_mode = st.sidebar.selectbox(
    "Session Mode",
    SESSION_OPTIONS,
    index=0,
)

effective_session_mode = normalize_session_mode(
    ticker,
    selected_session_mode,
)

st.sidebar.info(
    f"Detected/applied session: "
    f"**{effective_session_mode}**"
)

custom_start_time = time(
    hour=9,
    minute=30,
)

custom_end_time = time(
    hour=16,
    minute=0,
)

custom_include_weekends = False

if effective_session_mode == "Custom Session":
    custom_start_time = st.sidebar.time_input(
        "Custom session start",
        value=time(9, 30),
    )

    custom_end_time = st.sidebar.time_input(
        "Custom session end",
        value=time(16, 0),
    )

    custom_include_weekends = st.sidebar.checkbox(
        "Include weekends",
        value=False,
    )

st.sidebar.caption(
    f"Yahoo native interval: "
    f"`{TIMEFRAME_CONFIG[timeframe]['yf_interval']}`"
)


# ============================================================
# LOAD AND FILTER DATA
# ============================================================

with st.sidebar:
    with st.spinner(
        f"Loading {timeframe} data for {ticker}..."
    ):
        source_df = fetch_history(
            ticker,
            timeframe,
        )

filtered_df = pd.DataFrame()

if source_df.empty:
    st.sidebar.error(
        "No data returned. Check the ticker, timeframe, "
        "Yahoo availability, or rate limits."
    )

else:
    filtered_df = apply_session_filter(
        source_df,
        ticker,
        effective_session_mode,
    )

    if effective_session_mode == "Custom Session":
        filtered_df = apply_custom_session(
            filtered_df,
            custom_start_time,
            custom_end_time,
            custom_include_weekends,
        )

    if filtered_df.empty:
        st.sidebar.error(
            "Data was returned, but no bars matched "
            "the selected session."
        )

    else:
        available_dates = sorted(
            filtered_df["session_date"].unique()
        )

        first_date = available_dates[0]
        last_date = available_dates[-1]

        st.sidebar.success(
            f"{len(filtered_df):,} session bars\n"
            f"{first_date} → {last_date}"
        )

        practice_date_key = (
            f"practice_date_"
            f"{ticker}_"
            f"{timeframe}_"
            f"{effective_session_mode}"
        )

        if (
            practice_date_key not in st.session_state
            or st.session_state[practice_date_key]
            not in available_dates
        ):
            if len(available_dates) >= 3:
                default_date = available_dates[-3]
            else:
                default_date = available_dates[-1]

            st.session_state[
                practice_date_key
            ] = default_date

        practice_date = st.sidebar.date_input(
            "Practice Session",
            min_value=first_date,
            max_value=last_date,
            key=practice_date_key,
        )

        practice_rows = filtered_df[
            filtered_df["session_date"] == practice_date
        ].copy()

        selected_start_timestamp = None
        can_start = not practice_rows.empty

        if practice_rows.empty:
            st.sidebar.warning(
                "No bars exist for this session."
            )

        elif TIMEFRAME_CONFIG[timeframe]["intraday"]:
            labels = practice_rows[
                "timestamp"
            ].dt.strftime("%H:%M").tolist()

            default_index = min(
                10,
                len(labels) - 1,
            )

            start_key = (
                f"start_time_"
                f"{ticker}_"
                f"{timeframe}_"
                f"{practice_date}"
            )

            selected_label = st.sidebar.selectbox(
                "Replay Start Time",
                labels,
                index=default_index,
                key=start_key,
            )

            selected_start_timestamp = practice_rows[
                practice_rows["timestamp"].dt.strftime(
                    "%H:%M"
                ) == selected_label
            ]["timestamp"].iloc[0]

        else:
            selected_start_timestamp = practice_rows[
                "timestamp"
            ].iloc[0]


# ============================================================
# POSITION CARRY SETTINGS
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Position Behavior")

carry_positions = st.sidebar.checkbox(
    "Carry positions across sessions",
    value=True,
)

st.sidebar.caption(
    "When disabled, open positions are closed at "
    "the final available bar of each session."
)


# ============================================================
# INDICATOR SETTINGS
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Moving Averages")

show_ma = st.sidebar.checkbox(
    "Show moving averages",
    value=True,
)

ma_type = st.sidebar.radio(
    "MA Type",
    ["EMA", "SMA"],
    horizontal=True,
)

default_fast, default_slow = TIMEFRAME_CONFIG[
    timeframe
]["ema"]

fast_length = st.sidebar.number_input(
    "Fast MA length",
    min_value=2,
    max_value=500,
    value=int(default_fast),
    step=1,
)

slow_length = st.sidebar.number_input(
    "Slow MA length",
    min_value=2,
    max_value=500,
    value=int(default_slow),
    step=1,
)

show_volume_ma = st.sidebar.checkbox(
    "Show volume MA",
    value=True,
)

volume_ma_length = st.sidebar.number_input(
    "Volume MA length",
    min_value=2,
    max_value=200,
    value=20,
    step=1,
)


# ============================================================
# CONTEXT AND CAMERA SETTINGS
# ============================================================

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
    "The x-axis uses sequential session bars instead "
    "of calendar time. Closed-market gaps are removed."
)


# ============================================================
# DRAWING SETTINGS
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("✏️ Drawings")

draw_mode = st.sidebar.selectbox(
    "Drawing mode",
    DRAW_MODES,
    index=0,
)

if draw_mode != st.session_state.last_draw_mode:
    st.session_state.last_draw_mode = draw_mode
    st.session_state.last_draw_signature = None
    st.session_state.draw_clicks = []

if st.sidebar.button(
    "Undo last drawing"
):
    if st.session_state.drawings:
        st.session_state.drawings.pop()

if st.sidebar.button(
    "Clear all drawings"
):
    st.session_state.drawings = []
    st.session_state.draw_clicks = []


# ============================================================
# START / RESET REPLAY
# ============================================================

st.sidebar.markdown("---")

if st.sidebar.button(
    "🚀 Start / Reset Replay",
    use_container_width=True,
    disabled=(
        filtered_df.empty
        or not can_start
    ),
):
    active_df = filtered_df[
        filtered_df["session_date"] <= practice_date
    ].copy()

    active_df = active_df.reset_index(
        drop=True
    )

    matching_indices = active_df.index[
        active_df["timestamp"]
        == selected_start_timestamp
    ].tolist()

    day_indices = active_df.index[
        active_df["session_date"]
        == practice_date
    ].tolist()

    if not matching_indices or not day_indices:
        st.sidebar.error(
            "Could not initialize the replay."
        )

    else:
        st.session_state.df = active_df
        st.session_state.sim_start_idx = day_indices[0]
        st.session_state.step = matching_indices[0]

        st.session_state.active_ticker = ticker
        st.session_state.active_interval = timeframe
        st.session_state.active_session_mode = (
            effective_session_mode
        )
        st.session_state.active_practice_date = (
            practice_date
        )
        st.session_state.active_carry_mode = (
            carry_positions
        )

        st.session_state.markers = []
        st.session_state.drawings = []
        st.session_state.draw_clicks = []
        st.session_state.last_draw_signature = None

        st.session_state.camera_revision += 1
        st.session_state.camera_signature = None
        st.session_state.force_camera = True
        st.session_state.sim_active = True

        st.rerun()


# ============================================================
# MAIN APP VALIDATION
# ============================================================

st.title("💹 Market Replay Simulator")

st.caption(
    "Session-compressed replay • persistent paper account • "
    "native Yahoo candles"
)

if (
    not st.session_state.sim_active
    or st.session_state.df.empty
):
    st.info(
        "Choose a ticker, timeframe, session, practice date, "
        "and start time. Then click Start / Reset Replay."
    )
    st.stop()


# ============================================================
# ACTIVE REPLAY DATA
# ============================================================

dataframe = st.session_state.df
step = st.session_state.step
practice_date = st.session_state.active_practice_date

if step >= len(dataframe):
    step = len(dataframe) - 1
    st.session_state.step = step

revealed_df = dataframe.iloc[
    : step + 1
].copy()

current_row = dataframe.iloc[step]
current_price = float(current_row["close"])
current_timestamp = pd.Timestamp(
    current_row["timestamp"]
)
current_session_date = current_row["session_date"]

current_time_text = current_timestamp.strftime(
    "%Y-%m-%d %H:%M"
)

calc_df = revealed_df.copy()

if show_ma:
    calc_df["fast_ma"] = moving_average(
        calc_df["close"],
        fast_length,
        ma_type,
    )

    calc_df["slow_ma"] = moving_average(
        calc_df["close"],
        slow_length,
        ma_type,
    )

if show_volume_ma:
    calc_df["volume_ma"] = calc_df[
        "volume"
    ].rolling(
        int(volume_ma_length)
    ).mean()

chart_df = get_context_dataframe(
    calc_df,
    practice_date,
    context_preset,
).reset_index(drop=True)

if chart_df.empty:
    chart_df = calc_df.copy().reset_index(
        drop=True
    )

# The compressed x-axis.
# Each visible bar gets a sequential numeric x-value.
chart_df["x_index"] = np.arange(
    len(chart_df)
)

current_chart_indices = chart_df.index[
    chart_df["timestamp"] == current_timestamp
].tolist()

if current_chart_indices:
    current_x = current_chart_indices[-1]
else:
    current_x = len(chart_df) - 1


# ============================================================
# ACCOUNT METRICS
# ============================================================

account = load_account()
position = load_position()

cash = float(account["cash_balance"])
realized_pnl = float(account["realized_pnl"])

if position is None:
    quantity_signed = 0.0
    position_value = 0.0
    unrealized_pnl = 0.0
    position_text = "—"
    position_type = "FLAT"
else:
    quantity = float(position["quantity"])

    if position["side"] == "long":
        quantity_signed = quantity
        position_value = quantity * current_price
        unrealized_pnl = (
            current_price
            - float(position["entry_price"])
        ) * quantity
        position_type = "🟢 LONG"

    else:
        quantity_signed = -quantity
        position_value = -quantity * current_price
        unrealized_pnl = (
            float(position["entry_price"])
            - current_price
        ) * quantity
        position_type = "🔴 SHORT"

    position_text = (
        f"{abs(quantity_signed):.4f} units"
    )

equity = cash + position_value
total_pnl = equity - float(
    account["starting_balance"]
)

# ============================================================
# METRICS
# ============================================================

m1, m2, m3, m4, m5, m6 = st.columns(6)

m1.metric(
    "Price",
    f"${current_price:.2f}",
)

m2.metric(
    "Equity",
    f"${equity:.2f}",
    f"${total_pnl:.2f}",
)

m3.metric(
    "Cash",
    f"${cash:.2f}",
)

m4.metric(
    f"Position ({position_type})",
    position_text,
)

m5.metric(
    "Realized P&L",
    f"${realized_pnl:.2f}",
)

m6.metric(
    "Unrealized P&L",
    f"${unrealized_pnl:.2f}",
)

st.caption(
    f"Account cycle {account['cycle_number']} • "
    f"Starting balance ${account['starting_balance']:.2f} • "
    f"Session mode: {st.session_state.active_session_mode}"
)


# ============================================================
# CHART
# ============================================================

volume_colors = np.where(
    chart_df["close"] >= chart_df["open"],
    "rgba(38,166,154,0.58)",
    "rgba(239,83,80,0.58)",
)

figure = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.78, 0.22],
)

figure.add_trace(
    go.Candlestick(
        x=chart_df["x_index"],
        open=chart_df["open"],
        high=chart_df["high"],
        low=chart_df["low"],
        close=chart_df["close"],
        name=st.session_state.active_ticker,
        increasing=dict(
            line=dict(color="#26a69a"),
            fillcolor="#26a69a",
        ),
        decreasing=dict(
            line=dict(color="#ef5350"),
            fillcolor="#ef5350",
        ),
        customdata=np.column_stack(
            [
                chart_df["timestamp"].astype(str),
                chart_df["session_date"].astype(str),
            ]
        ),
        hovertemplate=(
            "Time: %{customdata[0]}<br>"
            "Session: %{customdata[1]}<br>"
            "Open: %{open:.2f}<br>"
            "High: %{high:.2f}<br>"
            "Low: %{low:.2f}<br>"
            "Close: %{close:.2f}"
            "<extra></extra>"
        ),
    ),
    row=1,
    col=1,
)

figure.add_trace(
    go.Bar(
        x=chart_df["x_index"],
        y=chart_df["volume"],
        marker_color=volume_colors,
        name="Volume",
    ),
    row=2,
    col=1,
)

if show_ma and "fast_ma" in chart_df.columns:
    figure.add_trace(
        go.Scatter(
            x=chart_df["x_index"],
            y=chart_df["fast_ma"],
            name=f"{ma_type}{int(fast_length)}",
            line=dict(
                color="#3b82f6",
                width=1.7,
            ),
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=chart_df["x_index"],
            y=chart_df["slow_ma"],
            name=f"{ma_type}{int(slow_length)}",
            line=dict(
                color="#f59e0b",
                width=1.7,
            ),
        ),
        row=1,
        col=1,
    )

if show_volume_ma and "volume_ma" in chart_df.columns:
    figure.add_trace(
        go.Scatter(
            x=chart_df["x_index"],
            y=chart_df["volume_ma"],
            name=f"Volume MA {int(volume_ma_length)}",
            line=dict(
                color="#ff6d00",
                width=1.7,
            ),
        ),
        row=2,
        col=1,
    )


# ============================================================
# POSITION LINES
# ============================================================

if position is not None:
    if position["stop_loss"] is not None:
        figure.add_hline(
            y=float(position["stop_loss"]),
            row=1,
            col=1,
            line=dict(
                color="#fb923c",
                width=2,
                dash="dash",
            ),
        )

    if position["target"] is not None:
        figure.add_hline(
            y=float(position["target"]),
            row=1,
            col=1,
            line=dict(
                color="#4ade80",
                width=2,
                dash="dash",
            ),
        )

    figure.add_hline(
        y=float(position["entry_price"]),
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
    "LONG": {
        "symbol": "triangle-up",
        "color": "#22c55e",
    },
    "SHORT": {
        "symbol": "triangle-down",
        "color": "#ef4444",
    },
    "SELL": {
        "symbol": "circle",
        "color": "#f97316",
    },
    "COVER": {
        "symbol": "circle",
        "color": "#38bdf8",
    },
}

for marker in st.session_state.markers:
    matching = chart_df.index[
        chart_df["timestamp"] == marker["timestamp"]
    ].tolist()

    if not matching:
        continue

    marker_x = matching[-1]

    style = marker_styles.get(
        marker["kind"],
        {
            "symbol": "circle",
            "color": "#ffffff",
        },
    )

    figure.add_trace(
        go.Scatter(
            x=[marker_x],
            y=[marker["price"]],
            mode="markers",
            showlegend=False,
            marker=dict(
                symbol=style["symbol"],
                size=12,
                color=style["color"],
                line=dict(
                    color="#ffffff",
                    width=1,
                ),
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
        figure.add_hline(
            y=drawing["price"],
            row=1,
            col=1,
            line=dict(
                color=drawing.get(
                    "color",
                    "#22d3ee",
                ),
                width=1.8,
            ),
        )

    elif drawing["type"] == "band":
        figure.add_hrect(
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
        x0_matches = chart_df.index[
            chart_df["timestamp"]
            == drawing["x0"]
        ].tolist()

        x1_matches = chart_df.index[
            chart_df["timestamp"]
            == drawing["x1"]
        ].tolist()

        if x0_matches and x1_matches:
            figure.add_trace(
                go.Scatter(
                    x=[
                        x0_matches[-1],
                        x1_matches[-1],
                    ],
                    y=[
                        drawing["y0"],
                        drawing["y1"],
                    ],
                    mode="lines",
                    showlegend=False,
                    line=dict(
                        color=drawing.get(
                            "color",
                            "#f472b6",
                        ),
                        width=2,
                    ),
                ),
                row=1,
                col=1,
            )


# ============================================================
# CAMERA / COMPRESSED AXIS
# ============================================================

context_signature = (
    context_preset,
    follow_replay,
    timeframe,
    practice_date,
    effective_session_mode,
)

if (
    st.session_state.camera_signature
    != context_signature
):
    st.session_state.camera_signature = (
        context_signature
    )
    st.session_state.camera_revision += 1
    st.session_state.force_camera = True

if follow_replay:
    focus_df = chart_df[
        chart_df["session_date"] == practice_date
    ].copy()

    if focus_df.empty:
        focus_df = chart_df.copy()

    focus_start = max(
        0,
        current_x - max(
            30,
            int(len(focus_df) * 0.8),
        ),
    )

    focus_end = min(
        len(chart_df) - 1,
        current_x + 8,
    )

else:
    focus_df = chart_df.copy()
    focus_start = 0
    focus_end = len(chart_df) - 1

price_range = get_price_axis_range(
    focus_df
)

volume_range = get_volume_axis_range(
    focus_df
)

if follow_replay:
    x_range = [
        focus_start,
        max(focus_start + 1, focus_end),
    ]
else:
    x_range = [
        0,
        max(1, len(chart_df) - 1),
    ]

tick_values = chart_df["x_index"].tolist()

if len(tick_values) > 12:
    tick_values = tick_values[
        :: max(1, len(tick_values) // 12)
    ]

tick_text = [
    pd.Timestamp(
        chart_df.iloc[int(index)]["timestamp"]
    ).strftime("%m-%d\n%H:%M")
    for index in tick_values
]

figure.update_layout(
    template="plotly_dark",
    height=640,
    dragmode="pan",
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    font=dict(color="#e5e7eb"),
    title=(
        f"{st.session_state.active_ticker} | "
        f"{st.session_state.active_interval} | "
        f"{current_time_text} | "
        f"Session-compressed axis"
    ),
    margin=dict(
        l=10,
        r=190,
        t=42,
        b=10,
    ),
    showlegend=True,
    hovermode="x",
    uirevision=(
        f"{st.session_state.camera_revision}-"
        f"{st.session_state.step}"
        if follow_replay
        else f"{st.session_state.camera_revision}"
    ),
)

for row_number in (1, 2):
    figure.update_xaxes(
        type="linear",
        range=x_range,
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_text,
        gridcolor="#1f2937",
        linecolor="#4b5563",
        zerolinecolor="#374151",
        showspikes=True,
        spikemode="across",
        spikecolor="#94a3b8",
        spikethickness=1,
        row=row_number,
        col=1,
    )

figure.update_yaxes(
    title_text="Price",
    gridcolor="#1f2937",
    linecolor="#4b5563",
    row=1,
    col=1,
)

figure.update_yaxes(
    title_text="Volume",
    gridcolor="#1f2937",
    linecolor="#4b5563",
    row=2,
    col=1,
)

if price_range is not None:
    figure.update_yaxes(
        range=price_range,
        row=1,
        col=1,
    )

if volume_range is not None:
    figure.update_yaxes(
        range=volume_range,
        row=2,
        col=1,
    )


# ============================================================
# CHART AND ADVANCE CONTROLS
# ============================================================

chart_col, control_col = st.columns(
    [6, 1]
)

with chart_col:
    st.plotly_chart(
        figure,
        use_container_width=True,
        key="market_chart",
        config={
            "scrollZoom": True,
            "displaylogo": False,
        },
    )

with control_col:
    st.markdown("#### ⏱️")
    st.caption(
        current_timestamp.strftime("%H:%M")
    )

    interval_minutes = TIMEFRAME_CONFIG[
        st.session_state.active_interval
    ]["minutes"]

    if interval_minutes >= 1440:
        step_one = 1
        step_two = 3
        step_three = 5
        label_one = "+1 day"
        label_two = "+3 days"
        label_three = "+5 days"
    else:
        step_one = 1
        step_two = max(
            1,
            round(15 / interval_minutes),
        )
        step_three = max(
            1,
            round(60 / interval_minutes),
        )
        label_one = f"+{interval_minutes}m"
        label_two = f"+{interval_minutes * step_two}m"
        label_three = (
            f"+{interval_minutes * step_three}m"
        )

    if st.button(
        f"▶️ {label_one}",
        use_container_width=True,
    ):
        advance_bars(step_one)
        st.rerun()

    if st.button(
        f"⏩ {label_two}",
        use_container_width=True,
    ):
        advance_bars(step_two)
        st.rerun()

    if st.button(
        f"⏭️ {label_three}",
        use_container_width=True,
    ):
        advance_bars(step_three)
        st.rerun()

    st.markdown("---")

    st.caption(
        f"Session: {current_session_date}"
    )

    st.caption(
        f"Bars shown: {len(chart_df):,}"
    )

    st.caption(
        f"Position carry: "
        f"{'ON' if st.session_state.active_carry_mode else 'OFF'}"
    )


# ============================================================
# TRADE ENTRY PANEL
# ============================================================

if position is None:
    st.markdown("### 🎮 Open a Position")

    e1, e2, e3, e4, e5 = st.columns(
        [1.2, 1.2, 1.2, 1, 1]
    )

    max_trade = max(
        10.0,
        float(cash),
    )

    with e1:
        trade_amount = st.number_input(
            "Trade Amount ($)",
            min_value=10.0,
            max_value=max_trade,
            value=min(100.0, max_trade),
            step=10.0,
        )

    with e2:
        stop_input = st.number_input(
            "Stop-Loss ($)",
            value=round(
                current_price * 0.99,
                2,
            ),
            step=0.01,
        )

    with e3:
        target_input = st.number_input(
            "Target ($, 0 = none)",
            value=0.0,
            step=0.01,
        )

    with e4:
        st.write("")
        st.write("")

        if st.button(
            "🟢 BUY",
            use_container_width=True,
        ):
            valid = True

            if stop_input >= current_price:
                st.error(
                    "Long stop must be below price."
                )
                valid = False

            if (
                target_input != 0
                and target_input <= current_price
            ):
                st.error(
                    "Long target must be above price."
                )
                valid = False

            if valid:
                open_position(
                    ticker=st.session_state.active_ticker,
                    price=current_price,
                    timestamp=current_timestamp,
                    side="long",
                    amount=trade_amount,
                    stop_loss=float(stop_input),
                    target=(
                        float(target_input)
                        if target_input > 0
                        else None
                    ),
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

        if st.button(
            "🔻 SHORT",
            use_container_width=True,
        ):
            valid = True

            if stop_input <= current_price:
                st.error(
                    "Short stop must be above price."
                )
                valid = False

            if (
                target_input != 0
                and target_input >= current_price
            ):
                st.error(
                    "Short target must be below price."
                )
                valid = False

            if valid:
                open_position(
                    ticker=st.session_state.active_ticker,
                    price=current_price,
                    timestamp=current_timestamp,
                    side="short",
                    amount=trade_amount,
                    stop_loss=float(stop_input),
                    target=(
                        float(target_input)
                        if target_input > 0
                        else None
                    ),
                )

                add_marker(
                    current_timestamp,
                    current_price,
                    "SHORT",
                )

                st.rerun()


# ============================================================
# POSITION MANAGEMENT PANEL
# ============================================================

else:
    position_side = position["side"]
    position_quantity = float(
        position["quantity"]
    )

    st.markdown(
        f"### 🎮 Manage "
        f"{'LONG' if position_side == 'long' else 'SHORT'} "
        f"Position"
    )

    m1, m2, m3, m4, m5 = st.columns(
        [1.2, 1.2, 1.2, 1, 1]
    )

    with m1:
        current_stop = (
            position["stop_loss"]
            if position["stop_loss"] is not None
            else current_price
        )

        new_stop = st.number_input(
            "Modify Stop-Loss",
            value=float(current_stop),
            step=0.01,
            key="modify_stop",
        )

    with m2:
        current_target = (
            position["target"]
            if position["target"] is not None
            else 0.0
        )

        new_target = st.number_input(
            "Modify Target",
            value=float(current_target),
            step=0.01,
            key="modify_target",
        )

    with m3:
        close_quantity = st.number_input(
            "Quantity to close",
            min_value=0.0,
            max_value=position_quantity,
            value=position_quantity,
            step=0.0001,
        )

    with m4:
        st.write("")
        st.write("")

        if st.button(
            "💾 Update SL/TP",
            use_container_width=True,
        ):
            valid = True

            if (
                position_side == "long"
                and new_stop >= current_price
            ):
                st.error(
                    "Long stop must be below price."
                )
                valid = False

            if (
                position_side == "short"
                and new_stop <= current_price
            ):
                st.error(
                    "Short stop must be above price."
                )
                valid = False

            if valid:
                position["stop_loss"] = float(
                    new_stop
                )

                position["target"] = (
                    float(new_target)
                    if new_target > 0
                    else None
                )

                save_position(position)
                st.rerun()

    with m5:
        st.write("")
        st.write("")

        close_label = (
            "🔴 SELL"
            if position_side == "long"
            else "🟢 COVER"
        )

        if st.button(
            close_label,
            use_container_width=True,
        ):
            if close_quantity > 0:
                close_position(
                    ticker=st.session_state.active_ticker,
                    price=current_price,
                    timestamp=current_timestamp,
                    reason="Manual close",
                    quantity_to_close=close_quantity,
                )

                add_marker(
                    current_timestamp,
                    current_price,
                    (
                        "SELL"
                        if position_side == "long"
                        else "COVER"
                    ),
                )

                maybe_reset_depleted_account(
                    current_price
                )

                st.rerun()


# ============================================================
# ACCOUNT HISTORY
# ============================================================

with st.expander(
    "📝 Persistent Trade History",
    expanded=False,
):
    connection = get_connection()

    trades = pd.read_sql_query(
        """
        SELECT
            cycle_number,
            ticker,
            asset_class,
            side,
            quantity,
            entry_price,
            exit_price,
            entry_time,
            exit_time,
            realized_pnl,
            reason
        FROM trades
        WHERE account_id = ?
        ORDER BY id DESC
        """,
        connection,
        params=(ACCOUNT_ID,),
    )

    connection.close()

    if trades.empty:
        st.info(
            "No closed trades have been recorded."
        )
    else:
        st.dataframe(
            trades,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# ACCOUNT EVENTS
# ============================================================

with st.expander(
    "🔁 Account Cycles and Events",
    expanded=False,
):
    connection = get_connection()

    events = pd.read_sql_query(
        """
        SELECT
            cycle_number,
            event_type,
            message,
            created_at
        FROM account_events
        WHERE account_id = ?
        ORDER BY id DESC
        """,
        connection,
        params=(ACCOUNT_ID,),
    )

    connection.close()

    if events.empty:
        st.info(
            "No account-cycle events have been recorded."
        )
    else:
        st.dataframe(
            events,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# REPLAY INFORMATION
# ============================================================

with st.expander(
    "📘 Replay Information",
    expanded=False,
):
    st.markdown(
        f"""
| Item | Value |
|---|---|
| Ticker | `{st.session_state.active_ticker}` |
| Asset class | `{detect_asset_class(st.session_state.active_ticker)}` |
| Native timeframe | `{st.session_state.active_interval}` |
| Session mode | `{st.session_state.active_session_mode}` |
| Practice session | `{practice_date}` |
| Current replay time | `{current_time_text}` |
| Current session date | `{current_session_date}` |
| Revealed bars | `{len(revealed_df):,}` |
| Rendered bars | `{len(chart_df):,}` |
| Follow replay | `{"ON" if follow_replay else "OFF"}` |
| Position carry | `{"ON" if st.session_state.active_carry_mode else "OFF"}` |
| Account cycle | `{account["cycle_number"]}` |
| Database | `{os.path.abspath(DB_PATH)}` |

### Session behavior

- US stocks and ETFs use regular market hours by default.
- Overnight, weekends, holidays, and regular-hours gaps are removed.
- Futures retain the overnight session but remove the daily maintenance break.
- Forex continues through weekdays and skips the weekend.
- Crypto retains all bars because crypto trades continuously.
- The chart uses sequential bar positions instead of calendar timestamps.
- Actual price gaps remain visible because OHLC prices are unchanged.

### Account behavior

- Cash and trades persist in SQLite.
- The wallet survives browser refreshes and app restarts.
- Positions remain open across sessions when position carry is enabled.
- Equity equals cash plus the current marked value of the open position.
- Account reset occurs only after the account is actually depleted.
"""
    )
