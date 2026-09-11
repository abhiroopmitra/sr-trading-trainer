# ============================================================
# HUMAN MARKET REPLAY SIMULATOR
#
# FEATURES
# - Native Yahoo candles
# - Session-aware filtering
# - Gapless sequential x-axis
# - 100/200-bar Follow mode for continuous markets
# - Current-session Follow mode for stocks
# - Persistent SQLite wallet and positions
# - Persistent manual drawings by ticker
# - Manual drawing helper
# - Structure labels and local S/R zones
# - Automatic S/R from 5 completed sessions
# - Developing current-session high and low
# - Dedicated right-side S/R label rail
# ============================================================

import json
import os
import sqlite3
from datetime import date, datetime, timedelta, time, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf


# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="Market Replay Simulator",
)


# ============================================================
# GLOBAL SETTINGS
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

SESSION_OPTIONS = [
    "Auto Detect",
    "US Stocks / ETFs - Regular Hours",
    "US Stocks / ETFs - Extended Hours",
    "CME Futures",
    "Forex 24/5",
    "Crypto 24/7",
    "Custom Session",
]

CONTEXT_PRESETS = [
    "Today",
    "1 Previous Day",
    "5 Previous Days",
    "20 Previous Days",
    "All Available History",
]

FOLLOW_WINDOW_OPTIONS = [100, 200]

DRAW_MODES = [
    "None",
    "Line at High",
    "Line at Close",
    "Line at Low",
    "Band",
    "Trend line",
]

AUTO_SR_LOOKBACK_SESSIONS = 5

# Right-side rail sizing.
MIN_LABEL_RAIL_BARS = 16
MAX_LABEL_RAIL_BARS = 34
LABEL_RAIL_PERCENT = 0.18

# Structure settings.
MAX_STRUCTURE_BARS = 700
MAX_SWING_LABELS = 90
MAX_BOS_LABELS = 30
MAX_CHOCH_LABELS = 20
MAX_ZONE_SWINGS = 180
MAX_LOCAL_ZONES_DRAWN = 10

SWING_K = 5
ZONE_TOL = 0.0012
MIN_DRAW_TOUCHES = 2
MIN_SWING_PCT = 0.0008
POLARITY_EDGE = 2
CHOCH_ENABLE = True

# Colors.
C_HH = {"fg": "#ffffff", "bg": "#16a34a"}
C_LH = {"fg": "#ffffff", "bg": "#dc2626"}
C_H = {"fg": "#ffffff", "bg": "#475569"}

C_BOS_UP = {"fg": "#111111", "bg": "#fde047"}
C_BOS_DN = {"fg": "#ffffff", "bg": "#dc2626"}

C_CH_UP = {"fg": "#111111", "bg": "#7dd3fc"}
C_CH_DN = {"fg": "#111111", "bg": "#fb923c"}

C_FLOOR = "#22c55e"
C_CEIL = "#ef4444"
C_BROKEN = "#a8a29e"
C_BOTH = "#eab308"

C_FAST_EMA = "#3b82f6"
C_SLOW_EMA = "#f59e0b"
C_VOL_MA = "#ff6d00"

C_AUTO_SUPPORT = "#22c55e"
C_AUTO_RESISTANCE = "#ef4444"
C_AUTO_PIVOT = "#eab308"

C_DEVELOPING_HIGH = "#38bdf8"
C_DEVELOPING_LOW = "#c084fc"

C_MANUAL_SR = "#eab308"


# ============================================================
# SESSION STATE
# ============================================================
SESSION_DEFAULTS = {
    "sim_active": False,
    "df": pd.DataFrame(),
    "step": 0,
    "active_ticker": None,
    "active_interval": None,
    "active_session_mode": None,
    "active_practice_date": None,
    "active_carry_mode": True,
    "markers": [],
    "draw_clicks": [],
    "last_draw_signature": None,
    "last_draw_mode": "None",
    "camera_signature": None,
    "camera_revision": 0,
    "force_camera": True,
}

for state_key, default_value in SESSION_DEFAULTS.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value


# ============================================================
# GENERAL HELPERS
# ============================================================
def utc_now_text():
    return datetime.now(timezone.utc).isoformat()


def to_pydate(value):
    if value is None:
        return None

    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    try:
        timestamp = pd.Timestamp(value)

        if pd.isna(timestamp):
            return None

        return timestamp.date()

    except Exception:
        return None


def detect_asset_class(ticker):
    symbol = str(ticker).upper().strip()

    if symbol.endswith("=F"):
        return "futures"

    if symbol.endswith("=X"):
        return "forex"

    if symbol.endswith("-USD"):
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


def is_continuous_mode(session_mode):
    return session_mode in {
        "CME Futures",
        "Forex 24/5",
        "Crypto 24/7",
    }


def price_decimals(price):
    try:
        value = abs(float(price))
    except Exception:
        value = 0.0

    if value >= 10:
        return 2

    if value >= 1:
        return 3

    if value >= 0.01:
        return 5

    return 8


def format_price(price, include_dollar=True):
    try:
        value = float(price)
    except Exception:
        value = 0.0

    decimals = price_decimals(value)
    prefix = "$" if include_dollar else ""

    return f"{prefix}{value:,.{decimals}f}"


def price_input_format(price):
    return f"%.{price_decimals(price)}f"


def price_input_step(price):
    return float(10 ** (-price_decimals(price)))


def bars_for_minutes(target_minutes, native_minutes):
    if native_minutes >= 1440:
        return 1

    return max(
        1,
        int(np.ceil(float(target_minutes) / float(native_minutes))),
    )


# ============================================================
# DATABASE
# ============================================================
def get_connection():
    connection = sqlite3.connect(
        DB_PATH,
        timeout=30,
    )

    connection.row_factory = sqlite3.Row
    return connection


def table_exists(connection, table_name):
    row = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table_name,),
    ).fetchone()

    return row is not None


def table_columns(connection, table_name):
    if not table_exists(connection, table_name):
        return set()

    return {
        row[1]
        for row in connection.execute(
            f"PRAGMA table_info({table_name})"
        )
    }


def row_value(row, key, default=None):
    if row is None:
        return default

    try:
        if key in row.keys() and row[key] is not None:
            return row[key]
    except Exception:
        pass

    return default


def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()
    now = utc_now_text()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS replay_accounts (
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
        CREATE TABLE IF NOT EXISTS replay_positions (
            account_id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            asset_class TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            entry_price REAL NOT NULL,
            entry_time TEXT NOT NULL,
            stop_loss REAL,
            target REAL,
            invested_amount REAL NOT NULL,
            last_mark_price REAL NOT NULL,
            last_mark_time TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS replay_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            cycle_number INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            asset_class TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT NOT NULL,
            realized_pnl REAL NOT NULL,
            fees REAL NOT NULL DEFAULT 0,
            reason TEXT DEFAULT ''
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS replay_events (
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
        """
        CREATE TABLE IF NOT EXISTS replay_drawings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            drawing_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    existing_account = cursor.execute(
        """
        SELECT account_id
        FROM replay_accounts
        WHERE account_id = ?
        """,
        (ACCOUNT_ID,),
    ).fetchone()

    if existing_account is None:
        starting_balance = STARTING_BALANCE
        cash_balance = STARTING_BALANCE
        realized_pnl = 0.0
        cycle_number = 1

        # Best-effort migration from the previous "accounts" table.
        if table_exists(connection, "accounts"):
            columns = table_columns(connection, "accounts")

            required = {
                "cash_balance",
                "realized_pnl",
                "cycle_number",
            }

            if required.issubset(columns):
                old_account = connection.execute(
                    """
                    SELECT *
                    FROM accounts
                    WHERE account_id = ?
                    LIMIT 1
                    """,
                    (ACCOUNT_ID,),
                ).fetchone()

                if old_account is not None:
                    starting_balance = float(
                        row_value(
                            old_account,
                            "starting_balance",
                            STARTING_BALANCE,
                        )
                    )

                    cash_balance = float(
                        row_value(
                            old_account,
                            "cash_balance",
                            STARTING_BALANCE,
                        )
                    )

                    realized_pnl = float(
                        row_value(
                            old_account,
                            "realized_pnl",
                            0.0,
                        )
                    )

                    cycle_number = int(
                        row_value(
                            old_account,
                            "cycle_number",
                            1,
                        )
                    )

        # Best-effort migration from the oldest "account" table.
        elif table_exists(connection, "account"):
            columns = table_columns(connection, "account")

            if "balance" in columns:
                if "status" in columns:
                    old_account = connection.execute(
                        """
                        SELECT *
                        FROM account
                        WHERE status = 'ACTIVE'
                        ORDER BY id DESC
                        LIMIT 1
                        """
                    ).fetchone()
                else:
                    old_account = connection.execute(
                        """
                        SELECT *
                        FROM account
                        ORDER BY id DESC
                        LIMIT 1
                        """
                    ).fetchone()

                if old_account is not None:
                    cash_balance = float(
                        row_value(
                            old_account,
                            "balance",
                            STARTING_BALANCE,
                        )
                    )

                    realized_pnl = float(
                        row_value(
                            old_account,
                            "realized_pnl",
                            0.0,
                        )
                    )

                    cycle_number = int(
                        row_value(
                            old_account,
                            "cycle",
                            1,
                        )
                    )

        cursor.execute(
            """
            INSERT INTO replay_accounts (
                account_id,
                starting_balance,
                cash_balance,
                realized_pnl,
                cycle_number,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ACCOUNT_ID,
                starting_balance,
                cash_balance,
                realized_pnl,
                cycle_number,
                now,
                now,
            ),
        )

    connection.commit()
    connection.close()


def load_account():
    connection = get_connection()

    row = connection.execute(
        """
        SELECT *
        FROM replay_accounts
        WHERE account_id = ?
        """,
        (ACCOUNT_ID,),
    ).fetchone()

    connection.close()

    if row is None:
        return {
            "account_id": ACCOUNT_ID,
            "starting_balance": STARTING_BALANCE,
            "cash_balance": STARTING_BALANCE,
            "realized_pnl": 0.0,
            "cycle_number": 1,
        }

    return dict(row)


def save_account(cash_balance, realized_pnl):
    connection = get_connection()

    connection.execute(
        """
        UPDATE replay_accounts
        SET cash_balance = ?,
            realized_pnl = ?,
            updated_at = ?
        WHERE account_id = ?
        """,
        (
            float(cash_balance),
            float(realized_pnl),
            utc_now_text(),
            ACCOUNT_ID,
        ),
    )

    connection.commit()
    connection.close()


def load_position():
    connection = get_connection()

    row = connection.execute(
        """
        SELECT *
        FROM replay_positions
        WHERE account_id = ?
        """,
        (ACCOUNT_ID,),
    ).fetchone()

    connection.close()

    if row is None:
        return None

    return dict(row)


def save_position(position):
    connection = get_connection()

    if position is None:
        connection.execute(
            """
            DELETE FROM replay_positions
            WHERE account_id = ?
            """,
            (ACCOUNT_ID,),
        )

    else:
        connection.execute(
            """
            INSERT INTO replay_positions (
                account_id,
                ticker,
                asset_class,
                side,
                quantity,
                entry_price,
                entry_time,
                stop_loss,
                target,
                invested_amount,
                last_mark_price,
                last_mark_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_id) DO UPDATE SET
                ticker = excluded.ticker,
                asset_class = excluded.asset_class,
                side = excluded.side,
                quantity = excluded.quantity,
                entry_price = excluded.entry_price,
                entry_time = excluded.entry_time,
                stop_loss = excluded.stop_loss,
                target = excluded.target,
                invested_amount = excluded.invested_amount,
                last_mark_price = excluded.last_mark_price,
                last_mark_time = excluded.last_mark_time
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
                float(position["last_mark_price"]),
                str(position.get("last_mark_time", "")),
            ),
        )

    connection.commit()
    connection.close()


def update_position_mark(ticker, price, timestamp):
    connection = get_connection()

    connection.execute(
        """
        UPDATE replay_positions
        SET last_mark_price = ?,
            last_mark_time = ?
        WHERE account_id = ?
          AND ticker = ?
        """,
        (
            float(price),
            str(timestamp),
            ACCOUNT_ID,
            ticker,
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
        INSERT INTO replay_trades (
            account_id,
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
            fees,
            reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ACCOUNT_ID,
            int(account["cycle_number"]),
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


def save_account_event(
    event_type,
    message,
    cycle_number=None,
):
    account = load_account()

    if cycle_number is None:
        cycle_number = int(account["cycle_number"])

    connection = get_connection()

    connection.execute(
        """
        INSERT INTO replay_events (
            account_id,
            cycle_number,
            event_type,
            message,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            ACCOUNT_ID,
            int(cycle_number),
            event_type,
            message,
            utc_now_text(),
        ),
    )

    connection.commit()
    connection.close()


def reset_account_cycle():
    account = load_account()
    old_cycle = int(account["cycle_number"])
    new_cycle = old_cycle + 1

    save_account_event(
        "ACCOUNT_DEPLETED",
        f"Account cycle {old_cycle} ended after equity depletion.",
        cycle_number=old_cycle,
    )

    connection = get_connection()

    connection.execute(
        """
        UPDATE replay_accounts
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
            new_cycle,
            utc_now_text(),
            ACCOUNT_ID,
        ),
    )

    connection.execute(
        """
        DELETE FROM replay_positions
        WHERE account_id = ?
        """,
        (ACCOUNT_ID,),
    )

    connection.commit()
    connection.close()

    save_account_event(
        "NEW_ACCOUNT_CYCLE",
        f"New paper account funded with ${STARTING_BALANCE:.2f}.",
        cycle_number=new_cycle,
    )


# ============================================================
# PERSISTENT DRAWINGS
# ============================================================
def add_persistent_drawing(
    ticker,
    drawing_type,
    payload,
):
    if not ticker:
        return

    connection = get_connection()

    connection.execute(
        """
        INSERT INTO replay_drawings (
            account_id,
            ticker,
            drawing_type,
            payload_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            ACCOUNT_ID,
            ticker,
            drawing_type,
            json.dumps(payload),
            utc_now_text(),
        ),
    )

    connection.commit()
    connection.close()


def load_persistent_drawings(ticker):
    if not ticker:
        return []

    connection = get_connection()

    rows = connection.execute(
        """
        SELECT *
        FROM replay_drawings
        WHERE account_id = ?
          AND ticker = ?
        ORDER BY id ASC
        """,
        (
            ACCOUNT_ID,
            ticker,
        ),
    ).fetchall()

    connection.close()

    drawings = []

    for row in rows:
        try:
            payload = json.loads(row["payload_json"])
            payload["id"] = row["id"]
            payload["type"] = row["drawing_type"]
            drawings.append(payload)
        except Exception:
            continue

    return drawings


def undo_last_persistent_drawing(ticker):
    if not ticker:
        return

    connection = get_connection()

    row = connection.execute(
        """
        SELECT id
        FROM replay_drawings
        WHERE account_id = ?
          AND ticker = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (
            ACCOUNT_ID,
            ticker,
        ),
    ).fetchone()

    if row is not None:
        connection.execute(
            """
            DELETE FROM replay_drawings
            WHERE id = ?
            """,
            (row["id"],),
        )

    connection.commit()
    connection.close()


def clear_persistent_drawings(ticker):
    if not ticker:
        return

    connection = get_connection()

    connection.execute(
        """
        DELETE FROM replay_drawings
        WHERE account_id = ?
          AND ticker = ?
        """,
        (
            ACCOUNT_ID,
            ticker,
        ),
    )

    connection.commit()
    connection.close()


def add_horizontal_line(
    price,
    label="",
    color="#22d3ee",
):
    add_persistent_drawing(
        st.session_state.active_ticker,
        "hline",
        {
            "price": float(price),
            "label": label,
            "color": color,
        },
    )


def add_band(price_1, price_2):
    add_persistent_drawing(
        st.session_state.active_ticker,
        "band",
        {
            "y0": min(float(price_1), float(price_2)),
            "y1": max(float(price_1), float(price_2)),
            "color": "rgba(34,197,94,0.18)",
            "line_color": "#22c55e",
        },
    )


def add_trend_line(x0, y0, x1, y1):
    add_persistent_drawing(
        st.session_state.active_ticker,
        "trend",
        {
            "x0": str(pd.Timestamp(x0)),
            "y0": float(y0),
            "x1": str(pd.Timestamp(x1)),
            "y1": float(y1),
            "color": "#f472b6",
        },
    )


# ============================================================
# MARKET CALENDAR
# ============================================================
def nth_weekday(year, month, weekday, occurrence):
    first_day = date(year, month, 1)
    offset = (weekday - first_day.weekday()) % 7

    return first_day + timedelta(
        days=offset + (occurrence - 1) * 7
    )


def last_weekday(year, month, weekday):
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)

    final_day = next_month - timedelta(days=1)
    offset = (final_day.weekday() - weekday) % 7

    return final_day - timedelta(days=offset)


def observed_fixed_holiday(year, month, day_value):
    actual = date(year, month, day_value)

    if actual.weekday() == 5:
        return actual - timedelta(days=1)

    if actual.weekday() == 6:
        return actual + timedelta(days=1)

    return actual


def calculate_easter(year):
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
    day_value = ((h + l - 7 * m + 114) % 31) + 1

    return date(year, month, day_value)


def us_market_holidays(year):
    holidays = {
        observed_fixed_holiday(year, 1, 1),
        nth_weekday(year, 1, 0, 3),
        nth_weekday(year, 2, 0, 3),
        calculate_easter(year) - timedelta(days=2),
        last_weekday(year, 5, 0),
        observed_fixed_holiday(year, 7, 4),
        nth_weekday(year, 9, 0, 1),
        nth_weekday(year, 11, 3, 4),
        observed_fixed_holiday(year, 12, 25),
    }

    if year >= 2022:
        holidays.add(
            observed_fixed_holiday(year, 6, 19)
        )

    return holidays


def is_us_market_holiday(day_value):
    day_value = to_pydate(day_value)

    if day_value is None:
        return False

    return day_value in us_market_holidays(day_value.year)


def is_early_close(day_value):
    day_value = to_pydate(day_value)

    if day_value is None:
        return False

    year = day_value.year

    return day_value in {
        date(year, 7, 3),
        nth_weekday(year, 11, 3, 4) + timedelta(days=1),
        date(year, 12, 24),
    }


def futures_session_date(timestamp):
    timestamp = pd.Timestamp(timestamp)
    session_date = timestamp.date()

    if timestamp.time() >= time(18, 0):
        session_date += timedelta(days=1)

    return session_date


def forex_session_date(timestamp):
    timestamp = pd.Timestamp(timestamp)
    session_date = timestamp.date()

    if timestamp.time() >= time(17, 0):
        session_date += timedelta(days=1)

    return session_date


# ============================================================
# DATA FUNCTIONS
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

    open_column = find_column(dataframe, ["Open"])
    high_column = find_column(dataframe, ["High"])
    low_column = find_column(dataframe, ["Low"])
    close_column = find_column(
        dataframe,
        ["Close", "Adj Close"],
    )
    volume_column = find_column(dataframe, ["Volume"])

    if any(
        column is None
        for column in [
            open_column,
            high_column,
            low_column,
            close_column,
        ]
    ):
        return pd.DataFrame()

    output = pd.DataFrame(
        {
            "timestamp": dataframe[dataframe.columns[0]],
            "open": dataframe[open_column],
            "high": dataframe[high_column],
            "low": dataframe[low_column],
            "close": dataframe[close_column],
            "volume": (
                dataframe[volume_column]
                if volume_column is not None
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

    valid_ohlc = (
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

    output = output[valid_ohlc].copy()

    if output.empty:
        return pd.DataFrame()

    output["bar_ratio"] = output["high"] / output["low"]

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

    reference = (
        rolling_median
        .fillna(expanding_median)
        .replace(0, np.nan)
    )

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

    return output.reset_index(drop=True)


@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def fetch_history(ticker, timeframe):
    configuration = TIMEFRAME_CONFIG[timeframe]

    for period in [
        configuration["period"],
        configuration["fallback_period"],
    ]:
        try:
            raw = yf.download(
                ticker,
                period=period,
                interval=configuration["yf_interval"],
                auto_adjust=True,
                progress=False,
                prepost=True,
                threads=False,
            )

            dataframe = normalize_ohlcv(
                raw,
                intraday=configuration["intraday"],
            )

            if not dataframe.empty:
                return dataframe

        except Exception:
            continue

    return pd.DataFrame()


def apply_session_filter(
    dataframe,
    session_mode,
    intraday=True,
):
    if dataframe.empty:
        return dataframe.copy()

    output = dataframe.copy()

    output["timestamp"] = pd.to_datetime(
        output["timestamp"]
    )

    if not intraday:
        if session_mode in {
            "US Stocks / ETFs - Regular Hours",
            "US Stocks / ETFs - Extended Hours",
        }:
            output = output[
                output["timestamp"].dt.weekday < 5
            ].copy()

            output = output[
                ~output["calendar_date"].apply(
                    is_us_market_holiday
                )
            ].copy()

        elif session_mode in {
            "CME Futures",
            "Forex 24/5",
        }:
            output = output[
                output["timestamp"].dt.weekday < 5
            ].copy()

        output["session_date"] = (
            output["calendar_date"].map(to_pydate)
        )

        return output.reset_index(drop=True)

    weekday = output["timestamp"].dt.weekday
    timestamp_time = output["timestamp"].dt.time

    if session_mode == "US Stocks / ETFs - Regular Hours":
        output = output[weekday < 5].copy()

        output = output[
            ~output["calendar_date"].apply(
                is_us_market_holiday
            )
        ].copy()

        minutes = (
            output["timestamp"].dt.hour * 60
            + output["timestamp"].dt.minute
        )

        closing_minutes = np.where(
            output["calendar_date"].apply(
                is_early_close
            ),
            13 * 60,
            16 * 60,
        )

        output = output[
            (minutes >= 9 * 60 + 30)
            & (minutes < closing_minutes)
        ].copy()

        output["session_date"] = (
            output["calendar_date"].map(to_pydate)
        )

    elif session_mode == "US Stocks / ETFs - Extended Hours":
        output = output[weekday < 5].copy()

        output = output[
            ~output["calendar_date"].apply(
                is_us_market_holiday
            )
        ].copy()

        minutes = (
            output["timestamp"].dt.hour * 60
            + output["timestamp"].dt.minute
        )

        output = output[
            (minutes >= 4 * 60)
            & (minutes < 20 * 60)
        ].copy()

        output["session_date"] = (
            output["calendar_date"].map(to_pydate)
        )

    elif session_mode == "CME Futures":
        sunday_valid = (
            (weekday == 6)
            & (timestamp_time >= time(18, 0))
        )

        monday_thursday_valid = (
            (weekday >= 0)
            & (weekday <= 3)
            & ~(
                (timestamp_time >= time(17, 0))
                & (timestamp_time < time(18, 0))
            )
        )

        friday_valid = (
            (weekday == 4)
            & (timestamp_time < time(17, 0))
        )

        output = output[
            sunday_valid
            | monday_thursday_valid
            | friday_valid
        ].copy()

        output["session_date"] = (
            output["timestamp"].apply(
                futures_session_date
            )
        )

    elif session_mode == "Forex 24/5":
        sunday_valid = (
            (weekday == 6)
            & (timestamp_time >= time(17, 0))
        )

        monday_thursday_valid = (
            (weekday >= 0)
            & (weekday <= 3)
        )

        friday_valid = (
            (weekday == 4)
            & (timestamp_time < time(17, 0))
        )

        output = output[
            sunday_valid
            | monday_thursday_valid
            | friday_valid
        ].copy()

        output["session_date"] = (
            output["timestamp"].apply(
                forex_session_date
            )
        )

    elif session_mode == "Crypto 24/7":
        output["session_date"] = (
            output["calendar_date"].map(to_pydate)
        )

    else:
        output["session_date"] = (
            output["calendar_date"].map(to_pydate)
        )

    output["session_date"] = (
        output["session_date"].map(to_pydate)
    )

    return (
        output
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def apply_custom_session(
    dataframe,
    start_time,
    end_time,
    include_weekends,
):
    if dataframe.empty:
        return dataframe.copy()

    output = dataframe.copy()

    if not include_weekends:
        output = output[
            output["timestamp"].dt.weekday < 5
        ].copy()

    minutes = (
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
            (minutes >= start_minutes)
            & (minutes < end_minutes)
        ].copy()

        output["session_date"] = (
            output["calendar_date"].map(to_pydate)
        )

    else:
        output = output[
            (minutes >= start_minutes)
            | (minutes < end_minutes)
        ].copy()

        def custom_session_date(timestamp):
            timestamp = pd.Timestamp(timestamp)
            result = timestamp.date()

            if timestamp.time() >= start_time:
                result += timedelta(days=1)

            return result

        output["session_date"] = (
            output["timestamp"].apply(
                custom_session_date
            )
        )

    return output.reset_index(drop=True)


def moving_average(
    series,
    length,
    average_type,
):
    if average_type == "EMA":
        return series.ewm(
            span=int(length),
            adjust=False,
        ).mean()

    return series.rolling(
        int(length)
    ).mean()


def get_context_dataframe(
    revealed_dataframe,
    current_session_date,
    preset,
):
    if revealed_dataframe.empty:
        return revealed_dataframe.copy()

    session_series = (
        revealed_dataframe["session_date"]
        .map(to_pydate)
    )

    session_dates = sorted(
        {
            session_date
            for session_date in session_series
            if session_date is not None
        }
    )

    current_session_date = to_pydate(
        current_session_date
    )

    if current_session_date not in session_dates:
        return revealed_dataframe.copy()

    previous_sessions_map = {
        "Today": 0,
        "1 Previous Day": 1,
        "5 Previous Days": 5,
        "20 Previous Days": 20,
        "All Available History": 999999,
    }

    current_index = session_dates.index(
        current_session_date
    )

    previous_sessions = previous_sessions_map.get(
        preset,
        0,
    )

    start_index = max(
        0,
        current_index - previous_sessions,
    )

    start_session = session_dates[start_index]

    return revealed_dataframe[
        session_series >= start_session
    ].copy()


def build_chart_dataframe(
    calculated_dataframe,
    current_session_date,
    context_preset,
    continuous_market,
    follow_enabled,
    follow_bar_count,
):
    if calculated_dataframe.empty:
        return calculated_dataframe.copy()

    if continuous_market and follow_enabled:
        return (
            calculated_dataframe
            .tail(int(follow_bar_count))
            .copy()
            .reset_index(drop=True)
        )

    return (
        get_context_dataframe(
            calculated_dataframe,
            current_session_date,
            context_preset,
        )
        .reset_index(drop=True)
    )


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
                )
                .dropna()
                .tolist()
            )

    if not values:
        return None

    low_value = float(min(values))
    high_value = float(max(values))

    median_price = float(
        pd.to_numeric(
            dataframe["close"],
            errors="coerce",
        ).median()
    )

    minimum_spread = max(
        abs(median_price) * 0.003,
        1e-12,
    )

    spread = max(
        high_value - low_value,
        minimum_spread,
    )

    padding = max(
        spread * 0.12,
        abs(median_price) * 0.001,
        1e-12,
    )

    return [
        low_value - padding,
        high_value + padding,
    ]


def get_volume_axis_range(dataframe):
    if dataframe.empty:
        return None

    maximum_volume = float(
        pd.to_numeric(
            dataframe["volume"],
            errors="coerce",
        )
        .fillna(0.0)
        .max()
    )

    if maximum_volume <= 0:
        return None

    return [
        0,
        maximum_volume * 1.10,
    ]


def disable_rangesliders(figure):
    figure.update_layout(
        xaxis_rangeslider_visible=False
    )

    figure.update_xaxes(
        rangeslider_visible=False
    )


# ============================================================
# AUTOMATIC 5-SESSION SUPPORT / RESISTANCE
# ============================================================
def compute_five_session_sr(
    revealed_dataframe,
    current_session_date,
    current_price,
    max_levels=6,
):
    if revealed_dataframe.empty:
        return [], [], 0.0

    working = revealed_dataframe.copy()

    working["_session"] = (
        working["session_date"].map(to_pydate)
    )

    working = working.dropna(
        subset=["_session"]
    )

    current_session_date = to_pydate(
        current_session_date
    )

    completed_session_dates = sorted(
        {
            session_date
            for session_date in working["_session"].unique()
            if session_date < current_session_date
        }
    )

    selected_sessions = completed_session_dates[
        -AUTO_SR_LOOKBACK_SESSIONS:
    ]

    if not selected_sessions:
        return [], [], 0.0

    candidates = []
    session_ranges = []

    for sequence, session_date_value in enumerate(
        selected_sessions,
        start=1,
    ):
        session_rows = working[
            working["_session"] == session_date_value
        ]

        if session_rows.empty:
            continue

        session_high = float(
            session_rows["high"].max()
        )

        session_low = float(
            session_rows["low"].min()
        )

        session_ranges.append(
            session_high - session_low
        )

        candidates.append(
            {
                "price": session_high,
                "kind": "H",
                "session": session_date_value,
                "recency": sequence,
            }
        )

        candidates.append(
            {
                "price": session_low,
                "kind": "L",
                "session": session_date_value,
                "recency": sequence,
            }
        )

    if not candidates:
        return [], selected_sessions, 0.0

    valid_ranges = [
        value
        for value in session_ranges
        if value > 0
    ]

    median_session_range = (
        float(np.median(valid_ranges))
        if valid_ranges
        else 0.0
    )

    merge_tolerance = max(
        abs(float(current_price)) * 0.0015,
        median_session_range * 0.08,
        1e-12,
    )

    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: candidate["price"],
    )

    clusters = []

    for candidate in sorted_candidates:
        if not clusters:
            clusters.append(
                {
                    "candidates": [candidate],
                }
            )
            continue

        previous_cluster = clusters[-1]

        cluster_center = float(
            np.median(
                [
                    item["price"]
                    for item in previous_cluster["candidates"]
                ]
            )
        )

        if (
            abs(candidate["price"] - cluster_center)
            <= merge_tolerance
        ):
            previous_cluster["candidates"].append(
                candidate
            )
        else:
            clusters.append(
                {
                    "candidates": [candidate],
                }
            )

    levels = []

    for cluster_index, cluster in enumerate(clusters):
        cluster_candidates = cluster["candidates"]

        cluster_prices = [
            item["price"]
            for item in cluster_candidates
        ]

        center_price = float(
            np.median(cluster_prices)
        )

        high_count = sum(
            1
            for item in cluster_candidates
            if item["kind"] == "H"
        )

        low_count = sum(
            1
            for item in cluster_candidates
            if item["kind"] == "L"
        )

        recency_score = sum(
            item["recency"]
            for item in cluster_candidates
        )

        levels.append(
            {
                "cluster_index": cluster_index,
                "price": center_price,
                "zone_low": float(min(cluster_prices)),
                "zone_high": float(max(cluster_prices)),
                "touches": len(cluster_candidates),
                "high_count": high_count,
                "low_count": low_count,
                "recency_score": recency_score,
                "sessions": sorted(
                    {
                        item["session"]
                        for item in cluster_candidates
                    }
                ),
            }
        )

    if not levels:
        return [], selected_sessions, merge_tolerance

    raw_minimum = min(
        candidate["price"]
        for candidate in candidates
    )

    raw_maximum = max(
        candidate["price"]
        for candidate in candidates
    )

    most_recent_session = selected_sessions[-1]

    mandatory_indices = set()

    for level_index, level in enumerate(levels):
        cluster_candidates = clusters[
            level["cluster_index"]
        ]["candidates"]

        cluster_prices = [
            item["price"]
            for item in cluster_candidates
        ]

        if raw_minimum in cluster_prices:
            mandatory_indices.add(level_index)

        if raw_maximum in cluster_prices:
            mandatory_indices.add(level_index)

        if any(
            item["session"] == most_recent_session
            for item in cluster_candidates
        ):
            mandatory_indices.add(level_index)

    def importance(level):
        distance = abs(
            level["price"] - float(current_price)
        )

        normalized_distance = (
            distance
            / max(abs(float(current_price)), 1e-12)
        )

        return (
            level["touches"] * 10
            + level["recency_score"]
            - normalized_distance
        )

    ranked_indices = sorted(
        range(len(levels)),
        key=lambda index: importance(levels[index]),
        reverse=True,
    )

    selected_indices = []

    for index in sorted(mandatory_indices):
        if index not in selected_indices:
            selected_indices.append(index)

        if len(selected_indices) >= max_levels:
            break

    for index in ranked_indices:
        if len(selected_indices) >= max_levels:
            break

        if index not in selected_indices:
            selected_indices.append(index)

    selected_levels = [
        levels[index]
        for index in selected_indices
    ]

    selected_levels = sorted(
        selected_levels,
        key=lambda level: level["price"],
    )

    return (
        selected_levels,
        selected_sessions,
        merge_tolerance,
    )


def get_developing_session_levels(
    revealed_dataframe,
    current_session_date,
):
    if revealed_dataframe.empty:
        return None

    current_session_date = to_pydate(
        current_session_date
    )

    session_rows = revealed_dataframe[
        revealed_dataframe["session_date"].map(to_pydate)
        == current_session_date
    ]

    if session_rows.empty:
        return None

    return {
        "high": float(session_rows["high"].max()),
        "low": float(session_rows["low"].min()),
        "bars": len(session_rows),
    }


# ============================================================
# LOCAL MARKET STRUCTURE
# ============================================================
def raw_swings(dataframe):
    if (
        dataframe is None
        or len(dataframe) < SWING_K * 2 + 1
    ):
        return []

    highs = dataframe["high"].values
    lows = dataframe["low"].values
    row_count = len(dataframe)

    reference_price = float(
        dataframe["close"].iloc[-1]
    )

    minimum_size = max(
        reference_price * MIN_SWING_PCT,
        1e-12,
    )

    raw = []

    for index in range(
        SWING_K,
        row_count - SWING_K,
    ):
        high_window = highs[
            index - SWING_K:
            index + SWING_K + 1
        ]

        low_window = lows[
            index - SWING_K:
            index + SWING_K + 1
        ]

        if (
            highs[index] == high_window.max()
            and highs[index] - low_window.min()
            >= minimum_size
        ):
            raw.append(
                (
                    index,
                    float(highs[index]),
                    "H",
                )
            )

        if (
            lows[index] == low_window.min()
            and high_window.max() - lows[index]
            >= minimum_size
        ):
            raw.append(
                (
                    index,
                    float(lows[index]),
                    "L",
                )
            )

    raw.sort(key=lambda value: value[0])

    cleaned = []

    for swing in raw:
        if (
            cleaned
            and cleaned[-1][2] == swing[2]
        ):
            previous = cleaned[-1]

            if (
                swing[2] == "H"
                and swing[1] >= previous[1]
            ):
                cleaned[-1] = swing

            elif (
                swing[2] == "L"
                and swing[1] <= previous[1]
            ):
                cleaned[-1] = swing
        else:
            cleaned.append(swing)

    labeled = []
    previous_high = None
    previous_low = None

    for index, price, swing_type in cleaned:
        if swing_type == "H":
            if previous_high is None:
                label = "H"
            elif price > previous_high:
                label = "HH"
            else:
                label = "LH"

            previous_high = price

        else:
            if previous_low is None:
                label = "L"
            elif price > previous_low:
                label = "HL"
            else:
                label = "LL"

            previous_low = price

        labeled.append(
            {
                "i": index,
                "price": price,
                "type": swing_type,
                "label": label,
            }
        )

    return labeled


def build_local_zones(
    labeled_swings,
    reference_price,
):
    zones = []

    for swing in labeled_swings:
        placed = False

        for zone in zones:
            distance = abs(
                swing["price"] - zone["mid"]
            )

            normalized_distance = (
                distance
                / max(reference_price, 1e-12)
            )

            if normalized_distance < ZONE_TOL:
                zone["prices"].append(
                    swing["price"]
                )

                zone["mid"] = float(
                    np.mean(zone["prices"])
                )

                zone["touches"] += 1

                if swing["type"] == "L":
                    zone["low_touches"] += 1
                else:
                    zone["high_touches"] += 1

                placed = True
                break

        if not placed:
            zones.append(
                {
                    "mid": swing["price"],
                    "prices": [swing["price"]],
                    "touches": 1,
                    "low_touches": (
                        1
                        if swing["type"] == "L"
                        else 0
                    ),
                    "high_touches": (
                        1
                        if swing["type"] == "H"
                        else 0
                    ),
                }
            )

    output = [
        zone
        for zone in zones
        if zone["touches"] >= MIN_DRAW_TOUCHES
    ]

    for zone in output:
        zone["min_px"] = min(zone["prices"])
        zone["max_px"] = max(zone["prices"])

    return output


def detect_bos_choch(
    dataframe,
    labeled_swings,
):
    bos_events = []
    choch_events = []

    last_hh = None
    last_hl = None
    last_lh = None
    last_ll = None

    bias = "NEUTRAL"
    pointer = 0

    for index in range(len(dataframe)):
        while (
            pointer < len(labeled_swings)
            and labeled_swings[pointer]["i"] + SWING_K
            <= index
        ):
            swing = labeled_swings[pointer]

            if swing["label"] == "HH":
                last_hh = swing["price"]
            elif swing["label"] == "HL":
                last_hl = swing["price"]
            elif swing["label"] == "LH":
                last_lh = swing["price"]
            elif swing["label"] == "LL":
                last_ll = swing["price"]

            pointer += 1

        close_price = float(
            dataframe["close"].iloc[index]
        )

        if (
            last_hh is not None
            and close_price > last_hh
        ):
            if bias != "BULL":
                bos_events.append(
                    {
                        "i": index,
                        "price": last_hh,
                        "dir": "up",
                    }
                )

                bias = "BULL"

            last_hh = None
            continue

        if (
            last_ll is not None
            and close_price < last_ll
        ):
            if bias != "BEAR":
                bos_events.append(
                    {
                        "i": index,
                        "price": last_ll,
                        "dir": "down",
                    }
                )

                bias = "BEAR"

            last_ll = None
            continue

        if CHOCH_ENABLE:
            if (
                bias == "BULL"
                and last_hl is not None
                and close_price < last_hl
            ):
                choch_events.append(
                    {
                        "i": index,
                        "price": last_hl,
                        "dir": "down",
                    }
                )

                last_hl = None
                bias = "NEUTRAL"

            elif (
                bias == "BEAR"
                and last_lh is not None
                and close_price > last_lh
            ):
                choch_events.append(
                    {
                        "i": index,
                        "price": last_lh,
                        "dir": "up",
                    }
                )

                last_lh = None
                bias = "NEUTRAL"

    return bos_events, choch_events


def local_zone_role(
    zone,
    close_price,
    noise,
):
    if (
        zone["low_touches"]
        >= zone["high_touches"] + POLARITY_EDGE
    ):
        role = "floor"

    elif (
        zone["high_touches"]
        >= zone["low_touches"] + POLARITY_EDGE
    ):
        role = "ceiling"

    else:
        role = "both"

    if (
        role == "floor"
        and close_price < zone["min_px"] - noise
    ):
        return "broken_floor"

    if (
        role == "ceiling"
        and close_price > zone["max_px"] + noise
    ):
        return "broken_ceiling"

    return role


def add_badge(
    figure,
    x_value,
    y_value,
    text,
    palette,
    y_shift=0,
    arrow=False,
):
    """
    Used only for candle-specific market-structure labels.
    These labels stay next to the swing candle.
    """
    figure.add_annotation(
        x=x_value,
        y=y_value,
        row=1,
        col=1,
        text=f"<b>{text}</b>",
        showarrow=arrow,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.2,
        arrowcolor="#e5e7eb",
        yshift=y_shift,
        font={
            "size": 10,
            "color": palette["fg"],
            "family": "Arial",
        },
        bgcolor=palette["bg"],
        bordercolor="#e5e7eb",
        borderwidth=1,
        borderpad=3,
        opacity=1,
    )


def add_rail_label(
    figure,
    x_value,
    y_value,
    text,
    background_color,
    foreground_color="#ffffff",
    border_color="#e5e7eb",
    font_size=9,
):
    """
    Places an S/R label in the dedicated right-side rail.
    The corresponding horizontal line remains at the exact price.
    """
    figure.add_annotation(
        x=x_value,
        y=float(y_value),
        row=1,
        col=1,
        xanchor="left",
        yanchor="middle",
        text=f"<b>{text}</b>",
        showarrow=False,
        font={
            "size": font_size,
            "color": foreground_color,
            "family": "Arial",
        },
        bgcolor=background_color,
        bordercolor=border_color,
        borderwidth=1,
        borderpad=3,
        opacity=0.94,
        align="left",
    )


# ============================================================
# TRADING
# ============================================================
def add_marker(
    ticker,
    timestamp,
    price,
    marker_type,
):
    st.session_state.markers.append(
        {
            "ticker": ticker,
            "timestamp": pd.Timestamp(timestamp),
            "price": float(price),
            "kind": marker_type,
        }
    )


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

    cash_balance = float(
        account["cash_balance"]
    )

    if side == "long":
        cash_balance -= float(amount)
    else:
        cash_balance += float(amount)

    save_account(
        cash_balance,
        float(account["realized_pnl"]),
    )

    save_position(
        {
            "ticker": ticker,
            "asset_class": detect_asset_class(ticker),
            "side": side,
            "quantity": quantity,
            "entry_price": float(price),
            "entry_time": str(timestamp),
            "stop_loss": stop_loss,
            "target": target,
            "invested_amount": float(amount),
            "last_mark_price": float(price),
            "last_mark_time": str(timestamp),
        }
    )


def close_position(
    price,
    timestamp,
    reason="",
    quantity_to_close=None,
):
    position = load_position()

    if position is None:
        return

    account = load_account()

    total_quantity = float(
        position["quantity"]
    )

    if quantity_to_close is None:
        quantity = total_quantity
    else:
        quantity = min(
            float(quantity_to_close),
            total_quantity,
        )

    if quantity <= 0:
        return

    entry_price = float(
        position["entry_price"]
    )

    exit_price = float(price)

    if position["side"] == "long":
        realized_pnl = (
            exit_price - entry_price
        ) * quantity

        new_cash = (
            float(account["cash_balance"])
            + exit_price * quantity
        )

    else:
        realized_pnl = (
            entry_price - exit_price
        ) * quantity

        new_cash = (
            float(account["cash_balance"])
            - exit_price * quantity
        )

    save_account(
        new_cash,
        float(account["realized_pnl"])
        + realized_pnl,
    )

    save_trade(
        ticker=position["ticker"],
        asset_class=position["asset_class"],
        side=position["side"],
        quantity=quantity,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_time=position["entry_time"],
        exit_time=timestamp,
        realized_pnl=realized_pnl,
        reason=reason,
    )

    remaining_quantity = (
        total_quantity - quantity
    )

    if remaining_quantity <= 1e-10:
        save_position(None)

    else:
        position["quantity"] = remaining_quantity

        position["invested_amount"] = (
            remaining_quantity * entry_price
        )

        position["last_mark_price"] = exit_price
        position["last_mark_time"] = str(timestamp)

        save_position(position)


def calculate_account_values(
    account,
    position,
):
    cash_balance = float(
        account["cash_balance"]
    )

    if position is None:
        return {
            "cash": cash_balance,
            "display_position_value": 0.0,
            "signed_position_value": 0.0,
            "equity": cash_balance,
            "unrealized_pnl": 0.0,
            "signed_quantity": 0.0,
        }

    quantity = float(position["quantity"])
    mark_price = float(position["last_mark_price"])
    entry_price = float(position["entry_price"])

    display_position_value = (
        quantity * mark_price
    )

    if position["side"] == "long":
        signed_position_value = (
            quantity * mark_price
        )

        unrealized_pnl = (
            mark_price - entry_price
        ) * quantity

        signed_quantity = quantity

    else:
        signed_position_value = (
            -quantity * mark_price
        )

        unrealized_pnl = (
            entry_price - mark_price
        ) * quantity

        signed_quantity = -quantity

    equity = (
        cash_balance
        + signed_position_value
    )

    return {
        "cash": cash_balance,
        "display_position_value": display_position_value,
        "signed_position_value": signed_position_value,
        "equity": equity,
        "unrealized_pnl": unrealized_pnl,
        "signed_quantity": signed_quantity,
    }


def maybe_reset_depleted_account():
    position = load_position()

    if position is not None:
        return

    account = load_account()

    if float(account["cash_balance"]) <= 0:
        reset_account_cycle()

        st.toast(
            "Account depleted. A new $1,000 account cycle started.",
            icon="⚠️",
        )


def check_account_depletion(
    active_ticker,
    current_price,
    current_timestamp,
):
    position = load_position()

    if position is None:
        maybe_reset_depleted_account()
        return

    if position["ticker"] != active_ticker:
        return

    update_position_mark(
        active_ticker,
        current_price,
        current_timestamp,
    )

    account = load_account()
    position = load_position()

    values = calculate_account_values(
        account,
        position,
    )

    if values["equity"] <= 0:
        marker_kind = (
            "SELL"
            if position["side"] == "long"
            else "COVER"
        )

        close_position(
            current_price,
            current_timestamp,
            reason="(ACCOUNT LIQUIDATION)",
        )

        add_marker(
            active_ticker,
            current_timestamp,
            current_price,
            marker_kind,
        )

        maybe_reset_depleted_account()


def advance_bars(number_of_bars):
    dataframe = st.session_state.df

    if dataframe.empty:
        return

    active_ticker = st.session_state.active_ticker
    maximum_step = len(dataframe) - 1

    for _ in range(int(number_of_bars)):
        if st.session_state.step >= maximum_step:
            st.toast(
                "End of available Yahoo history reached.",
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

        timestamp = pd.Timestamp(
            row["timestamp"]
        )

        position = load_position()

        position_is_active = (
            position is not None
            and position["ticker"] == active_ticker
        )

        if (
            position_is_active
            and not st.session_state.active_carry_mode
            and to_pydate(previous_row["session_date"])
            != to_pydate(row["session_date"])
        ):
            execution_price = float(
                previous_row["close"]
            )

            marker_kind = (
                "SELL"
                if position["side"] == "long"
                else "COVER"
            )

            close_position(
                execution_price,
                previous_row["timestamp"],
                reason="(SESSION FORCE-CLOSE)",
            )

            add_marker(
                active_ticker,
                previous_row["timestamp"],
                execution_price,
                marker_kind,
            )

            position = None
            position_is_active = False

        if position_is_active:
            stop_loss = position["stop_loss"]
            target = position["target"]

            if position["side"] == "long":
                if (
                    stop_loss is not None
                    and float(row["low"]) <= float(stop_loss)
                ):
                    execution_price = min(
                        float(stop_loss),
                        float(row["open"]),
                    )

                    close_position(
                        execution_price,
                        timestamp,
                        reason="(STOP-LOSS)",
                    )

                    add_marker(
                        active_ticker,
                        timestamp,
                        execution_price,
                        "SELL",
                    )

                    st.toast(
                        f"Long stopped at {format_price(execution_price)}",
                        icon="💥",
                    )
                    break

                if (
                    target is not None
                    and float(row["high"]) >= float(target)
                ):
                    execution_price = max(
                        float(target),
                        float(row["open"]),
                    )

                    close_position(
                        execution_price,
                        timestamp,
                        reason="(TARGET HIT)",
                    )

                    add_marker(
                        active_ticker,
                        timestamp,
                        execution_price,
                        "SELL",
                    )

                    st.toast(
                        f"Long target hit at {format_price(execution_price)}",
                        icon="🎉",
                    )
                    break

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
                        execution_price,
                        timestamp,
                        reason="(STOP-LOSS)",
                    )

                    add_marker(
                        active_ticker,
                        timestamp,
                        execution_price,
                        "COVER",
                    )

                    st.toast(
                        f"Short stopped at {format_price(execution_price)}",
                        icon="💥",
                    )
                    break

                if (
                    target is not None
                    and float(row["low"]) <= float(target)
                ):
                    execution_price = min(
                        float(target),
                        float(row["open"]),
                    )

                    close_position(
                        execution_price,
                        timestamp,
                        reason="(TARGET HIT)",
                    )

                    add_marker(
                        active_ticker,
                        timestamp,
                        execution_price,
                        "COVER",
                    )

                    st.toast(
                        f"Short target hit at {format_price(execution_price)}",
                        icon="🎉",
                    )
                    break

        check_account_depletion(
            active_ticker,
            float(row["close"]),
            timestamp,
        )

    maybe_reset_depleted_account()


# ============================================================
# DRAWING EVENT HELPERS
# ============================================================
def extract_selected_points(chart_event):
    if chart_event is None:
        return []

    try:
        return [
            dict(point)
            for point in chart_event.selection.points
        ]
    except Exception:
        pass

    try:
        return (
            chart_event
            .get("selection", {})
            .get("points", [])
        )
    except Exception:
        return []


def nearest_row_from_event(
    dataframe,
    point,
):
    if dataframe.empty:
        return None

    x_value = point.get("x")

    if x_value is not None:
        try:
            selected_index = int(
                round(float(x_value))
            )

            matching_rows = dataframe[
                dataframe["x_index"]
                == selected_index
            ]

            if not matching_rows.empty:
                return matching_rows.iloc[0]

        except Exception:
            pass

    point_index = point.get(
        "point_index",
        point.get("pointIndex"),
    )

    if point_index is not None:
        try:
            point_index = int(point_index)

            if 0 <= point_index < len(dataframe):
                return dataframe.iloc[point_index]

        except Exception:
            pass

    return dataframe.iloc[-1]


def process_drawing_click(
    mode,
    row,
    clicked_y=None,
):
    if row is None or mode == "None":
        return

    timestamp = pd.Timestamp(
        row["timestamp"]
    )

    try:
        anchor_price = float(clicked_y)
    except Exception:
        anchor_price = float(row["close"])

    if mode == "Line at High":
        price = float(row["high"])

        add_horizontal_line(
            price,
            label=(
                f"H {format_price(price, False)}"
            ),
            color="#ef4444",
        )

        st.toast(
            f"Persistent high line added at {format_price(price)}",
            icon="📌",
        )

    elif mode == "Line at Close":
        price = float(row["close"])

        add_horizontal_line(
            price,
            label=(
                f"C {format_price(price, False)}"
            ),
            color="#22d3ee",
        )

        st.toast(
            f"Persistent close line added at {format_price(price)}",
            icon="📌",
        )

    elif mode == "Line at Low":
        price = float(row["low"])

        add_horizontal_line(
            price,
            label=(
                f"L {format_price(price, False)}"
            ),
            color="#22c55e",
        )

        st.toast(
            f"Persistent low line added at {format_price(price)}",
            icon="📌",
        )

    elif mode == "Band":
        st.session_state.draw_clicks.append(
            {
                "timestamp": timestamp,
                "price": anchor_price,
            }
        )

        if len(st.session_state.draw_clicks) == 1:
            st.toast(
                "Band point one saved. Choose the second point.",
                icon="🖱️",
            )

        elif len(st.session_state.draw_clicks) >= 2:
            first = st.session_state.draw_clicks[0]
            second = st.session_state.draw_clicks[1]

            add_band(
                first["price"],
                second["price"],
            )

            st.session_state.draw_clicks = []

            st.toast(
                "Persistent band added.",
                icon="🟩",
            )

    elif mode == "Trend line":
        st.session_state.draw_clicks.append(
            {
                "timestamp": timestamp,
                "price": anchor_price,
            }
        )

        if len(st.session_state.draw_clicks) == 1:
            st.toast(
                "Trend point one saved. Choose the second point.",
                icon="🖱️",
            )

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

            st.toast(
                "Persistent trend line added.",
                icon="📈",
            )


# ============================================================
# INITIALIZE DATABASE
# ============================================================
initialize_database()


# ============================================================
# SIDEBAR — SETUP
# ============================================================
st.sidebar.header("⚙️ Setup")

ticker = (
    st.sidebar.text_input(
        "Ticker",
        value="NQ=F",
    )
    .upper()
    .strip()
)

timeframe = st.sidebar.selectbox(
    "Chart Timeframe",
    TIMEFRAME_OPTIONS,
    index=TIMEFRAME_OPTIONS.index("5m"),
)

timeframe_configuration = (
    TIMEFRAME_CONFIG[timeframe]
)

selected_session_mode = (
    st.sidebar.selectbox(
        "Session Mode",
        SESSION_OPTIONS,
        index=0,
    )
)

effective_session_mode = (
    normalize_session_mode(
        ticker,
        selected_session_mode,
    )
)

st.sidebar.info(
    "Detected/applied session: "
    f"**{effective_session_mode}**"
)

custom_start_time = time(9, 30)
custom_end_time = time(16, 0)
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

    custom_include_weekends = (
        st.sidebar.checkbox(
            "Include weekends",
            value=False,
        )
    )

st.sidebar.caption(
    "Yahoo native interval: "
    f"`{timeframe_configuration['yf_interval']}`"
)


# ============================================================
# LOAD AND FILTER DATA
# ============================================================
source_dataframe = pd.DataFrame()
filtered_dataframe = pd.DataFrame()

can_start = False
practice_date = None
selected_start_timestamp = None

if ticker:
    with st.sidebar:
        with st.spinner(
            f"Loading {timeframe} data for {ticker}..."
        ):
            source_dataframe = fetch_history(
                ticker,
                timeframe,
            )

st.sidebar.subheader(
    "📅 Practice Date & Time"
)

if source_dataframe.empty:
    st.sidebar.error(
        "No Yahoo data was returned. "
        "Check the ticker or timeframe."
    )

    st.sidebar.date_input(
        "Practice Session",
        value=date.today(),
        disabled=True,
        key="disabled_practice_date",
    )

    st.sidebar.selectbox(
        "Replay Start Time",
        ["--"],
        disabled=True,
        key="disabled_start_time",
    )

else:
    filtered_dataframe = apply_session_filter(
        source_dataframe,
        effective_session_mode,
        intraday=timeframe_configuration["intraday"],
    )

    if effective_session_mode == "Custom Session":
        filtered_dataframe = apply_custom_session(
            filtered_dataframe,
            custom_start_time,
            custom_end_time,
            custom_include_weekends,
        )

    if filtered_dataframe.empty:
        st.sidebar.error(
            "Yahoo returned data, but no bars "
            "matched the selected session."
        )

        st.sidebar.date_input(
            "Practice Session",
            value=date.today(),
            disabled=True,
            key="disabled_filtered_date",
        )

        st.sidebar.selectbox(
            "Replay Start Time",
            ["--"],
            disabled=True,
            key="disabled_filtered_time",
        )

    else:
        available_dates = sorted(
            {
                to_pydate(session_date)
                for session_date
                in filtered_dataframe["session_date"].unique()
                if to_pydate(session_date) is not None
            }
        )

        first_date = available_dates[0]
        last_date = available_dates[-1]

        if len(available_dates) >= 3:
            default_practice_date = available_dates[-3]
        else:
            default_practice_date = available_dates[-1]

        st.sidebar.success(
            f"{len(filtered_dataframe):,} session bars\n"
            f"{first_date} → {last_date}"
        )

        practice_date_key = (
            f"practice_date_{ticker}_"
            f"{timeframe}_"
            f"{effective_session_mode}"
        )

        saved_practice_date = to_pydate(
            st.session_state.get(practice_date_key)
        )

        if (
            saved_practice_date is None
            or saved_practice_date < first_date
            or saved_practice_date > last_date
        ):
            st.session_state[
                practice_date_key
            ] = default_practice_date

        practice_date = to_pydate(
            st.sidebar.date_input(
                "Practice Session",
                min_value=first_date,
                max_value=last_date,
                key=practice_date_key,
            )
        )

        practice_rows = (
            filtered_dataframe[
                filtered_dataframe["session_date"]
                .map(to_pydate)
                == practice_date
            ]
            .copy()
        )

        if practice_rows.empty:
            st.sidebar.warning(
                "No trading session exists for this date. "
                "For CME futures, Sunday evening belongs "
                "to the Monday trading session."
            )

            st.sidebar.selectbox(
                "Replay Start Time",
                ["--"],
                disabled=True,
                key=(
                    f"empty_start_{ticker}_"
                    f"{timeframe}_{practice_date}"
                ),
            )

        elif timeframe_configuration["intraday"]:
            practice_rows["start_label"] = (
                practice_rows["timestamp"]
                .dt.strftime("%Y-%m-%d %H:%M")
            )

            start_labels = (
                practice_rows["start_label"].tolist()
            )

            start_time_key = (
                f"start_time_{ticker}_"
                f"{timeframe}_{practice_date}"
            )

            selected_start_label = (
                st.sidebar.selectbox(
                    "Replay Start Time",
                    start_labels,
                    index=min(
                        10,
                        len(start_labels) - 1,
                    ),
                    key=start_time_key,
                )
            )

            selected_start_timestamp = (
                practice_rows[
                    practice_rows["start_label"]
                    == selected_start_label
                ]["timestamp"]
                .iloc[0]
            )

            can_start = True

        else:
            selected_start_timestamp = (
                practice_rows["timestamp"].iloc[0]
            )

            st.sidebar.caption(
                "Daily timeframe: replay starts "
                "at the selected session."
            )

            can_start = True


# ============================================================
# SIDEBAR — POSITION BEHAVIOR
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader(
    "📦 Position Behavior"
)

carry_positions = st.sidebar.checkbox(
    "Carry positions across sessions",
    value=True,
)

st.sidebar.caption(
    "When off, an open position is closed "
    "at the last bar of the trading session."
)


# ============================================================
# SIDEBAR — MOVING AVERAGES
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader(
    "📈 Moving Averages"
)

show_moving_averages = (
    st.sidebar.checkbox(
        "Show moving averages",
        value=True,
    )
)

moving_average_type = (
    st.sidebar.radio(
        "MA Type",
        ["EMA", "SMA"],
        horizontal=True,
    )
)

default_fast_length, default_slow_length = (
    timeframe_configuration["ema"]
)

fast_length = st.sidebar.number_input(
    "Fast MA length",
    min_value=2,
    max_value=500,
    value=int(default_fast_length),
    step=1,
    key=f"fast_length_{timeframe}",
)

slow_length = st.sidebar.number_input(
    "Slow MA length",
    min_value=2,
    max_value=500,
    value=int(default_slow_length),
    step=1,
    key=f"slow_length_{timeframe}",
)

show_volume_ma = st.sidebar.checkbox(
    "Show volume MA",
    value=True,
)

volume_ma_length = (
    st.sidebar.number_input(
        "Volume MA length",
        min_value=2,
        max_value=200,
        value=20,
        step=1,
        key=f"volume_ma_{timeframe}",
    )
)


# ============================================================
# SIDEBAR — STRUCTURE / S&R
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader(
    "🏗️ Structure & S/R"
)

show_structure_labels = (
    st.sidebar.checkbox(
        "Show structure labels",
        value=False,
    )
)

show_local_sr_zones = (
    st.sidebar.checkbox(
        "Show local chart S/R zones",
        value=False,
        help=(
            "Uses recent chart swings. "
            "This is more sensitive and may include noise."
        ),
    )
)

show_five_session_sr = (
    st.sidebar.checkbox(
        "Show automatic 5-session S/R",
        value=True,
        help=(
            "Uses highs and lows from the five "
            "completed sessions before the current session."
        ),
    )
)

maximum_auto_sr_levels = (
    st.sidebar.selectbox(
        "Maximum historical S/R levels",
        [4, 6, 8],
        index=1,
        disabled=not show_five_session_sr,
    )
)

show_developing_session_hl = (
    st.sidebar.checkbox(
        "Show developing current-session H/L",
        value=True,
    )
)

st.sidebar.caption(
    "S/R text is displayed in the right-side "
    "label rail so it does not cover candles."
)


# ============================================================
# SIDEBAR — CONTEXT / FOLLOW
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader(
    "📷 Chart Context"
)

context_preset = st.sidebar.selectbox(
    "Context preset",
    CONTEXT_PRESETS,
    index=2,
)

follow_replay = st.sidebar.checkbox(
    "Follow replay candle",
    value=True,
)

follow_bars = st.sidebar.radio(
    "Follow window (futures/forex/crypto)",
    FOLLOW_WINDOW_OPTIONS,
    index=1,
    horizontal=True,
)

st.sidebar.caption(
    "Continuous markets: Follow displays the "
    "most recent 100 or 200 trading bars. "
    "Maintenance breaks and weekends do not "
    "consume chart space.\n\n"
    "Stocks: Follow displays the current session.\n\n"
    "Follow OFF uses the selected context."
)


# ============================================================
# SIDEBAR — DRAWINGS
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

drawing_ticker = (
    st.session_state.active_ticker
    if st.session_state.sim_active
    else ticker
)

if st.sidebar.button(
    "Undo last drawing",
    width="stretch",
):
    undo_last_persistent_drawing(
        drawing_ticker
    )

    st.toast(
        "Removed the last saved drawing.",
        icon="↩️",
    )

    st.rerun()

if st.sidebar.button(
    "Clear all drawings",
    width="stretch",
):
    clear_persistent_drawings(
        drawing_ticker
    )

    st.session_state.draw_clicks = []
    st.session_state.last_draw_signature = None

    st.toast(
        f"Cleared saved drawings for {drawing_ticker}.",
        icon="🧹",
    )

    st.rerun()

if st.session_state.draw_clicks:
    st.sidebar.info(
        "Pending drawing points: "
        f"{len(st.session_state.draw_clicks)} / 2"
    )

st.sidebar.caption(
    "Manual drawings are saved in SQLite "
    "by ticker and survive replay resets."
)


# ============================================================
# START / RESET REPLAY
# ============================================================
st.sidebar.markdown("---")

if st.sidebar.button(
    "🚀 Start / Reset Replay",
    width="stretch",
    disabled=not can_start,
):
    active_dataframe = (
        filtered_dataframe
        .copy()
        .reset_index(drop=True)
    )

    matching_indices = (
        active_dataframe.index[
            active_dataframe["timestamp"]
            == selected_start_timestamp
        ]
        .tolist()
    )

    if not matching_indices:
        st.sidebar.error(
            "Could not find the selected replay start bar."
        )

    else:
        st.session_state.df = active_dataframe
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

        # Persistent drawings are intentionally not cleared.
        st.session_state.draw_clicks = []
        st.session_state.last_draw_signature = None
        st.session_state.camera_revision += 1
        st.session_state.camera_signature = None
        st.session_state.force_camera = True
        st.session_state.sim_active = True

        st.rerun()


# ============================================================
# MAIN APP
# ============================================================
st.title("💹 Market Replay Simulator")

st.caption(
    "Session-compressed replay • automatic 5-session S/R • "
    "right-side S/R label rail • persistent drawings • "
    "persistent paper account"
)

if (
    not st.session_state.sim_active
    or st.session_state.df.empty
):
    st.info(
        "Choose a ticker, timeframe, practice date, "
        "and start time. Then click Start / Reset Replay."
    )

    st.stop()


if (
    st.session_state.active_ticker != ticker
    or st.session_state.active_interval != timeframe
    or st.session_state.active_session_mode
    != effective_session_mode
):
    st.warning(
        "Sidebar setup differs from the active replay. "
        "Click Start / Reset Replay to apply it."
    )


# ============================================================
# CURRENT REPLAY DATA
# ============================================================
full_dataframe = st.session_state.df

st.session_state.step = min(
    st.session_state.step,
    len(full_dataframe) - 1,
)

current_step = st.session_state.step

revealed_dataframe = (
    full_dataframe
    .iloc[:current_step + 1]
    .copy()
)

current_row = full_dataframe.iloc[
    current_step
]

current_timestamp = pd.Timestamp(
    current_row["timestamp"]
)

current_price = float(
    current_row["close"]
)

current_session_date = to_pydate(
    current_row["session_date"]
)

active_ticker = (
    st.session_state.active_ticker
)

active_interval = (
    st.session_state.active_interval
)

active_session_mode = (
    st.session_state.active_session_mode
)

continuous_market = is_continuous_mode(
    active_session_mode
)


# ============================================================
# CALCULATIONS
# ============================================================
calculated_dataframe = (
    revealed_dataframe.copy()
)

if show_moving_averages:
    calculated_dataframe["fast_ma"] = (
        moving_average(
            calculated_dataframe["close"],
            fast_length,
            moving_average_type,
        )
    )

    calculated_dataframe["slow_ma"] = (
        moving_average(
            calculated_dataframe["close"],
            slow_length,
            moving_average_type,
        )
    )

if show_volume_ma:
    calculated_dataframe["volume_ma"] = (
        calculated_dataframe["volume"]
        .rolling(int(volume_ma_length))
        .mean()
    )


# ============================================================
# CHART DATA
# ============================================================
chart_dataframe = build_chart_dataframe(
    calculated_dataframe,
    current_session_date,
    context_preset,
    continuous_market,
    follow_replay,
    follow_bars,
)

if chart_dataframe.empty:
    chart_dataframe = (
        calculated_dataframe
        .copy()
        .reset_index(drop=True)
    )

chart_dataframe["x_index"] = np.arange(
    len(chart_dataframe)
)

current_chart_matches = (
    chart_dataframe.index[
        chart_dataframe["timestamp"]
        == current_timestamp
    ]
    .tolist()
)

if current_chart_matches:
    current_x_index = current_chart_matches[-1]
else:
    current_x_index = len(chart_dataframe) - 1


# ============================================================
# CAMERA WINDOW
# ============================================================
if follow_replay and continuous_market:
    start_index = max(
        0,
        current_x_index - int(follow_bars) + 1,
    )

    end_index = current_x_index

    follow_title = (
        f"Follow: {int(follow_bars)} bars"
    )

elif follow_replay:
    current_session_rows = (
        chart_dataframe[
            chart_dataframe["session_date"]
            .map(to_pydate)
            == current_session_date
        ]
    )

    if current_session_rows.empty:
        start_index = 0
        end_index = current_x_index
    else:
        start_index = int(
            current_session_rows["x_index"].iloc[0]
        )

        end_index = current_x_index

    follow_title = "Follow: current session"

else:
    start_index = 0
    end_index = len(chart_dataframe) - 1
    follow_title = "Follow: OFF"


visible_dataframe = (
    chart_dataframe
    .iloc[start_index:end_index + 1]
    .copy()
)

price_axis_range = get_price_axis_range(
    visible_dataframe
)

volume_axis_range = get_volume_axis_range(
    visible_dataframe
)


# ============================================================
# RIGHT-SIDE LABEL RAIL
# ============================================================
visible_bar_count = max(
    1,
    end_index - start_index + 1,
)

label_rail_width = int(
    round(
        visible_bar_count
        * LABEL_RAIL_PERCENT
    )
)

label_rail_width = max(
    MIN_LABEL_RAIL_BARS,
    min(
        MAX_LABEL_RAIL_BARS,
        label_rail_width,
    ),
)

label_rail_start_x = (
    end_index + 0.75
)

label_rail_label_x = (
    end_index + 2.0
)

label_rail_end_x = (
    end_index + label_rail_width
)


# ============================================================
# AUTOMATIC S/R
# ============================================================
automatic_sr_levels = []
automatic_sr_sessions = []
automatic_sr_tolerance = 0.0

if show_five_session_sr:
    (
        automatic_sr_levels,
        automatic_sr_sessions,
        automatic_sr_tolerance,
    ) = compute_five_session_sr(
        calculated_dataframe,
        current_session_date,
        current_price,
        max_levels=int(
            maximum_auto_sr_levels
        ),
    )

developing_session_levels = None

if show_developing_session_hl:
    developing_session_levels = (
        get_developing_session_levels(
            calculated_dataframe,
            current_session_date,
        )
    )


# ============================================================
# ACCOUNT / POSITION
# ============================================================
account = load_account()
position = load_position()

position_is_on_active_ticker = (
    position is not None
    and position["ticker"] == active_ticker
)

if position_is_on_active_ticker:
    update_position_mark(
        active_ticker,
        current_price,
        current_timestamp,
    )

    position = load_position()

account_values = calculate_account_values(
    account,
    position,
)

cash_balance = account_values["cash"]
position_value = account_values[
    "display_position_value"
]
equity = account_values["equity"]
unrealized_pnl = account_values[
    "unrealized_pnl"
]
signed_quantity = account_values[
    "signed_quantity"
]

realized_pnl = float(
    account["realized_pnl"]
)

total_pnl = (
    equity
    - float(account["starting_balance"])
)

if position is None:
    position_type = "FLAT"
    position_text = "—"

elif position["side"] == "long":
    position_type = "🟢 LONG"
    position_text = (
        f"{float(position['quantity']):.4f} units"
    )

else:
    position_type = "🔴 SHORT"
    position_text = (
        f"{float(position['quantity']):.4f} units"
    )


# ============================================================
# METRICS
# ============================================================
metric_1, metric_2, metric_3, metric_4, metric_5, metric_6 = (
    st.columns(6)
)

metric_1.metric(
    "Price",
    format_price(current_price),
)

metric_2.metric(
    "Cash",
    f"${cash_balance:,.2f}",
)

metric_3.metric(
    "Position Value",
    f"${position_value:,.2f}",
)

metric_4.metric(
    "Equity",
    f"${equity:,.2f}",
    f"${total_pnl:,.2f}",
)

metric_5.metric(
    "Realized P&L",
    f"${realized_pnl:,.2f}",
)

metric_6.metric(
    "Unrealized P&L",
    f"${unrealized_pnl:,.2f}",
)

position_caption = (
    f"Position ({position_type}): "
    f"{position_text} • "
    f"Cycle {account['cycle_number']} • "
    f"Session: {active_session_mode}"
)

if position is not None:
    position_caption += (
        f" • Position ticker: {position['ticker']}"
    )

st.caption(position_caption)

if (
    position is not None
    and not position_is_on_active_ticker
):
    st.warning(
        f"Your open position is on `{position['ticker']}`. "
        f"The current chart is `{active_ticker}`. "
        "Switch to the position ticker to manage it."
    )


# ============================================================
# BUILD CHART
# ============================================================
volume_colors = np.where(
    chart_dataframe["close"]
    >= chart_dataframe["open"],
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

price_precision = price_decimals(
    current_price
)

candlestick_hover = (
    "Time: %{customdata[0]}"
    "<br>Session: %{customdata[1]}"
    f"<br>O: %{{open:.{price_precision}f}}"
    f"<br>H: %{{high:.{price_precision}f}}"
    f"<br>L: %{{low:.{price_precision}f}}"
    f"<br>C: %{{close:.{price_precision}f}}"
    "<extra></extra>"
)

figure.add_trace(
    go.Candlestick(
        x=chart_dataframe["x_index"],
        open=chart_dataframe["open"],
        high=chart_dataframe["high"],
        low=chart_dataframe["low"],
        close=chart_dataframe["close"],
        name=active_ticker,
        increasing={
            "line": {
                "color": "#26a69a"
            },
            "fillcolor": "#26a69a",
        },
        decreasing={
            "line": {
                "color": "#ef5350"
            },
            "fillcolor": "#ef5350",
        },
        customdata=np.column_stack(
            [
                chart_dataframe["timestamp"].astype(str),
                chart_dataframe["session_date"].astype(str),
            ]
        ),
        hovertemplate=candlestick_hover,
    ),
    row=1,
    col=1,
)

figure.add_trace(
    go.Bar(
        x=chart_dataframe["x_index"],
        y=chart_dataframe["volume"],
        marker_color=volume_colors,
        name="Volume",
        customdata=chart_dataframe[
            "timestamp"
        ].astype(str),
        hovertemplate=(
            "Time: %{customdata}"
            "<br>Volume: %{y:,.0f}"
            "<extra></extra>"
        ),
    ),
    row=2,
    col=1,
)

if (
    show_moving_averages
    and "fast_ma" in chart_dataframe.columns
):
    figure.add_trace(
        go.Scatter(
            x=chart_dataframe["x_index"],
            y=chart_dataframe["fast_ma"],
            name=(
                f"{moving_average_type}"
                f"{int(fast_length)}"
            ),
            line={
                "color": C_FAST_EMA,
                "width": 1.7,
            },
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=chart_dataframe["x_index"],
            y=chart_dataframe["slow_ma"],
            name=(
                f"{moving_average_type}"
                f"{int(slow_length)}"
            ),
            line={
                "color": C_SLOW_EMA,
                "width": 1.7,
            },
        ),
        row=1,
        col=1,
    )

if (
    show_volume_ma
    and "volume_ma" in chart_dataframe.columns
):
    figure.add_trace(
        go.Scatter(
            x=chart_dataframe["x_index"],
            y=chart_dataframe["volume_ma"],
            name=(
                f"VolMA {int(volume_ma_length)}"
            ),
            line={
                "color": C_VOL_MA,
                "width": 1.7,
            },
        ),
        row=2,
        col=1,
    )


# ============================================================
# RIGHT-SIDE LABEL RAIL BACKGROUND
# ============================================================
figure.add_vrect(
    x0=label_rail_start_x,
    x1=label_rail_end_x,
    row=1,
    col=1,
    fillcolor="rgba(15,23,42,0.68)",
    line_width=0,
    layer="below",
)

figure.add_vline(
    x=label_rail_start_x,
    row=1,
    col=1,
    line_color="#64748b",
    line_width=1,
    line_dash="dot",
)


# ============================================================
# CLICK TARGET FOR DRAWING MODE
# ============================================================
if draw_mode != "None":
    figure.add_trace(
        go.Scatter(
            x=chart_dataframe["x_index"],
            y=chart_dataframe["close"],
            mode="markers",
            name="Drawing click targets",
            showlegend=False,
            hoverinfo="skip",
            marker={
                "size": 14,
                "color": "rgba(255,255,255,0.01)",
                "line": {
                    "width": 0
                },
            },
        ),
        row=1,
        col=1,
    )


# ============================================================
# LOCAL STRUCTURE LABELS / LOCAL ZONES
# ============================================================
if (
    show_structure_labels
    or show_local_sr_zones
):
    structure_dataframe = (
        chart_dataframe
        .tail(MAX_STRUCTURE_BARS)
        .copy()
        .reset_index(drop=True)
    )

    labeled_swings = raw_swings(
        structure_dataframe
    )

    confirmed_swings = [
        swing
        for swing in labeled_swings
        if swing["i"] + SWING_K
        <= len(structure_dataframe) - 1
    ]

else:
    structure_dataframe = pd.DataFrame()
    labeled_swings = []
    confirmed_swings = []


if (
    show_structure_labels
    and not structure_dataframe.empty
):
    bos_events, choch_events = (
        detect_bos_choch(
            structure_dataframe,
            labeled_swings,
        )
    )

    for swing in confirmed_swings[
        -MAX_SWING_LABELS:
    ]:
        x_value = structure_dataframe[
            "x_index"
        ].iloc[swing["i"]]

        if swing["label"] in {
            "HH",
            "HL",
        }:
            palette = C_HH

        elif swing["label"] in {
            "LH",
            "LL",
        }:
            palette = C_LH

        else:
            palette = C_H

        y_shift = (
            18
            if swing["type"] == "H"
            else -18
        )

        add_badge(
            figure,
            x_value,
            swing["price"],
            swing["label"],
            palette,
            y_shift=y_shift,
        )

    for event in bos_events[
        -MAX_BOS_LABELS:
    ]:
        event_index = event["i"]

        if (
            0
            <= event_index
            < len(structure_dataframe)
        ):
            x_value = (
                structure_dataframe[
                    "x_index"
                ].iloc[event_index]
            )

            if event["dir"] == "up":
                add_badge(
                    figure,
                    x_value,
                    float(
                        structure_dataframe[
                            "high"
                        ].iloc[event_index]
                    ),
                    "BOS↑",
                    C_BOS_UP,
                    y_shift=22,
                    arrow=True,
                )

            else:
                add_badge(
                    figure,
                    x_value,
                    float(
                        structure_dataframe[
                            "low"
                        ].iloc[event_index]
                    ),
                    "BOS↓",
                    C_BOS_DN,
                    y_shift=-22,
                    arrow=True,
                )

    for event in choch_events[
        -MAX_CHOCH_LABELS:
    ]:
        event_index = event["i"]

        if (
            0
            <= event_index
            < len(structure_dataframe)
        ):
            x_value = (
                structure_dataframe[
                    "x_index"
                ].iloc[event_index]
            )

            if event["dir"] == "up":
                add_badge(
                    figure,
                    x_value,
                    float(
                        structure_dataframe[
                            "high"
                        ].iloc[event_index]
                    ),
                    "CHoCH↑",
                    C_CH_UP,
                    y_shift=22,
                    arrow=True,
                )

            else:
                add_badge(
                    figure,
                    x_value,
                    float(
                        structure_dataframe[
                            "low"
                        ].iloc[event_index]
                    ),
                    "CHoCH↓",
                    C_CH_DN,
                    y_shift=-22,
                    arrow=True,
                )


if (
    show_local_sr_zones
    and confirmed_swings
):
    average_noise = (
        structure_dataframe["high"]
        - structure_dataframe["low"]
    ).tail(20).mean()

    if pd.isna(average_noise):
        average_noise = 0.0

    average_noise = max(
        float(average_noise),
        1e-12,
    )

    local_zones = build_local_zones(
        confirmed_swings[
            -MAX_ZONE_SWINGS:
        ],
        current_price,
    )

    local_zones = sorted(
        local_zones,
        key=lambda zone: (
            abs(zone["mid"] - current_price),
            -zone["touches"],
        ),
    )[:MAX_LOCAL_ZONES_DRAWN]

    for zone in local_zones:
        role = local_zone_role(
            zone,
            current_price,
            average_noise,
        )

        if role == "floor":
            color = C_FLOOR
            tag = "LOCAL FLOOR"

        elif role == "ceiling":
            color = C_CEIL
            tag = "LOCAL CEIL"

        elif role == "both":
            color = C_BOTH
            tag = "LOCAL BOTH"

        else:
            color = C_BROKEN
            tag = role.replace(
                "_",
                " ",
            ).upper()

        figure.add_hline(
            y=zone["mid"],
            row=1,
            col=1,
            line_color=color,
            line_width=min(
                1 + zone["touches"],
                4,
            ),
            line_dash="dot",
        )

        add_rail_label(
            figure,
            label_rail_label_x,
            zone["mid"],
            (
                f"{tag} "
                f"{format_price(zone['mid'], False)} "
                f"({zone['touches']}x)"
            ),
            color,
        )


# ============================================================
# AUTOMATIC 5-SESSION S/R
# ============================================================
for level in automatic_sr_levels:
    level_price = float(
        level["price"]
    )

    if (
        level["zone_low"]
        <= current_price
        <= level["zone_high"]
    ):
        level_role = "PIVOT"
        level_color = C_AUTO_PIVOT

    elif level_price < current_price:
        level_role = "SUP"
        level_color = C_AUTO_SUPPORT

    else:
        level_role = "RES"
        level_color = C_AUTO_RESISTANCE

    line_width = min(
        1.5 + level["touches"] * 0.5,
        4.0,
    )

    figure.add_hline(
        y=level_price,
        row=1,
        col=1,
        line_color=level_color,
        line_width=line_width,
        line_dash="dash",
    )

    add_rail_label(
        figure,
        label_rail_label_x,
        level_price,
        (
            f"5S {level_role} "
            f"{format_price(level_price, False)} "
            f"• {level['touches']} touch"
            f"{'es' if level['touches'] != 1 else ''}"
        ),
        level_color,
    )


# ============================================================
# DEVELOPING CURRENT-SESSION HIGH / LOW
# ============================================================
if developing_session_levels is not None:
    developing_high = (
        developing_session_levels["high"]
    )

    developing_low = (
        developing_session_levels["low"]
    )

    figure.add_hline(
        y=developing_high,
        row=1,
        col=1,
        line_color=C_DEVELOPING_HIGH,
        line_width=1.3,
        line_dash="dot",
    )

    add_rail_label(
        figure,
        label_rail_label_x,
        developing_high,
        (
            f"SESSION H "
            f"{format_price(developing_high, False)}"
        ),
        C_DEVELOPING_HIGH,
        foreground_color="#111111",
    )

    if abs(
        developing_low - developing_high
    ) > 1e-12:
        figure.add_hline(
            y=developing_low,
            row=1,
            col=1,
            line_color=C_DEVELOPING_LOW,
            line_width=1.3,
            line_dash="dot",
        )

        add_rail_label(
            figure,
            label_rail_label_x,
            developing_low,
            (
                f"SESSION L "
                f"{format_price(developing_low, False)}"
            ),
            C_DEVELOPING_LOW,
            foreground_color="#111111",
        )


# ============================================================
# POSITION LINES
# ============================================================
if position_is_on_active_ticker:
    if position["stop_loss"] is not None:
        figure.add_hline(
            y=float(position["stop_loss"]),
            row=1,
            col=1,
            line_color="#fb923c",
            line_width=2,
            line_dash="dash",
        )

    if position["target"] is not None:
        figure.add_hline(
            y=float(position["target"]),
            row=1,
            col=1,
            line_color="#4ade80",
            line_width=2,
            line_dash="dash",
        )

    figure.add_hline(
        y=float(position["entry_price"]),
        row=1,
        col=1,
        line_color="#22d3ee",
        line_width=1.3,
        line_dash="dot",
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
    marker_ticker = marker.get(
        "ticker",
        active_ticker,
    )

    if marker_ticker != active_ticker:
        continue

    marker_matches = (
        chart_dataframe.index[
            chart_dataframe["timestamp"]
            == marker["timestamp"]
        ]
        .tolist()
    )

    if not marker_matches:
        continue

    style = marker_styles.get(
        marker["kind"],
        {
            "symbol": "circle",
            "color": "#ffffff",
        },
    )

    figure.add_trace(
        go.Scatter(
            x=[
                chart_dataframe[
                    "x_index"
                ].iloc[marker_matches[-1]]
            ],
            y=[marker["price"]],
            mode="markers",
            showlegend=False,
            marker={
                "symbol": style["symbol"],
                "size": 12,
                "color": style["color"],
                "line": {
                    "color": "#ffffff",
                    "width": 1,
                },
            },
        ),
        row=1,
        col=1,
    )


# ============================================================
# PERSISTENT MANUAL DRAWINGS
# ============================================================
persistent_drawings = (
    load_persistent_drawings(
        active_ticker
    )
)

for drawing in persistent_drawings:
    if drawing["type"] == "hline":
        drawing_price = float(
            drawing["price"]
        )

        drawing_color = drawing.get(
            "color",
            C_MANUAL_SR,
        )

        figure.add_hline(
            y=drawing_price,
            row=1,
            col=1,
            line_color=drawing_color,
            line_width=1.8,
        )

        add_rail_label(
            figure,
            label_rail_label_x,
            drawing_price,
            drawing.get(
                "label",
                (
                    f"S/R "
                    f"{format_price(drawing_price, False)}"
                ),
            ),
            drawing_color,
        )

    elif drawing["type"] == "band":
        figure.add_hrect(
            y0=float(drawing["y0"]),
            y1=float(drawing["y1"]),
            row=1,
            col=1,
            fillcolor=drawing.get(
                "color",
                "rgba(34,197,94,0.18)",
            ),
            line_width=1,
            line_color=drawing.get(
                "line_color",
                "#22c55e",
            ),
            layer="below",
        )

    elif drawing["type"] == "trend":
        first_timestamp = pd.Timestamp(
            drawing["x0"]
        )

        second_timestamp = pd.Timestamp(
            drawing["x1"]
        )

        first_matches = (
            chart_dataframe.index[
                chart_dataframe["timestamp"]
                == first_timestamp
            ]
            .tolist()
        )

        second_matches = (
            chart_dataframe.index[
                chart_dataframe["timestamp"]
                == second_timestamp
            ]
            .tolist()
        )

        if first_matches and second_matches:
            first_x = chart_dataframe[
                "x_index"
            ].iloc[first_matches[-1]]

            second_x = chart_dataframe[
                "x_index"
            ].iloc[second_matches[-1]]

            figure.add_trace(
                go.Scatter(
                    x=[
                        first_x,
                        second_x,
                    ],
                    y=[
                        float(drawing["y0"]),
                        float(drawing["y1"]),
                    ],
                    mode="lines",
                    showlegend=False,
                    line={
                        "color": drawing.get(
                            "color",
                            "#f472b6",
                        ),
                        "width": 2,
                    },
                ),
                row=1,
                col=1,
            )


# ============================================================
# TICKS / CAMERA
# ============================================================
number_of_ticks = min(
    12,
    visible_bar_count,
)

tick_values = np.unique(
    np.linspace(
        start_index,
        end_index,
        number_of_ticks,
        dtype=int,
    )
).tolist()

tick_text = [
    pd.Timestamp(
        chart_dataframe.iloc[
            tick_index
        ]["timestamp"]
    ).strftime("%m-%d\n%H:%M")
    for tick_index in tick_values
]

camera_signature = (
    context_preset,
    follow_replay,
    int(follow_bars),
    active_interval,
    current_session_date,
    active_session_mode,
)

if (
    st.session_state.camera_signature
    != camera_signature
):
    st.session_state.camera_signature = (
        camera_signature
    )

    st.session_state.camera_revision += 1
    st.session_state.force_camera = True

apply_camera = (
    follow_replay
    or st.session_state.force_camera
)

chart_drag_mode = (
    "select"
    if draw_mode != "None"
    else "pan"
)

if follow_replay:
    ui_revision = (
        f"follow-"
        f"{st.session_state.camera_revision}-"
        f"{current_step}"
    )
else:
    ui_revision = (
        f"manual-"
        f"{st.session_state.camera_revision}"
    )

figure.update_layout(
    template="plotly_dark",
    height=640,
    dragmode=chart_drag_mode,
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    font={
        "color": "#e5e7eb"
    },
    title=(
        f"{active_ticker} | "
        f"{active_interval} | "
        f"{current_timestamp.strftime('%Y-%m-%d %H:%M')} | "
        f"{follow_title}"
    ),
    margin={
        "l": 10,
        "r": 25,
        "t": 75,
        "b": 10,
    },
    showlegend=True,
    legend={
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "left",
        "x": 0,
        "bgcolor": "rgba(0,0,0,0.45)",
    },
    hovermode="x",
    uirevision=ui_revision,
    xaxis_rangeslider_visible=False,
)

disable_rangesliders(figure)

for row_number in (1, 2):
    figure.update_xaxes(
        type="linear",
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_text,
        gridcolor="#1f2937",
        linecolor="#4b5563",
        zerolinecolor="#374151",
        rangeslider_visible=False,
        showspikes=True,
        spikemode="across",
        spikecolor="#94a3b8",
        spikethickness=1,
        row=row_number,
        col=1,
    )

figure.update_yaxes(
    title_text="Price",
    tickformat=f".{price_precision}f",
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

if apply_camera:
    x_axis_range = [
        start_index - 0.5,
        label_rail_end_x + 0.5,
    ]

    figure.update_xaxes(
        range=x_axis_range,
        row=1,
        col=1,
    )

    figure.update_xaxes(
        range=x_axis_range,
        row=2,
        col=1,
    )

    if price_axis_range is not None:
        figure.update_yaxes(
            range=price_axis_range,
            autorange=False,
            row=1,
            col=1,
        )

    if volume_axis_range is not None:
        figure.update_yaxes(
            range=volume_axis_range,
            autorange=False,
            row=2,
            col=1,
        )

    if not follow_replay:
        st.session_state.force_camera = False

else:
    figure.update_yaxes(
        autorange=True,
        row=1,
        col=1,
    )

    figure.update_yaxes(
        autorange=True,
        row=2,
        col=1,
    )

disable_rangesliders(figure)


# ============================================================
# CHART + RIGHT CONTROLS
# ============================================================
chart_column, control_column = (
    st.columns([6, 1])
)

with chart_column:
    chart_event = None

    if draw_mode == "None":
        st.plotly_chart(
            figure,
            width="stretch",
            key="market_chart_view",
            config={
                "scrollZoom": True,
                "displaylogo": False,
            },
        )

    else:
        st.info(
            "Drawing mode is active. Click or drag-select "
            "a candle. If chart selection is unreliable, "
            "use Manual drawing helpers below."
        )

        try:
            chart_event = st.plotly_chart(
                figure,
                width="stretch",
                key="market_chart_draw",
                on_select="rerun",
                selection_mode="points",
                config={
                    "scrollZoom": True,
                    "displaylogo": False,
                },
            )

        except TypeError:
            st.plotly_chart(
                figure,
                width="stretch",
                key="market_chart_draw_fallback",
                config={
                    "scrollZoom": True,
                    "displaylogo": False,
                },
            )

        selected_points = (
            extract_selected_points(
                chart_event
            )
        )

        if selected_points:
            selected_point = selected_points[-1]

            selected_row = (
                nearest_row_from_event(
                    chart_dataframe,
                    selected_point,
                )
            )

            clicked_y = selected_point.get("y")

            signature = (
                draw_mode,
                (
                    str(selected_row["timestamp"])
                    if selected_row is not None
                    else ""
                ),
                str(clicked_y),
            )

            if (
                signature
                != st.session_state.last_draw_signature
            ):
                st.session_state.last_draw_signature = (
                    signature
                )

                process_drawing_click(
                    draw_mode,
                    selected_row,
                    clicked_y,
                )

                st.rerun()


with control_column:
    st.markdown("#### ⏱️")

    st.caption(
        current_timestamp.strftime("%H:%M")
    )

    st.caption(
        f"Native {active_interval}"
    )

    native_minutes = (
        TIMEFRAME_CONFIG[
            active_interval
        ]["minutes"]
    )

    if native_minutes >= 1440:
        advance_buttons = [
            ("▶️ +1 day", 1),
            ("⏩ +3 days", 3),
            ("⏭️ +5 days", 5),
            ("⏭️ +10 days", 10),
        ]

    else:
        advance_buttons = [
            (
                "▶️ +5m",
                bars_for_minutes(
                    5,
                    native_minutes,
                ),
            ),
            (
                "⏩ +15m",
                bars_for_minutes(
                    15,
                    native_minutes,
                ),
            ),
            (
                "⏭️ +30m",
                bars_for_minutes(
                    30,
                    native_minutes,
                ),
            ),
            (
                "⏭️ +60m",
                bars_for_minutes(
                    60,
                    native_minutes,
                ),
            ),
        ]

    for button_label, bars_to_advance in (
        advance_buttons
    ):
        if st.button(
            button_label,
            width="stretch",
        ):
            advance_bars(
                bars_to_advance
            )

            st.rerun()

    st.markdown("---")

    if signed_quantity > 0:
        st.success("🟢 LONG")

    elif signed_quantity < 0:
        st.error("🔴 SHORT")

    else:
        st.info("FLAT")

    st.caption(
        f"Session: {current_session_date}"
    )

    st.caption(
        f"Visible bars: {len(visible_dataframe):,}"
    )

    st.caption(
        f"Rendered bars: {len(chart_dataframe):,}"
    )

    if continuous_market:
        st.caption(
            f"Follow window: {int(follow_bars)} bars"
        )
    else:
        st.caption(
            "Follow window: session"
        )

    st.caption(
        "Carry: "
        f"{'ON' if st.session_state.active_carry_mode else 'OFF'}"
    )

    st.caption(
        f"5S S/R levels: {len(automatic_sr_levels)}"
    )


# ============================================================
# MANUAL DRAWING HELPERS
# ============================================================
with st.expander(
    "✏️ Manual drawing helpers",
    expanded=False,
):
    st.caption(
        "Manual drawings are stored in SQLite by ticker. "
        "Horizontal-line labels appear in the right-side rail."
    )

    if chart_dataframe.empty:
        st.info(
            "No chart bars are available."
        )

    else:
        helper_dataframe = (
            chart_dataframe.copy()
        )

        helper_dataframe["choice"] = (
            helper_dataframe["timestamp"]
            .dt.strftime("%Y-%m-%d %H:%M")
        )

        helper_choices = (
            helper_dataframe["choice"]
            .tolist()[::-1]
        )

        selected_helper_choice = (
            st.selectbox(
                "Select candle",
                helper_choices,
                key=(
                    f"manual_candle_{active_ticker}"
                ),
            )
        )

        selected_helper_row = (
            helper_dataframe[
                helper_dataframe["choice"]
                == selected_helper_choice
            ]
            .iloc[0]
        )

        selected_timestamp_key = int(
            pd.Timestamp(
                selected_helper_row["timestamp"]
            ).value
        )

        manual_price = st.number_input(
            "S/R price or drawing point",
            value=float(
                selected_helper_row["close"]
            ),
            step=price_input_step(
                current_price
            ),
            format=price_input_format(
                current_price
            ),
            key=(
                f"manual_sr_price_"
                f"{active_ticker}_"
                f"{selected_timestamp_key}"
            ),
        )

        helper_1, helper_2, helper_3, helper_4, helper_5, helper_6 = (
            st.columns(6)
        )

        with helper_1:
            if st.button(
                "High line",
                width="stretch",
            ):
                high_price = float(
                    selected_helper_row["high"]
                )

                add_horizontal_line(
                    high_price,
                    (
                        f"H "
                        f"{format_price(high_price, False)}"
                    ),
                    "#ef4444",
                )

                st.rerun()

        with helper_2:
            if st.button(
                "Close line",
                width="stretch",
            ):
                close_price = float(
                    selected_helper_row["close"]
                )

                add_horizontal_line(
                    close_price,
                    (
                        f"C "
                        f"{format_price(close_price, False)}"
                    ),
                    "#22d3ee",
                )

                st.rerun()

        with helper_3:
            if st.button(
                "Low line",
                width="stretch",
            ):
                low_price = float(
                    selected_helper_row["low"]
                )

                add_horizontal_line(
                    low_price,
                    (
                        f"L "
                        f"{format_price(low_price, False)}"
                    ),
                    "#22c55e",
                )

                st.rerun()

        with helper_4:
            if st.button(
                "S/R at price",
                width="stretch",
            ):
                add_horizontal_line(
                    float(manual_price),
                    (
                        f"S/R "
                        f"{format_price(manual_price, False)}"
                    ),
                    C_MANUAL_SR,
                )

                st.rerun()

        with helper_5:
            if st.button(
                "Band / trend point",
                width="stretch",
            ):
                if draw_mode in {
                    "Band",
                    "Trend line",
                }:
                    helper_mode = draw_mode
                else:
                    helper_mode = "Band"

                process_drawing_click(
                    helper_mode,
                    selected_helper_row,
                    float(manual_price),
                )

                st.rerun()

        with helper_6:
            if st.button(
                "Clear pending",
                width="stretch",
            ):
                st.session_state.draw_clicks = []
                st.session_state.last_draw_signature = None
                st.rerun()

        st.caption(
            f"Saved drawings for {active_ticker}: "
            f"{len(persistent_drawings)} • "
            f"Pending points: "
            f"{len(st.session_state.draw_clicks)}"
        )


# ============================================================
# TRADE ENTRY / MANAGEMENT
# ============================================================
position = load_position()

position_is_on_active_ticker = (
    position is not None
    and position["ticker"] == active_ticker
)

price_step = price_input_step(
    current_price
)

price_format = price_input_format(
    current_price
)

if position is None:
    st.markdown("### 🎮 Open a Position")

    entry_1, entry_2, entry_3, entry_4, entry_5 = (
        st.columns(
            [1.2, 1.2, 1.2, 1, 1]
        )
    )

    maximum_trade_amount = max(
        0.0,
        float(cash_balance),
    )

    if maximum_trade_amount <= 0:
        st.error(
            "No cash is available. The account will "
            "reset after true equity depletion."
        )

    else:
        minimum_trade_amount = min(
            10.0,
            maximum_trade_amount,
        )

        default_trade_amount = min(
            100.0,
            maximum_trade_amount,
        )

        with entry_1:
            trade_amount = st.number_input(
                "Trade Amount ($)",
                min_value=float(
                    minimum_trade_amount
                ),
                max_value=float(
                    maximum_trade_amount
                ),
                value=float(
                    default_trade_amount
                ),
                step=float(
                    min(
                        10.0,
                        maximum_trade_amount,
                    )
                ),
            )

        with entry_2:
            stop_input = st.number_input(
                "Stop-Loss",
                value=float(
                    current_price * 0.99
                ),
                step=price_step,
                format=price_format,
            )

        with entry_3:
            target_input = st.number_input(
                "Target (0 = none)",
                value=0.0,
                step=price_step,
                format=price_format,
            )

        with entry_4:
            st.write("")
            st.write("")

            if st.button(
                "🟢 BUY (Long)",
                width="stretch",
            ):
                if stop_input >= current_price:
                    st.error(
                        "Long stop must be below current price."
                    )

                elif (
                    target_input != 0
                    and target_input <= current_price
                ):
                    st.error(
                        "Long target must be above current price."
                    )

                else:
                    open_position(
                        active_ticker,
                        current_price,
                        current_timestamp,
                        "long",
                        trade_amount,
                        float(stop_input),
                        (
                            float(target_input)
                            if target_input > 0
                            else None
                        ),
                    )

                    add_marker(
                        active_ticker,
                        current_timestamp,
                        current_price,
                        "LONG",
                    )

                    st.rerun()

        with entry_5:
            st.write("")
            st.write("")

            if st.button(
                "🔻 SHORT",
                width="stretch",
            ):
                if stop_input <= current_price:
                    st.error(
                        "Short stop must be above current price."
                    )

                elif (
                    target_input != 0
                    and target_input >= current_price
                ):
                    st.error(
                        "Short target must be below current price."
                    )

                else:
                    open_position(
                        active_ticker,
                        current_price,
                        current_timestamp,
                        "short",
                        trade_amount,
                        float(stop_input),
                        (
                            float(target_input)
                            if target_input > 0
                            else None
                        ),
                    )

                    add_marker(
                        active_ticker,
                        current_timestamp,
                        current_price,
                        "SHORT",
                    )

                    st.rerun()


elif not position_is_on_active_ticker:
    st.markdown("### 🎮 Open Position")

    st.warning(
        f"The account currently has a "
        f"{position['side'].upper()} position on "
        f"`{position['ticker']}`. Switch to that "
        "ticker to manage it."
    )


else:
    st.markdown(
        "### 🎮 Manage "
        f"{position['side'].upper()} Position"
    )

    manage_1, manage_2, manage_3, manage_4, manage_5 = (
        st.columns(
            [1.2, 1.2, 1.2, 1, 1]
        )
    )

    with manage_1:
        modified_stop = st.number_input(
            "Modify Stop-Loss",
            value=float(
                position["stop_loss"]
                if position["stop_loss"] is not None
                else current_price
            ),
            step=price_step,
            format=price_format,
            key=(
                f"modify_stop_{active_ticker}"
            ),
        )

    with manage_2:
        modified_target = st.number_input(
            "Modify Target (0 = none)",
            value=float(
                position["target"]
                if position["target"] is not None
                else 0.0
            ),
            step=price_step,
            format=price_format,
            key=(
                f"modify_target_{active_ticker}"
            ),
        )

    with manage_3:
        close_quantity = st.number_input(
            "Quantity to close",
            min_value=0.0,
            max_value=float(
                position["quantity"]
            ),
            value=float(
                position["quantity"]
            ),
            step=0.0001,
        )

    with manage_4:
        st.write("")
        st.write("")

        if st.button(
            "💾 Update SL/TP",
            width="stretch",
        ):
            valid_update = True

            if (
                position["side"] == "long"
                and modified_stop >= current_price
            ):
                st.error(
                    "Long stop must be below current price."
                )
                valid_update = False

            if (
                position["side"] == "short"
                and modified_stop <= current_price
            ):
                st.error(
                    "Short stop must be above current price."
                )
                valid_update = False

            if valid_update:
                position["stop_loss"] = float(
                    modified_stop
                )

                position["target"] = (
                    float(modified_target)
                    if modified_target > 0
                    else None
                )

                position["last_mark_price"] = (
                    current_price
                )

                position["last_mark_time"] = str(
                    current_timestamp
                )

                save_position(position)
                st.rerun()

    with manage_5:
        st.write("")
        st.write("")

        if position["side"] == "long":
            close_button_label = "🔴 SELL"
            close_marker_kind = "SELL"
        else:
            close_button_label = "🟢 COVER"
            close_marker_kind = "COVER"

        if st.button(
            close_button_label,
            width="stretch",
        ):
            if close_quantity > 0:
                close_position(
                    current_price,
                    current_timestamp,
                    reason="Manual close",
                    quantity_to_close=(
                        close_quantity
                    ),
                )

                add_marker(
                    active_ticker,
                    current_timestamp,
                    current_price,
                    close_marker_kind,
                )

                maybe_reset_depleted_account()
                st.rerun()


# ============================================================
# AUTOMATIC S/R DETAILS
# ============================================================
with st.expander(
    "🧭 Automatic 5-session S/R details",
    expanded=False,
):
    if not show_five_session_sr:
        st.info(
            "Automatic five-session S/R is disabled."
        )

    elif not automatic_sr_levels:
        st.info(
            "Not enough completed-session history is available."
        )

    else:
        sessions_text = ", ".join(
            str(session_date)
            for session_date in automatic_sr_sessions
        )

        st.caption(
            f"Completed sessions used: {sessions_text}"
        )

        st.caption(
            "Automatic merge tolerance: "
            f"{format_price(automatic_sr_tolerance)}"
        )

        sr_rows = []

        for level in automatic_sr_levels:
            if (
                level["zone_low"]
                <= current_price
                <= level["zone_high"]
            ):
                role = "Pivot"

            elif level["price"] < current_price:
                role = "Support"

            else:
                role = "Resistance"

            sr_rows.append(
                {
                    "role_now": role,
                    "price": format_price(
                        level["price"],
                        include_dollar=False,
                    ),
                    "touches": level["touches"],
                    "source_highs": level["high_count"],
                    "source_lows": level["low_count"],
                    "source_sessions": ", ".join(
                        str(session_date)
                        for session_date
                        in level["sessions"]
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(sr_rows),
            hide_index=True,
            width="stretch",
        )

        st.caption(
            "Historical levels exclude the current "
            "partial session. SESSION H and SESSION L "
            "are calculated separately."
        )


# ============================================================
# TRADE HISTORY
# ============================================================
with st.expander(
    "📝 Persistent Trade History",
    expanded=False,
):
    connection = get_connection()

    trades_dataframe = pd.read_sql_query(
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
        FROM replay_trades
        WHERE account_id = ?
        ORDER BY id DESC
        """,
        connection,
        params=(ACCOUNT_ID,),
    )

    connection.close()

    if trades_dataframe.empty:
        st.info(
            "No closed trades are recorded."
        )
    else:
        st.dataframe(
            trades_dataframe,
            hide_index=True,
            width="stretch",
        )


# ============================================================
# ACCOUNT EVENTS
# ============================================================
with st.expander(
    "🔁 Account Cycles and Events",
    expanded=False,
):
    connection = get_connection()

    events_dataframe = pd.read_sql_query(
        """
        SELECT
            cycle_number,
            event_type,
            message,
            created_at
        FROM replay_events
        WHERE account_id = ?
        ORDER BY id DESC
        """,
        connection,
        params=(ACCOUNT_ID,),
    )

    connection.close()

    if events_dataframe.empty:
        st.info(
            "No account-cycle events are recorded."
        )
    else:
        st.dataframe(
            events_dataframe,
            hide_index=True,
            width="stretch",
        )


# ============================================================
# REPLAY INFORMATION
# ============================================================
with st.expander(
    "📘 Replay Information",
    expanded=False,
):
    completed_sessions_text = (
        ", ".join(
            str(session_date)
            for session_date
            in automatic_sr_sessions
        )
        if automatic_sr_sessions
        else "Insufficient history"
    )

    st.markdown(
        f"""
| Item | Value |
|---|---|
| Ticker | `{active_ticker}` |
| Asset class | `{detect_asset_class(active_ticker)}` |
| Native timeframe | `{active_interval}` |
| Session mode | `{active_session_mode}` |
| Current replay time | `{current_timestamp}` |
| Current trading session | `{current_session_date}` |
| Revealed bars | `{len(revealed_dataframe):,}` |
| Rendered bars | `{len(chart_dataframe):,}` |
| Visible bars | `{len(visible_dataframe):,}` |
| Follow mode | `{follow_title}` |
| Context preset | `{context_preset}` |
| Position carry | `{"ON" if st.session_state.active_carry_mode else "OFF"}` |
| Five-session S/R | `{"ON" if show_five_session_sr else "OFF"}` |
| Historical S/R levels | `{len(automatic_sr_levels)}` |
| Sessions used for S/R | `{completed_sessions_text}` |
| Developing session H/L | `{"ON" if show_developing_session_hl else "OFF"}` |
| Persistent drawings | `{len(persistent_drawings)}` |
| S/R label rail width | `{label_rail_width} bars` |
| Account cycle | `{account["cycle_number"]}` |
| SQLite database | `{os.path.abspath(DB_PATH)}` |

### S/R label rail

- Horizontal S/R lines remain at their exact prices.
- `5S SUP / RES / PIVOT` labels are placed in the right rail.
- `LOCAL FLOOR / CEIL / BOTH / BROKEN` labels are placed in the right rail.
- `SESSION H / SESSION L` labels are placed in the right rail.
- Persistent manual horizontal-line labels are placed in the right rail.
- HH, HL, LH, LL, BOS, and CHoCH remain attached to their relevant candles.
"""
    )
