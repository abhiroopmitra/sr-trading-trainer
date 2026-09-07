import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="S&R Paper Trading Sim")

# ==========================================
# STRUCTURE DETECTION CONSTANTS (from bot)
# ==========================================
SWING_K = 5
ZONE_TOL = 0.0012
MIN_DRAW_TOUCHES = 2
MIN_SWING_PCT = 0.0008
POLARITY_EDGE = 2
CHOCH_ENABLE = True

# Colors tuned for BLACK background
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

def badge(fig, x, y, text, pal, yshift=0, arrow=False):
    fig.add_annotation(
        x=x, y=y, text=f"<b>{text}</b>", row=1, col=1,
        showarrow=arrow, arrowhead=2, arrowsize=1, arrowwidth=1.4,
        arrowcolor="#e5e7eb", yshift=yshift,
        font=dict(size=11, color=pal["fg"], family="Arial"),
        bgcolor=pal["bg"], bordercolor="#e5e7eb", borderwidth=1, borderpad=3,
        opacity=1, align="center",
    )

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
# 5. TIME ADVANCE + SL/TARGET CHECKS
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

# ----- Moving Average controls (per-timeframe) -----
st.sidebar.markdown("---")
show_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
ma_type = st.sidebar.radio("MA type", ["EMA", "SMA"], horizontal=True)
st.sidebar.markdown("**EMA lengths per timeframe**")
with st.sidebar.expander("⚙️ 1m settings", expanded=(chart_tf == "1m")):
    fast_1m = st.number_input("1m Fast", 3, 100, 9, 1, key="f1")
    slow_1m = st.number_input("1m Slow", 5, 300, 21, 1, key="s1")
with st.sidebar.expander("⚙️ 5m settings", expanded=(chart_tf == "5m")):
    fast_5m = st.number_input("5m Fast", 3, 100, 20, 1, key="f5")
    slow_5m = st.number_input("5m Slow", 5, 300, 50, 1, key="s5")
with st.sidebar.expander("⚙️ 15m settings", expanded=(chart_tf == "15m")):
    fast_15m = st.number_input("15m Fast", 3, 100, 50, 1, key="f15")
    slow_15m = st.number_input("15m Slow", 5, 300, 200, 1, key="s15")
ma_settings = {"1m": (fast_1m, slow_1m), "5m": (fast_5m, slow_5m), "15m": (fast_15m, slow_15m)}
fast_len, slow_len = ma_settings[chart_tf]
st.sidebar.caption(f"📊 {chart_tf} view → {ma_type}{fast_len} & {ma_type}{slow_len}")

# ----- Structure overlays -----
st.sidebar.markdown("---")
show_struct = st.sidebar.checkbox("Show structure labels (HH/HL/BOS/CHoCH)", value=True)
show_zones = st.sidebar.checkbox("Show S/R zones", value=True)
show_context = st.sidebar.checkbox("Show previous days (context)", value=False)

st.sidebar.markdown("**✏️ Draw:** line tool in chart toolbar. Scroll = zoom, drag = pan.")

# ==========================================
# 7. MAIN DASHBOARD
# ==========================================
st.title("💹 Market Replay Simulator")

if st.session_state.sim_active:
    step = st.session_state.step
    full_df = st.session_state.df
    sim0 = st.session_state.sim_start_idx

    # Visible bars for candles: today only (or with context)
    today_1m = full_df.iloc[sim0: step + 1]
    if show_context:
        visible_1m = full_df.iloc[: step + 1]
    else:
        visible_1m = today_1m

    # ---- KEY: For EMA, use the FULL history up to now (all days) ----
    hist_1m = full_df.iloc[: step + 1]

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

    # ---- Resample views ----
    view_df = resample_view(visible_1m, chart_tf)      # for candles (today / context)
    hist_view = resample_view(hist_1m, chart_tf)       # for EMA (full history)

    # ---- 2-row layout ----
    vol_colors = np.where(
        view_df["close"] >= view_df["open"],
        "rgba(38,166,154,0.6)", "rgba(239,83,80,0.6)"
    )
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.78, 0.22])

    fig.add_trace(go.Candlestick(
        x=view_df["label"], open=view_df["open"], high=view_df["high"],
        low=view_df["low"], close=view_df["close"], name=ticker,
        increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"),
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=view_df["label"], y=view_df["volume"], marker_color=vol_colors, name="Volume",
    ), row=2, col=1)

    # ---- EMAs computed on FULL history, then merged onto visible labels ----
    if show_ma and len(hist_view) >= 2:
        hist_view = hist_view.copy()
        hist_view["fast_ma"] = moving_avg(hist_view["close"], fast_len, ma_type)
        hist_view["slow_ma"] = moving_avg(hist_view["close"], slow_len, ma_type)
        # keep only the labels currently visible on the chart
        ema_plot = hist_view[hist_view["label"].isin(view_df["label"])]
        fig.add_trace(go.Scatter(
            x=ema_plot["label"], y=ema_plot["fast_ma"],
            line=dict(color="#3b82f6", width=1.6),   # blue = fast
            name=f"{ma_type}{fast_len} ({chart_tf})",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=ema_plot["label"], y=ema_plot["slow_ma"],
            line=dict(color="#f59e0b", width=1.6),    # amber = slow
            name=f"{ma_type}{slow_len} ({chart_tf})",
        ), row=1, col=1)

    # ---- Structure labels + zones (computed on visible view) ----
    noise = float((view_df["high"] - view_df["low"]).tail(20).mean() or 0.3)
    if show_struct or show_zones:
        labeled = _raw_swings(view_df)
        conf = [s for s in labeled if s["i"] + SWING_K <= len(view_df) - 1]

    if show_struct and len(view_df) > SWING_K * 2 + 1:
        bos, choch = detect_bos_choch(view_df, labeled)
        for s in conf:
            x = view_df["label"].iloc[s["i"]]
            if s["label"] in ("HH", "HL"):
                pal, ys = C_HH, (18 if s["type"] == "H" else -18)
            elif s["label"] in ("LH", "LL"):
                pal, ys = C_LH, (18 if s["type"] == "H" else -18)
            else:
                pal, ys = C_H, (18 if s["type"] == "H" else -18)
            badge(fig, x, s["price"], s["label"], pal, yshift=ys)
        for b in bos:
            if b["i"] < len(view_df):
                x = view_df["label"].iloc[b["i"]]
                if b["dir"] == "up":
                    badge(fig, x, float(view_df["high"].iloc[b["i"]]), "BOS↑", C_BOS_UP, yshift=22, arrow=True)
                else:
                    badge(fig, x, float(view_df["low"].iloc[b["i"]]), "BOS↓", C_BOS_DN, yshift=-22, arrow=True)
        for h in choch:
            if h["i"] < len(view_df):
                x = view_df["label"].iloc[h["i"]]
                if h["dir"] == "up":
                    badge(fig, x, float(view_df["high"].iloc[h["i"]]), "CHoCH↑", C_CH_UP, yshift=22, arrow=True)
                else:
                    badge(fig, x, float(view_df["low"].iloc[h["i"]]), "CHoCH↓", C_CH_DN, yshift=-22, arrow=True)

    if show_zones and len(view_df) > SWING_K * 2 + 1:
        zones = build_zones(conf, current_price)
        last_x = view_df["label"].iloc[-1]
        for z in zones:
            role = zone_role(z, current_price, noise)
            if role == "floor":
                col, tag = C_FLOOR, "FLOOR"
            elif role == "ceiling":
                col, tag = C_CEIL, "CEIL"
            elif role == "both":
                col, tag = "#eab308", "BOTH"
            else:
                col, tag = C_BROKEN, role.replace("_", " ").upper()
            fig.add_hline(y=z["mid"], row=1, col=1,
                          line=dict(color=col, width=min(1 + z["touches"], 4), dash="dot"))
            fig.add_annotation(
                x=last_x, y=z["mid"], xanchor="left", xref="x", row=1, col=1,
                text=f"<b> {z['mid']:.2f} {tag} {z['low_touches']}L/{z['high_touches']}H</b>",
                showarrow=False,
                font=dict(size=10, color="#ffffff", family="Arial"),
                bgcolor=col, bordercolor="#e5e7eb", borderwidth=1, borderpad=3,
            )

    # ---- Position lines ----
    if pos != 0:
        if st.session_state.stop_loss:
            fig.add_hline(y=st.session_state.stop_loss, row=1, col=1,
                          line=dict(color="#fb923c", width=2, dash="dash"))
        if st.session_state.target:
            fig.add_hline(y=st.session_state.target, row=1, col=1,
                          line=dict(color="#4ade80", width=2, dash="dash"))
        if st.session_state.entry_price:
            fig.add_hline(y=st.session_state.entry_price, row=1, col=1,
                          line=dict(color="#22d3ee", width=1, dash="dot"))

    # ---- Camera zoom: TODAY only (regardless of EMA history) ----
    today_mask = view_df["timestamp"].dt.date == selected_date
    today_view = view_df.loc[today_mask]
    labels = view_df["label"].tolist()
    if len(today_view) >= 1:
        x0 = today_view["label"].iloc[0]
        x1 = today_view["label"].iloc[-1]
    else:
        x0, x1 = view_df["label"].iloc[0], view_df["label"].iloc[-1]
    try:
        i0 = labels.index(x0)
        i1 = labels.index(x1)
    except ValueError:
        i0, i1 = 0, len(labels) - 1

    # ---- BLACK THEME layout ----
    fig.update_layout(
        template="plotly_dark", height=720, dragmode="pan",
        paper_bgcolor="#000000", plot_bgcolor="#000000",
        font=dict(color="#e5e7eb"),
        xaxis_rangeslider_visible=False,
        title=f"{ticker} | {chart_tf} view | Sim Time: {current_time}",
        margin=dict(l=10, r=160, t=40, b=10),
        newshape=dict(line_color="#22d3ee", line_width=2),
        uirevision="keep-drawings", showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="#e5e7eb")),
    )
    for r in (1, 2):
        fig.update_xaxes(
            type="category", range=[i0 - 0.5, i1 + 0.5], nticks=12,
            rangeslider_visible=False, gridcolor="#1f2937",
            linecolor="#4b5563", zerolinecolor="#374151", row=r, col=1,
        )
    fig.update_yaxes(title_text="Price", gridcolor="#1f2937",
                     linecolor="#4b5563", row=1, col=1)

    # Volume axis cap (opening-spike fix)
    if len(view_df) and view_df["volume"].max() > 0:
        cap = max(view_df["volume"].quantile(0.95) * 1.15, view_df["volume"].median() * 2)
        fig.update_yaxes(range=[0, cap], title_text="Volume", gridcolor="#1f2937",
                         linecolor="#4b5563", row=2, col=1)
    else:
        fig.update_yaxes(title_text="Volume", gridcolor="#1f2937",
                         linecolor="#4b5563", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True, config={
        "scrollZoom": True,
        "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
        "displaylogo": False,
    })

    # ==========================================
    # ENTRY PANEL
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
