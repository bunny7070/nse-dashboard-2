import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nsepython import nsefetch
from streamlit_autorefresh import st_autorefresh
import requests
from datetime import date, timedelta

# ===========================================
# PAGE CONFIG + DARK THEME
# ===========================================
st.set_page_config(
    page_title="NSE Trading Dashboard",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 NSE Trading Dashboard – Watchlist, Scanner, Options, Backtest, Calculator & Screeners")

# ===========================================
# SIDEBAR – REFRESH + TELEGRAM
# ===========================================
refresh_rate = st.sidebar.slider("⏳ Auto Refresh (seconds)", 5, 60, 15)

st.sidebar.markdown("### 📲 Telegram Alerts (optional)")
tg_enable = st.sidebar.checkbox("Enable Alerts")
tg_token = st.sidebar.text_input("Bot Token", type="password")
tg_chat_id = st.sidebar.text_input("Chat ID / Group ID")

# Auto-refresh
st_autorefresh(interval=refresh_rate * 1000, key="refresh_counter")


def send_telegram(msg: str):
    if not (tg_enable and tg_token and tg_chat_id):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{tg_token}/sendMessage",
            data={"chat_id": tg_chat_id, "text": msg},
            timeout=5
        )
    except Exception:
        pass


# ===========================================
# LOAD SYMBOL LIST (EQUITY_L.CSV)
# ===========================================
@st.cache_data
def load_symbols():
    df = pd.read_csv("https://archives.nseindia.com/content/equities/EQUITY_L.csv")
    return df["SYMBOL"].dropna().sort_values().tolist()


all_symbols = load_symbols()

# ===========================================
# INDICATORS
# ===========================================
def rsi(price: pd.Series, period: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series):
    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    macd_line = e12 - e26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: int = 3):
    hl2 = (df["High"] + df["Low"]) / 2

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift()).abs()
    tr3 = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    trend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(len(df)):
        if i == 0:
            trend.iloc[i] = upper.iloc[i]
            direction.iloc[i] = 1
        else:
            if df["Close"].iloc[i] > trend.iloc[i - 1]:
                direction.iloc[i] = 1
                trend.iloc[i] = lower.iloc[i]
            else:
                direction.iloc[i] = -1
                trend.iloc[i] = upper.iloc[i]

    return trend, direction


# ===========================================
# HISTORICAL DATA FROM NSE (NO YFINANCE)
# ===========================================
def get_history_from_nse(symbol: str, days_back: int = 180) -> pd.DataFrame | None:
    """
    Fetch recent EOD candles for symbol from NSE historical API.
    """
    try:
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=days_back)
        start_str = start_dt.strftime("%d-%m-%Y")
        end_str = end_dt.strftime("%d-%m-%Y")

        url = (
            "https://www.nseindia.com/api/historical/cm/equity"
            f"?symbol={symbol}&series=[%22EQ%22]&from={start_str}&to={end_str}"
        )
        payload = nsefetch(url)
        raw = pd.DataFrame.from_records(payload["data"])
        if raw.empty:
            return None

        df = raw.copy()
        df["Date"] = pd.to_datetime(df["CH_TIMESTAMP"])
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        df["Open"] = df["CH_OPENING_PRICE"].astype(float)
        df["High"] = df["CH_TRADE_HIGH_PRICE"].astype(float)
        df["Low"] = df["CH_TRADE_LOW_PRICE"].astype(float)
        df["Close"] = df["CH_CLOSING_PRICE"].astype(float)
        df["Volume"] = df["CH_TOT_TRADED_QTY"].astype(float)

        # EMAs
        df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

        # Indicators
        df["RSI"] = rsi(df["Close"])
        df["MACD"], df["MACD_SIGNAL"] = macd(df["Close"])
        df["VOL20"] = df["Volume"].rolling(20).mean()
        df["HIGH20"] = df["High"].rolling(20).max()
        st_line, st_dir = supertrend(df)
        df["SUPERTREND"] = st_line
        df["ST_DIR"] = st_dir

        # EMA9/21 crossover
        df["CROSS"] = np.where(
            (df["EMA9"] > df["EMA21"]) & (df["EMA9"].shift(1) <= df["EMA21"].shift(1)),
            "Bullish",
            np.where(
                (df["EMA9"] < df["EMA21"]) & (df["EMA9"].shift(1) >= df["EMA21"].shift(1)),
                "Bearish",
                ""
            )
        )

        return df

    except Exception:
        return None


# ===========================================
# STRATEGY ENGINE (DAILY)
# ===========================================
def compute_strategy_signal(last_row: pd.Series,
                            vol_spike: bool,
                            breakout: bool) -> str:
    ema20 = last_row.get("EMA20", np.nan)
    ema50 = last_row.get("EMA50", np.nan)
    rsi_val = last_row.get("RSI", np.nan)
    macd_val = last_row.get("MACD", np.nan)
    macd_sig = last_row.get("MACD_SIGNAL", np.nan)
    st_dir_val = last_row.get("ST_DIR", 0)

    up_trend = st_dir_val == 1
    ema_bull = not np.isnan(ema20) and not np.isnan(ema50) and ema20 > ema50
    macd_bull = not (np.isnan(macd_val) or np.isnan(macd_sig)) and macd_val > macd_sig
    rsi_bull = not np.isnan(rsi_val) and rsi_val > 55

    if up_trend and ema_bull and macd_bull and rsi_bull and breakout and vol_spike:
        return "STRONG BUY ✅"

    bullish_count = sum([up_trend, ema_bull, macd_bull, rsi_bull])
    if bullish_count >= 3 and (breakout or vol_spike):
        return "BUY ✅"

    if st_dir_val == -1 or (not np.isnan(rsi_val) and rsi_val < 45 and not macd_bull):
        return "SELL ❌"

    return "HOLD ⏸"


# ===========================================
# MULTI-TF ENGINE (DAILY SKELETON)
# ===========================================
def compute_multitf_signal(daily_row: pd.Series,
                           tf15_row: pd.Series | None = None,
                           tf5_row: pd.Series | None = None) -> str:
    ema20 = daily_row.get("EMA20", np.nan)
    ema50 = daily_row.get("EMA50", np.nan)
    rsi_val = daily_row.get("RSI", np.nan)
    macd_val = daily_row.get("MACD", np.nan)
    macd_sig = daily_row.get("MACD_SIGNAL", np.nan)
    st_dir_val = daily_row.get("ST_DIR", 0)

    daily_up = st_dir_val == 1
    ema_bull = not np.isnan(ema20) and not np.isnan(ema50) and ema20 > ema50
    macd_bull = not (np.isnan(macd_val) or np.isnan(macd_sig)) and macd_val > macd_sig
    rsi_bull = not np.isnan(rsi_val) and rsi_val > 55

    score = sum([daily_up, ema_bull, macd_bull, rsi_bull])

    if score >= 3:
        return "MULTI-TF LONG BIAS ✅"
    elif st_dir_val == -1 or (not np.isnan(rsi_val) and rsi_val < 45):
        return "MULTI-TF SHORT BIAS ❌"
    else:
        return "MULTI-TF NEUTRAL ⏸"


# ===========================================
# BACKTEST ENGINE (DAILY)
# ===========================================
def run_backtest(symbol: str, days_back: int = 400):
    df = get_history_from_nse(symbol, days_back=days_back)
    if df is None or df.empty:
        return None, None

    df = df.reset_index(drop=True)
    if len(df) < 50:
        return None, None

    capital = 100000.0
    cash = capital
    position = 0
    entry_price = 0.0
    trades = []

    for i in range(25, len(df)):
        row = df.iloc[i]
        breakout = row["Close"] > row["HIGH20"] if not np.isnan(row["HIGH20"]) else False
        vol_spike = False

        signal = compute_strategy_signal(row, vol_spike, breakout)

        if position == 0 and ("BUY" in signal):
            qty = int(cash // row["Close"])
            if qty > 0:
                entry_price = row["Close"]
                position = qty
                cash -= qty * entry_price
                trades.append({
                    "Date": row["Date"],
                    "Type": "BUY",
                    "Price": entry_price,
                    "Qty": qty
                })

        exit_condition = (
            (position > 0 and "SELL" in signal) or
            (position > 0 and row["Close"] < entry_price * 0.97) or
            (position > 0 and i == len(df) - 1)
        )

        if exit_condition and position > 0:
            exit_price = row["Close"]
            cash += position * exit_price
            trades.append({
                "Date": row["Date"],
                "Type": "SELL",
                "Price": exit_price,
                "Qty": position,
                "PnL": (exit_price - entry_price) * position
            })
            position = 0
            entry_price = 0.0

    last_close = df.iloc[-1]["Close"]
    equity = cash + position * last_close

    trades_df = pd.DataFrame(trades)
    return equity, trades_df


# ===========================================
# CANDLE PATTERN HELPERS
# ===========================================
def is_bullish_engulfing(df: pd.DataFrame, idx: int) -> bool:
    if idx < 1:
        return False
    prev = df.iloc[idx - 1]
    cur = df.iloc[idx]
    prev_red = prev["Close"] < prev["Open"]
    cur_green = cur["Close"] > cur["Open"]
    body_prev_low = min(prev["Open"], prev["Close"])
    body_prev_high = max(prev["Open"], prev["Close"])
    body_cur_low = min(cur["Open"], cur["Close"])
    body_cur_high = max(cur["Open"], cur["Close"])
    return prev_red and cur_green and body_cur_low <= body_prev_low and body_cur_high >= body_prev_high


def is_hammer(df: pd.DataFrame, idx: int) -> bool:
    cur = df.iloc[idx]
    body = abs(cur["Close"] - cur["Open"])
    total_range = cur["High"] - cur["Low"]
    if total_range == 0:
        return False
    lower_shadow = min(cur["Open"], cur["Close"]) - cur["Low"]
    return (lower_shadow > 2 * body) and (body / total_range < 0.3)


def is_inside_bar(df: pd.DataFrame, idx: int) -> bool:
    if idx < 1:
        return False
    prev = df.iloc[idx - 1]
    cur = df.iloc[idx]
    return cur["High"] < prev["High"] and cur["Low"] > prev["Low"]


# ========= PIVOTS, ZONES, TRENDLINES =========
def get_pivots(df: pd.DataFrame, left: int = 3, right: int = 3):
    pivot_highs = []
    pivot_lows = []
    for i in range(left, len(df) - right):
        window = df.iloc[i-left:i+right+1]
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        if high == window["High"].max():
            pivot_highs.append(i)
        if low == window["Low"].min():
            pivot_lows.append(i)
    return pivot_highs, pivot_lows


def build_zones_from_pivots(df: pd.DataFrame, pivot_idx_list, kind: str = "demand", max_zones: int = 3):
    zones = []
    for idx in reversed(pivot_idx_list):
        bar = df.iloc[idx]
        if kind == "demand":
            price_low = bar["Low"]
            price_high = bar["Low"] * 1.005
        else:
            price_high = bar["High"]
            price_low = bar["High"] * 0.995

        x0 = bar["Date"]
        x1 = df["Date"].iloc[-1]
        zones.append((x0, x1, price_low, price_high))
        if len(zones) >= max_zones:
            break
    return zones


def build_trendline_points(df: pd.DataFrame, pivot_idx_list, use="low"):
    if len(pivot_idx_list) < 2:
        return None
    i1, i2 = pivot_idx_list[-2], pivot_idx_list[-1]
    if use == "low":
        y_vals = df["Low"].iloc[[i1, i2]].tolist()
    else:
        y_vals = df["High"].iloc[[i1, i2]].tolist()
    x_vals = df["Date"].iloc[[i1, i2]].tolist()
    return x_vals, y_vals


# ========= INDEX MEMBERS LOADER =========
def load_index_members(index_name: str):
    # index_name like "NIFTY 50", "NIFTY 100", "NIFTY 500"
    encoded = index_name.replace(" ", "%20")
    payload = nsefetch(f"https://www.nseindia.com/api/equity-stockIndices?index={encoded}")
    return sorted([row["symbol"] for row in payload["data"]])


# ===========================================
# TABS
# ===========================================
(
    tab_watch,
    tab_chart,
    tab_opt,
    tab_scan,
    tab_bt,
    tab_calc,
    tab_scr50,
    tab_scr100,
    tab_scr500,
) = st.tabs([
    "📈 Watchlist",
    "📊 Chart",
    "📉 Option Chain",
    "🚨 NIFTY500 Scanner",
    "📜 Backtest",
    "🧮 Calculator",
    "📌 NIFTY50 Screener",
    "📌 NIFTY100 Screener",
    "📌 NIFTY500 Screener",
])

# ===========================================
# TAB 1 – WATCHLIST
# ===========================================
with tab_watch:
    st.subheader("📈 Live Watchlist + Strategy Signals (Daily)")

    default_stocks = ["RELIANCE", "AXISCADES", "JIOFIN"]
    valid_defaults = [s for s in default_stocks if s in all_symbols]

    selected = st.multiselect(
        "Select Stocks",
        all_symbols,
        default=valid_defaults,
    )

    rows = []
    alerts = []

    for sym in selected:
        try:
            quote = nsefetch(
                f"https://www.nseindia.com/api/quote-equity?symbol={sym}"
            )
            price_info = quote["priceInfo"]
            last_price = price_info["lastPrice"]
            prev_close = price_info["previousClose"]
            total_vol = quote.get("preOpenMarket", {}).get("totalTradedVolume", 0)
            pct_change = (
                (last_price - prev_close) / prev_close * 100
                if prev_close else None
            )

            hist = get_history_from_nse(sym, days_back=180)

            ema9 = ema21 = ema20 = ema50 = rsi14 = macd_val = macd_sig = st_dir = None
            vol20 = high20 = None
            vol_spike = breakout = False
            cross_text = ""
            signal_text = "NA"
            multi_tf_signal = "NA"

            if hist is not None and len(hist) >= 25:
                last = hist.iloc[-1]
                ema9 = float(last["EMA9"])
                ema21 = float(last["EMA21"])
                ema20 = float(last["EMA20"])
                ema50 = float(last["EMA50"])
                rsi14 = float(last["RSI"])
                macd_val = float(last["MACD"])
                macd_sig = float(last["MACD_SIGNAL"])
                vol20 = float(last["VOL20"])
                high20 = float(last["HIGH20"])
                st_dir = "Up" if last["ST_DIR"] == 1 else "Down"
                cross_text = last["CROSS"]

                if vol20 and not np.isnan(vol20) and total_vol > 1.5 * vol20:
                    vol_spike = True
                if high20 and not np.isnan(high20) and last_price > high20:
                    breakout = True

                signal_text = compute_strategy_signal(last, vol_spike, breakout)
                multi_tf_signal = compute_multitf_signal(last)

            rows.append([
                sym,
                last_price,
                round(pct_change, 2) if pct_change is not None else None,
                int(total_vol),
                int(vol20) if vol20 and not np.isnan(vol20) else None,
                "Yes" if vol_spike else "",
                prev_close,
                ema9,
                ema21,
                ema20,
                ema50,
                rsi14,
                macd_val,
                macd_sig,
                st_dir,
                high20,
                "Yes" if breakout else "",
                cross_text,
                signal_text,
                multi_tf_signal
            ])

            if "BUY" in signal_text:
                if pct_change is not None:
                    alerts.append(
                        f"🚀 {sym} | {signal_text} | Price: {last_price} | Chg: {pct_change:.2f}%"
                    )
                else:
                    alerts.append(
                        f"🚀 {sym} | {signal_text} | Price: {last_price}"
                    )

        except Exception as e:
            st.warning(f"Error loading {sym}: {e}")

    df_watch = pd.DataFrame(
        rows,
        columns=[
            "Symbol", "Price", "% Chg", "Volume", "20D Vol",
            "Vol Spike", "Prev Close",
            "EMA9", "EMA21", "EMA20", "EMA50",
            "RSI", "MACD", "MACD Signal",
            "ST Dir", "20D High", "Breakout",
            "EMA9/21 Cross", "Signal", "Multi-TF Bias"
        ]
    )

    st.markdown("### 📋 Watchlist Table with Strategy Signals")
    st.dataframe(df_watch, use_container_width=True)

    if not df_watch.empty:
        st.markdown("### 📊 Price Comparison")
        fig_price = go.Figure(
            data=[go.Bar(x=df_watch["Symbol"], y=df_watch["Price"])]
        )
        fig_price.update_layout(height=300, xaxis_title="Symbol", yaxis_title="Price")
        st.plotly_chart(fig_price, use_container_width=True)

    if alerts:
        st.success("Signals this refresh:\n" + "\n".join(alerts))
        send_telegram("\n".join(alerts))


# ===========================================
# TAB 2 – CHART
# ===========================================
with tab_chart:
    st.subheader("📊 Candlestick Chart with EMA, Supertrend, Patterns & Zones (Daily)")

    chart_symbol = st.selectbox(
        "Select symbol for chart",
        all_symbols,
        index=all_symbols.index("RELIANCE") if "RELIANCE" in all_symbols else 0
    )

    hist_chart = get_history_from_nse(chart_symbol, days_back=220)

    if hist_chart is None or hist_chart.empty:
        st.warning("No historical data available from NSE for this symbol.")
    else:
        dfc = hist_chart.copy()
        st.markdown(f"#### {chart_symbol} – Last {len(dfc)} Daily Candles")

        fig_candle = go.Figure()
        fig_candle.add_trace(go.Candlestick(
            x=dfc["Date"],
            open=dfc["Open"],
            high=dfc["High"],
            low=dfc["Low"],
            close=dfc["Close"],
            name="Price"
        ))

        fig_candle.add_trace(go.Scatter(
            x=dfc["Date"], y=dfc["EMA9"], mode="lines", name="EMA9",
            line=dict(color="green")
        ))
        fig_candle.add_trace(go.Scatter(
            x=dfc["Date"], y=dfc["EMA21"], mode="lines", name="EMA21",
            line=dict(color="red")
        ))
        fig_candle.add_trace(go.Scatter(
            x=dfc["Date"], y=dfc["EMA20"], mode="lines", name="EMA20",
            line=dict(color="blue")
        ))
        fig_candle.add_trace(go.Scatter(
            x=dfc["Date"], y=dfc["EMA50"], mode="lines", name="EMA50",
            line=dict(color="purple")
        ))
        fig_candle.add_trace(go.Scatter(
            x=dfc["Date"], y=dfc["SUPERTREND"], mode="lines", name="Supertrend"
        ))

        bullish_points = dfc[dfc["CROSS"] == "Bullish"]
        bearish_points = dfc[dfc["CROSS"] == "Bearish"]

        fig_candle.add_trace(go.Scatter(
            x=bullish_points["Date"],
            y=bullish_points["Close"],
            mode="markers+text",
            text=["🔼"] * len(bullish_points),
            textposition="top center",
            marker=dict(size=14, color="green"),
            name="Bullish Cross"
        ))

        fig_candle.add_trace(go.Scatter(
            x=bearish_points["Date"],
            y=bearish_points["Close"],
            mode="markers+text",
            text=["🔽"] * len(bearish_points),
            textposition="bottom center",
            marker=dict(size=14, color="red"),
            name="Bearish Cross"
        ))

        dfc["BullishEngulfing"] = [
            is_bullish_engulfing(dfc, i) for i in range(len(dfc))
        ]
        dfc["HammerPattern"] = [
            is_hammer(dfc, i) for i in range(len(dfc))
        ]
        dfc["InsideBar"] = [
            is_inside_bar(dfc, i) for i in range(len(dfc))
        ]
        dfc["Breakout20"] = dfc["Close"] > dfc["HIGH20"].shift()

        be = dfc[dfc["BullishEngulfing"]]
        fig_candle.add_trace(go.Scatter(
            x=be["Date"], y=be["Low"] * 0.995,
            mode="text",
            text=["🟢 BE"] * len(be),
            textposition="bottom center",
            name="Bullish Engulfing"
        ))

        ha = dfc[dfc["HammerPattern"]]
        fig_candle.add_trace(go.Scatter(
            x=ha["Date"], y=ha["Low"] * 0.995,
            mode="text",
            text=["🔨"] * len(ha),
            textposition="bottom center",
            name="Hammer"
        ))

        ib = dfc[dfc["InsideBar"]]
        fig_candle.add_trace(go.Scatter(
            x=ib["Date"], y=ib["High"] * 1.005,
            mode="text",
            text=["📦"] * len(ib),
            textposition="top center",
            name="Inside Bar"
        ))

        bo = dfc[dfc["Breakout20"]]
        fig_candle.add_trace(go.Scatter(
            x=bo["Date"], y=bo["High"] * 1.01,
            mode="text",
            text=["🚀"] * len(bo),
            textposition="top center",
            name="Breakout 20D High"
        ))

        ph_idx, pl_idx = get_pivots(dfc, left=3, right=3)

        demand_zones = build_zones_from_pivots(dfc, pl_idx, kind="demand", max_zones=3)
        supply_zones = build_zones_from_pivots(dfc, ph_idx, kind="supply", max_zones=3)

        for x0, x1, y0, y1 in demand_zones:
            fig_candle.add_shape(
                type="rect",
                xref="x", yref="y",
                x0=x0, x1=x1,
                y0=y0, y1=y1,
                fillcolor="rgba(0, 255, 0, 0.12)",
                line=dict(width=0),
                layer="below"
            )

        for x0, x1, y0, y1 in supply_zones:
            fig_candle.add_shape(
                type="rect",
                xref="x", yref="y",
                x0=x0, x1=x1,
                y0=y0, y1=y1,
                fillcolor="rgba(255, 0, 0, 0.12)",
                line=dict(width=0),
                layer="below"
            )

        low_tl = build_trendline_points(dfc, pl_idx, use="low")
        high_tl = build_trendline_points(dfc, ph_idx, use="high")

        if low_tl is not None:
            x_vals, y_vals = low_tl
            fig_candle.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="lines",
                name="Support TL",
                line=dict(color="lime", width=2, dash="dash")
            ))

        if high_tl is not None:
            x_vals, y_vals = high_tl
            fig_candle.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="lines",
                name="Resistance TL",
                line=dict(color="red", width=2, dash="dash")
            ))

        fig_candle.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig_candle, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**RSI (14)**")
            fig_rsi = go.Figure(
                data=[go.Scatter(x=dfc["Date"], y=dfc["RSI"], mode="lines", name="RSI")]
            )
            fig_rsi.add_hline(y=70, line_dash="dot")
            fig_rsi.add_hline(y=30, line_dash="dot")
            fig_rsi.update_layout(height=250)
            st.plotly_chart(fig_rsi, use_container_width=True)

        with c2:
            st.markdown("**MACD**")
            fig_macd = go.Figure()
            fig_macd.add_trace(
                go.Scatter(x=dcf["Date"], y=dfc["MACD"], mode="lines", name="MACD")
            )
            fig_macd.add_trace(
                go.Scatter(x=dfc["Date"], y=dfc["MACD_SIGNAL"], mode="lines", name="Signal")
            )
            fig_macd.update_layout(height=250)
            st.plotly_chart(fig_macd, use_container_width=True)
