import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nsepython import nsefetch
from streamlit_autorefresh import st_autorefresh
import requests
from datetime import date, timedelta

# ===========================================
# PAGE CONFIG + DARK THEME OVERRIDE
# ===========================================
st.set_page_config(
    page_title="NSE Watchlist + Strategy + Scanner + Option Chain",
    layout="wide"
)

# Simple dark-style override (works even if Streamlit theme is default)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stDataFrame, .stMetric, .stMarkdown, .stSelectbox, .stButton, .stSlider {
        color: #fafafa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 NSE Live Watchlist + Strategy + Scanner + Option Chain (NSE Data Only)")

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
        # Do not crash UI if Telegram fails
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
        df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
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
    """
    Multi-condition strategy (daily).
    """
    ema20 = last_row.get("EMA20", np.nan)
    ema50 = last_row.get("EMA50", np.nan)
    rsi_val = last_row.get("RSI", np.nan)
    macd_val = last_row.get("MACD", np.nan)
    macd_sig = last_row.get("MACD_SIGNAL", np.nan)
    st_dir_val = last_row.get("ST_DIR", 0)

    up_trend = st_dir_val == 1
    ema_bull = not np.isnan(ema20) and not np.isnan(ema50) and ema20 > ema50
    macd_bull = not np.isnan(macd_val) and not np.isnan(macd_sig) and macd_val > macd_sig
    rsi_bull = not np.isnan(rsi_val) and rsi_val > 55

    # Strong buy: all alignment + breakout + volume spike
    if up_trend and ema_bull and macd_bull and rsi_bull and breakout and vol_spike:
        return "STRONG BUY ✅"

    # Buy: trend + at least two confirmations
    bullish_count = sum([up_trend, ema_bull, macd_bull, rsi_bull])
    if bullish_count >= 3 and (breakout or vol_spike):
        return "BUY ✅"

    # Sell: clear downtrend / bearish momentum
    if st_dir_val == -1 or (not np.isnan(rsi_val) and rsi_val < 45 and not macd_bull):
        return "SELL ❌"

    return "HOLD ⏸"


# ===========================================
# MULTI-TF STRATEGY ENGINE (DAILY-ONLY SKELETON)
# ===========================================
def compute_multitf_signal(
    daily_row: pd.Series,
    tf15_row: pd.Series | None = None,
    tf5_row: pd.Series | None = None
) -> str:
    """
    Multi-timeframe decision engine.
    Currently uses DAILY row only.
    Later you can plug 15m / 5m rows from broker intraday data (Angel websocket).
    """
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
# SIMPLE BACKTEST ENGINE (DAILY)
# ===========================================
def run_backtest(symbol: str, days_back: int = 400):
    """
    Simple long-only backtest using daily candles and compute_strategy_signal.
    """
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

    for i in range(25, len(df)):  # need indicators to be ready
        row = df.iloc[i]
        breakout = row["Close"] > row["HIGH20"] if not np.isnan(row["HIGH20"]) else False
        vol_spike = False  # can reuse volume logic if required

        signal = compute_strategy_signal(row, vol_spike, breakout)

        # Entry: Buy when STRONG BUY or BUY and no position
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

        # Exit conditions
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


# ========= SUPPORT / RESISTANCE + ZONES + TRENDLINES HELPERS =========
def get_pivots(df: pd.DataFrame, left: int = 3, right: int = 3):
    """Return indices of pivot highs and lows using swing structure."""
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
    """
    Build simple supply/demand zones from pivot indices.
    kind = 'demand' (from swing lows) or 'supply' (from swing highs)
    Returns list of (x0, x1, y0, y1)
    """
    zones = []
    for idx in reversed(pivot_idx_list):
        bar = df.iloc[idx]
        if kind == "demand":
            price_low = bar["Low"]
            price_high = bar["Low"] * 1.005  # 0.5% band
        else:  # supply
            price_high = bar["High"]
            price_low = bar["High"] * 0.995  # 0.5% band

        x0 = bar["Date"]
        x1 = df["Date"].iloc[-1]  # extend zone to latest candle

        zones.append((x0, x1, price_low, price_high))
        if len(zones) >= max_zones:
            break
    return zones


def build_trendline_points(df: pd.DataFrame, pivot_idx_list, use="low"):
    """
    From a list of pivot indices, return two points for a trendline.
    use='low' -> connect lows, use='high' -> connect highs.
    """
    if len(pivot_idx_list) < 2:
        return None
    i1, i2 = pivot_idx_list[-2], pivot_idx_list[-1]
    if use == "low":
        y_vals = df["Low"].iloc[[i1, i2]].tolist()
    else:
        y_vals = df["High"].iloc[[i1, i2]].tolist()
    x_vals = df["Date"].iloc[[i1, i2]].tolist()
    return x_vals, y_vals


# ===========================================
# TABS
# ===========================================
tab_watch, tab_chart, tab_opt, tab_scan, tab_bt, tab_calc = st.tabs(
    ["📈 Watchlist", "📊 Chart", "📉 Option Chain", "🚨 NIFTY500 Scanner", "📜 Backtest", "🧮 Calculator"]
)

# ===========================================
# TAB 1 – WATCHLIST WITH STRATEGY SIGNALS
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
# TAB 2 – CHART (CANDLESTICKS + INDICATORS + ZONES + PATTERNS)
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

        # EMA lines
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

        # EMA9/21 crossover arrows
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

        # ==== PATTERN MARKERS ON CHART ====
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

        be = dfc[dfc["BullishEngulfing"] == True]
        fig_candle.add_trace(go.Scatter(
            x=be["Date"], y=be["Low"] * 0.995,
            mode="text",
            text=["🟢 BE"] * len(be),
            textposition="bottom center",
            name="Bullish Engulfing"
        ))

        ha = dfc[dfc["HammerPattern"] == True]
        fig_candle.add_trace(go.Scatter(
            x=ha["Date"], y=ha["Low"] * 0.995,
            mode="text",
            text=["🔨"] * len(ha),
            textposition="bottom center",
            name="Hammer"
        ))

        ib = dfc[dfc["InsideBar"] == True]
        fig_candle.add_trace(go.Scatter(
            x=ib["Date"], y=ib["High"] * 1.005,
            mode="text",
            text=["📦"] * len(ib),
            textposition="top center",
            name="Inside Bar"
        ))

        bo = dfc[dfc["Breakout20"] == True]
        fig_candle.add_trace(go.Scatter(
            x=bo["Date"], y=bo["High"] * 1.01,
            mode="text",
            text=["🚀"] * len(bo),
            textposition="top center",
            name="Breakout 20D High"
        ))

        # ========= AUTO ZONES & TRENDLINES =========
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
                go.Scatter(x=dfc["Date"], y=dfc["MACD"], mode="lines", name="MACD")
            )
            fig_macd.add_trace(
                go.Scatter(x=dfc["Date"], y=dfc["MACD_SIGNAL"], mode="lines", name="Signal")
            )
            fig_macd.update_layout(height=250)
            st.plotly_chart(fig_macd, use_container_width=True)


# ===========================================
# TAB 3 – OPTION CHAIN + OI CHANGE HEATMAP
# ===========================================
with tab_opt:
    st.subheader("📉 Option Chain – OI, PCR, Support & Resistance + OI Change Heatmap")

    oc_symbol = st.selectbox(
        "Select F&O symbol",
        ["NIFTY", "BANKNIFTY"] + all_symbols
    )

    if "prev_oc" not in st.session_state:
        st.session_state["prev_oc"] = None

    try:
        if oc_symbol in ["NIFTY", "BANKNIFTY"]:
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={oc_symbol}"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={oc_symbol}"

        payload = nsefetch(url)
        data = payload["records"]["data"]

        oc_rows = []
        for r in data:
            strike = r.get("strikePrice")
            ce_oi = r.get("CE", {}).get("openInterest", 0)
            pe_oi = r.get("PE", {}).get("openInterest", 0)
            oc_rows.append([strike, ce_oi, pe_oi])

        oc = pd.DataFrame(oc_rows, columns=["Strike", "CE_OI", "PE_OI"]).dropna()
        if oc.empty:
            st.warning("No option chain data available.")
        else:
            prev_oc = st.session_state.get("prev_oc")

            if prev_oc is not None:
                oc = oc.merge(prev_oc[["Strike", "CE_OI", "PE_OI"]],
                              on="Strike", how="left", suffixes=("", "_prev"))
                oc["CE_OI_prev"].fillna(oc["CE_OI"], inplace=True)
                oc["PE_OI_prev"].fillna(oc["PE_OI"], inplace=True)
                oc["CE_OI_CHG"] = oc["CE_OI"] - oc["CE_OI_prev"]
                oc["PE_OI_CHG"] = oc["PE_OI"] - oc["PE_OI_prev"]
            else:
                oc["CE_OI_CHG"] = 0
                oc["PE_OI_CHG"] = 0

            st.session_state["prev_oc"] = oc.copy()

            total_ce = oc["CE_OI"].sum()
            total_pe = oc["PE_OI"].sum()
            pcr = total_pe / total_ce if total_ce else None

            res_strike = oc.loc[oc["CE_OI"].idxmax(), "Strike"]
            sup_strike = oc.loc[oc["PE_OI"].idxmax(), "Strike"]

            c1, c2, c3 = st.columns(3)
            c1.metric("PCR", f"{pcr:.2f}" if pcr else "NA")
            c2.metric("Resistance (Max CE OI)", res_strike)
            c3.metric("Support (Max PE OI)", sup_strike)

            st.markdown("### 🔍 OI by Strike")
            fig_oi = go.Figure()
            fig_oi.add_trace(go.Bar(x=oc["Strike"], y=oc["CE_OI"], name="Call OI"))
            fig_oi.add_trace(go.Bar(x=oc["Strike"], y=oc["PE_OI"], name="Put OI"))
            fig_oi.update_layout(
                barmode="group",
                height=400,
                xaxis_title="Strike",
                yaxis_title="Open Interest"
            )
            st.plotly_chart(fig_oi, use_container_width=True)

            st.markdown("### 🌡 OI Change Heatmap (vs Previous Refresh)")
            heat_df = pd.DataFrame({
                "Strike": oc["Strike"].astype(str).tolist() * 2,
                "Type": ["CE"] * len(oc) + ["PE"] * len(oc),
                "OI_Change": list(oc["CE_OI_CHG"]) + list(oc["PE_OI_CHG"])
            })

            pivot = heat_df.pivot(index="Type", columns="Strike", values="OI_Change")

            fig_hm = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorbar_title="Δ OI"
            ))
            fig_hm.update_layout(height=300, xaxis_title="Strike", yaxis_title="Type")
            st.plotly_chart(fig_hm, use_container_width=True)

            st.markdown("### ⚡ Top OI Change Strikes")
            top_changes = oc.assign(
                Total_OI_Change=oc["CE_OI_CHG"].abs() + oc["PE_OI_CHG"].abs()
            ).sort_values("Total_OI_Change", ascending=False).head(10)
            st.dataframe(
                top_changes[["Strike", "CE_OI_CHG", "PE_OI_CHG", "CE_OI", "PE_OI"]],
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error fetching option chain: {e}")
        st.info("Sometimes NSE blocks frequent calls; try again after a few seconds.")


# ===========================================
# TAB 4 – NIFTY500 SCANNER WITH PATTERNS & CROSS
# ===========================================
with tab_scan:
    st.subheader("🚨 NIFTY500 Volume Spike + Breakout + Pattern Scanner")

    max_scan = st.slider("Max symbols to scan", 20, 200, 60)
    run_scan = st.button("▶ Run Scan")

    if run_scan:
        try:
            n500_payload = nsefetch(
                "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500"
            )
            symbols500 = sorted([row["symbol"] for row in n500_payload["data"]])
        except Exception as e:
            st.error(f"Error loading NIFTY500 list: {e}")
            symbols500 = []

        scan_rows = []
        alerts_scan = []

        with st.spinner("Scanning NIFTY500..."):
            for sym in symbols500[:max_scan]:
                hist = get_history_from_nse(sym, days_back=200)
                if hist is None or len(hist) < 25:
                    continue

                dfh = hist.reset_index(drop=True)
                last_idx = len(dfh) - 1
                last = dfh.iloc[last_idx]

                close = float(last["Close"])
                volume = float(last["Volume"])
                vol20 = float(last["VOL20"]) if not np.isnan(last["VOL20"]) else None
                high20 = float(last["HIGH20"]) if not np.isnan(last["HIGH20"]) else None

                vol_spike = vol20 is not None and volume > 1.5 * vol20
                breakout = high20 is not None and close > high20
                be = is_bullish_engulfing(dfh, last_idx)
                hammer = is_hammer(dfh, last_idx)
                inside = is_inside_bar(dfh, last_idx)
                cross_text = dfh.iloc[last_idx]["CROSS"]

                if vol_spike or breakout or be or hammer or inside or cross_text != "":
                    scan_rows.append([
                        sym,
                        close,
                        int(volume),
                        int(vol20) if vol20 is not None else None,
                        vol_spike,
                        breakout,
                        be,
                        hammer,
                        inside,
                        cross_text
                    ])
                    alerts_scan.append(
                        f"{sym} | Close={close} | VolSpike={vol_spike} | Breakout={breakout} "
                        f"| BullEng={be} | Hammer={hammer} | Inside={inside} | Cross={cross_text}"
                    )

        df_scan = pd.DataFrame(
            scan_rows,
            columns=[
                "Symbol", "Close", "Volume", "20D Vol",
                "Vol Spike", "Breakout",
                "Bullish Engulfing", "Hammer", "Inside Bar",
                "EMA9/21 Cross"
            ]
        )
        if df_scan.empty:
            st.warning("No stocks matched conditions in scanned universe.")
        else:
            st.dataframe(df_scan, use_container_width=True)
            if alerts_scan:
                send_telegram("🚨 NIFTY500 Scan Alerts:\n" + "\n".join(alerts_scan[:20]))


# ===========================================
# TAB 5 – BACKTEST (DAILY)
# ===========================================
with tab_bt:
    st.subheader("📜 Backtest (Daily Strategy)")

    bt_symbol = st.selectbox(
        "Symbol to backtest",
        all_symbols,
        index=all_symbols.index("RELIANCE") if "RELIANCE" in all_symbols else 0
    )

    days_back = st.slider("Days of history", 100, 800, 400)

    if st.button("▶ Run Backtest"):
        equity, trades_df = run_backtest(bt_symbol, days_back=days_back)
        if equity is None:
            st.warning("Not enough data to backtest.")
        else:
            st.metric("Final Equity (Starting 1,00,000)", f"{equity:,.0f} ₹")
            if trades_df is not None and not trades_df.empty:
                st.markdown("### 🧾 Trades")
                st.dataframe(trades_df, use_container_width=True)

                pnl_series = trades_df[trades_df["Type"] == "SELL"]["PnL"].cumsum()
                fig_bt = go.Figure(data=[go.Scatter(
                    x=pnl_series.index, y=pnl_series.values, mode="lines", name="Cumulative PnL"
                )])
                fig_bt.update_layout(height=300, xaxis_title="Trade #", yaxis_title="PnL")
                st.plotly_chart(fig_bt, use_container_width=True)
            else:
                st.info("No completed trades generated for this period/strategy.")


# ===========================================
# TAB 6 – INVESTMENT CALCULATOR (COMPARE + SIP + CHART)
# ===========================================
with tab_calc:
    st.subheader("🧮 Investment Return Calculator – Lumpsum & SIP")

    calc_symbols = st.multiselect(
        "Select Stocks (for comparison)",
        all_symbols,
        default=["RELIANCE", "TCS", "HDFCBANK"]
    )

    col_lump, col_sip = st.columns(2)

    # ---------- Lumpsum ----------
    with col_lump:
        st.markdown("### 💰 Lumpsum Calculator")
        invest_amt = st.number_input("Investment Amount (₹ per stock)", min_value=1000, max_value=50000000, value=100000)
        years = st.slider("Years ago (Lumpsum)", 1, 15, 5, key="years_lump")

        if st.button("Calculate Lumpsum Returns"):
            results = []
            growth_data = {}

            for sym in calc_symbols:
                df_hist = get_history_from_nse(sym, days_back=years * 365)
                if df_hist is None or df_hist.empty or len(df_hist) < 10:
                    continue

                start_price = df_hist.iloc[0]["Close"]
                current_price = df_hist.iloc[-1]["Close"]

                qty = invest_amt / start_price
                today_value = qty * current_price
                returns_pct = (today_value / invest_amt - 1) * 100
                total_years = (df_hist.iloc[-1]["Date"] - df_hist.iloc[0]["Date"]).days / 365
                if total_years <= 0:
                    total_years = years
                cagr = ((today_value / invest_amt) ** (1 / total_years) - 1) * 100

                results.append([
                    sym, f"{start_price:.2f}", f"{current_price:.2f}",
                    f"{qty:.2f}", f"{today_value:,.0f}", f"{returns_pct:.2f}%", f"{cagr:.2f}%"
                ])

                growth_curve = df_hist["Close"] / start_price * invest_amt
                growth_data[sym] = (df_hist["Date"], growth_curve.tolist())

            if not results:
                st.error("No valid historical data found to compute.")
            else:
                st.markdown("#### 🥇 Lumpsum Leaderboard")
                df_res = pd.DataFrame(results, columns=[
                    "Stock", "Start Price", "Current Price", "Qty", "Value Today", "Total Return %", "CAGR %"
                ]).sort_values("Value Today", ascending=False)

                st.dataframe(df_res, use_container_width=True)

                st.markdown("#### 📈 Lumpsum Value Growth Curve")
                fig_gc = go.Figure()
                for sym, (dates, curve) in growth_data.items():
                    fig_gc.add_trace(go.Scatter(
                        x=dates,
                        y=curve,
                        mode="lines",
                        name=sym
                    ))
                fig_gc.update_layout(
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Value (₹)"
                )
                st.plotly_chart(fig_gc, use_container_width=True)

    # ---------- SIP ----------
    with col_sip:
        st.markdown("### 📆 SIP Calculator (Monthly)")
        sip_symbol = st.selectbox(
            "SIP Stock",
            all_symbols,
            index=all_symbols.index("RELIANCE") if "RELIANCE" in all_symbols else 0,
            key="sip_symbol"
        )
        sip_amt = st.number_input("Monthly SIP Amount (₹)", min_value=500, max_value=1000000, value=5000)
        sip_years = st.slider("SIP Duration (years)", 1, 15, 5, key="sip_years")

        if st.button("Calculate SIP Returns"):
            df_sip = get_history_from_nse(sip_symbol, days_back=sip_years * 365 + 60)
            if df_sip is None or df_sip.empty or len(df_sip) < 20:
                st.error("Not enough historical data for SIP calculation.")
            else:
                df_sip["YearMonth"] = df_sip["Date"].dt.to_period("M")
                monthly = df_sip.groupby("YearMonth").first().reset_index()
                monthly.sort_values("Date", inplace=True)

                # Use only last `sip_years` years
                if len(monthly) > sip_years * 12:
                    monthly = monthly.iloc[-sip_years * 12:]

                units = 0.0
                cash_flows = []
                dates_cf = []
                for _, row in monthly.iterrows():
                    price = row["Close"]
                    units += sip_amt / price
                    cash_flows.append(-sip_amt)
                    dates_cf.append(row["Date"])

                last_price = df_sip.iloc[-1]["Close"]
                current_value = units * last_price
                total_invested = sip_amt * len(monthly)
                profit = current_value - total_invested
                total_years_sip = (df_sip.iloc[-1]["Date"] - monthly.iloc[0]["Date"]).days / 365

                if total_years_sip <= 0:
                    total_years_sip = sip_years
                cagr_sip = ((current_value / total_invested) ** (1 / total_years_sip) - 1) * 100

                st.metric("Total Invested", f"₹{total_invested:,.0f}")
                st.metric("Current Value", f"₹{current_value:,.0f}")
                st.metric("Profit", f"₹{profit:,.0f}")
                st.metric("Approx. CAGR", f"{cagr_sip:.2f}%")

                st.success(
                    f"SIP of ₹{sip_amt:,.0f} / month in {sip_symbol} for ~{total_years_sip:.1f} years → "
                    f"₹{current_value:,.0f} (Invested ₹{total_invested:,.0f})"
                )

                # SIP growth curve
                st.markdown("#### 📈 SIP Portfolio Growth Over Time")
                portfolio_values = []
                cum_units = 0.0
                for _, row in monthly.iterrows():
                    cum_units += sip_amt / row["Close"]
                    portfolio_values.append(cum_units * row["Close"])

                fig_sip = go.Figure(
                    data=[go.Scatter(
                        x=monthly["Date"],
                        y=portfolio_values,
                        mode="lines+markers",
                        name="SIP Value"
                    )]
                )
                fig_sip.update_layout(
                    height=350,
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value (₹)"
                )
                st.plotly_chart(fig_sip, use_container_width=True)
