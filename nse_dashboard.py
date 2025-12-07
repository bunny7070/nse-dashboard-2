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
st.set_page_config(page_title="NSE Trading Dashboard", layout="wide")

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

st.title("📊 NSE Trading Dashboard – Watchlist | Chart | Options | Scanner | Backtest | Calculator")

# ===========================================
# SIDEBAR – REFRESH + TELEGRAM (OPTIONAL)
# ===========================================
refresh_rate = st.sidebar.slider("⏳ Auto Refresh (seconds)", 5, 60, 15)

st.sidebar.markdown("### 📲 Telegram Alerts (optional)")
tg_enable = st.sidebar.checkbox("Enable Alerts")
tg_token = st.sidebar.text_input("Bot Token", type="password")
tg_chat_id = st.sidebar.text_input("Chat ID / Group ID")

st_autorefresh(interval=refresh_rate * 1000, key="refresh_counter")


def send_telegram(msg: str):
    if not (tg_enable and tg_token and tg_chat_id):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{tg_token}/sendMessage",
            data={"chat_id": tg_chat_id, "text": msg},
            timeout=5,
        )
    except Exception:
        pass


# ===========================================
# SYMBOL LIST
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
# HISTORICAL DATA FROM NSE
# ===========================================
def get_history_from_nse(symbol: str, days_back: int = 200) -> pd.DataFrame | None:
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

        df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

        df["RSI"] = rsi(df["Close"])
        df["MACD"], df["MACD_SIGNAL"] = macd(df["Close"])
        df["VOL20"] = df["Volume"].rolling(20).mean()
        df["HIGH20"] = df["High"].rolling(20).max()

        st_line, st_dir = supertrend(df)
        df["SUPERTREND"] = st_line
        df["ST_DIR"] = st_dir

        df["CROSS"] = np.where(
            (df["EMA9"] > df["EMA21"]) & (df["EMA9"].shift(1) <= df["EMA21"].shift(1)),
            "Bullish",
            np.where(
                (df["EMA9"] < df["EMA21"]) & (df["EMA9"].shift(1) >= df["EMA21"].shift(1)),
                "Bearish",
                "",
            ),
        )

        return df

    except Exception:
        return None


# ===========================================
# STRATEGY ENGINE (DAILY)
# ===========================================
def compute_strategy_signal(last_row: pd.Series, vol_spike: bool, breakout: bool) -> str:
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
                trades.append(
                    {"Date": row["Date"], "Type": "BUY", "Price": entry_price, "Qty": qty}
                )

        exit_condition = (
            (position > 0 and "SELL" in signal)
            or (position > 0 and row["Close"] < entry_price * 0.97)
            or (position > 0 and i == len(df) - 1)
        )

        if exit_condition and position > 0:
            exit_price = row["Close"]
            cash += position * exit_price
            trades.append(
                {
                    "Date": row["Date"],
                    "Type": "SELL",
                    "Price": exit_price,
                    "Qty": position,
                    "PnL": (exit_price - entry_price) * position,
                }
            )
            position = 0
            entry_price = 0.0

    last_close = df.iloc[-1]["Close"]
    equity = cash + position * last_close
    return equity, pd.DataFrame(trades)


# ===========================================
# CANDLE PATTERNS
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
    return (
        prev_red
        and cur_green
        and body_cur_low <= body_prev_low
        and body_cur_high >= body_prev_high
    )


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
) = st.tabs(
    [
        "📈 Watchlist",
        "📊 Chart",
        "📉 Option Chain",
        "🚨 NIFTY500 Scanner",
        "📜 Backtest",
        "🧮 Calculator",
    ]
)

# ===========================================
# TAB 1 – WATCHLIST
# ===========================================
with tab_watch:
    st.subheader("📈 Live Watchlist + Strategy Signals (Daily)")

    defaults = [s for s in ["RELIANCE", "AXISCADES", "JIOFIN"] if s in all_symbols]
    selected = st.multiselect("Select Stocks", all_symbols, default=defaults)

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
                (last_price - prev_close) / prev_close * 100 if prev_close else None
            )

            hist = get_history_from_nse(sym, days_back=200)

            ema9 = ema20 = ema50 = rsi14 = macd_val = macd_sig = None
            vol20 = high20 = None
            st_dir = ""
            vol_spike = breakout = False
            cross_text = ""
            signal_text = "NA"

            if hist is not None and len(hist) >= 25:
                last = hist.iloc[-1]
                ema9 = float(last["EMA9"])
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

            rows.append(
                [
                    sym,
                    last_price,
                    round(pct_change, 2) if pct_change is not None else None,
                    int(total_vol),
                    int(vol20) if vol20 and not np.isnan(vol20) else None,
                    "Yes" if vol_spike else "",
                    prev_close,
                    ema9,
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
                ]
            )

            if "BUY" in signal_text:
                alerts.append(
                    f"🚀 {sym} | {signal_text} | Price: {last_price} | Chg: {pct_change:.2f}%"
                    if pct_change is not None
                    else f"🚀 {sym} | {signal_text} | Price: {last_price}"
                )

        except Exception as e:
            st.warning(f"Error loading {sym}: {e}")

    df_watch = pd.DataFrame(
        rows,
        columns=[
            "Symbol",
            "Price",
            "% Chg",
            "Volume",
            "20D Vol",
            "Vol Spike",
            "Prev Close",
            "EMA9",
            "EMA20",
            "EMA50",
            "RSI",
            "MACD",
            "MACD Signal",
            "ST Dir",
            "20D High",
            "Breakout",
            "EMA9/21 Cross",
            "Signal",
        ],
    )

    st.dataframe(df_watch, use_container_width=True)

    if alerts:
        st.success("Signals:\n" + "\n".join(alerts))
        send_telegram("\n".join(alerts))


# ===========================================
# TAB 2 – CHART
# ===========================================
with tab_chart:
    st.subheader("📊 Candlestick Chart + Indicators (Daily)")

    chart_symbol = st.selectbox(
        "Select symbol for chart",
        all_symbols,
        index=all_symbols.index("RELIANCE") if "RELIANCE" in all_symbols else 0,
    )

    hist_chart = get_history_from_nse(chart_symbol, days_back=250)

    if hist_chart is None or hist_chart.empty:
        st.warning("No historical data available.")
    else:
        dfc = hist_chart.copy()

        fig_candle = go.Figure()
        fig_candle.add_trace(
            go.Candlestick(
                x=dfc["Date"],
                open=dfc["Open"],
                high=dfc["High"],
                low=dfc["Low"],
                close=dfc["Close"],
                name="Price",
            )
        )

        fig_candle.add_trace(
            go.Scatter(
                x=dfc["Date"],
                y=dfc["EMA9"],
                mode="lines",
                name="EMA9",
                line=dict(color="green"),
            )
        )
        fig_candle.add_trace(
            go.Scatter(
                x=dfc["Date"],
                y=dfc["EMA21"],
                mode="lines",
                name="EMA21",
                line=dict(color="red"),
            )
        )
        fig_candle.add_trace(
            go.Scatter(
                x=dfc["Date"],
                y=dfc["EMA20"],
                mode="lines",
                name="EMA20",
                line=dict(color="blue"),
            )
        )
        fig_candle.add_trace(
            go.Scatter(
                x=dfc["Date"],
                y=dfc["EMA50"],
                mode="lines",
                name="EMA50",
                line=dict(color="purple"),
            )
        )
        fig_candle.add_trace(
            go.Scatter(
                x=dfc["Date"],
                y=dfc["SUPERTREND"],
                mode="lines",
                name="Supertrend",
            )
        )

        bullish_points = dfc[dfc["CROSS"] == "Bullish"]
        bearish_points = dfc[dfc["CROSS"] == "Bearish"]

        fig_candle.add_trace(
            go.Scatter(
                x=bullish_points["Date"],
                y=bullish_points["Close"],
                mode="markers+text",
                text=["🔼"] * len(bullish_points),
                textposition="top center",
                marker=dict(size=14, color="green"),
                name="Bullish Cross",
            )
        )

        fig_candle.add_trace(
            go.Scatter(
                x=bearish_points["Date"],
                y=bearish_points["Close"],
                mode="markers+text",
                text=["🔽"] * len(bearish_points),
                textposition="bottom center",
                marker=dict(size=14, color="red"),
                name="Bearish Cross",
            )
        )

        dfc["BullishEngulfing"] = [
            is_bullish_engulfing(dfc, i) for i in range(len(dfc))
        ]
        dfc["HammerPattern"] = [is_hammer(dfc, i) for i in range(len(dfc))]
        dfc["InsideBar"] = [is_inside_bar(dfc, i) for i in range(len(dfc))]
        dfc["Breakout20"] = dfc["Close"] > dfc["HIGH20"].shift()

        be = dfc[dfc["BullishEngulfing"]]
        fig_candle.add_trace(
            go.Scatter(
                x=be["Date"],
                y=be["Low"] * 0.995,
                mode="text",
                text=["🟢 BE"] * len(be),
                textposition="bottom center",
                name="Bullish Engulfing",
            )
        )

        ha = dfc[dfc["HammerPattern"]]
        fig_candle.add_trace(
            go.Scatter(
                x=ha["Date"],
                y=ha["Low"] * 0.995,
                mode="text",
                text=["🔨"] * len(ha),
                textposition="bottom center",
                name="Hammer",
            )
        )

        ib = dfc[dfc["InsideBar"]]
        fig_candle.add_trace(
            go.Scatter(
                x=ib["Date"],
                y=ib["High"] * 1.005,
                mode="text",
                text=["📦"] * len(ib),
                textposition="top center",
                name="Inside Bar",
            )
        )

        bo = dfc[dfc["Breakout20"]]
        fig_candle.add_trace(
            go.Scatter(
                x=bo["Date"],
                y=bo["High"] * 1.01,
                mode="text",
                text=["🚀"] * len(bo),
                textposition="top center",
                name="20D Breakout",
            )
        )

        fig_candle.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
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
                go.Scatter(
                    x=dfc["Date"], y=dfc["MACD"], mode="lines", name="MACD"
                )
            )
            fig_macd.add_trace(
                go.Scatter(
                    x=dfc["Date"], y=dfc["MACD_SIGNAL"], mode="lines", name="Signal"
                )
            )
            fig_macd.update_layout(height=250)
            st.plotly_chart(fig_macd, use_container_width=True)


# ===========================================
# TAB 3 – ADVANCED OPTION CHAIN (C)
# ===========================================
with tab_opt:
    st.subheader("📉 Option Chain – OI, PCR, Max Pain, Basic Greeks")

    oc_symbol = st.selectbox(
        "Select F&O symbol",
        ["NIFTY", "BANKNIFTY"] + all_symbols,
    )

    try:
        if oc_symbol in ["NIFTY", "BANKNIFTY"]:
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={oc_symbol}"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={oc_symbol}"

        payload = nsefetch(url)
        data = payload["records"]["data"]

        rows_ce = []
        rows_pe = []

        for r in data:
            strike = r.get("strikePrice")
            ce = r.get("CE")
            pe = r.get("PE")

            if ce:
                rows_ce.append(
                    [
                        strike,
                        ce.get("lastPrice", 0),
                        ce.get("openInterest", 0),
                        ce.get("changeinOpenInterest", 0),
                        ce.get("impliedVolatility", None),
                        ce.get("totalTradedVolume", 0),
                    ]
                )
            if pe:
                rows_pe.append(
                    [
                        strike,
                        pe.get("lastPrice", 0),
                        pe.get("openInterest", 0),
                        pe.get("changeinOpenInterest", 0),
                        pe.get("impliedVolatility", None),
                        pe.get("totalTradedVolume", 0),
                    ]
                )

        df_ce = pd.DataFrame(
            rows_ce,
            columns=[
                "Strike",
                "CE_LTP",
                "CE_OI",
                "CE_OI_Chg",
                "CE_IV",
                "CE_Vol",
            ],
        )
        df_pe = pd.DataFrame(
            rows_pe,
            columns=[
                "Strike",
                "PE_LTP",
                "PE_OI",
                "PE_OI_Chg",
                "PE_IV",
                "PE_Vol",
            ],
        )

        oc = pd.merge(df_ce, df_pe, on="Strike", how="outer").fillna(0)
        oc.sort_values("Strike", inplace=True)

        total_ce = oc["CE_OI"].sum()
        total_pe = oc["PE_OI"].sum()
        pcr = total_pe / total_ce if total_ce else None

        oc["Total_OI"] = oc["CE_OI"] + oc["PE_OI"]
        max_pain_strike = oc.loc[oc["Total_OI"].idxmax(), "Strike"]

        c1, c2, c3 = st.columns(3)
        c1.metric("PCR", f"{pcr:.2f}" if pcr else "NA")
        c2.metric("Max Pain (Max Total OI)", f"{max_pain_strike}")
        c3.metric("Strikes Count", len(oc))

        st.markdown("### Option Chain Table")
        st.dataframe(oc, use_container_width=True)

        st.markdown("### OI by Strike")
        fig_oi = go.Figure()
        fig_oi.add_trace(
            go.Bar(x=oc["Strike"], y=oc["CE_OI"], name="Call OI")
        )
        fig_oi.add_trace(
            go.Bar(x=oc["Strike"], y=oc["PE_OI"], name="Put OI")
        )
        fig_oi.update_layout(
            barmode="group",
            height=400,
            xaxis_title="Strike",
            yaxis_title="Open Interest",
        )
        st.plotly_chart(fig_oi, use_container_width=True)

        st.markdown("### OI Change Heatmap")
        heat_df = pd.DataFrame(
            {
                "Strike": oc["Strike"].astype(str).tolist() * 2,
                "Type": ["CE"] * len(oc) + ["PE"] * len(oc),
                "OI_Change": list(oc["CE_OI_Chg"]) + list(oc["PE_OI_Chg"]),
            }
        )
        pivot = heat_df.pivot(index="Type", columns="Strike", values="OI_Change")
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorbar_title="ΔOI",
            )
        )
        fig_hm.update_layout(
            height=300, xaxis_title="Strike", yaxis_title="Type"
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading option chain: {e}")
        st.info("NSE may temporarily block too frequent calls. Try again in a bit.")


# ===========================================
# TAB 4 – NIFTY500 SCANNER
# ===========================================
with tab_scan:
    st.subheader("🚨 NIFTY500 Scanner – EMA / Volume / RSI / MACD")

    max_scan = st.slider("Max symbols to scan", 20, 200, 60)
    run_scan = st.button("Run Scanner ▶")

    if run_scan:
        try:
            payload = nsefetch(
                "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500"
            )
            symbols500 = sorted([row["symbol"] for row in payload["data"]])
        except Exception as e:
            st.error(f"Error loading NIFTY500 list: {e}")
            symbols500 = []

        rows_scan = []

        with st.spinner("Scanning..."):
            for sym in symbols500[:max_scan]:
                df = get_history_from_nse(sym, days_back=220)
                if df is None or len(df) < 50:
                    continue

                last = df.iloc[-1]
                volume = last["Volume"]
                vol20 = last["VOL20"]

                cond_trend = last["EMA9"] > last["EMA21"] > last["EMA50"]
                cond_vol = (
                    last["EMA9"] > last["EMA20"]
                    and vol20
                    and not np.isnan(vol20)
                    and volume > 1.5 * vol20
                )
                cond_rsi = last["EMA9"] > last["EMA20"] and last["RSI"] > 60
                cond_macd = (
                    last["EMA9"] > last["EMA20"]
                    and last["MACD"] > last["MACD_SIGNAL"]
                )

                if cond_trend or cond_vol or cond_rsi or cond_macd:
                    rows_scan.append(
                        [
                            sym,
                            float(last["Close"]),
                            int(volume),
                            bool(cond_trend),
                            bool(cond_vol),
                            bool(cond_rsi),
                            bool(cond_macd),
                        ]
                    )

        df_scan = pd.DataFrame(
            rows_scan,
            columns=[
                "Symbol",
                "Close",
                "Volume",
                "Trend (EMA9>21>50)",
                "EMA+Vol Spike",
                "EMA+RSI>60",
                "EMA+MACD Bullish",
            ],
        )
        if df_scan.empty:
            st.warning("No matching stocks found.")
        else:
            st.dataframe(df_scan, use_container_width=True)


# ===========================================
# TAB 5 – BACKTEST
# ===========================================
with tab_bt:
    st.subheader("📜 Backtest (Daily Strategy)")

    bt_symbol = st.selectbox(
        "Symbol to backtest",
        all_symbols,
        index=all_symbols.index("RELIANCE") if "RELIANCE" in all_symbols else 0,
    )
    days_back = st.slider("Days of history", 100, 800, 400)

    if st.button("Run Backtest ▶"):
        equity, trades_df = run_backtest(bt_symbol, days_back=days_back)
        if equity is None:
            st.warning("Not enough data to backtest.")
        else:
            st.metric("Final Equity (Starting 1,00,000)", f"{equity:,.0f} ₹")
            if trades_df is not None and not trades_df.empty:
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.info("No completed trades generated.")


# ===========================================
# TAB 6 – CALCULATOR (Lumpsum + SIP)
# ===========================================
with tab_calc:
    st.subheader("🧮 Investment Calculator – Lumpsum & SIP")

    col_lump, col_sip = st.columns(2)

    # ----- Lumpsum -----
    with col_lump:
        st.markdown("### 💰 Lumpsum Calculator")
        ls_symbol = st.selectbox(
            "Lumpsum Stock",
            all_symbols,
            index=all_symbols.index("RELIANCE") if "RELIANCE" in all_symbols else 0,
        )
        invest_amt = st.number_input(
            "Investment Amount (₹)", min_value=1000, max_value=50000000, value=100000
        )
        years = st.slider("Years ago", 1, 15, 5)

        if st.button("Calculate Lumpsum ▶"):
            df = get_history_from_nse(ls_symbol, days_back=years * 365)
            if df is None or df.empty or len(df) < 10:
                st.error("Not enough data.")
            else:
                start_price = df.iloc[0]["Close"]
                current_price = df.iloc[-1]["Close"]
                qty = invest_amt / start_price
                today_value = qty * current_price
                ret_pct = (today_value / invest_amt - 1) * 100
                total_years = (df.iloc[-1]["Date"] - df.iloc[0]["Date"]).days / 365
                if total_years <= 0:
                    total_years = years
                cagr = (today_value / invest_amt) ** (1 / total_years) - 1

                st.metric("Start Price", f"₹{start_price:.2f}")
                st.metric("Current Price", f"₹{current_price:.2f}")
                st.metric("Quantity", f"{qty:.2f}")
                st.metric("Current Value", f"₹{today_value:,.0f}")
                st.metric("Total Return", f"{ret_pct:.2f}%")
                st.metric("CAGR", f"{cagr*100:.2f}%")

    # ----- SIP -----
    with col_sip:
        st.markdown("### 📆 SIP Calculator (Monthly)")
        sip_symbol = st.selectbox(
            "SIP Stock",
            all_symbols,
            index=all_symbols.index("RELIANCE") if "RELIANCE" in all_symbols else 0,
        )
        sip_amt = st.number_input(
            "Monthly SIP Amount (₹)", min_value=500, max_value=1000000, value=5000
        )
        sip_years = st.slider("SIP Duration (years)", 1, 15, 5)

        if st.button("Calculate SIP ▶"):
            df_sip = get_history_from_nse(sip_symbol, days_back=sip_years * 365 + 60)
            if df_sip is None or df_sip.empty or len(df_sip) < 20:
                st.error("Not enough data.")
            else:
                df_sip["YearMonth"] = df_sip["Date"].dt.to_period("M")
                monthly = df_sip.groupby("YearMonth").first().reset_index()
                monthly.sort_values("Date", inplace=True)

                if len(monthly) > sip_years * 12:
                    monthly = monthly.iloc[-sip_years * 12 :]

                units = 0.0
                for _, row in monthly.iterrows():
                    units += sip_amt / row["Close"]

                last_price = df_sip.iloc[-1]["Close"]
                current_value = units * last_price
                total_invested = sip_amt * len(monthly)
                profit = current_value - total_invested
                total_years = (df_sip.iloc[-1]["Date"] - monthly.iloc[0]["Date"]).days / 365
                if total_years <= 0:
                    total_years = sip_years
                cagr_sip = (current_value / total_invested) ** (1 / total_years) - 1

                st.metric("Total Invested", f"₹{total_invested:,.0f}")
                st.metric("Current Value", f"₹{current_value:,.0f}")
                st.metric("Profit", f"₹{profit:,.0f}")
                st.metric("Approx. CAGR", f"{cagr_sip*100:.2f}%")

                st.success(
                    f"SIP of ₹{sip_amt:,.0f}/month in {sip_symbol} for ~{total_years:.1f} years "
                    f"→ ₹{current_value:,.0f} (Invested ₹{total_invested:,.0f})"
                )
