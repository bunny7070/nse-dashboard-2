import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nsepython import nsefetch
from streamlit_autorefresh import st_autorefresh
import requests
from datetime import date, timedelta
import math

# ===========================================
# STREAMLIT CONFIG
# ===========================================
st.set_page_config(page_title="NSE Trading Dashboard", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #fafafa; }
</style>
""", unsafe_allow_html=True)

st_autorefresh(interval=10000, key="refresh")  # auto-refresh 10 sec

st.title("📊 NSE Trading Dashboard – Pro Edition")
st.caption("Live Market | Screener | Options | AI Research | Intelligence | Heatmap")

# ===========================================
# LOAD SYMBOLS FROM NSE
# ===========================================
@st.cache_data
def load_symbols():
    df = pd.read_csv("https://archives.nseindia.com/content/equities/EQUITY_L.csv")
    return df["SYMBOL"].dropna().sort_values().tolist()

all_symbols = load_symbols()

# ===========================================
# Technical Indicators
# ===========================================
def rsi(price: pd.Series, period: int = 14):
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(close):
    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    m = e12 - e26
    s = m.ewm(span=9, adjust=False).mean()
    return m, s

def supertrend(df, period=10, multiplier=3):
    hl2 = (df["High"] + df["Low"]) / 2
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    trend = pd.Series(index=df.index)
    direction = pd.Series(index=df.index)

    for i in range(len(df)):
        if i == 0:
            trend.iloc[i] = upper.iloc[i]
            direction.iloc[i] = 1
        else:
            if df["Close"].iloc[i] > trend.iloc[i-1]:
                direction.iloc[i] = 1
                trend.iloc[i] = lower.iloc[i]
            else:
                direction.iloc[i] = -1
                trend.iloc[i] = upper.iloc[i]

    return trend, direction

# ===========================================
# GET NSE PRICE HISTORY
# ===========================================
def get_history(symbol, days_back=220):
    try:
        end = date.today()
        start = end - timedelta(days=days_back)
        url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}&series=[%22EQ%22]&from={start.strftime('%d-%m-%Y')}&to={end.strftime('%d-%m-%Y')}"
        payload = nsefetch(url)
        df = pd.DataFrame(payload["data"])
        if df.empty:
            return None

        df["Date"] = pd.to_datetime(df["CH_TIMESTAMP"])
        df.sort_values("Date", inplace=True)
        df["Open"] = df["CH_OPENING_PRICE"].astype(float)
        df["High"] = df["CH_TRADE_HIGH_PRICE"].astype(float)
        df["Low"] = df["CH_TRADE_LOW_PRICE"].astype(float)
        df["Close"] = df["CH_CLOSING_PRICE"].astype(float)
        df["Volume"] = df["CH_TOT_TRADED_QTY"].astype(float)

        df["EMA9"] = df["Close"].ewm(span=9).mean()
        df["EMA21"] = df["Close"].ewm(span=21).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()
        df["VOL20"] = df["Volume"].rolling(20).mean()
        df["HIGH20"] = df["High"].rolling(20).max()
        df["RSI"] = rsi(df["Close"])
        df["MACD"], df["MACD_SIGNAL"] = macd(df["Close"])
        df["SUPERTREND"], df["ST_DIR"] = supertrend(df)

        return df
    except Exception:
        return None

# ===========================================
# FUNDAMENTALS (NSE ONLY)
# ===========================================
def get_nse_fundamentals(symbol):
    try:
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        headers = {"User-Agent": "Mozilla/5.0"}
        session = requests.Session()
        resp = session.get(url, headers=headers, timeout=8)
        data = resp.json()
        sh = data.get("shareholding", {}).get("summary", [])

        return {
            "MarketCap (Cr)": data.get("marketDeptOrderBook", {}).get("marketCap"),
            "PE": data.get("metadata", {}).get("pe"),
            "PB": data.get("metadata", {}).get("pb"),
            "ROE": data.get("financials", {}).get("roe"),
            "ROCE": data.get("financials", {}).get("roce"),
            "Debt/Equity": data.get("financials", {}).get("debtEquityRatio"),
            "Promoter%": next((x["percent"] for x in sh if x["name"]=="Promoter & Promoter Group"), None),
            "FII%": next((x["percent"] for x in sh if x["name"]=="Foreign Institutions"), None),
            "DII%": next((x["percent"] for x in sh if x["name"]=="Domestic Institutions"), None),
            "Public%": next((x["percent"] for x in sh if x["name"]=="Public"), None),
        }
    except:
        return {}

def fundamentals_score_nse(f):
    if not f:
        return 50, "No fundamentals available"
    score = 50
    if f["PE"] and f["PE"] < 20: score += 10
    if f["ROE"] and f["ROE"] > 15: score += 10
    if f["ROCE"] and f["ROCE"] > 15: score += 10
    if f["Debt/Equity"] and f["Debt/Equity"] < 0.5: score += 10
    if f["Debt/Equity"] and f["Debt/Equity"] > 1.5: score -= 10
    return min(max(score,0),100), "Calculated from NSE financial metrics"

# ===========================================
# SUPPORT / RESISTANCE + AI REPORT
# ===========================================
def detect_support_resistance(df):
    recent = df.tail(40)
    return recent["Low"].min(), recent["High"].max()

def generate_ai_report(symbol, tech_score, fund_score, news_score, df, f):
    support, resistance = detect_support_resistance(df)
    last = df.iloc[-1]
    roe = f.get("ROE")
    promoter = f.get("Promoter%")

    rsi_comment = "overbought" if last["RSI"]>70 else "oversold" if last["RSI"]<30 else "balanced"
    macd_comment = "bullish crossover building" if last["MACD"]>last["MACD_SIGNAL"] else "bearish crossover dominating"
    trend_comment = "uptrend continuation" if last["EMA9"]>last["EMA21"]>last["EMA50"] else "trend weakening" if last["EMA21"]>last["EMA50"] else "downtrend pressure"

    text = f"""
{symbol} is currently trading near ₹{last['Close']:.2f} with {trend_comment} characteristics driven by EMA alignment and price structure analysis.
RSI at {last['RSI']:.1f} indicates {rsi_comment}, while MACD reflects {macd_comment}, signalling momentum direction.
Key structural support sits near ₹{support:.0f}, while resistance overhead lies around ₹{resistance:.0f}, marking an important breakout zone.
{"Promoter holding of " + str(promoter) + "% strengthens long-term management confidence." if promoter else ""}
{"Strong ROE of " + str(round(roe,2)) + "% supports financial resilience." if roe else ""}
Technical score at {tech_score}/100 and fundamental score at {fund_score}/100 combine with sentiment score {news_score}/100 to form the conviction outlook.
Based on the confluence zone between ₹{support:.0f} and ₹{resistance:.0f}, traders may observe price reaction closely for directional confirmation.
"""
    return text
# ===========================================
# TABS LAYOUT
# ===========================================
(
    tab_watch,
    tab_chart,
    tab_options,
    tab_scanner,
    tab_backtest,
    tab_calc,
    tab_intel,
    tab_heatmap,
) = st.tabs([
    "📈 Watchlist",
    "📊 Chart",
    "📉 Option Chain",
    "🚨 NIFTY500 Scanner",
    "📜 Backtest",
    "🧮 Calculator",
    "🧠 Intelligence",
    "🌍 Sentiment Heatmap",
])

# ===========================================
# TAB 1 — WATCHLIST
# ===========================================
with tab_watch:
    st.subheader("📈 Live Watchlist + Strategy Signals")
    default = [s for s in ["RELIANCE", "AXISCADES", "VEDL", "JIOFIN"] if s in all_symbols]
    selected = st.multiselect("Select Stocks", all_symbols, default=default)

    rows = []
    for sym in selected:
        try:
            data = nsefetch(f"https://www.nseindia.com/api/quote-equity?symbol={sym}")
            ltp = data["priceInfo"]["lastPrice"]
            prev = data["priceInfo"]["previousClose"]
            pct = ((ltp-prev)/prev*100) if prev else None
            vol = data.get("preOpenMarket", {}).get("totalTradedVolume", 0)

            hist = get_history(sym)
            if hist is not None and len(hist)>30:
                last = hist.iloc[-1]
                rows.append([
                    sym,
                    ltp,
                    round(pct,2) if pct else None,
                    int(vol),
                    int(last["VOL20"]) if last["VOL20"] else None,
                    round(last["RSI"],1),
                    round(last["EMA9"],1),
                    round(last["EMA21"],1),
                    round(last["EMA50"],1),
                    round(last["MACD"],2),
                    round(last["MACD_SIGNAL"],2),
                    "Up" if last["ST_DIR"]==1 else "Down"
                ])
        except:
            pass

    df_watch = pd.DataFrame(rows, columns=[
        "Symbol", "LTP", "% Chg", "Volume", "20D Vol",
        "RSI", "EMA9", "EMA21", "EMA50", "MACD", "MACD Sig", "Trend"
    ])
    st.dataframe(df_watch, use_container_width=True)

# ===========================================
# TAB 2 — CHART
# ===========================================
with tab_chart:
    st.subheader("📊 Chart – Candles + EMA + RSI + MACD + Supertrend")

    s = st.selectbox("Choose stock", all_symbols, index=all_symbols.index("RELIANCE") if "RELIANCE" in all_symbols else 0)
    df = get_history(s, days_back=250)

    if df is None:
        st.error("No data available")
    else:
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
        ))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA9"], mode="lines", name="EMA9", line=dict(color="green")))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA21"], mode="lines", name="EMA21", line=dict(color="red")))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA50"], mode="lines", name="EMA50", line=dict(color="purple")))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SUPERTREND"], mode="lines", name="Supertrend", line=dict(color="yellow")))

        fig.update_layout(height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("RSI (14)")
            fig_rsi = go.Figure(go.Scatter(x=df["Date"], y=df["RSI"], mode="lines"))
            fig_rsi.add_hline(y=70, line_dash="dot")
            fig_rsi.add_hline(y=30, line_dash="dot")
            fig_rsi.update_layout(height=260)
            st.plotly_chart(fig_rsi, use_container_width=True)

        with col2:
            st.subheader("MACD")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], mode="lines", name="MACD"))
            fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["MACD_SIGNAL"], mode="lines", name="Signal"))
            st.plotly_chart(fig_macd, use_container_width=True)

# ===========================================
# TAB 3 — OPTION CHAIN
# ===========================================
with tab_options:
    st.subheader("📉 Options Chain – OI, OI Change, PCR, Max Pain")

    oc_symbol = st.selectbox("Select F&O", ["NIFTY", "BANKNIFTY"] + all_symbols)

    try:
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={oc_symbol}" if oc_symbol in ["NIFTY","BANKNIFTY"] else f"https://www.nseindia.com/api/option-chain-equities?symbol={oc_symbol}"
        payload = nsefetch(url)
        chain = payload["records"]["data"]
        rows_ce = []
        rows_pe = []

        for r in chain:
            strike = r.get("strikePrice")
            ce = r.get("CE")
            pe = r.get("PE")

            if ce:
                rows_ce.append([
                    strike,
                    ce.get("lastPrice",0),
                    ce.get("openInterest",0),
                    ce.get("changeinOpenInterest",0),
                    ce.get("totalTradedVolume",0),
                ])

            if pe:
                rows_pe.append([
                    strike,
                    pe.get("lastPrice",0),
                    pe.get("openInterest",0),
                    pe.get("changeinOpenInterest",0),
                    pe.get("totalTradedVolume",0),
                ])

        df_ce = pd.DataFrame(rows_ce, columns=["Strike", "CE_LTP", "CE_OI", "CE_OI_Chg", "CE_Vol"])
        df_pe = pd.DataFrame(rows_pe, columns=["Strike", "PE_LTP", "PE_OI", "PE_OI_Chg", "PE_Vol"])
        oc = df_ce.merge(df_pe, on="Strike", how="outer").fillna(0)

        pcr = oc["PE_OI"].sum() / oc["CE_OI"].sum()
        oc["TotalOI"] = oc["CE_OI"] + oc["PE_OI"]
        maxpain = oc.loc[oc["TotalOI"].idxmax(), "Strike"]

        c1, c2 = st.columns(2)
        c1.metric("PCR", f"{pcr:.2f}")
        c2.metric("Max Pain", maxpain)

        st.dataframe(oc, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading option chain: {e}")

# ===========================================
# TAB 4 — NIFTY500 SCANNER
# ===========================================
with tab_scanner:
    st.subheader("🚨 NIFTY500 Breakout & Volume Scanner")
    if st.button("Scan ▶"):
        try:
            payload = nsefetch("https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500")
            symbols = sorted(x["symbol"] for x in payload["data"])
        except:
            st.error("Could not load Nifty500 list")
            symbols = []

        results = []

        with st.spinner("Scanning market..."):
            for sym in symbols[:120]:
                df = get_history(sym)
                if df is None or len(df)<30:
                    continue
                last = df.iloc[-1]

                cond = (
                    last["EMA9"] > last["EMA21"] > last["EMA50"] or
                    (last["Volume"] > 1.5 * last["VOL20"]) or
                    (last["Close"] > last["HIGH20"])
                )

                if cond:
                    results.append([
                        sym,
                        last["Close"],
                        last["Volume"],
                        round(last["RSI"],1),
                        "Yes" if last["Volume"] > 1.5 * last["VOL20"] else "",
                        "Breakout" if last["Close"] > last["HIGH20"] else "",
                    ])

        df_scan = pd.DataFrame(results, columns=["Symbol","Close","Volume","RSI","Vol Spike","Breakout"])
        st.dataframe(df_scan, use_container_width=True)

# ===========================================
# TAB 5 — BACKTEST
# ===========================================
with tab_backtest:
    st.subheader("📜 Strategy Backtest (Daily)")
    symbol = st.selectbox("Select stock to backtest", all_symbols)
    days = st.slider("History (days)", 150, 800, 400)

    if st.button("Run Backtest"):
        df = get_history(symbol, days_back=days)
        if df is None:
            st.error("No data")
        else:
            capital = 100000
            cash = capital
            qty = 0
            entry = 0
            trades = []

            for i in range(30, len(df)):
                row = df.iloc[i]
                buy = row["EMA9"] > row["EMA21"] > row["EMA50"] and row["RSI"]>55
                sell = row["EMA9"] < row["EMA21"] or row["RSI"]<40

                if buy and qty==0:
                    qty = int(cash//row["Close"])
                    entry = row["Close"]
                    cash -= qty * entry
                    trades.append(["BUY", row["Date"], entry])

                if sell and qty>0:
                    cash += qty * row["Close"]
                    trades.append(["SELL", row["Date"], row["Close"]])
                    qty = 0

            final_value = cash + qty * df.iloc[-1]["Close"]
            st.metric("Final Equity", f"₹{final_value:,.0f}")

            st.dataframe(pd.DataFrame(trades, columns=["Type","Date","Price"]), use_container_width=True)

# ===========================================
# TAB 6 — CALCULATOR
# ===========================================
with tab_calc:
    st.subheader("🧮 Investment Calculator")

    colL, colS = st.columns(2)

    # Lumpsum
    with colL:
        s2 = st.selectbox("Stock", all_symbols, key="lump")
        amount = st.number_input("Investment (₹)", value=100000)
        years = st.slider("Years", 1, 10, 5)

        if st.button("Calculate Lumpsum"):
            df = get_history(s2, days_back=years*365)
            if df is None:
                st.error("No data")
            else:
                start = df.iloc[0]["Close"]
                now = df.iloc[-1]["Close"]
                qty = amount/start
                value = qty*now
                st.metric("Start Price", f"₹{start:.2f}")
                st.metric("Current Price", f"₹{now:.2f}")
                st.metric("Current Value", f"₹{value:,.0f}")
# ===========================================
# NEWS SENTIMENT ANALYSIS
# ===========================================
POS_WORDS = ["gain","surge","rally","up","profit","bull","beat","record","strong","growth","buy","upgrade"]
NEG_WORDS = ["fall","down","loss","fraud","bear","scam","weak","cut","sell","crash","default","probe"]

def get_latest_news(symbol, n=6):
    try:
        url = (
            "https://newsapi.org/v2/everything?"
            f"q={symbol}&language=en&sortBy=publishedAt&pageSize={n}&apiKey=4e1e23b0eaa54e48af4d228dfbf85fc0"
        )
        r = requests.get(url, timeout=10)
        data = r.json()
        return data.get("articles", [])
    except:
        return []

def sentiment_score(text):
    score = 0
    t = text.lower()
    for w in POS_WORDS:
        if w in t: score += 1
    for w in NEG_WORDS:
        if w in t: score -= 1
    return score

def aggregate_sentiment(articles):
    if not articles:
        return 50, []
    total = 0
    scored = []
    for a in articles:
        score = sentiment_score(a["title"] + " " + str(a.get("description","")))
        total += score
        scored.append({**a, "sentiment": score})
    avg = total/len(articles)
    converted = int((avg+3)/6*100)  # normalize -3..+3 => 0..100
    return converted, scored

# ===========================================
# TAB 7 — STOCK INTELLIGENCE
# ===========================================
with tab_intel:
    st.subheader("🧠 Stock Intelligence – AI Research Report")

    intel_symbol = st.selectbox("Select Stock", all_symbols)

    df = get_history(intel_symbol)
    if df is None or len(df)<60:
        st.warning("Not enough price data available")
    else:
        last = df.iloc[-1]
        tech_score = 50
        if last["EMA9"]>last["EMA21"]>last["EMA50"]: tech_score += 20
        if last["MACD"]>last["MACD_SIGNAL"]: tech_score += 10
        if last["RSI"]>60: tech_score += 10
        if last["Volume"] > 1.5 * last["VOL20"]: tech_score += 10

        fundamentals = get_nse_fundamentals(intel_symbol)
        fund_score, _ = fundamentals_score_nse(fundamentals)

        with st.spinner("Fetching news sentiment..."):
            articles = get_latest_news(intel_symbol, 8)
            news_score, news_list = aggregate_sentiment(articles)

        combined = int(0.5*tech_score + 0.3*news_score + 0.2*fund_score)

        colA, colB, colC = st.columns(3)
        colA.metric("Technical", f"{tech_score}/100")
        colB.metric("Fundamental", f"{fund_score}/100")
        colC.metric("Sentiment", f"{news_score}/100")

        st.subheader(f"✨ Final Score: {combined}/100")

        st.markdown("### 📄 AI Research Report")
        report = generate_ai_report(intel_symbol, tech_score, fund_score, news_score, df, fundamentals)
        st.write(report)

        st.markdown("### 📊 Fundamentals")
        st.json(fundamentals)

        st.markdown("### 📰 News Headlines")
        if news_list:
            df_news = pd.DataFrame(news_list)[["publishedAt","source","title","sentiment","url"]]
            st.dataframe(df_news, use_container_width=True)
        else:
            st.info("No recent news available")

# ===========================================
# TAB 8 — SENTIMENT HEATMAP
# ===========================================
with tab_heatmap:
    st.subheader("🌍 Market Sentiment – NIFTY500 Heatmap")

    if st.button("Run Sentiment Scan ▶"):
        try:
            payload = nsefetch("https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500")
            symbols = sorted(x["symbol"] for x in payload["data"])
        except:
            st.error("Unable to load NIFTY500")
            symbols = []

        results = []
        with st.spinner("Scanning news sentiment across market..."):
            for sym in symbols[:80]:
                news = get_latest_news(sym, 4)
                score, _ = aggregate_sentiment(news)
                results.append([sym, score])

        df_heat = pd.DataFrame(results, columns=["Symbol","Sentiment"]).sort_values("Sentiment", ascending=False)

        st.markdown("### 🥇 Top 10 Bullish")
        st.dataframe(df_heat.head(10), use_container_width=True)

        st.markdown("### 🔻 Top 10 Bearish")
        st.dataframe(df_heat.tail(10), use_container_width=True)

        st.markdown("### 🟦 Heatmap")
        fig = go.Figure(data=go.Heatmap(
            z=df_heat["Sentiment"],
            x=df_heat["Symbol"],
            y=["Sentiment"]*len(df_heat),
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

# ===========================================
# END
# ===========================================
