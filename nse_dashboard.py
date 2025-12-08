import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

st.set_page_config(
    page_title="Option Chain Pro Terminal",
    layout="wide",
    page_icon="🧠"
)

st.title("🧠 Option Chain Pro Terminal – Advanced")
st.write("NIFTY • BANKNIFTY • F&O Stocks • PCR • Max Pain • Support & Resistance • Signals")

symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"])
refresh = st.slider("Auto Refresh (Seconds)", 10, 300, 30)


@st.cache_data(ttl=60)
def fetch_oc(symbol: str):
    """
    Safely fetch option chain from NSE.
    Returns: (records_list, expiry_list, error_message)
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Connection": "keep-alive",
        "Referer": "https://www.nseindia.com/",
    }

    if symbol in ["NIFTY", "BANKNIFTY"]:
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    else:
        url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

    try:
        session = requests.Session()
        # Hit homepage once to get cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=5)

        resp = session.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return [], [], f"NSE returned HTTP {resp.status_code}. Try again later or run locally."

        try:
            data = resp.json()
        except ValueError:
            return [], [], "NSE did not return JSON (maybe blocked / HTML page)."

        records = data.get("records", {}).get("data", [])
        if not isinstance(records, list) or len(records) == 0:
            return [], [], "No option chain records found in NSE response."

        exp_set = set()
        for item in records:
            exp = item.get("expiryDate")
            if exp:
                exp_set.add(exp)
        expiry_list = sorted(exp_set)

        if not expiry_list:
            return [], [], "No expiry dates found in response."

        return records, expiry_list, ""

    except Exception as e:
        return [], [], f"Error while fetching data: {e}"


# --------- FETCH DATA ----------
records, expiries, error_msg = fetch_oc(symbol)

if error_msg:
    st.error(error_msg)
    st.stop()

if not records or not expiries:
    st.error("No data available. Please try again later.")
    st.stop()

expiry = st.selectbox("Select Expiry", expiries)

# --------- BUILD DATAFRAME ----------
rows = []
for entry in records:
    if entry.get("expiryDate") != expiry:
        continue

    strike = entry.get("strikePrice")
    ce = entry.get("CE")
    pe = entry.get("PE")

    if ce:
        rows.append([
            "CE",
            strike,
            ce.get("openInterest", 0),
            ce.get("changeinOpenInterest", 0),
            ce.get("lastPrice", 0.0),
            ce.get("impliedVolatility", 0.0),
        ])
    if pe:
        rows.append([
            "PE",
            strike,
            pe.get("openInterest", 0),
            pe.get("changeinOpenInterest", 0),
            pe.get("lastPrice", 0.0),
            pe.get("impliedVolatility", 0.0),
        ])

if not rows:
    st.warning("No option chain rows found for this expiry.")
    st.stop()

df = pd.DataFrame(rows, columns=["Type", "Strike", "OI", "ChgOI", "LTP", "IV"])

ce = df[df["Type"] == "CE"].sort_values("Strike")
pe = df[df["Type"] == "PE"].sort_values("Strike")

# make sure we can merge
if ce.empty or pe.empty:
    st.warning("Either CE or PE data is empty for this expiry.")
    st.dataframe(df)
    st.stop()

merged = ce.merge(pe, on="Strike", suffixes=("_CE", "_PE"))

# --------- METRICS (with safety) ----------
total_ce_oi = merged["OI_CE"].sum()
total_pe_oi = merged["OI_PE"].sum()

if total_ce_oi == 0 or total_pe_oi == 0:
    pcr = 0
else:
    pcr = round(total_pe_oi / total_ce_oi, 2)

try:
    maxpain = merged.iloc[(merged["OI_PE"] - merged["OI_CE"]).abs().argsort()[:1]]["Strike"].values[0]
except Exception:
    maxpain = None

if pcr > 1.1:
    bias = "📈 Bullish"
elif pcr < 0.9:
    bias = "📉 Bearish"
else:
    bias = "🔁 Neutral"

# approximate ATM = strike closest to middle of strikes
try:
    atm = merged["Strike"].iloc[merged["Strike"].sub(merged["Strike"].median()).abs().idxmin()]
except Exception:
    atm = None

# --------- HEADER CARDS ----------
st.markdown("### 🔥 Market Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("PCR", pcr)
c2.metric("Max Pain", maxpain if maxpain is not None else "-")
c3.metric("Trend", bias)
c4.metric("Total OI (CE / PE)", f"{int(total_ce_oi):,} / {int(total_pe_oi):,}")

# --------- SUPPORT / RESISTANCE ----------
st.markdown("### 🧱 Support & Resistance Zones (by OI)")
s1, s2 = st.columns(2)
supports = merged.nlargest(5, "OI_PE")["Strike"].tolist()
resists = merged.nlargest(5, "OI_CE")["Strike"].tolist()
s1.success(f"🟢 Supports (PE OI): {supports}")
s2.error(f"🔴 Resistances (CE OI): {resists}")

# --------- OPTION CHAIN TABLE ----------
st.markdown("### 📊 Option Chain Table (Heatmap)")
styled = (
    merged.style
    .background_gradient(subset=["OI_CE"], cmap="Reds")
    .background_gradient(subset=["OI_PE"], cmap="Greens")
)
st.dataframe(styled, use_container_width=True)

# --------- OI CHART ----------
st.markdown("### 📈 Open Interest by Strike")
fig = go.Figure()
fig.add_trace(go.Bar(x=merged["Strike"], y=merged["OI_CE"], name="CE OI"))
fig.add_trace(go.Bar(x=merged["Strike"], y=merged["OI_PE"], name="PE OI"))
fig.update_layout(
    title="Open Interest vs Strike",
    xaxis_title="Strike",
    yaxis_title="Open Interest",
    barmode="group",
    height=450,
)
st.plotly_chart(fig, use_container_width=True)

# --------- SIGNAL / INTERPRETATION ----------
st.markdown("### 🧠 Signal Interpretation (Beginner Friendly)")

if bias.startswith("📈"):
    msg = f"Market bias: **Bullish**.\n\nStrong put writing near supports {supports[:2]} suggests demand zones. Dips towards these strikes may find buying interest."
elif bias.startswith("📉"):
    msg = f"Market bias: **Bearish**.\n\nStrong call writing near resistances {resists[:2]} suggests supply zones. Rallies towards these strikes may face selling pressure."
else:
    msg = f"Market bias: **Sideways/Neutral**.\n\nBest to trade range between supports {supports[:2]} and resistances {resists[:2]}. Option selling strategies (strangle/straddle) may work near ATM, if risk-managed."
st.info(msg)

st.markdown("### 📘 Beginner Guide (What These Numbers Mean)")
st.markdown(
"""
| Indicator | What it is | How to read it |
|-----------|------------|----------------|
| **OI (Open Interest)** | Number of open contracts | High OI = strong level (support/resistance) |
| **Chg OI** | Change in OI today | High PE Chg OI → new support building; high CE Chg OI → new resistance building |
| **PCR (Put/Call Ratio)** | PE OI / CE OI | > 1.1 = Bullish bias, < 0.9 = Bearish bias, 0.9–1.1 = Range / neutral |
| **Max Pain** | Strike where option sellers lose least | Price often gravitates here near expiry (not guaranteed) |
| **Supports (PE OI)** | Strikes with highest PE OI | Price often bounces from these levels |
| **Resistances (CE OI)** | Strikes with highest CE OI | Price often reverses or slows near these levels |
"""
)
