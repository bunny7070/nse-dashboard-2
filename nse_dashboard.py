import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Option Chain Pro Terminal",
                   layout="wide",
                   page_icon="🧠")

st.title("🧠 Option Chain Pro Terminal – Advanced v2.0")
st.write("NIFTY • BANKNIFTY • F&O Stocks • Greeks • Max Pain • Signal Engine")

symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"])
refresh = st.slider("Auto Refresh (Seconds)", 10, 300, 30)

@st.cache_data(ttl=refresh)
def fetch_oc(symbol):
    headers = {
        "user-agent": "Mozilla/5.0",
        "accept": "application/json",
    }

    if symbol in ["NIFTY", "BANKNIFTY"]:
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    else:
        url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

    s = requests.Session()
    s.get("https://www.nseindia.com", headers=headers)
    r = s.get(url, headers=headers).json()

    exp_list = list(set([i["expiryDate"] for i in r["records"]["data"] if "CE" in i or "PE" in i]))
    exp_list.sort()
    return r, exp_list

data, expiries = fetch_oc(symbol)
expiry = st.selectbox("Select Expiry", expiries)

# filter by expiry
records = []
for entry in data["records"]["data"]:
    if entry["expiryDate"] == expiry:
        strike = entry["strikePrice"]
        if "CE" in entry:
            records.append(["CE", strike,
                            entry["CE"]["openInterest"],
                            entry["CE"]["changeinOpenInterest"],
                            entry["CE"]["lastPrice"],
                            entry["CE"]["impliedVolatility"]])
        if "PE" in entry:
            records.append(["PE", strike,
                            entry["PE"]["openInterest"],
                            entry["PE"]["changeinOpenInterest"],
                            entry["PE"]["lastPrice"],
                            entry["PE"]["impliedVolatility"]])

df = pd.DataFrame(records, columns=["Type", "Strike", "OI", "ChgOI", "LTP", "IV"])

ce = df[df.Type == "CE"].sort_values("Strike")
pe = df[df.Type == "PE"].sort_values("Strike")
merged = ce.merge(pe, on="Strike", suffixes=("_CE", "_PE"))

# PCR / Max Pain / Bias
pcr = round(merged["OI_PE"].sum() / merged["OI_CE"].sum(), 2)
maxpain = merged.iloc[(merged["OI_PE"] - merged["OI_CE"]).abs().argsort()[:1]]["Strike"].values[0]
bias = "📈 Bullish" if pcr > 1.1 else "📉 Bearish" if pcr < 0.9 else "🔁 Neutral"

atm = merged.iloc[(merged["LTP_CE"] - merged["LTP_PE"]).abs().argsort()[:1]]["Strike"].values[0]

sup = merged.nlargest(5, "OI_PE")["Strike"].tolist()
res = merged.nlargest(5, "OI_CE")["Strike"].tolist()

# HEADER CARDS
st.markdown("### 🔥 Market Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("PCR", pcr)
c2.metric("Max Pain", maxpain)
c3.metric("Trend", bias)
c4.metric("ATM Strike", atm)

# Support / Resistance
st.markdown("### 🧱 Support & Resistance Levels")
s1, s2 = st.columns(2)
s1.success(f"Supports (PE OI) → {sup}")
s2.error(f"Resistances (CE OI) → {res}")

# OPTION CHAIN TABLE
st.markdown("### 📊 Option Chain Table (Heatmap)")
st.dataframe(merged.style.background_gradient(subset=["OI_CE"], cmap="Reds")
                          .background_gradient(subset=["OI_PE"], cmap="Greens").highlight_max(subset=["OI_CE","OI_PE"]))

# CHART
fig = go.Figure()
fig.add_trace(go.Bar(x=merged["Strike"], y=merged["OI_CE"], name="CE OI"))
fig.add_trace(go.Bar(x=merged["Strike"], y=merged["OI_PE"], name="PE OI"))
fig.update_layout(title="Open Interest by Strike",
                  barmode='group', height=400)
st.plotly_chart(fig, use_container_width=True)

# Strategy Suggestions
st.markdown("### 🧠 Strategy Engine Suggestions")
suggestion = ""

if bias == "📈 Bullish":
    suggestion = f"Try Bull Put Spread / PE Selling near supports: {sup[:2]}"
elif bias == "📉 Bearish":
    suggestion = f"Try Bear Call Spread / CE Selling near resistances: {res[:2]}"
else:
    suggestion = f"Try Short Straddle or Short Strangle around ATM {atm}"

st.warning(suggestion)

st.markdown("### 📕 Beginner Learning Guide")
st.info("""
**OI** → Strong level indicator  
**Chg OI** → Where fresh positions being built  
**PCR** > 1.1 Bullish, < 0.9 Bearish  
**Max Pain** → Expiry magnet  
**Support/Resistance** from high OI zones  
""")
