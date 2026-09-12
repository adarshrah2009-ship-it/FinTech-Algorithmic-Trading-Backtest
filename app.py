import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pro Quant Terminal",
    layout="wide",
    page_icon="📈"
)

st.title("Institutional Quant & AI Research Terminal")

POPULAR_ASSETS = [
    "SUZLON.NS", "RELIANCE.NS", "TATAMOTORS.NS", "TCS.NS", "INFY.NS", 
    "ADANIENT.NS", "BTC-USD", "ETH-USD", "GC=F", "AAPL", "NVDA", "TSLA"
]

# Sidebar asset picker
selected_asset = st.sidebar.selectbox(
    "Select Asset Ticker", 
    options=POPULAR_ASSETS,
    accept_new_options=True
).upper().strip()

selected_period = st.sidebar.selectbox("Data Horizon", ["1y", "2y", "5y"], index=1)

# Dynamic data loader
@st.cache_data(ttl=300)
def fetch_data(symbol, period):
    data = yf.download(symbol, period=period, progress=False)
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        try:
            if symbol in data.columns.get_level_values(1):
                data = data.xs(symbol, axis=1, level=1)
            else:
                data.columns = [col[0] for col in data.columns]
        except Exception:
            data.columns = [col[0] for col in data.columns]
    return data

def calculate_indicators(df):
    data = df.copy()
    data["50_DMA"] = data["Close"].rolling(50).mean()
    data["200_DMA"] = data["Close"].rolling(200).mean()

    # RSI (14)
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # ADX (14)
    high_diff = data["High"].diff()
    low_diff = -data["Low"].diff()
    pos_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    neg_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr1 = data["High"] - data["Low"]
    tr2 = (data["High"] - data["Close"].shift(1)).abs()
    tr3 = (data["Low"] - data["Close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * (pd.Series(pos_dm).rolling(14).mean() / atr)
    minus_di = 100 * (pd.Series(neg_dm).rolling(14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    data["ADX"] = dx.rolling(14).mean()
    data["Volume_MA"] = data["Volume"].rolling(20).mean()

    return data

# Fetch data
df_raw = fetch_data(selected_asset, selected_period)

if df_raw.empty or len(df_raw) < 200:
    st.error(f"Insufficient data for '{selected_asset}'. Try selecting a longer horizon (e.g. 5y).")
    st.stop()

df = calculate_indicators(df_raw)

# Latest numbers
latest = df.iloc[-1]
price = float(latest["Close"])
rsi = float(latest["RSI"])
adx = float(latest["ADX"])
sma50 = float(latest["50_DMA"])
sma200 = float(latest["200_DMA"])
vol_confirm = bool(latest["Volume"] > latest["Volume_MA"])

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Multi-Factor Matrix",
    "⚠️ Chop Filter",
    "🧪 Backtest Engine",
    "🤖 AI Research Agent",
    "🧮 Risk & ROI Calculator"
])

# TAB 1: MATRIX
with tab1:
    st.subheader(f"Technical Scorecard ({selected_asset})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"{price:.2f}")
    c2.metric("RSI (14)", f"{rsi:.1f}")
    c3.metric("ADX (Trend Strength)", f"{adx:.1f}")
    c4.metric("High Volume?", "YES" if vol_confirm else "NO")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color="white")))
    fig.add_trace(go.Scatter(x=df.index, y=df["50_DMA"], name="50 DMA", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df.index, y=df["200_DMA"], name="200 DMA", line=dict(color="red")))
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: CHOP FILTER
with tab2:
    st.subheader("Market Trend vs Sideways Range")
    ma_diff_pct = abs(sma50 - sma200) / price * 100
    if adx < 20 or ma_diff_pct < 1.5:
        st.error("SIDEWAYS / CHOPPY MARKET - High risk of false signals!")
    else:
        st.success("CLEAR TREND DETECTED - Signals carry higher accuracy.")

# TAB 3: BACKTEST
with tab3:
    st.subheader("Backtest Strategy vs Buy & Hold")
    bt_df = df.copy()
    bt_df["Signal"] = 0
    bt_df.loc[(bt_df["50_DMA"] > bt_df["200_DMA"]) & (bt_df["RSI"] > 50) & (bt_df["ADX"] > 20), "Signal"] = 1
    bt_df["Bench_Ret"] = bt_df["Close"].pct_change()
    bt_df["Strat_Ret"] = bt_df["Bench_Ret"] * bt_df["Signal"].shift(1)
    
    clean_bt = bt_df.dropna(subset=["Strat_Ret"]).copy()
    if not clean_bt.empty:
        clean_bt["Cum_Strat"] = (1 + clean_bt["Strat_Ret"]).cumprod()
        clean_bt["Cum_Bench"] = (1 + clean_bt["Bench_Ret"]).cumprod()
        
        b1, b2 = st.columns(2)
        b1.metric("Strategy Return", f"{(clean_bt['Cum_Strat'].iloc[-1] - 1)*100:.2f}%")
        b2.metric("Buy & Hold Return", f"{(clean_bt['Cum_Bench'].iloc[-1] - 1)*100:.2f}%")

# TAB 4: AI RESEARCH AGENT
with tab4:
    st.subheader("Ask the AI Analyst (Free via Google Gemini)")
    st.write("Analyze whether you should **BUY**, **HOLD**, or **CASH OUT** using Google's free API.")

    user_api_key = st.text_input("Paste your Google Gemini API Key (starts with AIzaSy...):", type="password")

    if st.button("Run Free AI Decision Engine"):
        cleaned_key = user_api_key.strip()
        if not cleaned_key:
            st.warning("Please paste your Google Gemini API Key above to run the AI for free!")
        else:
            with st.spinner(f"AI is analyzing market signals for {selected_asset}..."):
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={cleaned_key}"
                    headers = {"Content-Type": "application/json"}
                    
                    prompt_text = f"""
                    You are an expert Quantitative Investment Advisor.
                    Analyze this stock: {selected_asset}
                    - Price: {price:.2f}
                    - 50-DMA: {sma50:.2f}
                    - 200-DMA: {sma200:.2f}
                    - RSI: {rsi:.1f}
                    - ADX Trend Strength: {adx:.1f}
                    - Technical Regime: {"Bullish" if sma50 > sma200 else "Bearish"}

                    Task: Tell the user if they should BUY, CASH_OUT, or HOLD. Give a confidence score from 0 to 100 and a short reasoning.

                    Return ONLY a JSON object:
                    {{
                        "action": "BUY or CASH_OUT or HOLD",
                        "confidence": 85,
                        "reasoning": "Your short explanation here."
                    }}
                    """

                    payload = {
                        "contents": [{
                            "parts": [{"text": prompt_text}]
                        }],
                        "generationConfig": {
                            "response_mime_type": "application/json"
                        }
                    }

                    response = requests.post(url, headers=headers, json=payload, timeout=30)
                    res_data = response.json()

                    if response.status_code != 200:
                        st.error(f"API Error ({response.status_code}): {res_data.get('error', {}).get('message', 'Unknown Error')}")
                    else:
                        raw_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                        result = json.loads(raw_text)

                        act = str(result.get("action", "HOLD")).upper()
                        conf = result.get("confidence", 0)
                        reason = result.get("reasoning", "")

                        st.markdown("---")
                        col_a, col_b = st.columns(2)
                        if "BUY" in act:
                            col_a.success("### Signal: 🟢 **BUY**")
                        elif "CASH" in act or "SELL" in act:
                            col_a.error("### Signal: 🔴 **CASH OUT**")
                        else:
                            col_a.warning("### Signal: 🟡 **HOLD**")

                        col_b.metric("AI Confidence", f"{conf}%")
                        st.info(f"**AI Rationale:**\n{reason}")

                except Exception as e:
                    st.error(f"Error calling Gemini API: {e}")

# TAB 5: RISK & ROI CALCULATOR
with tab5:
    st.subheader(f"Position Sizing & Return Calculator ({selected_asset})")
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        account_balance = st.number_input("Total Portfolio / Trading Capital ($ / ₹)", min_value=100.0, value=10000.0, step=500.0)
        risk_per_trade_pct = st.number_input("Risk Per Trade (%)", min_value=0.1, max_value=100.0, value=2.0, step=0.5)
        entry_price = st.number_input("Planned Entry Price", min_value=0.01, value=price, step=1.0)
        
    with col_input2:
        stop_loss_price = st.number_input("Stop Loss Price", min_value=0.01, value=round(price * 0.95, 2), step=1.0)
        take_profit_price = st.number_input("Target Take-Profit Price", min_value=0.01, value=round(price * 1.15, 2), step=1.0)

    # Calculation logic
    risk_per_share = abs(entry_price - stop_loss_price)
    reward_per_share = abs(take_profit_price - entry_price)
    
    max_capital_risk = account_balance * (risk_per_trade_pct / 100.0)
    
    if risk_per_share > 0:
        position_size_units = max_capital_risk / risk_per_share
        total_position_value = position_size_units * entry_price
        
        potential_profit = position_size_units * reward_per_share
        roi_on_capital = (potential_profit / account_balance) * 100.0
        roi_on_position = (reward_per_share / entry_price) * 100.0
        risk_reward_ratio = reward_per_share / risk_per_share
        
        st.markdown("---")
        st.markdown("### 📊 Trade Metrics Breakdown")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recommended Position Size", f"{position_size_units:.2f} Units")
        m2.metric("Total Investment Value", f"{total_position_value:.2f}")
        m3.metric("Max Dollar/Rupee Risk", f"{max_capital_risk:.2f}")
        m4.metric("Risk-to-Reward Ratio", f"1 : {risk_reward_ratio:.2f}")
        
        m5, m6, m7 = st.columns(3)
        m5.metric("Potential Profit", f"{potential_profit:.2f}", delta=f"{roi_on_position:.2f}% (Price)")
        m6.metric("Expected Portfolio ROI", f"{roi_on_capital:.2f}%")
        
        if risk_reward_ratio >= 2.0:
            m7.success("Excellent Risk-Reward Ratio (>= 1:2)")
        elif risk_reward_ratio >= 1.0:
            m7.warning("Moderate Risk-Reward Ratio (1:1 - 1:2)")
        else:
            m7.error("Poor Risk-Reward (Risk exceeds reward!)")
    else:
        st.error("Stop Loss Price cannot be equal to the Entry Price.")
