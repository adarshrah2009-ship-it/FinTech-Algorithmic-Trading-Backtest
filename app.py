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

selected_asset = st.sidebar.selectbox(
    "Select Asset Ticker", 
    options=POPULAR_ASSETS,
    accept_new_options=True
).upper().strip()

selected_period = st.sidebar.selectbox("Data Horizon", ["1y", "2y", "5y"], index=1)

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

    # RSI
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # ADX
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

df_raw = fetch_data(selected_asset, selected_period)

if df_raw.empty or len(df_raw) < 200:
    st.error(f"Insufficient data for '{selected_asset}'. Try selecting a longer horizon (e.g. 5y).")
    st.stop()

df = calculate_indicators(df_raw)

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Multi-Factor Matrix",
    "⚠️ Chop Filter",
    "🧪 Backtest Engine",
    "🤖 AI Technical Agent",
    "🧮 Risk & ROI Calculator",
    "🌐 Macro & News Intelligence",
    "🎲 Monte Carlo Simulation"
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

# TAB 4: AI TECHNICAL AGENT
with tab4:
    st.subheader("Ask the AI Technical Analyst")
    user_api_key = st.text_input("Paste your Google Gemini API Key:", type="password", key="tech_key")

    if st.button("Run AI Technical Decision Engine"):
        cleaned_key = user_api_key.strip()
        if not cleaned_key:
            st.warning("Please paste your Google Gemini API Key above.")
        else:
            with st.spinner(f"AI is analyzing technicals for {selected_asset}..."):
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

                    Task: Tell the user if they should BUY, CASH_OUT, or HOLD. Return JSON with action, confidence, reasoning.
                    """

                    payload = {
                        "contents": [{"parts": [{"text": prompt_text}]}],
                        "generationConfig": {"response_mime_type": "application/json"}
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
        account_balance = st.number_input("Total Portfolio / Trading Capital", min_value=100.0, value=10000.0, step=500.0)
        risk_per_trade_pct = st.number_input("Risk Per Trade (%)", min_value=0.1, max_value=100.0, value=2.0, step=0.5)
        entry_price = st.number_input("Planned Entry Price", min_value=0.01, value=price, step=1.0)
    with col_input2:
        stop_loss_price = st.number_input("Stop Loss Price", min_value=0.01, value=round(price * 0.95, 2), step=1.0)
        take_profit_price = st.number_input("Target Take-Profit Price", min_value=0.01, value=round(price * 1.15, 2), step=1.0)

    risk_per_share = abs(entry_price - stop_loss_price)
    reward_per_share = abs(take_profit_price - entry_price)
    max_capital_risk = account_balance * (risk_per_trade_pct / 100.0)
    
    if risk_per_share > 0:
        position_size_units = max_capital_risk / risk_per_share
        total_position_value = position_size_units * entry_price
        potential_profit = position_size_units * reward_per_share
        roi_on_capital = (potential_profit / account_balance) * 100.0
        risk_reward_ratio = reward_per_share / risk_per_share
        
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recommended Position Size", f"{position_size_units:.2f} Units")
        m2.metric("Total Investment Value", f"{total_position_value:.2f}")
        m3.metric("Max Capital Risk", f"{max_capital_risk:.2f}")
        m4.metric("Risk-to-Reward", f"1 : {risk_reward_ratio:.2f}")

# TAB 6: MACRO & NEWS INTELLIGENCE
with tab6:
    st.subheader(f"🌐 Real-Time Macro & Policy Scanner ({selected_asset})")
    macro_api_key = st.text_input("Paste your Google Gemini API Key:", type="password", key="macro_key")
    custom_query = st.text_input("Custom Policy Topic:", placeholder="e.g. Rate cuts, regulatory policies, earnings")

    if st.button("Run Live Macro Search"):
        cleaned_macro_key = macro_api_key.strip()
        if not cleaned_macro_key:
            st.warning("Please paste your Google Gemini API Key above.")
        else:
            with st.spinner("AI is scanning news & macro policy..."):
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={cleaned_macro_key}"
                    headers = {"Content-Type": "application/json"}
                    user_topic = custom_query if custom_query.strip() else "latest news earnings macro policies"
                    macro_prompt = f"Audit '{selected_asset}' focusing on {user_topic}. Provide news, policy impact, and macro verdict."

                    payload = {
                        "contents": [{"parts": [{"text": macro_prompt}]}],
                        "tools": [{"google_search": {}}]
                    }
                    response = requests.post(url, headers=headers, json=payload, timeout=45)
                    res_data = response.json()

                    if response.status_code == 429:
                        st.warning("⚠️ Live Search rate limit reached. Falling back to Gemini knowledge base...")
                        payload_fallback = {"contents": [{"parts": [{"text": macro_prompt}]}]}
                        response = requests.post(url, headers=headers, json=payload_fallback, timeout=30)
                        res_data = response.json()

                    if response.status_code != 200:
                        st.error(f"API Error ({response.status_code}): {res_data.get('error', {}).get('message', 'Unknown Error')}")
                    else:
                        candidate = res_data["candidates"][0]
                        parts = candidate["content"]["parts"]
                        full_analysis = "".join([p["text"] for p in parts if "text" in p])
                        st.markdown("---")
                        st.markdown(full_analysis)
                except Exception as e:
                    st.error(f"Error: {e}")

# Replace Tab 7 in your app.py with this Hybrid AI + Monte Carlo implementation:

with tab7:
    st.subheader(f"🎲 AI-Assisted Monte Carlo Engine ({selected_asset})")
    st.write("Combines live Google Search policy audits with stochastic Geometric Brownian Motion to calculate realistic probability distributions.")

    mc_api_key = st.text_input("Paste your Google Gemini API Key:", type="password", key="mc_key")

    mc_col1, mc_col2, mc_col3 = st.columns(3)
    with mc_col1:
        num_simulations = st.slider("Simulations", min_value=100, max_value=2000, value=500, step=100)
    with mc_col2:
        forecast_days = st.slider("Trading Days Horizon", min_value=10, max_value=252, value=60, step=10)
    with mc_col3:
        target_price = st.number_input("Target Price to Evaluate", min_value=0.01, value=round(price * 1.15, 2), step=1.0)

    if st.button("Run AI-Augmented Monte Carlo Simulation"):
        cleaned_mc_key = mc_api_key.strip()
        
        # Default baseline stats from historical data
        log_returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        base_u = float(log_returns.mean())
        base_stdev = float(log_returns.std())
        
        drift_multiplier = 1.0
        vol_multiplier = 1.0
        ai_policy_summary = "Using baseline historical metrics without AI macro adjustment."

        if cleaned_mc_key:
            with st.spinner("AI is evaluating live global news and policy factors to adjust Monte Carlo parameters..."):
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={cleaned_mc_key}"
                    headers = {"Content-Type": "application/json"}
                    
                    mc_prompt = f"""
                    Perform a quick news and policy check for asset: '{selected_asset}'.
                    Determine if near-term news, government policy, earnings, or interest rates present a major TAILWIND (bullish), HEADWIND (bearish), or HIGH UNCERTAINTY.

                    Return ONLY a JSON object:
                    {{
                        "drift_multiplier": 1.2,  // 1.0 = normal, >1.0 for bullish policy, <1.0 for bearish policy
                        "volatility_multiplier": 1.1, // 1.0 = normal, >1.0 if high uncertainty/regulatory risk
                        "reasoning": "Short 2-sentence summary of the news and policy impact."
                    }}
                    """

                    payload = {
                        "contents": [{"parts": [{"text": mc_prompt}]}],
                        "tools": [{"google_search": {}}],
                        "generationConfig": {"response_mime_type": "application/json"}
                    }

                    response = requests.post(url, headers=headers, json=payload, timeout=30)
                    
                    # Fallback if search rate limit (429) triggers
                    if response.status_code == 429:
                        payload_fallback = {
                            "contents": [{"parts": [{"text": mc_prompt}]}],
                            "generationConfig": {"response_mime_type": "application/json"}
                        }
                        response = requests.post(url, headers=headers, json=payload_fallback, timeout=30)

                    if response.status_code == 200:
                        res_data = response.json()
                        raw_json = res_data["candidates"][0]["content"]["parts"][0]["text"]
                        mc_ai_res = json.loads(raw_json)
                        
                        drift_multiplier = float(mc_ai_res.get("drift_multiplier", 1.0))
                        vol_multiplier = float(mc_ai_res.get("volatility_multiplier", 1.0))
                        ai_policy_summary = mc_ai_res.get("reasoning", "")
                        st.success(f"**AI Macro Parameter Adjustments Applied!**\n\n*Rationale:* {ai_policy_summary}")
                except Exception as e:
                    st.warning(f"Could not fetch AI adjustments due to key error or rate limit. Running pure statistical model instead. (Error: {e})")

        # Adjust parameters based on AI reasoning
        adjusted_u = base_u * drift_multiplier
        adjusted_stdev = base_stdev * vol_multiplier
        var = adjusted_stdev ** 2
        drift = adjusted_u - (0.5 * var)

        # Run Stochastic Geometric Brownian Motion
        daily_returns = np.exp(drift + adjusted_stdev * np.random.normal(size=(forecast_days, num_simulations)))
        price_paths = np.zeros_like(daily_returns)
        price_paths[0] = price

        for t in range(1, forecast_days):
            price_paths[t] = price_paths[t - 1] * daily_returns[t]

        ending_prices = price_paths[-1]
        mean_ending_price = np.mean(ending_prices)
        
        # Calculate empirical probability of reaching target price
        successful_paths = np.sum(ending_prices >= target_price)
        prob_success = (successful_paths / num_simulations) * 100.0

        # Render Chart
        mc_fig = go.Figure()
        for i in range(min(num_simulations, 80)):
            mc_fig.add_trace(go.Scatter(y=price_paths[:, i], mode='lines', line=dict(width=0.5, color='rgba(150, 150, 150, 0.15)'), showlegend=False))

        mc_fig.add_trace(go.Scatter(y=np.mean(price_paths, axis=1), mode='lines', name='AI Expected Mean Path', line=dict(color='cyan', width=3)))
        mc_fig.add_trace(go.Scatter(y=[target_price] * forecast_days, mode='lines', name=f'Target Price ({target_price})', line=dict(color='yellow', dash='dash')))

        mc_fig.update_layout(template="plotly_dark", height=450, title=f"AI Stochastic Projection ({forecast_days} Trading Days)", xaxis_title="Days", yaxis_title="Price")
        st.plotly_chart(mc_fig, use_container_width=True)

        # Output Summary Metrics
        st.markdown("---")
        st.markdown("### 📊 AI Probability Analysis")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Current Price", f"{price:.2f}")
        p2.metric("Target Price", f"{target_price:.2f}")
        p3.metric("Probability of Hitting Target", f"{prob_success:.1f}%")
        p4.metric("AI Mean Price Horizon", f"{mean_ending_price:.2f}", delta=f"{((mean_ending_price - price) / price) * 100:.2f}%")
