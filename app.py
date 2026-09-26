import os
import tempfile
import requests

# Set cache path
os.environ["YFINANCE_CACHE_DIR"] = tempfile.gettempdir()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from arch import arch_model
import google.generativeai as genai
from scipy.stats import norm

# ==========================================
# 1. PAGE SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Institutional Quant & AI Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Institutional Quant & AI Terminal")

# ==========================================
# 2. PAYWALL & SUBSCRIPTION LOCK SYSTEM
# ==========================================
query_params = st.query_params
is_pro_user = query_params.get("status") == "pro"

st.sidebar.header("💳 Membership Tier")

if is_pro_user:
    st.sidebar.success("Pro Tier Active! 🔥")
else:
    st.sidebar.warning("Free Version")
    st.sidebar.markdown("[👉 Upgrade to Pro Access](https://buy.stripe.com/your_checkout_link)")

st.sidebar.divider()

# ==========================================
# 3. SIDEBAR INPUTS & SECURE API HANDLING
# ==========================================
st.sidebar.header("System Controls")

# Mapping tickers for reliable Stooq fetching
ticker_map = {
    "BTC-USD": "BTCUSD",
    "ETH-USD": "ETHUSD",
    "S&P 500": "SPX",
    "AAPL": "AAPL.US",
    "NVDA": "NVDA.US",
    "SUZLON": "SUZLON.IN"
}

ticker_display = st.sidebar.selectbox(
    "Select Asset Ticker",
    list(ticker_map.keys()),
    index=0
)
ticker_symbol = ticker_map[ticker_display]

horizon_map = {"1y": 365, "2y": 730, "5y": 1825}
horizon_label = st.sidebar.selectbox("Data Horizon", ["1y", "2y", "5y"], index=1)
days_back = horizon_map[horizon_label]

api_key = st.secrets.get("GEMINI_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input(
        "🔑 Enter Gemini API Key", 
        type="password", 
        help="Visitors can provide their own key here, or configure GEMINI_API_KEY in Streamlit Secrets."
    )

# Robust data fetching function
@st.cache_data(ttl=600)
def load_data(symbol, days):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days)
    
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=10)
    
    # Check if response is valid CSV rather than XML/HTML error
    if response.status_code != 200 or response.text.startswith("<?xml") or response.text.startswith("<Error"):
        raise ValueError(f"Unable to fetch data for symbol {symbol}. Provider returned invalid response.")
        
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    
    if df.empty or 'Date' not in df.columns or 'Close' not in df.columns:
        raise ValueError(f"No pricing data found for {symbol}.")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df.set_index('Date', inplace=True)
    
    df = df[df.index >= start_date]
    df['Returns'] = df['Close'].pct_change()
    return df.dropna()

try:
    df = load_data(ticker_symbol, days_back)
    current_price = float(df['Close'].iloc[-1])
except Exception as e:
    st.error(f"⚠️ Data Retrieval Error: {e}")
    st.info("Tip: Try switching the asset ticker or selecting a different horizon.")
    st.stop()

# ==========================================
# 4. NAVIGATION TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Market Matrix", 
    "📈 Strategy Backtest", 
    "🤖 AI Technical Signal", 
    "🧮 Value at Risk & Position Sizer", 
    "🎲 AI Monte Carlo Engine"
])

# ------------------------------------------
# TAB 1: MARKET MATRIX & CHOP AUDIT (FREE)
# ------------------------------------------
with tab1:
    st.header("Market Structure & Price Action")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name=ticker_display
        ))
        fig.update_layout(title=f"{ticker_display} Price Chart", template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Asset Metrics")
        st.metric("Latest Close", f"${current_price:,.2f}")
        st.metric("Annualized Volatility", f"{df['Returns'].std() * np.sqrt(252) * 100:.2f}%")
        st.metric("Max Daily Gain", f"{df['Returns'].max() * 100:.2f}%")
        st.metric("Max Daily Loss", f"{df['Returns'].min() * 100:.2f}%")

# ------------------------------------------
# TAB 2: STRATEGY BACKTEST (FREE)
# ------------------------------------------
with tab2:
    st.header("Simple Moving Average Crossover Backtest")
    fast_window = st.slider("Fast SMA", 5, 50, 20)
    slow_window = st.slider("Slow SMA", 20, 200, 50)
    
    df_bt = df.copy()
    df_bt['Fast_SMA'] = df_bt['Close'].rolling(window=fast_window).mean()
    df_bt['Slow_SMA'] = df_bt['Close'].rolling(window=slow_window).mean()
    df_bt['Signal'] = np.where(df_bt['Fast_SMA'] > df_bt['Slow_SMA'], 1, 0)
    df_bt['Strat_Returns'] = df_bt['Signal'].shift(1) * df_bt['Returns']
    
    cum_bench = (1 + df_bt['Returns']).cumprod()
    cum_strat = (1 + df_bt['Strat_Returns']).cumprod()
    
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=df_bt.index, y=cum_bench, name="Buy & Hold"))
    fig_bt.add_trace(go.Scatter(x=df_bt.index, y=cum_strat, name="SMA Strategy"))
    fig_bt.update_layout(title="Strategy vs Benchmark Performance", template="plotly_dark")
    st.plotly_chart(fig_bt, use_container_width=True)

# ------------------------------------------
# TAB 3: AI TECHNICAL SIGNAL (PAID - LOCKED)
# ------------------------------------------
with tab3:
    st.header("AI Technical Analysis & Executive Audit")
    
    if not is_pro_user:
        st.error("🔒 This feature is locked! Please upgrade to Pro in the sidebar to access the AI Chief Risk Officer.")
    else:
        if not api_key:
            st.warning("⚠️ Please configure your GEMINI_API_KEY in Streamlit Secrets or enter your key in the sidebar.")
        else:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                if st.button("Run AI Technical Engine"):
                    with st.spinner("Analyzing market dynamics & generating risk summary..."):
                        recent_returns = df['Returns'].tail(10).values
                        prompt = f"""
                        You are a Chief Risk Officer at an institutional quantitative fund.
                        Target Asset: {ticker_display}
                        Horizon: {horizon_label}
                        Current Price: {current_price}
                        Recent 10-Day Returns: {recent_returns}
                        
                        Provide a concise executive audit detailing:
                        1. Current Volatility & Risk Profile.
                        2. Recommended Stop-Loss Strategy.
                        3. Position Sizing Guidance for high-volatility scenarios.
                        """
                        response = model.generate_content(prompt)
                        st.markdown("### 📋 Executive Audit Summary")
                        st.write(response.text)
            except Exception as e:
                st.error(f"Failed to connect to Gemini API: {e}")

# ------------------------------------------
# TAB 4: VALUE AT RISK & POSITION SIZER (PAID - LOCKED)
# ------------------------------------------
with tab4:
    st.header("Parametric Value at Risk (VaR) & Position Sizer")
    
    if not is_pro_user:
        st.error("🔒 This feature is locked! Please upgrade to Pro in the sidebar to calculate Parametric VaR.")
    else:
        portfolio_val = st.number_input("Portfolio Size ($)", value=100000, step=5000)
        confidence_level = st.selectbox("Confidence Level", [0.95, 0.99], index=0)
        
        mean_ret = df['Returns'].mean()
        std_ret = df['Returns'].std()
        
        z_score = norm.ppf(confidence_level)
        var_1d_pct = (z_score * std_ret) - mean_ret
        var_1d_dollar = portfolio_val * var_1d_pct
        
        col_var1, col_var2 = st.columns(2)
        col_var1.metric(f"1-Day VaR ({int(confidence_level*100)}%) Dollar Exposure", f"${var_1d_dollar:,.2f}")
        col_var2.metric(f"1-Day VaR ({int(confidence_level*100)}%) Percentage", f"{var_1d_pct*100:.2f}%")
        
        st.info("VaR measures the maximum expected loss over a 1-day period under normal market conditions.")

# ------------------------------------------
# TAB 5: AI MONTE CARLO & GARCH ENGINE (PAID - LOCKED)
# ------------------------------------------
with tab5:
    st.header("GARCH Volatility & Monte Carlo Engine")
    
    if not is_pro_user:
        st.error("🔒 This feature is locked! Please upgrade to Pro in the sidebar to run Monte Carlo simulations & GARCH forecasting.")
    else:
        sim_days = st.slider("Simulation Horizon (Days)", 10, 252, 30)
        num_sims = st.slider("Number of Simulation Paths", 100, 2000, 500)
        
        if st.button("Run Stochastic Engine"):
            with st.spinner("Fitting GARCH(1,1) model and running Geometric Brownian Motion..."):
                garch = arch_model(df['Returns'] * 100, vol='Garch', p=1, q=1)
                res = garch.fit(disp='off')
                forecast_vol = np.sqrt(res.forecast().variance.iloc[-1, -1]) / 100
                
                dt = 1 / 252
                daily_drift = (df['Returns'].mean() - 0.5 * (forecast_vol ** 2)) * dt
                daily_vol = forecast_vol * np.sqrt(dt)
                
                sim_paths = np.zeros((sim_days, num_sims))
                sim_paths[0] = current_price
                
                for t in range(1, sim_days):
                    shock = np.random.normal(0, 1, num_sims)
                    sim_paths[t] = sim_paths[t-1] * np.exp(daily_drift + daily_vol * shock)
                
                fig_mc = go.Figure()
                for i in range(min(num_sims, 100)):
                    fig_mc.add_trace(go.Scatter(y=sim_paths[:, i], mode='lines', line=dict(width=0.5), showlegend=False))
                fig_mc.update_layout(title=f"{num_sims}-Path Monte Carlo Simulation ({sim_days} Days)", template="plotly_dark")
                st.plotly_chart(fig_mc, use_container_width=True)
                
                ending_prices = sim_paths[-1]
                cvar_95 = current_price - np.mean(ending_prices[ending_prices <= np.percentile(ending_prices, 5)])
                st.warning(f"Estimated 95% CVaR (Expected Tail Loss over {sim_days} days): **${cvar_95:,.2f}** per share.")
