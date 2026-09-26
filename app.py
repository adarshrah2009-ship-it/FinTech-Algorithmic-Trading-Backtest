import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from arch import arch_model
import google.generativeai as genai
from fpdf import FPDF
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

ticker = st.sidebar.selectbox(
    "Select Asset Ticker",
    ["BTC-USD", "ETH-USD", "^GSPC", "SUZLON.NS", "AAPL", "NVDA"],
    index=0
)

horizon = st.sidebar.selectbox("Data Horizon", ["1y", "2y", "5y"], index=1)

# Securely grab key from Streamlit Secrets or prompt visitor for input
api_key = st.secrets.get("AQ.Ab8RN6JEjmiDrYKPT4UgXDGu6sbVs6WcFfjsXyWSXpZovNEFTQ", "")
if not api_key:
    api_key = st.sidebar.text_input(
        "🔑 Enter Gemini API Key", 
        type="password", 
        help="Visitors can provide their own key here, or configure GEMINI_API_KEY in Streamlit Secrets."
    )

# Fetch Market Data
@st.cache_data(ttl=600)
def load_data(symbol, period):
    df = yf.download(symbol, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, level=1, axis=1)
    df['Returns'] = df['Close'].pct_change().dropna()
    return df.dropna()

try:
    df = load_data(ticker, horizon)
    current_price = float(df['Close'].iloc[-1])
except Exception as e:
    st.error(f"Error fetching ticker data: {e}")
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
            name=ticker
        ))
        fig.update_layout(title=f"{ticker} Price Chart", template="plotly_dark", xaxis_rangeslider_visible=False)
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
                        Target Asset: {ticker}
                        Horizon: {horizon}
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
                # 1. Fit GARCH(1,1)
                garch = arch_model(df['Returns'] * 100, vol='Garch', p=1, q=1)
                res = garch.fit(disp='off')
                forecast_vol = np.sqrt(res.forecast().variance.iloc[-1, -1]) / 100
                
                # 2. Run Monte Carlo
                dt = 1 / 252
                daily_drift = (df['Returns'].mean() - 0.5 * (forecast_vol ** 2)) * dt
                daily_vol = forecast_vol * np.sqrt(dt)
                
                sim_paths = np.zeros((sim_days, num_sims))
                sim_paths[0] = current_price
                
                for t in range(1, sim_days):
                    shock = np.random.normal(0, 1, num_sims)
                    sim_paths[t] = sim_paths[t-1] * np.exp(daily_drift + daily_vol * shock)
                
                # 3. Plot Paths
                fig_mc = go.Figure()
                for i in range(min(num_sims, 100)):
                    fig_mc.add_trace(go.Scatter(y=sim_paths[:, i], mode='lines', line=dict(width=0.5), showlegend=False))
                fig_mc.update_layout(title=f"{num_sims}-Path Monte Carlo Simulation ({sim_days} Days)", template="plotly_dark")
                st.plotly_chart(fig_mc, use_container_width=True)
                
                ending_prices = sim_paths[-1]
                cvar_95 = current_price - np.mean(ending_prices[ending_prices <= np.percentile(ending_prices, 5)])
                st.warning(f"Estimated 95% CVaR (Expected Tail Loss over {sim_days} days): **${cvar_95:,.2f}** per share.")
