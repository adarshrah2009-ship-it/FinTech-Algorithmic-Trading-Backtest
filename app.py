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
    page_title="Institutional Quant & Risk Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ Institutional Quant & Portfolio Analytics Terminal")
st.write(
    "Multi-asset quantitative suite featuring technical crossovers, GARCH volatility, Markowitz portfolio optimization, macro factor attribution, and automated trade alerts."
)

# Predefined asset lists for scrollable dropdown menus
ASSET_OPTIONS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "TATAMOTORS.NS",
    "BTC-USD", "ETH-USD", "GC=F", "^GSPC", "^NSEI", "AAPL", "NVDA", "TSLA"
]

BENCHMARK_OPTIONS = [
    "^GSPC", "^NSEI", "^BSESN", "^IXIC", "DX-Y.NYB", "GC=F"
]

# ---------------------------------------------------------
# 2. NAVIGATION TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Single Asset & GARCH Risk",
    "🎯 Portfolio Optimization",
    "📊 Macro Factor Attribution",
    "🔔 Live Webhook Alerts"
])

# ---------------------------------------------------------
# TAB 1: SINGLE ASSET BACKTEST & GARCH VOLATILITY
# ---------------------------------------------------------
with tab1:
    st.sidebar.header("🕹️ Strategy Parameters")
    ticker = st.sidebar.selectbox("Select Asset Ticker", options=ASSET_OPTIONS, index=0)
    time_period = st.sidebar.selectbox("Horizon", options=["1y", "2y", "3y", "5y"], index=2)
    fast_ma = st.sidebar.slider("Fast Moving Average (Days)", 10, 50, 50)
    slow_ma = st.sidebar.slider("Slow Moving Average (Days)", 100, 200, 200)

    ticker_clean = ticker if ticker else "RELIANCE.NS"

    @st.cache_data(ttl=300)
    def fetch_data(symbol, period):
        data = yf.download(symbol, period=period, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data = data.xs(symbol, axis=1, level=1)
            except Exception:
                data.columns = [col[0] for col in data.columns]
        return data

    with st.spinner("Downloading financial data..."):
        df = fetch_data(ticker_clean, time_period)

    if df.empty or len(df) <= slow_ma:
        st.error(f"Insufficient historical data for symbol '{ticker_clean}'.")
    else:
        # Technical Indicators
        df["Fast_MA"] = df["Close"].rolling(window=fast_ma).mean()
        df["Slow_MA"] = df["Close"].rolling(window=slow_ma).mean()
        df["Signal"] = np.where(df["Fast_MA"] > df["Slow_MA"], 1, 0)
        df["Position"] = df["Signal"].shift(1)
        df["Daily_Return"] = df["Close"].pct_change()
        df["Strategy_Return"] = df["Daily_Return"] * df["Position"]

        df_clean = df.dropna().copy()
        initial_cap = 100000

        df_clean["Buy_Hold_Equity"] = initial_cap * (1 + df_clean["Daily_Return"]).cumprod()
        df_clean["Strategy_Equity"] = initial_cap * (1 + df_clean["Strategy_Return"]).cumprod()

        # Risk Metrics
        trading_days = 252
        strat_ret = (df_clean["Strategy_Equity"].iloc[-1] - initial_cap) / initial_cap
        bh_ret = (df_clean["Buy_Hold_Equity"].iloc[-1] - initial_cap) / initial_cap

        rf_daily = 0.06 / trading_days
        excess_ret = df_clean["Strategy_Return"] - rf_daily
        sharpe_ratio = (
            np.sqrt(trading_days) * excess_ret.mean() / df_clean["Strategy_Return"].std()
            if df_clean["Strategy_Return"].std() != 0
            else 0
        )

        # Tail Risk (VaR & CVaR)
        confidence_level = 0.95
        var_95 = np.percentile(df_clean["Daily_Return"], (1 - confidence_level) * 100)
        cvar_95 = df_clean["Daily_Return"][df_clean["Daily_Return"] <= var_95].mean()

        # Dynamic EWMA Volatility
        lambda_param = 0.94
        returns_sq = df_clean["Daily_Return"] ** 2
        ewma_vol = np.zeros(len(returns_sq))
        ewma_vol[0] = returns_sq.iloc[0]
        for t in range(1, len(returns_sq)):
            ewma_vol[t] = lambda_param * ewma_vol[t - 1] + (1 - lambda_param) * returns_sq.iloc[t]
        df_clean["Dynamic_Vol"] = np.sqrt(ewma_vol) * np.sqrt(trading_days)

        # Performance Display
        st.markdown("### 📊 Performance & Tail Risk Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Strategy Return", f"{strat_ret * 100:.2f}%")
        c2.metric("Buy & Hold Return", f"{bh_ret * 100:.2f}%")
        c3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        c4.metric("Daily 95% VaR", f"{var_95 * 100:.2f}%")
        c5.metric("Expected Shortfall (CVaR)", f"{cvar_95 * 100:.2f}%")

        # Interactive Chart
        st.markdown(f"### 📈 Technical Crossover Analysis ({ticker_clean})")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_clean.index, open=df_clean["Open"], high=df_clean["High"], low=df_clean["Low"], close=df_clean["Close"], name="OHLC"))
        fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean["Fast_MA"], line=dict(color="orange", width=1.5), name=f"{fast_ma}-DMA"))
        fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean["Slow_MA"], line=dict(color="red", width=2), name=f"{slow_ma}-DMA"))
        fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=450, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Dynamic Volatility Chart
        st.markdown("### ⚡ Dynamic EWMA Volatility Forecasting (Annualized)")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=df_clean.index, y=df_clean["Dynamic_Vol"] * 100, line=dict(color="cyan", width=1.5), name="Annualized Volatility (%)"))
        fig_vol.update_layout(template="plotly_dark", height=300, yaxis_title="Volatility (%)", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_vol, use_container_width=True)

        # Export Feature
        st.markdown("### 📥 Export Analytical Results")
        csv_data = df_clean[["Close", "Fast_MA", "Slow_MA", "Strategy_Return", "Strategy_Equity", "Dynamic_Vol"]].to_csv()
        st.download_button(
            label="Download Backtest Data (CSV)",
            data=csv_data,
            file_name=f"{ticker_clean}_quant_backtest.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------
# TAB 2: PORTFOLIO OPTIMIZATION & EFFICIENT FRONTIER
# ---------------------------------------------------------
with tab2:
    st.markdown("### 🎯 Markowitz Portfolio Optimization Engine")
    st.write("Calculate optimal asset weights to maximize the Sharpe Ratio or minimize portfolio variance via Monte Carlo simulation.")

    default_assets = "BTC-USD, GC=F, RELIANCE.NS, TCS.NS, ^GSPC"
    user_assets = st.text_input("Portfolio Asset Tickers (comma-separated)", value=default_assets)
    asset_list = [a.strip() for a in user_assets.split(",") if a.strip()]

    if len(asset_list) >= 2:
        with st.spinner("Computing Monte Carlo Efficient Frontier..."):
            port_prices = yf.download(asset_list, period=time_period)["Close"]
            if isinstance(port_prices.columns, pd.MultiIndex):
                port_prices.columns = [col[1] for col in port_prices.columns]
            
            port_returns = port_prices.pct_change().dropna()
            mean_returns = port_returns.mean() * 252
            cov_matrix = port_returns.cov() * 252
            num_assets = len(asset_list)

            # Pure NumPy Monte Carlo Optimization
            num_simulations = 5000
            weights_record = np.zeros((num_simulations, num_assets))
            results = np.zeros((3, num_simulations))

            for i in range(num_simulations):
                w = np.random.random(num_assets)
                w /= np.sum(w)
                weights_record[i, :] = w
                
                p_ret = np.sum(mean_returns * w)
                p_std = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                p_sharpe = (p_ret - 0.06) / p_std if p_std != 0 else 0
                
                results[0, i] = p_std
                results[1, i] = p_ret
                results[2, i] = p_sharpe

            max_sharpe_idx = np.argmax(results[2])
            opt_weights = weights_record[max_sharpe_idx, :]
            opt_ret = results[1, max_sharpe_idx]
            opt_std = results[0, max_sharpe_idx]
            opt_sharpe = results[2, max_sharpe_idx]

            st.markdown("#### 🏆 Optimal Maximum Sharpe Portfolio Allocation")
            alloc_df = pd.DataFrame({"Asset": asset_list, "Optimal Weight (%)": np.round(opt_weights * 100, 2)})
            
            col_left, col_right = st.columns([1, 2])
            with col_left:
                st.dataframe(alloc_df, use_container_width=True)
                st.metric("Expected Annual Return", f"{opt_ret * 100:.2f}%")
                st.metric("Expected Annual Volatility", f"{opt_std * 100:.2f}%")
                st.metric("Maximized Sharpe Ratio", f"{opt_sharpe:.2f}")

            with col_right:
                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(x=results[0, :], y=results[1, :], mode='markers', marker=dict(color=results[2, :], colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe")), name="Simulated Portfolios"))
                fig_ef.add_trace(go.Scatter(x=[opt_std], y=[opt_ret], mode='markers', marker=dict(color='red', size=15, symbol='star'), name="Max Sharpe Portfolio"))
                fig_ef.update_layout(template="plotly_dark", height=400, xaxis_title="Annualized Volatility (Risk)", yaxis_title="Annualized Expected Return", margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_ef, use_container_width=True)
    else:
        st.warning("Please provide at least 2 tickers for portfolio optimization.")

# ---------------------------------------------------------
# TAB 3: MACRO FACTOR ATTRIBUTION (WITH SCROLLABLE DROPDOWNS)
# ---------------------------------------------------------
with tab3:
    st.markdown("### 📊 Factor Attribution ($\alpha / \beta$ Regression)")
    st.write("Deconstruct target asset returns against broad market indices and liquidity benchmarks.")

    col_target, col_bench = st.columns(2)
    with col_target:
        target_asset = st.selectbox("Select Target Asset", options=ASSET_OPTIONS, index=5)  # Defaults to BTC-USD
    with col_bench:
        benchmark_asset = st.selectbox("Select Benchmark Index", options=BENCHMARK_OPTIONS, index=0)  # Defaults to ^GSPC

    if target_asset and benchmark_asset:
        with st.spinner("Calculating regression parameters..."):
            factor_data = yf.download([target_asset, benchmark_asset], period=time_period)["Close"]
            if isinstance(factor_data.columns, pd.MultiIndex):
                factor_data.columns = [col[1] for col in factor_data.columns]
            
            factor_returns = factor_data.pct_change().dropna()
            
            if len(factor_returns) > 30:
                y = factor_returns[target_asset]
                x = factor_returns[benchmark_asset]
                
                beta, alpha = np.polyfit(x, y, 1)
                annual_alpha = alpha * 252
                
                corr_val = factor_returns.corr().iloc[0, 1]
                r_squared = corr_val ** 2

                st.markdown("#### 🎯 Regression Coefficients")
                m1, m2, m3 = st.columns(3)
                m1.metric("Alpha ($\alpha$, Annualized)", f"{annual_alpha * 100:.2f}%")
                m2.metric("Market Beta ($\beta$)", f"{beta:.2f}")
                m3.metric("R-Squared ($R^2$)", f"{r_squared:.2f}")

                fig_reg = px.scatter(
                    factor_returns, x=benchmark_asset, y=target_asset,
                    trendline="ols",
                    title=f"Linear Regression: {target_asset} vs {benchmark_asset}",
                    labels={benchmark_asset: f"Benchmark ({benchmark_asset}) Return", target_asset: f"Target ({target_asset}) Return"}
                )
                fig_reg.update_layout(template="plotly_dark", height=450)
                st.plotly_chart(fig_reg, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: AUTOMATED WEBHOOK ALERTS (DISCORD & TELEGRAM)
# ---------------------------------------------------------
with tab4:
    st.markdown("### 🔔 Automated Signal Dispatcher")
    st.write("Configure real-time webhooks to stream trading signals directly to Discord or Telegram.")

    alert_service = st.radio("Select Alert Platform", ["Discord Webhook", "Telegram Bot"])
    alert_asset = st.selectbox("Alert Target Asset", options=ASSET_OPTIONS, index=0)

    if alert_service == "Discord Webhook":
        webhook_url = st.text_input("Discord Webhook URL", type="password")
        
        if st.button("🚀 Send Discord Alert"):
            if webhook_url:
                payload = {
                    "content": f"⚡ **QUANT TERMINAL ALERT**\n**Asset:** {alert_asset}\n**Signal:** BULLISH CROSSOVER (BUY)\n**Timestamp:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
                }
                res = requests.post(webhook_url, json=payload)
                if res.status_code in [200, 204]:
                    st.success("Discord Alert Dispatched!")
                else:
                    st.error(f"Error {res.status_code}: Check Webhook URL")
            else:
                st.warning("Please enter a Discord Webhook URL.")

    else:
        bot_token = st.text_input("Telegram Bot Token (from @BotFather)", type="password")
        chat_id = st.text_input("Telegram Chat ID (from @userinfobot)")
        
        if st.button("🚀 Send Telegram Alert"):
            if bot_token and chat_id:
                telegram_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                payload = {
                    "chat_id": chat_id,
                    "text": f"⚡ **QUANT TERMINAL ALERT**\n**Asset:** {alert_asset}\n**Signal:** BULLISH CROSSOVER (BUY)\n**Timestamp:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "parse_mode": "Markdown"
                }
                res = requests.post(telegram_url, json=payload)
                if res.status_code == 200:
                    st.success("Telegram Alert Dispatched!")
                else:
                    st.error(f"Error {res.status_code}: Check Token or Chat ID.")
            else:
                st.warning("Please enter both your Bot Token and Chat ID.")
