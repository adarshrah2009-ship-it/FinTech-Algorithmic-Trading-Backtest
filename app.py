import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# 1. Dashboard Configuration
st.set_page_config(
    page_title="Institutional Hedge Fund Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ Hedge Fund Analytics & Risk Terminal")
st.write(
    "Multi-asset risk engine featuring Value at Risk (VaR), Expected Shortfall (CVaR), Technical Indicators, and Correlation Modeling."
)

# 2. Navigation Tabs
tab1, tab2 = st.tabs(["📈 Single Asset Backtest & Risk", "🔥 Portfolio Correlation Matrix"])

# ---------------------------------------------------------
# TAB 1: SINGLE ASSET RISK & BACKTEST
# ---------------------------------------------------------
with tab1:
    st.sidebar.header("🕹️ Strategy Parameters")
    ticker = st.sidebar.text_input("Enter Primary Stock Ticker", value="RELIANCE.NS").strip()
    time_period = st.sidebar.selectbox("Horizon", options=["1y", "2y", "3y", "5y"], index=2)
    fast_ma = st.sidebar.slider("Fast Moving Average (Days)", 10, 50, 50)
    slow_ma = st.sidebar.slider("Slow Moving Average (Days)", 100, 200, 200)

    ticker_clean = ticker if ticker else "RELIANCE.NS"

    @st.cache_data(ttl=300)
    def get_single_data(symbol, period):
        data = yf.download(symbol, period=period, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data = data.xs(symbol, axis=1, level=1)
            except Exception:
                data.columns = [col[0] for col in data.columns]
        return data

    with st.spinner("Analyzing asset metrics..."):
        df = get_single_data(ticker_clean, time_period)

    if df.empty or len(df) <= slow_ma:
        st.error(f"Insufficient data for symbol '{ticker_clean}'.")
    else:
        # Strategy Logic
        df["Fast_MA"] = df["Close"].rolling(window=fast_ma).mean()
        df["Slow_MA"] = df["Close"].rolling(window=slow_ma).mean()
        df["Signal"] = np.where(df["Fast_MA"] > df["Slow_MA"], 1, 0)
        df["Position"] = df["Signal"].shift(1)
        df["Daily_Return"] = df["Close"].pct_change()
        df["Strategy_Return"] = df["Daily_Return"] * df["Position"]

        # RSI Calculation (14-period)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

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

        peak = df_clean["Strategy_Equity"].cummax()
        drawdown = (df_clean["Strategy_Equity"] - peak) / peak
        max_drawdown = drawdown.min() * 100

        # Institutional Tail Risk Metrics
        confidence_level = 0.95
        var_95 = np.percentile(df_clean["Daily_Return"], (1 - confidence_level) * 100)
        cvar_95 = df_clean["Daily_Return"][df_clean["Daily_Return"] <= var_95].mean()

        # Metrics Display
        st.markdown("### 📊 Performance & Tail Risk Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Strategy Return", f"{strat_ret * 100:.2f}%")
        c2.metric("Benchmark Return", f"{bh_ret * 100:.2f}%")
        c3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        c4.metric("Daily 95% VaR", f"{var_95 * 100:.2f}%")
        c5.metric("Expected Shortfall (CVaR)", f"{cvar_95 * 100:.2f}%")

        # Interactive Price & RSI Chart
        st.markdown(f"### 📈 Technical Analysis & Momentum ({ticker_clean})")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_clean.index, open=df_clean["Open"], high=df_clean["High"], low=df_clean["Low"], close=df_clean["Close"], name="OHLC"))
        fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean["Fast_MA"], line=dict(color="orange", width=1.5), name=f"{fast_ma}-DMA"))
        fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean["Slow_MA"], line=dict(color="red", width=2), name=f"{slow_ma}-DMA"))
        fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=450, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: PORTFOLIO CORRELATION MATRIX
# ---------------------------------------------------------
with tab2:
    st.markdown("### 🎲 Multi-Asset Correlation Heatmap")
    st.write("Evaluate portfolio diversification by measuring cross-asset return dependencies.")
    
    default_tickers = "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS"
    user_tickers = st.text_input("Enter Tickers (separated by commas)", value=default_tickers)
    
    ticker_list = [t.strip() for t in user_tickers.split(",") if t.strip()]
    
    if len(ticker_list) >= 2:
        with st.spinner("Building correlation engine..."):
            port_data = yf.download(ticker_list, period=time_period)["Close"]
            if isinstance(port_data.columns, pd.MultiIndex):
                port_data.columns = [col[1] for col in port_data.columns]
            
            daily_returns = port_data.pct_change().dropna()
            corr_matrix = daily_returns.corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Asset Return Correlation Matrix",
            )
            fig_corr.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Please enter at least 2 stock tickers to calculate correlation.")
