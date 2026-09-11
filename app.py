import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Page setup
st.set_page_config(
    page_title="Institutional Quant Dashboard", layout="wide"
)
st.title("⚡ Quantitative Trading & Risk Management Engine")

# Sidebar
st.sidebar.header("🕹️ Strategy & Risk Settings")
ticker = st.sidebar.text_input("Enter NSE Ticker", value="RELIANCE.NS").strip()
time_period = st.sidebar.selectbox(
    "Backtest Horizon", options=["1y", "2y", "3y", "5y"], index=2
)
fast_ma = st.sidebar.slider("Fast Moving Average (Days)", 10, 50, 50)
slow_ma = st.sidebar.slider("Slow Moving Average (Days)", 100, 200, 200)

# Fetch Data
@st.cache_data(ttl=300)
def get_data(symbol, period):
    return yf.download(symbol, period=period)

with st.spinner("Downloading financial data..."):
    df = get_data(ticker, time_period)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=1)

if not df.empty and len(df) > slow_ma:
    # 1. Indicator Calculations
    df["Fast_MA"] = df["Close"].rolling(window=fast_ma).mean()
    df["Slow_MA"] = df["Close"].rolling(window=slow_ma).mean()

    # Position & Returns
    df["Signal"] = np.where(df["Fast_MA"] > df["Slow_MA"], 1, 0)
    df["Position"] = df["Signal"].shift(1)
    df["Daily_Return"] = df["Close"].pct_change()
    df["Strategy_Return"] = df["Daily_Return"] * df["Position"]

    df_clean = df.dropna().copy()
    initial_cap = 100000

    # Portfolio Growth
    df_clean["Buy_Hold_Equity"] = initial_cap * (
        1 + df_clean["Daily_Return"]
    ).cumprod()
    df_clean["Strategy_Equity"] = initial_cap * (
        1 + df_clean["Strategy_Return"]
    ).cumprod()

    # Advanced Quant Metrics
    trading_days = 252
    strat_ret = (
        df_clean["Strategy_Equity"].iloc[-1] - initial_cap
    ) / initial_cap
    bh_ret = (df_clean["Buy_Hold_Equity"].iloc[-1] - initial_cap) / initial_cap

    # Sharpe Ratio (assuming 6% risk-free rate)
    rf_daily = 0.06 / trading_days
    excess_returns = df_clean["Strategy_Return"] - rf_daily
    sharpe_ratio = (
        np.sqrt(trading_days)
        * excess_returns.mean()
        / df_clean["Strategy_Return"].std()
    )

    # Maximum Drawdown (MDD)
    peak = df_clean["Strategy_Equity"].cummax()
    drawdown = (df_clean["Strategy_Equity"] - peak) / peak
    max_drawdown = drawdown.min() * 100

    # 2. Executive Metric Cards
    st.markdown("### 📊 Performance & Risk Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Strategy Total Return", f"{strat_ret * 100:.2f}%")
    m2.metric("Buy & Hold Return", f"{bh_ret * 100:.2f}%")
    m3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    m4.metric("Max Drawdown", f"{max_drawdown:.2f}%")

    # 3. Interactive Candlestick & Moving Average Chart
    st.markdown("### 📈 Interactive Price & Technical Analysis")
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df_clean.index,
            open=df_clean["Open"],
            high=df_clean["High"],
            low=df_clean["Low"],
            close=df_clean["Close"],
            name="OHLC",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_clean.index,
            y=df_clean["Fast_MA"],
            line=dict(color="orange", width=1.5),
            name=f"{fast_ma}-DMA",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_clean.index,
            y=df_clean["Slow_MA"],
            line=dict(color="red", width=2),
            name=f"{slow_ma}-DMA",
        )
    )
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4. Monte Carlo Simulation Engine
    st.markdown("### 🎲 Monte Carlo Risk Simulation (100-Day Forward Projection)")
    st.write(
        "Simulating 200 possible future price paths based on historical daily volatility."
    )

    returns = df_clean["Daily_Return"]
    last_price = df_clean["Close"].iloc[-1]
    num_simulations = 200
    num_days = 100

    simulation_df = pd.DataFrame()
    for x in range(num_simulations):
        price_series = [last_price]
        for y in range(num_days):
            simulated_return = np.random.normal(returns.mean(), returns.std())
            price_series.append(price_series[-1] * (1 + simulated_return))
        simulation_df[x] = price_series

    mc_fig = go.Figure()
    for col in simulation_df.columns:
        mc_fig.add_trace(
            go.Scatter(
                y=simulation_df[col],
                mode="lines",
                line=dict(width=0.5),
                showlegend=False,
            )
        )
    mc_fig.update_layout(
        template="plotly_dark",
        height=400,
        yaxis_title="Projected Price (INR)",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(mc_fig, use_container_width=True)

else:
    st.error(
        f"Unable to fetch sufficient historical price data for '{ticker}'. Please adjust the parameters."
    )
