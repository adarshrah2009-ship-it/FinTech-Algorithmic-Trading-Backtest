import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# 1. Page Configuration
st.set_page_config(
    page_title="Institutional Quant Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ Quantitative Trading & Risk Engine")
st.write(
    "Institutional backtester and probabilistic risk model evaluating moving average crossovers against market benchmarks."
)

# 2. Sidebar Parameters
st.sidebar.header("🕹️ Strategy Parameters")
ticker = st.sidebar.text_input("Enter Stock Symbol", value="RELIANCE.NS").strip()
time_period = st.sidebar.selectbox(
    "Backtest Period", options=["1y", "2y", "3y", "5y"], index=2
)

fast_ma = st.sidebar.slider("Fast Moving Average (Days)", 10, 50, 50)
slow_ma = st.sidebar.slider("Slow Moving Average (Days)", 100, 200, 200)

ticker_clean = ticker if ticker else "RELIANCE.NS"


# 3. Robust Data Fetcher
@st.cache_data(ttl=300)
def fetch_data(symbol, period):
    data = yf.download(symbol, period=period, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.xs(symbol, axis=1, level=1)
        except Exception:
            data.columns = [col[0] for col in data.columns]
    return data


with st.spinner(f"Fetching market data for {ticker_clean}..."):
    df = fetch_data(ticker_clean, time_period)

if df.empty or len(df) <= slow_ma:
    st.error(
        f"Insufficient historical data for symbol '{ticker_clean}'. Try selecting a longer timeframe or checking the symbol."
    )
else:
    # 4. Strategy & Indicator Calculations
    df["Fast_MA"] = df["Close"].rolling(window=fast_ma).mean()
    df["Slow_MA"] = df["Close"].rolling(window=slow_ma).mean()

    df["Signal"] = np.where(df["Fast_MA"] > df["Slow_MA"], 1, 0)
    df["Position"] = df["Signal"].shift(1)

    df["Daily_Return"] = df["Close"].pct_change()
    df["Strategy_Return"] = df["Daily_Return"] * df["Position"]

    df_clean = df.dropna().copy()
    initial_cap = 100000

    df_clean["Buy_Hold_Equity"] = initial_cap * (
        1 + df_clean["Daily_Return"]
    ).cumprod()
    df_clean["Strategy_Equity"] = initial_cap * (
        1 + df_clean["Strategy_Return"]
    ).cumprod()

    # 5. Risk Performance Metrics
    trading_days = 252
    strat_ret = (
        df_clean["Strategy_Equity"].iloc[-1] - initial_cap
    ) / initial_cap
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

    # 6. Display Dashboard Metrics
    st.markdown("### 📊 Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategy Total Return", f"{strat_ret * 100:.2f}%")
    c2.metric("Buy & Hold Return", f"{bh_ret * 100:.2f}%")
    c3.metric("Sharpe Ratio (Annualized)", f"{sharpe_ratio:.2f}")
    c4.metric("Max Drawdown", f"{max_drawdown:.2f}%")

    # 7. Interactive Candlestick Chart
    st.markdown(f"### 📈 Interactive Technical Analysis ({ticker_clean})")
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df_clean.index,
            open=df_clean["Open"],
            high=df_clean["High"],
            low=df_clean["Low"],
            close=df_clean["Close"],
            name="OHLC Price",
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

    # 8. Monte Carlo Simulation Engine
    st.markdown("### 🎲 Monte Carlo Risk Simulation (100-Day Forward Projection)")
    st.write(
        "Simulating 200 random forward price trajectories based on historical daily volatility."
    )

    returns = df_clean["Daily_Return"]
    last_price = float(df_clean["Close"].iloc[-1])
    num_simulations = 200
    num_days = 100

    sim_matrix = np.zeros((num_days + 1, num_simulations))
    sim_matrix[0] = last_price

    daily_mean = returns.mean()
    daily_std = returns.std()

    for day in range(1, num_days + 1):
        random_shocks = np.random.normal(daily_mean, daily_std, num_simulations)
        sim_matrix[day] = sim_matrix[day - 1] * (1 + random_shocks)

    simulation_df = pd.DataFrame(sim_matrix)

    # Calculate Percentile Boundaries
    mean_path = simulation_df.mean(axis=1)
    p95 = simulation_df.quantile(0.95, axis=1)
    p05 = simulation_df.quantile(0.05, axis=1)

    mc_fig = go.Figure()

    # Plot Background Trajectories
    for col in simulation_df.columns:
        mc_fig.add_trace(
            go.Scatter(
                y=simulation_df[col],
                mode="lines",
                line=dict(width=0.3, color="gray"),
                showlegend=False,
                opacity=0.3,
            )
        )

    # Overlay Statistical Percentiles
    mc_fig.add_trace(
        go.Scatter(
            y=p95,
            mode="lines",
            line=dict(color="green", width=2.5),
            name="95th Percentile (Best-Case Boundary)",
        )
    )
    mc_fig.add_trace(
        go.Scatter(
            y=mean_path,
            mode="lines",
            line=dict(color="yellow", width=2.5),
            name="Expected Mean Path",
        )
    )
    mc_fig.add_trace(
        go.Scatter(
            y=p05,
            mode="lines",
            line=dict(color="red", width=2.5),
            name="5th Percentile (Worst-Case Floor)",
        )
    )

    mc_fig.update_layout(
        template="plotly_dark",
        height=450,
        yaxis_title="Projected Price",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(mc_fig, use_container_width=True)
