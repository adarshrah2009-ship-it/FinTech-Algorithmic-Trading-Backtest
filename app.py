import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# 1. Dashboard Title & Configuration
st.set_page_config(
    page_title="Quant Backtesting Dashboard", layout="wide"
)
st.title("📈 Quantitative Multi-Stock Scanner & Backtester")

# 2. Sidebar Settings & Controls
st.sidebar.header("User Settings")
raw_ticker = st.sidebar.text_input("Enter NSE Stock Ticker", value="RELIANCE.NS")
time_period = st.sidebar.selectbox(
    "Select Period", options=["1y", "2y", "3y", "5y"], index=2
)

# Live Refresh Controls
st.sidebar.subheader("🔄 Refresh Controls")
if st.sidebar.button("Fetch Latest Market Data"):
    st.cache_data.clear()
    st.rerun()

auto_refresh = st.sidebar.checkbox("Enable 60s Auto-Refresh", value=False)

# Ensure ticker fallback
ticker = raw_ticker.strip() if raw_ticker.strip() else "RELIANCE.NS"

# 3. Cached Data Fetcher
@st.cache_data(ttl=60)
def load_stock_data(symbol, period):
    return yf.download(symbol, period=period)

with st.spinner(f"Loading market data for {ticker}..."):
    try:
        df = load_stock_data(ticker, time_period)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=1)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        df = pd.DataFrame()

# 4. Display Real-Time Price Ticker & Metrics
if not df.empty and len(df) >= 50:
    latest_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]
    day_change = latest_close - prev_close
    pct_change = (day_change / prev_close) * 100

    # Dynamic color indicator
    delta_color = "normal" if day_change >= 0 else "inverse"

    st.subheader(f"Live Ticker: {ticker}")
    st.metric(
        label="Latest Market Close Price",
        value=f"₹{latest_close:,.2f}",
        delta=f"{day_change:+,.2f} ({pct_change:+.2f}%)",
    )

    # Strategy Calculations
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["MA200"] = df["Close"].rolling(window=200).mean()
    df["Position"] = np.where(df["MA50"] > df["MA200"], 1, 0)
    df["Position"] = df["Position"].shift(1)

    df["Daily_Return"] = df["Close"].pct_change()
    df["Strategy_Return"] = df["Daily_Return"] * df["Position"]

    df_clean = df.dropna().copy()
    initial_capital = 100000

    df_clean["Buy_Hold"] = initial_capital * (
        1 + df_clean["Daily_Return"]
    ).cumprod()
    df_clean["Strategy"] = initial_capital * (
        1 + df_clean["Strategy_Return"]
    ).cumprod()

    # Backtest Performance Overview
    st.subheader("Strategy vs Benchmark Returns")
    col1, col2, col3 = st.columns(3)
    final_bh = df_clean["Buy_Hold"].iloc[-1]
    final_strat = df_clean["Strategy"].iloc[-1]
    ret_bh = ((final_bh - initial_capital) / initial_capital) * 100
    ret_strat = ((final_strat - initial_capital) / initial_capital) * 100

    col1.metric("Buy & Hold Return", f"{ret_bh:.2f}%", f"₹{final_bh:,.2f}")
    col2.metric("Dual MA Strategy Return", f"{ret_strat:.2f}%", f"₹{final_strat:,.2f}")
    col3.metric("Alpha Generated", f"{ret_strat - ret_bh:.2f}%")

    # Interactive Chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_clean["Close"], label="Close Price", color="blue", alpha=0.4)
    ax.plot(df_clean["MA50"], label="50-DMA", color="orange")
    ax.plot(df_clean["MA200"], label="200-DMA", color="red")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    st.warning(f"Insufficient historical data available for '{ticker}'.")

# Auto-refresh loop (60s timer)
if auto_refresh:
    import time

    time.sleep(60)
    st.rerun()
