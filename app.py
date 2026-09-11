import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# 1. Page Title & Configuration
st.set_page_config(
    page_title="Quant Backtesting Dashboard", layout="wide"
)
st.title("📈 Quantitative Multi-Stock Scanner & Backtester")
st.write(
    "Interactive financial tool analyzing 50-DMA and 200-DMA crossovers against Buy & Hold benchmarks."
)

# 2. Sidebar Inputs
st.sidebar.header("User Settings")
raw_ticker = st.sidebar.text_input("Enter NSE Stock Ticker", value="RELIANCE.NS")
time_period = st.sidebar.selectbox(
    "Select Period", options=["1y", "2y", "3y", "5y"], index=2
)

# Ensure ticker is never empty
ticker = raw_ticker.strip() if raw_ticker.strip() else "RELIANCE.NS"

# 3. Setup Session
session = None
try:
    from curl_cffi import requests

    session = requests.Session(impersonate="chrome")
except Exception:
    session = None

# 4. Fetch Stock Data Safely
with st.spinner(f"Fetching market data for {ticker}..."):
    try:
        if session:
            stock = yf.Ticker(ticker, session=session)
            df = stock.history(period=time_period)
        else:
            df = yf.download(ticker, period=time_period)
    except Exception as e:
        st.error(f"Error connecting to Yahoo Finance: {e}")
        df = pd.DataFrame()

# 5. Process & Display Data
if df.empty or len(df) < 50:
    st.warning(
        f"No data returned for '{ticker}'. Please check the symbol (e.g., RELIANCE.NS, TATAMOTORS.NS, AAPL)."
    )
else:
    # Calculations
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

    # Display Metrics
    col1, col2, col3 = st.columns(3)
    final_bh = df_clean["Buy_Hold"].iloc[-1]
    final_strat = df_clean["Strategy"].iloc[-1]
    ret_bh = ((final_bh - initial_capital) / initial_capital) * 100
    ret_strat = ((final_strat - initial_capital) / initial_capital) * 100

    col1.metric("Buy & Hold Return", f"{ret_bh:.2f}%", f"₹{final_bh:,.2f}")
    col2.metric("Dual MA Strategy Return", f"{ret_strat:.2f}%", f"₹{final_strat:,.2f}")
    col3.metric("Alpha Generated", f"{ret_strat - ret_bh:.2f}%")

    # Plot Chart
    st.subheader(f"Price & Moving Average Chart ({ticker})")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_clean["Close"], label="Close Price", color="blue", alpha=0.4)
    ax.plot(df_clean["MA50"], label="50-DMA", color="orange")
    ax.plot(df_clean["MA200"], label="200-DMA", color="red")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
