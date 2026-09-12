import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Page setup
st.set_page_config(
    page_title="Pro Quant Terminal", layout="wide", page_icon="📈"
)
st.title("📈 Advanced Quantitative Trading Terminal")

POPULAR_ASSETS = [
    "RELIANCE.NS",
    "TATAMOTORS.NS",
    "SUZLON.NS",
    "BTC-USD",
    "AAPL",
    "NVDA",
]
selected_asset = st.sidebar.selectbox("Select Asset", POPULAR_ASSETS)
selected_period = st.sidebar.selectbox(
    "Data Horizon", ["1y", "2y", "5y"], index=1
)


# Helper Functions: Data & Indicators
@st.cache_data(ttl=3600)
def fetch_data(ticker, period):
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def calculate_indicators(df):
    data = df.copy()
    # Moving Averages
    data["50_DMA"] = data["Close"].rolling(50).mean()
    data["200_DMA"] = data["Close"].rolling(200).mean()

    # RSI (14-period)
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # ADX & DMI (14-period)
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

    # Volume 20-period Moving Average
    data["Volume_MA"] = data["Volume"].rolling(20).mean()

    return data


df_raw = fetch_data(selected_asset, selected_period)

if df_raw.empty or len(df_raw) < 200:
    st.error(
        f"Insufficient historical data for {selected_asset}. Please select a longer horizon."
    )
    st.stop()

df = calculate_indicators(df_raw)

# Extract Latest Values
latest = df.iloc[-1]
price = latest["Close"]
rsi = latest["RSI"]
adx = latest["ADX"]
sma50 = latest["50_DMA"]
sma200 = latest["200_DMA"]
vol_confirm = latest["Volume"] > latest["Volume_MA"]

# Tabs Layout
tab1, tab2, tab3 = st.tabs(
    [
        "📊 Multi-Factor Matrix",
        "⚠️ Chop / Volatility Filter",
        "🧪 Backtest Engine",
    ]
)

# ---------------------------------------------------------
# FEATURE 1: MULTI-FACTOR COMPOSITE MATRIX
# ---------------------------------------------------------
with tab1:
    st.subheader("Composite Technical Scorecard")

    # Score calculation (Max +4 Bullish, Min -4 Bearish)
    score = 0
    ma_signal = 1 if sma50 > sma200 else -1
    rsi_signal = 1 if rsi > 50 and rsi < 70 else (-1 if rsi < 50 else 0)
    adx_signal = 1 if adx > 25 else 0
    vol_signal = 1 if vol_confirm else 0

    score = ma_signal + rsi_signal + (adx_signal if ma_signal > 0 else -adx_signal)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"₹{price:.2f}" if ".NS" in selected_asset else f"${price:.2f}")
    col2.metric("RSI (14)", f"{rsi:.1f}", delta="Bullish Zone" if rsi > 50 else "Bearish Zone")
    col3.metric("ADX (Trend Strength)", f"{adx:.1f}", delta="Strong Trend" if adx > 25 else "Weak/Choppy")
    col4.metric("Volume Above 20-MA", "YES" if vol_confirm else "NO")

    st.markdown("---")

    if score >= 2 and adx > 25:
        st.success(f"🟢 **STRONG BULLISH CONFIRMATION** (Composite Score: {score}/4)")
    elif score <= -2 and adx > 25:
        st.error(f"🔴 **STRONG BEARISH CONFIRMATION** (Composite Score: {score}/4)")
    else:
        st.warning(f"🟡 **NEUTRAL / MIXED SIGNALS** (Composite Score: {score}/4) — Avoid heavy directional bets.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close Price", line=dict(color="white")))
    fig.add_trace(go.Scatter(x=df.index, y=df["50_DMA"], name="50 DMA", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df.index, y=df["200_DMA"], name="200 DMA", line=dict(color="red")))
    fig.update_layout(title=f"{selected_asset} Price & Moving Averages", template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# FEATURE 2: VOLATILE MARKET CHOP FILTER
# ---------------------------------------------------------
with tab2:
    st.subheader("Market Regime & Sideways Filter")

    # Flag sideways condition when ADX < 20 or MAs are converging within 1.5% range
    ma_diff_pct = abs(sma50 - sma200) / price * 100
    is_sideways = (adx < 20) or (ma_diff_pct < 1.5)

    if is_sideways:
        st.error("🚫 **SIDEWAYS / CHOPPY MARKET DETECTED**")
        st.write("Market trend strength is insufficient. Moving average crossovers are prone to false signals (whipsaws). **Recommendation: Stay on Sidelines.**")
    else:
        st.success("✅ **TRENDING MARKET DETECTED**")
        st.write(f"Trend strength is sufficient (ADX = {adx:.1f}). Crossover signals carry higher statistical validity.")

    st.markdown("#### **Regime Breakdown**")
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.info(f"**ADX Reading:** {adx:.2f}\n* (ADX > 25 indicates trend presence; < 20 indicates consolidated range)")
    with r_col2:
        st.info(f"**MA Convergence Gap:** {ma_diff_pct:.2f}%\n* (Gaps under 1.5% signal tight consolidation and chop risk)")

# ---------------------------------------------------------
# FEATURE 3: STRATEGY BACKTESTING ENGINE
# ---------------------------------------------------------
with tab3:
    st.subheader("Historical Performance Backtest")
    st.write("Evaluate how a **Multi-Factor Trend Strategy** performed historically compared to Buy & Hold.")

    # Backtest logic: Long when 50-DMA > 200-DMA AND RSI > 50 AND ADX > 20
    bt_df = df.dropna().copy()
    bt_df["Signal"] = 0
    bt_df.loc[(bt_df["50_DMA"] > bt_df["200_DMA"]) & (bt_df["RSI"] > 50) & (bt_df["ADX"] > 20), "Signal"] = 1

    bt_df["Strategy_Returns"] = bt_df["Close"].pct_change() * bt_df["Signal"].shift(1)
    bt_df["Benchmark_Returns"] = bt_df["Close"].pct_change()

    bt_df["Cum_Strategy"] = (1 + bt_df["Strategy_Returns"].fillna(0)).cumprod()
    bt_df["Cum_Benchmark"] = (1 + bt_df["Benchmark_Returns"].fillna(0)).cumprod()

    # Metrics
    total_strat_ret = (bt_df["Cum_Strategy"].iloc[-1] - 1) * 100
    total_bench_ret = (bt_df["Cum_Benchmark"].iloc[-1] - 1) * 100

    trades = bt_df["Signal"].diff().abs()
    num_trades = int(trades.sum() / 2)

    winning_days = bt_df[bt_df["Strategy_Returns"] > 0]["Strategy_Returns"].count()
    active_days = bt_df[bt_df["Signal"].shift(1) == 1]["Strategy_Returns"].count()
    win_rate = (winning_days / active_days * 100) if active_days > 0 else 0

    peak = bt_df["Cum_Strategy"].cummax()
    drawdown = (bt_df["Cum_Strategy"] - peak) / peak
    max_drawdown = drawdown.min() * 100

    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    b_col1.metric("Strategy Return", f"{total_strat_ret:.2f}%")
    b_col2.metric("Buy & Hold Return", f"{total_bench_ret:.2f}%")
    b_col3.metric("Win Rate", f"{win_rate:.1f}%")
    b_col4.metric("Max Drawdown", f"{max_drawdown:.2f}%")

    # Chart
    bt_fig = go.Figure()
    bt_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["Cum_Strategy"], name="Multi-Factor Strategy", line=dict(color="cyan")))
    bt_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["Cum_Benchmark"], name="Buy & Hold Benchmark", line=dict(color="gray", dash="dash")))
    bt_fig.update_layout(title="Cumulative Returns Comparison", template="plotly_dark", height=400)
    st.plotly_chart(bt_fig, use_container_width=True)
