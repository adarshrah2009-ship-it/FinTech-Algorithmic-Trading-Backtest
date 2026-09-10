# FinTech-Algorithmic-Trading-Backtest
"Python-based backtest comparing Single 50-DMA and Dual 50/200-DMA crossover strategies against Buy &amp; Hold for Indian equities."
# Quantitative Equity Backtest: Dual Moving Average Crossover

## Overview
An algorithmic trading backtest built in Python analyzing historical price data for Indian equities (NSE: RELIANCE). The project compares a single 50-Day Moving Average strategy against a Dual (50/200-DMA) Golden Cross strategy and a traditional Buy & Hold benchmark.

## Key Insights & Results
- **Single 50-DMA Failure:** High trade frequency during sideways consolidation led to performance drag due to whipsawing.
- **Dual Crossover Alpha:** Using a 50/200-DMA filter eliminated false signals, outperforming Buy & Hold by **+3.47%** over a 3-year backtest horizon by preserving capital during prolonged bear phases.

## Tech & Libraries
- Python, Pandas, NumPy, Matplotlib, `yfinance`, `curl_cffi`
