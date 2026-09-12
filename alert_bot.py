import pandas as pd
import numpy as np
import yfinance as yf
import requests

# 1. Configured Tickers & Discord Webhook URL
WATCHLIST = ["SUZLON.NS", "RELIANCE.NS", "TATAMOTORS.NS", "BTC-USD"]
DISCORD_WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL_HERE"

def check_signals_and_alert():
    print("🤖 Checking market signals...")
    
    for ticker in WATCHLIST:
        # Download recent data
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty or len(df) < 200:
            continue
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Calculate Moving Averages
        df["50_DMA"] = df["Close"].rolling(50).mean()
        df["200_DMA"] = df["Close"].rolling(200).mean()

        # Get latest 2 days to check for exact crossover events
        today_50 = df["50_DMA"].iloc[-1]
        today_200 = df["200_DMA"].iloc[-1]
        yesterday_50 = df["50_DMA"].iloc[-2]
        yesterday_200 = df["200_DMA"].iloc[-2]
        price = df["Close"].iloc[-1]

        signal_event = None

        # Detect Golden Cross (Buy Signal)
        if yesterday_50 <= yesterday_200 and today_50 > today_200:
            signal_event = "🚀 **GOLDEN CROSS (BULLISH BUY)**"

        # Detect Death Cross (Sell Signal)
        elif yesterday_50 >= yesterday_200 and today_50 < today_200:
            signal_event = "🔴 **DEATH CROSS (BEARISH SELL)**"

        # If a crossover occurred today, send alert
        if signal_event:
            message = {
                "content": f"{signal_event}\n**Asset:** {ticker}\n**Price:** {price:.2f}\n**50-DMA:** {today_50:.2f} | **200-DMA:** {today_200:.2f}"
            }
            requests.post(DISCORD_WEBHOOK_URL, json=message)
            print(f"Sent alert for {ticker}")

if __name__ == "__main__":
    check_signals_and_alert()
