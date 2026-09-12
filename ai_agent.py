import os
import json
import requests
import yfinance as yf
import pandas as pd
from openai import OpenAI

# Initialize client (Set OPENAI_API_KEY or DEEPSEEK_API_KEY in system environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_asset_context(ticker):
    """Fetches technical context for the AI Agent."""
    df = yf.download(ticker, period="6m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df["50_DMA"] = df["Close"].rolling(50).mean()
    df["200_DMA"] = df["Close"].rolling(200).mean()

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / loss)))

    latest = df.iloc[-1]
    
    return {
        "ticker": ticker,
        "price": float(latest["Close"]),
        "sma_50": float(latest["50_DMA"]),
        "sma_200": float(latest["200_DMA"]),
        "rsi": float(latest["RSI"]),
        "regime": "Bullish" if latest["50_DMA"] > latest["200_DMA"] else "Bearish"
    }

def run_ai_research_agent(ticker):
    """Queries the LLM for investment research and cash-out signals."""
    context = get_asset_context(ticker)

    prompt = f"""
    You are an expert Quantitative Investment Analyst. Analyze the following market data for asset: {context['ticker']}

    Technical Context:
    - Current Price: {context['price']}
    - 50-Day Moving Average: {context['sma_50']}
    - 200-Day Moving Average: {context['sma_200']}
    - RSI (14): {context['rsi']}
    - Technical Regime: {context['regime']}

    Task:
    Provide an autonomous decision whether to INVEST (BUY), CASH OUT (SELL/TAKE PROFIT), or HOLD (STAY IN CASH/POSITION).

    Return ONLY a raw JSON object with this structure:
    {{
        "action": "BUY" | "CASH_OUT" | "HOLD",
        "confidence": 85,
        "summary": "1-2 sentence executive summary of the research reasoning."
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    decision = json.loads(response.choices[0].message.content)
    return context, decision

if __name__ == "__main__":
    context, decision = run_ai_research_agent("SUZLON.NS")
    print("Market Context:", context)
    print("AI Agent Decision:", decision)
