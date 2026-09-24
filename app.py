import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from arch import arch_model
import google.generativeai as genai
from fpdf import FPDF

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Institutional Quant & AI Terminal", layout="wide")

st.title("Institutional Quant & AI Terminal")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Configuration")
ticker = st.sidebar.selectbox("Select Asset Ticker", ["BTC-USD", "ETH-USD", "^GSPC", "SUZLON.NS"], index=0)
horizon = st.sidebar.selectbox("Data Horizon", ["1y", "2y", "5y"], index=1)

# --- SECURE GEMINI API KEY LOADING ---
# Checks Streamlit Cloud Secrets first; falls back to manual user input if secrets are missing.
api_key = st.secrets.get("GEMINI_API_KEY", "")

if not api_key:
    api_key = st.sidebar.text_input("🔑 Gemini API Key", type="password", help="Enter your key or set it in Streamlit Secrets.")

# --- NAVIGATION TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Market Matrix", 
    "📈 Strategy Backtest", 
    "🤖 AI Technical Signal", 
    "🧮 Value at Risk & Position Sizer", 
    "🎲 AI Monte Carlo Engine"
])

# --- TAB 3: AI TECHNICAL ANALYSIS LOGIC ---
with tab3:
    st.header("AI Technical Analysis")
    
    if not api_key:
        st.warning("⚠️ Please configure your GEMINI_API_KEY in Streamlit Secrets or enter it in the sidebar.")
    else:
        # Configure Gemini API dynamically
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            if st.button("Run AI Technical Engine"):
                with st.spinner("Generating institutional audit..."):
                    prompt = f"""
                    You are a Chief Risk Officer at a quantitative hedge fund. 
                    Analyze the asset {ticker} over a horizon of {horizon}. 
                    Provide an executive summary focusing on volatility structure, risk posture, and position management.
                    """
                    response = model.generate_content(prompt)
                    st.markdown("### 📋 Executive Audit Summary")
                    st.write(response.text)
        except Exception as e:
            st.error(f"Failed to connect to Gemini API: {e}")
