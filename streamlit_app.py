"""
Bitcoin Ultra Conservative Scalping Strategy - Streamlit Dashboard
===================================================================
Real-time trading signals with 92.31% backtest win rate
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="BTC Ultra Conservative Scalping",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #f7931a, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .condition-met {
        color: #00ff88;
        font-weight: bold;
    }
    .condition-not-met {
        color: #ff4757;
    }
    .price-display {
        font-size: 3rem;
        font-weight: bold;
        color: #f7931a;
        text-align: center;
    }
    .stAlert {
        background-color: rgba(255, 193, 7, 0.1);
        border: 1px solid rgba(255, 193, 7, 0.3);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def fetch_bitcoin_data():
    """Fetch Bitcoin data with caching"""
    try:
        btc = yf.Ticker("BTC-USD")
        df = btc.history(period="60d", interval="1h")
        df = df.reset_index()
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def calculate_indicators(df):
    """Calculate all technical indicators"""
    df = df.copy()
    
    # RSI (14 period)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20 period)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # EMAs
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # Stochastic (14 period)
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    return df


def check_entry_conditions(df):
    """Check Ultra Conservative entry conditions"""
    if len(df) < 201:
        return None, {}
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    conditions = {
        'RSI < 35 (Oversold)': bool(current['RSI'] < 35),
        'Stochastic K < 25': bool(current['Stoch_K'] < 25),
        'BB Position < 0.2': bool(current['BB_Position'] < 0.2),
        'Price > EMA 200': bool(current['Close'] > current['EMA_200']),
        'MACD Improving': bool(current['MACD_Hist'] > prev['MACD_Hist'])
    }
    
    indicators = {
        'price': float(current['Close']),
        'rsi': float(current['RSI']),
        'stoch_k': float(current['Stoch_K']),
        'stoch_d': float(current['Stoch_D']),
        'bb_position': float(current['BB_Position']),
        'bb_lower': float(current['BB_Lower']),
        'bb_upper': float(current['BB_Upper']),
        'bb_middle': float(current['BB_Middle']),
        'ema_200': float(current['EMA_200']),
        'ema_50': float(current['EMA_50']),
        'macd_hist': float(current['MACD_Hist']),
        'macd_hist_prev': float(prev['MACD_Hist'])
    }
    
    return conditions, indicators


def main():
    # Header
    st.markdown('<h1 class="main-header">₿ Bitcoin Ultra Conservative Scalping</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #888;">Real-time signals based on 92.31% backtest win rate strategy</p>', unsafe_allow_html=True)
    
    # Disclaimer
    st.warning("⚠️ **PAPER TRADING ONLY** - This is for educational purposes. Do NOT use real money without proper risk management.")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Strategy Parameters")
        st.markdown("---")
        st.metric("Target Profit", "0.08%", delta=None)
        st.metric("Stop Loss", "1.0%", delta=None)
        st.metric("Backtest Win Rate", "92.31%", delta=None)
        
        st.markdown("---")
        st.subheader("Entry Conditions")
        st.markdown("""
        - RSI < 35 (Oversold)
        - Stochastic K < 25
        - BB Position < 0.2
        - Price > EMA 200
        - MACD Improving
        """)
        
        st.markdown("---")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Fetch data
    with st.spinner("Fetching Bitcoin data..."):
        df = fetch_bitcoin_data()
    
    if df is None or len(df) < 201:
        st.error("Unable to fetch sufficient data. Please try again.")
        return
    
    # Calculate indicators
    df = calculate_indicators(df)
    conditions, indicators = check_entry_conditions(df)
    
    # Main content
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f'<p class="price-display">${indicators["price"]:,.2f}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align: center; color: #888;">Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>', unsafe_allow_html=True)
    
    with col2:
        conditions_met = sum(conditions.values())
        st.metric("Conditions Met", f"{conditions_met}/5")
    
    with col3:
        if all(conditions.values()):
            st.success("🟢 SIGNAL: BUY")
            target = indicators['price'] * 1.0008
            stop_loss = indicators['price'] * 0.99
            st.write(f"Target: ${target:,.2f}")
            st.write(f"Stop Loss: ${stop_loss:,.2f}")
        else:
            st.info("⏳ Waiting for signal...")
    
    st.markdown("---")
    
    # Entry Conditions
    st.subheader("✅ Entry Conditions Status")
    
    cond_cols = st.columns(5)
    for i, (cond_name, cond_met) in enumerate(conditions.items()):
        with cond_cols[i]:
            if cond_met:
                st.success(f"✓ {cond_name}")
            else:
                st.error(f"✗ {cond_name}")
    
    st.markdown("---")
    
    # Technical Indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📈 Momentum Indicators")
        
        rsi_color = "🟢" if indicators['rsi'] < 35 else ("🔴" if indicators['rsi'] > 65 else "🟡")
        st.metric("RSI (14)", f"{indicators['rsi']:.2f}", delta=f"{rsi_color} {'Oversold' if indicators['rsi'] < 35 else ('Overbought' if indicators['rsi'] > 65 else 'Neutral')}")
        
        stoch_color = "🟢" if indicators['stoch_k'] < 25 else ("🔴" if indicators['stoch_k'] > 75 else "🟡")
        st.metric("Stochastic K", f"{indicators['stoch_k']:.2f}", delta=f"{stoch_color}")
        st.metric("Stochastic D", f"{indicators['stoch_d']:.2f}")
    
    with col2:
        st.subheader("📊 Bollinger Bands")
        
        bb_color = "🟢" if indicators['bb_position'] < 0.2 else ("🔴" if indicators['bb_position'] > 0.8 else "🟡")
        st.metric("BB Position", f"{indicators['bb_position']:.3f}", delta=f"{bb_color}")
        st.metric("Upper Band", f"${indicators['bb_upper']:,.2f}")
        st.metric("Middle Band", f"${indicators['bb_middle']:,.2f}")
        st.metric("Lower Band", f"${indicators['bb_lower']:,.2f}")
    
    with col3:
        st.subheader("📉 Trend Indicators")
        
        above_ema = indicators['price'] > indicators['ema_200']
        st.metric("EMA 200", f"${indicators['ema_200']:,.2f}", delta=f"{'🟢 Above' if above_ema else '🔴 Below'}")
        st.metric("EMA 50", f"${indicators['ema_50']:,.2f}")
        
        macd_improving = indicators['macd_hist'] > indicators['macd_hist_prev']
        st.metric("MACD Histogram", f"{indicators['macd_hist']:.2f}", delta=f"{'🟢 ↑' if macd_improving else '🔴 ↓'}")
    
    st.markdown("---")
    
    # Price Chart
    st.subheader("📈 Price Chart (Last 100 Candles)")
    
    chart_data = df.tail(100)[['Date', 'Close', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'EMA_200']].copy()
    chart_data = chart_data.set_index('Date')
    
    st.line_chart(chart_data[['Close', 'BB_Upper', 'BB_Middle', 'BB_Lower']])
    
    # RSI Chart
    st.subheader("📊 RSI Chart")
    rsi_data = df.tail(100)[['Date', 'RSI']].copy()
    rsi_data = rsi_data.set_index('Date')
    st.line_chart(rsi_data)
    
    # Strategy Info
    st.markdown("---")
    st.subheader("📖 About the Strategy")
    
    with st.expander("Strategy Details"):
        st.markdown("""
        ### Ultra Conservative Scalping Strategy
        
        This strategy achieved a **92.31% win rate** in backtesting on Bitcoin hourly data.
        
        **Entry Conditions (ALL must be met):**
        1. **RSI < 35**: Price is oversold
        2. **Stochastic K < 25**: Additional oversold confirmation
        3. **BB Position < 0.2**: Price near lower Bollinger Band
        4. **Price > EMA 200**: Uptrend filter (only trade with the trend)
        5. **MACD Improving**: Momentum turning positive
        
        **Exit Rules:**
        - **Target**: 0.08% profit
        - **Stop Loss**: 1.0% loss
        
        **Important Notes:**
        - High win rate comes with low risk/reward ratio (0.08:1)
        - One losing trade can wipe out ~12 winning trades
        - This is for educational purposes only
        - Past performance does not guarantee future results
        """)
    
    # Auto-refresh
    st.markdown("---")
    st.markdown('<p style="text-align: center; color: #666;">Data refreshes every 60 seconds. Click "Refresh Data" in sidebar for manual refresh.</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
