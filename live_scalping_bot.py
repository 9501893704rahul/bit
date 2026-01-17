"""
Bitcoin Ultra Conservative Scalping - Live Market Signal Generator
===================================================================
Real-time scalping signals based on the Ultra Conservative strategy
that achieved 92.31% win rate in backtesting.

Strategy Parameters:
- Target: 0.08% profit
- Stop Loss: 1.0% loss
- Entry Conditions:
  * RSI < 35 (oversold)
  * Stochastic K < 25 (oversold)
  * Price near lower Bollinger Band (BB Position < 0.2)
  * Price above 200 EMA (uptrend filter)
  * MACD Histogram improving

⚠️ DISCLAIMER: This is for EDUCATIONAL purposes only.
   Do NOT use real money without proper risk management.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import threading
from flask import Flask, render_template_string, jsonify
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global state
live_data = {
    'current_price': 0,
    'indicators': {},
    'signal': None,
    'position': None,
    'trades': [],
    'last_update': None,
    'candles': []
}

def fetch_live_data():
    """Fetch latest Bitcoin data"""
    try:
        btc = yf.Ticker("BTC-USD")
        # Fetch more data to ensure we have enough for 200 EMA
        df = btc.history(period="60d", interval="1h")
        df = df.reset_index()
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        print(f"Fetched {len(df)} candles")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
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

def check_ultra_conservative_signal(df):
    """Check for Ultra Conservative strategy entry signal"""
    if len(df) < 201:
        return None, {}
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    indicators = {
        'price': round(float(current['Close']), 2),
        'rsi': round(float(current['RSI']), 2),
        'stoch_k': round(float(current['Stoch_K']), 2),
        'stoch_d': round(float(current['Stoch_D']), 2),
        'bb_position': round(float(current['BB_Position']), 3),
        'bb_lower': round(float(current['BB_Lower']), 2),
        'bb_upper': round(float(current['BB_Upper']), 2),
        'bb_middle': round(float(current['BB_Middle']), 2),
        'ema_200': round(float(current['EMA_200']), 2),
        'ema_50': round(float(current['EMA_50']), 2),
        'macd': round(float(current['MACD']), 2),
        'macd_signal': round(float(current['MACD_Signal']), 2),
        'macd_hist': round(float(current['MACD_Hist']), 2),
        'macd_hist_prev': round(float(prev['MACD_Hist']), 2),
        'macd_improving': bool(current['MACD_Hist'] > prev['MACD_Hist'])
    }
    
    # Check conditions
    conditions = {
        'rsi_oversold': bool(current['RSI'] < 35),
        'stoch_oversold': bool(current['Stoch_K'] < 25),
        'near_bb_lower': bool(current['BB_Position'] < 0.2),
        'above_ema200': bool(current['Close'] > current['EMA_200']),
        'macd_improving': bool(current['MACD_Hist'] > prev['MACD_Hist'])
    }
    
    indicators['conditions'] = conditions
    indicators['conditions_met'] = int(sum(conditions.values()))
    indicators['total_conditions'] = int(len(conditions))
    
    # All conditions must be met for a signal
    if all(conditions.values()):
        entry_price = current['Close']
        target = entry_price * 1.0008  # 0.08% target
        stop_loss = entry_price * 0.99  # 1% stop loss
        
        signal = {
            'type': 'LONG',
            'entry_price': round(float(entry_price), 2),
            'target': round(float(target), 2),
            'stop_loss': round(float(stop_loss), 2),
            'target_pct': 0.08,
            'stop_loss_pct': 1.0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return signal, indicators
    
    return None, indicators

def update_position(current_price):
    """Update position status if we have an open position"""
    global live_data
    
    if live_data['position'] is None:
        return
    
    pos = live_data['position']
    
    # Check if target hit
    if current_price >= pos['target']:
        pnl = (pos['target'] - pos['entry_price']) / pos['entry_price'] * 100
        trade = {
            'type': 'LONG',
            'entry_price': pos['entry_price'],
            'exit_price': pos['target'],
            'entry_time': pos['entry_time'],
            'exit_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pnl_pct': round(pnl, 3),
            'exit_reason': 'TARGET',
            'is_winner': True
        }
        live_data['trades'].insert(0, trade)
        live_data['position'] = None
        return
    
    # Check if stop loss hit
    if current_price <= pos['stop_loss']:
        pnl = (pos['stop_loss'] - pos['entry_price']) / pos['entry_price'] * 100
        trade = {
            'type': 'LONG',
            'entry_price': pos['entry_price'],
            'exit_price': pos['stop_loss'],
            'entry_time': pos['entry_time'],
            'exit_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pnl_pct': round(pnl, 3),
            'exit_reason': 'STOP_LOSS',
            'is_winner': False
        }
        live_data['trades'].insert(0, trade)
        live_data['position'] = None

def data_updater():
    """Background thread to update data"""
    global live_data
    
    while True:
        try:
            df = fetch_live_data()
            if df is not None and len(df) > 0:
                df = calculate_indicators(df)
                
                current_price = float(df.iloc[-1]['Close'])
                live_data['current_price'] = round(current_price, 2)
                
                # Update position if exists
                update_position(current_price)
                
                # Check for new signal (only if no position)
                if live_data['position'] is None:
                    signal, indicators = check_ultra_conservative_signal(df)
                    live_data['signal'] = signal
                    live_data['indicators'] = indicators
                    
                    # Auto-enter position on signal (paper trading)
                    if signal:
                        live_data['position'] = {
                            'type': 'LONG',
                            'entry_price': signal['entry_price'],
                            'target': signal['target'],
                            'stop_loss': signal['stop_loss'],
                            'entry_time': signal['timestamp']
                        }
                        live_data['signal'] = None
                else:
                    # Still update indicators
                    _, indicators = check_ultra_conservative_signal(df)
                    live_data['indicators'] = indicators
                
                # Store recent candles for chart
                recent = df.tail(50)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                recent['Date'] = recent['Date'].astype(str)
                live_data['candles'] = recent.to_dict('records')
                
                live_data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            time.sleep(60)  # Update every minute
            
        except Exception as e:
            print(f"Error in data updater: {e}")
            time.sleep(30)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC Ultra Conservative Scalping - LIVE</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        
        h1 {
            text-align: center;
            font-size: 2.5em;
            background: linear-gradient(90deg, #f7931a, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }
        .subtitle { text-align: center; color: #888; margin-bottom: 20px; }
        
        .live-badge {
            display: inline-block;
            background: #ff4757;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8em;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .disclaimer {
            background: rgba(255, 193, 7, 0.15);
            border: 1px solid rgba(255, 193, 7, 0.4);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            text-align: center;
            color: #ffc107;
        }
        
        .price-box {
            background: linear-gradient(135deg, rgba(247, 147, 26, 0.2), rgba(0, 255, 136, 0.1));
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            margin-bottom: 20px;
            border: 2px solid rgba(247, 147, 26, 0.3);
        }
        .price-box h2 {
            font-size: 3.5em;
            color: #f7931a;
            margin-bottom: 10px;
        }
        .price-box .update-time { color: #888; font-size: 0.9em; }
        
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
        
        .card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .card h3 {
            color: #f7931a;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        
        .indicator-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .indicator-label { color: #888; }
        .indicator-value { font-weight: bold; }
        .bullish { color: #00ff88; }
        .bearish { color: #ff4757; }
        .neutral { color: #f7931a; }
        
        .condition-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.02);
        }
        .condition-met { background: rgba(0, 255, 136, 0.15); border-left: 3px solid #00ff88; }
        .condition-not-met { background: rgba(255, 71, 87, 0.15); border-left: 3px solid #ff4757; }
        
        .signal-box {
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.2), rgba(0, 255, 136, 0.05));
            border: 2px solid #00ff88;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            margin-bottom: 20px;
        }
        .signal-box.no-signal {
            background: rgba(255, 255, 255, 0.03);
            border-color: rgba(255, 255, 255, 0.1);
        }
        .signal-box h3 { font-size: 1.5em; margin-bottom: 10px; }
        .signal-box .signal-type { font-size: 2em; color: #00ff88; font-weight: bold; }
        
        .position-box {
            background: linear-gradient(135deg, rgba(247, 147, 26, 0.2), rgba(247, 147, 26, 0.05));
            border: 2px solid #f7931a;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
        }
        .position-box h3 { color: #f7931a; margin-bottom: 15px; }
        .position-detail { display: flex; justify-content: space-between; padding: 8px 0; }
        
        .pnl-positive { color: #00ff88; font-size: 1.5em; font-weight: bold; }
        .pnl-negative { color: #ff4757; font-size: 1.5em; font-weight: bold; }
        
        .trades-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }
        .trades-table th, .trades-table td {
            padding: 12px;
            text-align: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }
        .trades-table th {
            background: rgba(247, 147, 26, 0.15);
            color: #f7931a;
        }
        
        .progress-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff4757, #ffa502, #00ff88);
            border-radius: 4px;
            transition: width 0.3s;
        }
        
        .strategy-info {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            font-size: 0.9em;
            color: #aaa;
        }
        .strategy-info strong { color: #f7931a; }
    </style>
</head>
<body>
    <div class="container">
        <h1>₿ BTC Ultra Conservative Scalping</h1>
        <p class="subtitle">
            <span class="live-badge">🔴 LIVE</span>
            Real-time signals based on 92.31% win rate strategy
        </p>
        
        <div class="disclaimer">
            ⚠️ <strong>PAPER TRADING ONLY</strong> - This is for educational purposes. 
            Do NOT use real money without proper risk management and understanding of the risks involved.
        </div>
        
        <div class="strategy-info">
            <strong>Strategy:</strong> Ultra Conservative Scalping | 
            <strong>Target:</strong> 0.08% | 
            <strong>Stop Loss:</strong> 1.0% | 
            <strong>Backtest Win Rate:</strong> 92.31% | 
            <strong>Timeframe:</strong> 1H
        </div>
        
        <div class="price-box">
            <h2 id="currentPrice">$--,---</h2>
            <p>Bitcoin (BTC/USD)</p>
            <p class="update-time">Last update: <span id="lastUpdate">--</span></p>
        </div>
        
        <div id="positionBox" style="display: none;" class="position-box">
            <h3>📊 Active Position</h3>
            <div class="position-detail">
                <span>Entry Price:</span>
                <span id="posEntry">$--</span>
            </div>
            <div class="position-detail">
                <span>Target (0.08%):</span>
                <span id="posTarget" class="bullish">$--</span>
            </div>
            <div class="position-detail">
                <span>Stop Loss (1.0%):</span>
                <span id="posStopLoss" class="bearish">$--</span>
            </div>
            <div class="position-detail">
                <span>Current P&L:</span>
                <span id="posPnl">--</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="pnlProgress" style="width: 50%;"></div>
            </div>
        </div>
        
        <div id="signalBox" class="signal-box no-signal">
            <h3>Signal Status</h3>
            <div id="signalContent">
                <p style="color: #888;">Waiting for entry conditions...</p>
                <p id="conditionsMet" style="margin-top: 10px;">Conditions: 0/5</p>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📈 Entry Conditions</h3>
                <div id="conditionsList"></div>
            </div>
            
            <div class="card">
                <h3>📊 Technical Indicators</h3>
                <div class="indicator-row">
                    <span class="indicator-label">RSI (14)</span>
                    <span class="indicator-value" id="rsiValue">--</span>
                </div>
                <div class="indicator-row">
                    <span class="indicator-label">Stochastic K</span>
                    <span class="indicator-value" id="stochKValue">--</span>
                </div>
                <div class="indicator-row">
                    <span class="indicator-label">Stochastic D</span>
                    <span class="indicator-value" id="stochDValue">--</span>
                </div>
                <div class="indicator-row">
                    <span class="indicator-label">BB Position</span>
                    <span class="indicator-value" id="bbPosValue">--</span>
                </div>
                <div class="indicator-row">
                    <span class="indicator-label">MACD Histogram</span>
                    <span class="indicator-value" id="macdHistValue">--</span>
                </div>
                <div class="indicator-row">
                    <span class="indicator-label">EMA 200</span>
                    <span class="indicator-value" id="ema200Value">--</span>
                </div>
            </div>
            
            <div class="card">
                <h3>📉 Bollinger Bands</h3>
                <div class="indicator-row">
                    <span class="indicator-label">Upper Band</span>
                    <span class="indicator-value neutral" id="bbUpper">--</span>
                </div>
                <div class="indicator-row">
                    <span class="indicator-label">Middle Band</span>
                    <span class="indicator-value" id="bbMiddle">--</span>
                </div>
                <div class="indicator-row">
                    <span class="indicator-label">Lower Band</span>
                    <span class="indicator-value neutral" id="bbLower">--</span>
                </div>
                <div class="indicator-row">
                    <span class="indicator-label">Price vs EMA200</span>
                    <span class="indicator-value" id="priceVsEma">--</span>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>📋 Recent Trades (Paper)</h3>
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Entry Time</th>
                        <th>Exit Time</th>
                        <th>Entry</th>
                        <th>Exit</th>
                        <th>P&L</th>
                        <th>Reason</th>
                        <th>Result</th>
                    </tr>
                </thead>
                <tbody id="tradesBody">
                    <tr><td colspan="7" style="color: #888;">No trades yet</td></tr>
                </tbody>
            </table>
        </div>
        
        <div class="card" style="margin-top: 20px;">
            <h3>📊 Price Chart (Last 50 Candles)</h3>
            <div style="height: 300px;">
                <canvas id="priceChart"></canvas>
            </div>
        </div>
    </div>
    
    <script>
        let priceChart = null;
        
        async function fetchData() {
            try {
                const response = await fetch('/api/live');
                const data = await response.json();
                updateUI(data);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }
        
        function updateUI(data) {
            // Update price
            document.getElementById('currentPrice').textContent = '$' + data.current_price.toLocaleString();
            document.getElementById('lastUpdate').textContent = data.last_update || '--';
            
            // Update indicators
            const ind = data.indicators;
            if (ind && ind.rsi !== undefined) {
                document.getElementById('rsiValue').textContent = ind.rsi;
                document.getElementById('rsiValue').className = 'indicator-value ' + (ind.rsi < 35 ? 'bullish' : (ind.rsi > 65 ? 'bearish' : ''));
                
                document.getElementById('stochKValue').textContent = ind.stoch_k;
                document.getElementById('stochKValue').className = 'indicator-value ' + (ind.stoch_k < 25 ? 'bullish' : (ind.stoch_k > 75 ? 'bearish' : ''));
                
                document.getElementById('stochDValue').textContent = ind.stoch_d;
                document.getElementById('bbPosValue').textContent = ind.bb_position;
                document.getElementById('bbPosValue').className = 'indicator-value ' + (ind.bb_position < 0.2 ? 'bullish' : (ind.bb_position > 0.8 ? 'bearish' : ''));
                
                document.getElementById('macdHistValue').textContent = ind.macd_hist + (ind.macd_improving ? ' ↑' : ' ↓');
                document.getElementById('macdHistValue').className = 'indicator-value ' + (ind.macd_improving ? 'bullish' : 'bearish');
                
                document.getElementById('ema200Value').textContent = '$' + ind.ema_200.toLocaleString();
                document.getElementById('bbUpper').textContent = '$' + ind.bb_upper.toLocaleString();
                document.getElementById('bbMiddle').textContent = '$' + ind.bb_middle.toLocaleString();
                document.getElementById('bbLower').textContent = '$' + ind.bb_lower.toLocaleString();
                
                const aboveEma = ind.price > ind.ema_200;
                document.getElementById('priceVsEma').textContent = aboveEma ? 'ABOVE ✓' : 'BELOW ✗';
                document.getElementById('priceVsEma').className = 'indicator-value ' + (aboveEma ? 'bullish' : 'bearish');
                
                // Update conditions
                if (ind.conditions) {
                    const conditionsHtml = `
                        <div class="condition-item ${ind.conditions.rsi_oversold ? 'condition-met' : 'condition-not-met'}">
                            <span>RSI < 35</span>
                            <span>${ind.conditions.rsi_oversold ? '✓' : '✗'} (${ind.rsi})</span>
                        </div>
                        <div class="condition-item ${ind.conditions.stoch_oversold ? 'condition-met' : 'condition-not-met'}">
                            <span>Stochastic K < 25</span>
                            <span>${ind.conditions.stoch_oversold ? '✓' : '✗'} (${ind.stoch_k})</span>
                        </div>
                        <div class="condition-item ${ind.conditions.near_bb_lower ? 'condition-met' : 'condition-not-met'}">
                            <span>BB Position < 0.2</span>
                            <span>${ind.conditions.near_bb_lower ? '✓' : '✗'} (${ind.bb_position})</span>
                        </div>
                        <div class="condition-item ${ind.conditions.above_ema200 ? 'condition-met' : 'condition-not-met'}">
                            <span>Price > EMA 200</span>
                            <span>${ind.conditions.above_ema200 ? '✓' : '✗'}</span>
                        </div>
                        <div class="condition-item ${ind.conditions.macd_improving ? 'condition-met' : 'condition-not-met'}">
                            <span>MACD Improving</span>
                            <span>${ind.conditions.macd_improving ? '✓' : '✗'}</span>
                        </div>
                    `;
                    document.getElementById('conditionsList').innerHTML = conditionsHtml;
                    document.getElementById('conditionsMet').textContent = `Conditions: ${ind.conditions_met}/${ind.total_conditions}`;
                }
            }
            
            // Update position
            const posBox = document.getElementById('positionBox');
            if (data.position) {
                posBox.style.display = 'block';
                document.getElementById('posEntry').textContent = '$' + data.position.entry_price.toLocaleString();
                document.getElementById('posTarget').textContent = '$' + data.position.target.toLocaleString();
                document.getElementById('posStopLoss').textContent = '$' + data.position.stop_loss.toLocaleString();
                
                const pnl = ((data.current_price - data.position.entry_price) / data.position.entry_price * 100).toFixed(3);
                document.getElementById('posPnl').textContent = pnl + '%';
                document.getElementById('posPnl').className = parseFloat(pnl) >= 0 ? 'pnl-positive' : 'pnl-negative';
                
                // Progress bar (from -1% to +0.08%)
                const progress = Math.min(100, Math.max(0, (parseFloat(pnl) + 1) / 1.08 * 100));
                document.getElementById('pnlProgress').style.width = progress + '%';
                
                // Update signal box
                document.getElementById('signalBox').className = 'signal-box';
                document.getElementById('signalContent').innerHTML = `
                    <div class="signal-type">📈 LONG POSITION ACTIVE</div>
                    <p style="margin-top: 10px;">Entry: $${data.position.entry_price.toLocaleString()}</p>
                `;
            } else {
                posBox.style.display = 'none';
                document.getElementById('signalBox').className = 'signal-box no-signal';
                document.getElementById('signalContent').innerHTML = `
                    <p style="color: #888;">Waiting for entry conditions...</p>
                    <p id="conditionsMet" style="margin-top: 10px;">Conditions: ${ind?.conditions_met || 0}/${ind?.total_conditions || 5}</p>
                `;
            }
            
            // Update trades
            if (data.trades && data.trades.length > 0) {
                const tradesHtml = data.trades.slice(0, 10).map(t => `
                    <tr>
                        <td>${t.entry_time}</td>
                        <td>${t.exit_time}</td>
                        <td>$${t.entry_price.toLocaleString()}</td>
                        <td>$${t.exit_price.toLocaleString()}</td>
                        <td class="${t.pnl_pct >= 0 ? 'bullish' : 'bearish'}">${t.pnl_pct}%</td>
                        <td>${t.exit_reason}</td>
                        <td class="${t.is_winner ? 'bullish' : 'bearish'}">${t.is_winner ? '✅ WIN' : '❌ LOSS'}</td>
                    </tr>
                `).join('');
                document.getElementById('tradesBody').innerHTML = tradesHtml;
            }
            
            // Update chart
            if (data.candles && data.candles.length > 0) {
                updateChart(data.candles);
            }
        }
        
        function updateChart(candles) {
            const ctx = document.getElementById('priceChart').getContext('2d');
            
            if (priceChart) {
                priceChart.destroy();
            }
            
            priceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: candles.map(c => c.Date.split(' ')[1] || c.Date.split('T')[1]?.substring(0,5) || ''),
                    datasets: [{
                        label: 'BTC Price',
                        data: candles.map(c => c.Close),
                        borderColor: '#f7931a',
                        backgroundColor: 'rgba(247, 147, 26, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            ticks: { color: '#888' }
                        },
                        x: {
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            ticks: { color: '#888', maxTicksLimit: 10 }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }
        
        // Initial fetch and set interval
        fetchData();
        setInterval(fetchData, 5000);  // Update every 5 seconds
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/live')
def get_live_data():
    return jsonify(live_data)

def main():
    print("=" * 60)
    print("BITCOIN ULTRA CONSERVATIVE SCALPING - LIVE SIGNALS")
    print("=" * 60)
    print("\n⚠️  PAPER TRADING MODE - For educational purposes only!")
    print("\nStrategy Parameters:")
    print("  - Target: 0.08%")
    print("  - Stop Loss: 1.0%")
    print("  - Backtest Win Rate: 92.31%")
    print("\nStarting data updater thread...")
    
    # Start background data updater
    updater_thread = threading.Thread(target=data_updater, daemon=True)
    updater_thread.start()
    
    print("\nStarting web server...")
    print("Access the dashboard at: http://0.0.0.0:12001")
    
    app.run(host='0.0.0.0', port=12001, debug=False, threaded=True)

if __name__ == '__main__':
    main()
