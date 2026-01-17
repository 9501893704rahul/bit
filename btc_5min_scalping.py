"""
Bitcoin 5-Minute Scalping Strategy Finder
==========================================
Finding the best scalping strategy on 5-minute timeframe for maximum win rate.

Strategies Tested:
1. RSI Extreme Reversal
2. Bollinger Band Bounce
3. EMA Pullback
4. Stochastic Oversold/Overbought
5. MACD Momentum
6. Price Action (Engulfing)
7. Multi-Indicator Confluence
8. Ultra Conservative (Tight Target, Wide Stop)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')


def fetch_bitcoin_5min_data():
    """Fetch Bitcoin 5-minute data (max 60 days available)"""
    print("Fetching Bitcoin 5-minute data...")
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period="60d", interval="5m")
    df = df.reset_index()
    if 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'Date'})
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    print(f"Fetched {len(df)} candles from {df['Date'].min()} to {df['Date'].max()}")
    return df


def calculate_indicators(df):
    """Calculate all technical indicators"""
    df = df.copy()
    
    # RSI with multiple periods
    for period in [7, 9, 14, 21]:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    for period in [10, 20]:
        df[f'BB_Middle_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'BB_Std_{period}'] = df['Close'].rolling(window=period).std()
        df[f'BB_Upper_{period}'] = df[f'BB_Middle_{period}'] + (df[f'BB_Std_{period}'] * 2)
        df[f'BB_Lower_{period}'] = df[f'BB_Middle_{period}'] - (df[f'BB_Std_{period}'] * 2)
        df[f'BB_Position_{period}'] = (df['Close'] - df[f'BB_Lower_{period}']) / (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'])
    
    # EMAs
    for period in [5, 8, 13, 21, 34, 55, 89, 144, 200]:
        df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    
    # SMAs
    for period in [10, 20, 50]:
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
    
    # Stochastic
    for period in [5, 9, 14]:
        low_n = df['Low'].rolling(window=period).min()
        high_n = df['High'].rolling(window=period).max()
        df[f'Stoch_K_{period}'] = 100 * (df['Close'] - low_n) / (high_n - low_n)
        df[f'Stoch_D_{period}'] = df[f'Stoch_K_{period}'].rolling(window=3).mean()
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['ATR_Pct'] = df['ATR'] / df['Close'] * 100
    
    # Volume
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price momentum
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['ROC_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
    
    # Candle patterns
    df['Body'] = df['Close'] - df['Open']
    df['Body_Pct'] = abs(df['Body']) / df['Open'] * 100
    df['Is_Bullish'] = df['Close'] > df['Open']
    df['Is_Bearish'] = df['Close'] < df['Open']
    
    # Previous candle
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Open'] = df['Open'].shift(1)
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    df['Prev_Is_Bullish'] = df['Is_Bullish'].shift(1)
    df['Prev_Is_Bearish'] = df['Is_Bearish'].shift(1)
    
    # Engulfing patterns
    df['Bullish_Engulfing'] = (df['Prev_Is_Bearish'] & df['Is_Bullish'] & 
                               (df['Open'] <= df['Prev_Close']) & 
                               (df['Close'] >= df['Prev_Open']))
    df['Bearish_Engulfing'] = (df['Prev_Is_Bullish'] & df['Is_Bearish'] & 
                               (df['Open'] >= df['Prev_Close']) & 
                               (df['Close'] <= df['Prev_Open']))
    
    return df


def execute_trade(df, entry_idx, trade_type, entry_price, target, stop_loss, max_bars=60):
    """Execute a trade and return the result"""
    for j in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
        candle = df.iloc[j]
        
        if trade_type == 'LONG':
            if candle['High'] >= target:
                return {
                    'type': trade_type,
                    'entry_idx': entry_idx,
                    'exit_idx': j,
                    'entry_date': df.iloc[entry_idx]['Date'],
                    'exit_date': candle['Date'],
                    'entry_price': entry_price,
                    'exit_price': target,
                    'pnl_pct': (target - entry_price) / entry_price * 100,
                    'exit_reason': 'TARGET',
                    'is_winner': True,
                    'bars_held': j - entry_idx
                }
            elif candle['Low'] <= stop_loss:
                return {
                    'type': trade_type,
                    'entry_idx': entry_idx,
                    'exit_idx': j,
                    'entry_date': df.iloc[entry_idx]['Date'],
                    'exit_date': candle['Date'],
                    'entry_price': entry_price,
                    'exit_price': stop_loss,
                    'pnl_pct': (stop_loss - entry_price) / entry_price * 100,
                    'exit_reason': 'STOP_LOSS',
                    'is_winner': False,
                    'bars_held': j - entry_idx
                }
        else:  # SHORT
            if candle['Low'] <= target:
                return {
                    'type': trade_type,
                    'entry_idx': entry_idx,
                    'exit_idx': j,
                    'entry_date': df.iloc[entry_idx]['Date'],
                    'exit_date': candle['Date'],
                    'entry_price': entry_price,
                    'exit_price': target,
                    'pnl_pct': (entry_price - target) / entry_price * 100,
                    'exit_reason': 'TARGET',
                    'is_winner': True,
                    'bars_held': j - entry_idx
                }
            elif candle['High'] >= stop_loss:
                return {
                    'type': trade_type,
                    'entry_idx': entry_idx,
                    'exit_idx': j,
                    'entry_date': df.iloc[entry_idx]['Date'],
                    'exit_date': candle['Date'],
                    'entry_price': entry_price,
                    'exit_price': stop_loss,
                    'pnl_pct': (entry_price - stop_loss) / entry_price * 100,
                    'exit_reason': 'STOP_LOSS',
                    'is_winner': False,
                    'bars_held': j - entry_idx
                }
    
    # Time exit
    exit_idx = min(entry_idx + max_bars, len(df) - 1)
    exit_price = df.iloc[exit_idx]['Close']
    if trade_type == 'LONG':
        pnl = (exit_price - entry_price) / entry_price * 100
    else:
        pnl = (entry_price - exit_price) / entry_price * 100
    
    return {
        'type': trade_type,
        'entry_idx': entry_idx,
        'exit_idx': exit_idx,
        'entry_date': df.iloc[entry_idx]['Date'],
        'exit_date': df.iloc[exit_idx]['Date'],
        'entry_price': entry_price,
        'exit_price': exit_price,
        'pnl_pct': pnl,
        'exit_reason': 'TIME_EXIT',
        'is_winner': pnl > 0,
        'bars_held': exit_idx - entry_idx
    }


# ============== SCALPING STRATEGIES FOR 5-MIN ==============

def strategy_rsi_extreme(df, rsi_period=7, rsi_oversold=20, rsi_overbought=80, 
                         target_pct=0.1, stop_loss_pct=0.3, max_bars=30):
    """RSI Extreme Mean Reversion for 5-min"""
    trades = []
    rsi_col = f'RSI_{rsi_period}'
    
    i = 200
    while i < len(df) - max_bars:
        row = df.iloc[i]
        
        # Long signal
        if row[rsi_col] < rsi_oversold and row['Close'] > row['EMA_200']:
            entry_price = row['Close']
            target = entry_price * (1 + target_pct / 100)
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'LONG', entry_price, target, stop_loss, max_bars)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        # Short signal
        elif row[rsi_col] > rsi_overbought and row['Close'] < row['EMA_200']:
            entry_price = row['Close']
            target = entry_price * (1 - target_pct / 100)
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'SHORT', entry_price, target, stop_loss, max_bars)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        i += 1
    
    return trades


def strategy_bb_bounce(df, bb_period=10, bb_threshold=0.1, 
                       target_pct=0.08, stop_loss_pct=0.25, max_bars=24):
    """Bollinger Band Bounce for 5-min"""
    trades = []
    bb_pos_col = f'BB_Position_{bb_period}'
    
    i = 200
    while i < len(df) - max_bars:
        row = df.iloc[i]
        
        # Long - near lower band
        if row[bb_pos_col] < bb_threshold and row['RSI_14'] < 40:
            entry_price = row['Close']
            target = entry_price * (1 + target_pct / 100)
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'LONG', entry_price, target, stop_loss, max_bars)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        # Short - near upper band
        elif row[bb_pos_col] > (1 - bb_threshold) and row['RSI_14'] > 60:
            entry_price = row['Close']
            target = entry_price * (1 - target_pct / 100)
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'SHORT', entry_price, target, stop_loss, max_bars)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        i += 1
    
    return trades


def strategy_ema_crossover(df, fast_ema=8, slow_ema=21, 
                           target_pct=0.12, stop_loss_pct=0.2, max_bars=36):
    """EMA Crossover for 5-min"""
    trades = []
    fast_col = f'EMA_{fast_ema}'
    slow_col = f'EMA_{slow_ema}'
    
    i = 200
    while i < len(df) - max_bars:
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Bullish crossover
        if prev[fast_col] <= prev[slow_col] and row[fast_col] > row[slow_col]:
            if row['Close'] > row['EMA_55']:  # Trend filter
                entry_price = row['Close']
                target = entry_price * (1 + target_pct / 100)
                stop_loss = entry_price * (1 - stop_loss_pct / 100)
                
                trade = execute_trade(df, i, 'LONG', entry_price, target, stop_loss, max_bars)
                if trade:
                    trades.append(trade)
                    i = trade['exit_idx']
        
        # Bearish crossover
        elif prev[fast_col] >= prev[slow_col] and row[fast_col] < row[slow_col]:
            if row['Close'] < row['EMA_55']:  # Trend filter
                entry_price = row['Close']
                target = entry_price * (1 - target_pct / 100)
                stop_loss = entry_price * (1 + stop_loss_pct / 100)
                
                trade = execute_trade(df, i, 'SHORT', entry_price, target, stop_loss, max_bars)
                if trade:
                    trades.append(trade)
                    i = trade['exit_idx']
        
        i += 1
    
    return trades


def strategy_stoch_extreme(df, stoch_period=5, stoch_oversold=15, stoch_overbought=85,
                           target_pct=0.1, stop_loss_pct=0.3, max_bars=24):
    """Stochastic Extreme for 5-min"""
    trades = []
    stoch_k = f'Stoch_K_{stoch_period}'
    stoch_d = f'Stoch_D_{stoch_period}'
    
    i = 200
    while i < len(df) - max_bars:
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Long - oversold with bullish crossover
        if (prev[stoch_k] < stoch_oversold and row[stoch_k] > prev[stoch_k] and 
            row[stoch_k] > row[stoch_d]):
            entry_price = row['Close']
            target = entry_price * (1 + target_pct / 100)
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'LONG', entry_price, target, stop_loss, max_bars)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        # Short - overbought with bearish crossover
        elif (prev[stoch_k] > stoch_overbought and row[stoch_k] < prev[stoch_k] and 
              row[stoch_k] < row[stoch_d]):
            entry_price = row['Close']
            target = entry_price * (1 - target_pct / 100)
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'SHORT', entry_price, target, stop_loss, max_bars)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        i += 1
    
    return trades


def strategy_macd_momentum(df, target_pct=0.15, stop_loss_pct=0.25, max_bars=30):
    """MACD Momentum for 5-min"""
    trades = []
    
    i = 200
    while i < len(df) - max_bars:
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Bullish momentum
        if (row['MACD'] > row['MACD_Signal'] and 
            row['MACD_Hist'] > 0 and row['MACD_Hist'] > prev['MACD_Hist'] and
            row['RSI_14'] > 45 and row['RSI_14'] < 70):
            entry_price = row['Close']
            target = entry_price * (1 + target_pct / 100)
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'LONG', entry_price, target, stop_loss, max_bars)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        # Bearish momentum
        elif (row['MACD'] < row['MACD_Signal'] and 
              row['MACD_Hist'] < 0 and row['MACD_Hist'] < prev['MACD_Hist'] and
              row['RSI_14'] < 55 and row['RSI_14'] > 30):
            entry_price = row['Close']
            target = entry_price * (1 - target_pct / 100)
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'SHORT', entry_price, target, stop_loss, max_bars)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        i += 1
    
    return trades


def strategy_engulfing(df, target_pct=0.1, stop_loss_pct=0.2, max_bars=20):
    """Engulfing Pattern for 5-min"""
    trades = []
    
    i = 200
    while i < len(df) - max_bars:
        row = df.iloc[i]
        
        # Bullish engulfing in uptrend
        if row['Bullish_Engulfing'] and row['Close'] > row['EMA_55']:
            entry_price = row['Close']
            target = entry_price * (1 + target_pct / 100)
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'LONG', entry_price, target, stop_loss, max_bars)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        # Bearish engulfing in downtrend
        elif row['Bearish_Engulfing'] and row['Close'] < row['EMA_55']:
            entry_price = row['Close']
            target = entry_price * (1 - target_pct / 100)
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'SHORT', entry_price, target, stop_loss, max_bars)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        i += 1
    
    return trades


def strategy_multi_confluence(df, target_pct=0.08, stop_loss_pct=0.4, max_bars=30):
    """Multi-Indicator Confluence for 5-min - High Win Rate"""
    trades = []
    
    i = 200
    while i < len(df) - max_bars:
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Long - multiple confirmations
        long_conditions = (
            row['RSI_7'] < 35 and
            row['Stoch_K_5'] < 25 and
            row['BB_Position_10'] < 0.15 and
            row['Close'] > row['EMA_200'] and
            row['MACD_Hist'] > prev['MACD_Hist']
        )
        
        if long_conditions:
            entry_price = row['Close']
            target = entry_price * (1 + target_pct / 100)
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'LONG', entry_price, target, stop_loss, max_bars)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        # Short - multiple confirmations
        short_conditions = (
            row['RSI_7'] > 65 and
            row['Stoch_K_5'] > 75 and
            row['BB_Position_10'] > 0.85 and
            row['Close'] < row['EMA_200'] and
            row['MACD_Hist'] < prev['MACD_Hist']
        )
        
        if short_conditions:
            entry_price = row['Close']
            target = entry_price * (1 - target_pct / 100)
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'SHORT', entry_price, target, stop_loss, max_bars)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        i += 1
    
    return trades


def strategy_ultra_conservative_5min(df, target_pct=0.05, stop_loss_pct=0.5, max_bars=60):
    """Ultra Conservative for 5-min - Maximum Win Rate"""
    trades = []
    
    i = 200
    while i < len(df) - max_bars:
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Very strict long conditions
        long_conditions = (
            row['RSI_7'] < 30 and
            row['RSI_14'] < 35 and
            row['Stoch_K_5'] < 20 and
            row['Stoch_K_14'] < 25 and
            row['BB_Position_10'] < 0.1 and
            row['BB_Position_20'] < 0.15 and
            row['Close'] > row['EMA_200'] and
            row['MACD_Hist'] > prev['MACD_Hist']
        )
        
        if long_conditions:
            entry_price = row['Close']
            target = entry_price * (1 + target_pct / 100)
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'LONG', entry_price, target, stop_loss, max_bars)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        i += 1
    
    return trades


def strategy_quick_scalp(df, target_pct=0.03, stop_loss_pct=0.15, max_bars=12):
    """Quick Scalp for 5-min - Very fast trades"""
    trades = []
    
    i = 200
    while i < len(df) - max_bars:
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Quick long
        if (row['RSI_7'] < 40 and row['RSI_7'] > prev['RSI_7'] and
            row['Stoch_K_5'] < 30 and row['Stoch_K_5'] > row['Stoch_D_5'] and
            row['Close'] > row['EMA_21']):
            entry_price = row['Close']
            target = entry_price * (1 + target_pct / 100)
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'LONG', entry_price, target, stop_loss, max_bars)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        # Quick short
        elif (row['RSI_7'] > 60 and row['RSI_7'] < prev['RSI_7'] and
              row['Stoch_K_5'] > 70 and row['Stoch_K_5'] < row['Stoch_D_5'] and
              row['Close'] < row['EMA_21']):
            entry_price = row['Close']
            target = entry_price * (1 - target_pct / 100)
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'SHORT', entry_price, target, stop_loss, max_bars)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        i += 1
    
    return trades


def calculate_metrics(trades, strategy_name=""):
    """Calculate trading metrics"""
    if not trades or len(trades) == 0:
        return None
    
    trades_df = pd.DataFrame(trades)
    total_trades = len(trades)
    
    if total_trades == 0:
        return None
    
    winning_trades = trades_df['is_winner'].sum()
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades) * 100
    
    avg_win = trades_df[trades_df['is_winner']]['pnl_pct'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[~trades_df['is_winner']]['pnl_pct'].mean() if losing_trades > 0 else 0
    
    total_pnl = trades_df['pnl_pct'].sum()
    avg_pnl = trades_df['pnl_pct'].mean()
    
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_bars = trades_df['bars_held'].mean()
    
    return {
        'strategy': strategy_name,
        'total_trades': int(total_trades),
        'winning_trades': int(winning_trades),
        'losing_trades': int(losing_trades),
        'win_rate': round(win_rate, 2),
        'avg_win_pct': round(float(avg_win), 3),
        'avg_loss_pct': round(float(avg_loss), 3),
        'total_pnl_pct': round(float(total_pnl), 2),
        'avg_pnl_pct': round(float(avg_pnl), 3),
        'risk_reward': round(float(risk_reward), 2),
        'profit_factor': round(float(profit_factor), 2) if profit_factor != float('inf') else 999,
        'avg_bars_held': round(float(avg_bars), 1)
    }


def run_all_strategies(df):
    """Run all strategies with parameter optimization"""
    all_results = []
    
    print("\n" + "="*70)
    print("TESTING 5-MINUTE SCALPING STRATEGIES")
    print("="*70)
    
    # Strategy 1: RSI Extreme
    print("\n1. Testing RSI Extreme Strategy...")
    for rsi_period in [7, 9, 14]:
        for rsi_os in [15, 20, 25]:
            for target in [0.05, 0.08, 0.1, 0.12, 0.15]:
                for sl in [0.2, 0.3, 0.4, 0.5]:
                    try:
                        trades = strategy_rsi_extreme(df, rsi_period=rsi_period, rsi_oversold=rsi_os,
                                                      rsi_overbought=100-rsi_os, target_pct=target, 
                                                      stop_loss_pct=sl)
                        metrics = calculate_metrics(trades, f"RSI_Extreme_{rsi_period}")
                        if metrics and metrics['total_trades'] >= 20:
                            metrics['params'] = {'rsi_period': rsi_period, 'rsi_oversold': rsi_os,
                                                'target_pct': target, 'stop_loss_pct': sl}
                            all_results.append(metrics)
                    except:
                        continue
    
    # Strategy 2: BB Bounce
    print("2. Testing Bollinger Band Bounce Strategy...")
    for bb_period in [10, 20]:
        for bb_thresh in [0.05, 0.1, 0.15]:
            for target in [0.05, 0.08, 0.1, 0.12]:
                for sl in [0.2, 0.25, 0.3, 0.4]:
                    try:
                        trades = strategy_bb_bounce(df, bb_period=bb_period, bb_threshold=bb_thresh,
                                                    target_pct=target, stop_loss_pct=sl)
                        metrics = calculate_metrics(trades, f"BB_Bounce_{bb_period}")
                        if metrics and metrics['total_trades'] >= 20:
                            metrics['params'] = {'bb_period': bb_period, 'bb_threshold': bb_thresh,
                                                'target_pct': target, 'stop_loss_pct': sl}
                            all_results.append(metrics)
                    except:
                        continue
    
    # Strategy 3: EMA Crossover
    print("3. Testing EMA Crossover Strategy...")
    for fast in [5, 8]:
        for slow in [13, 21]:
            for target in [0.1, 0.12, 0.15, 0.2]:
                for sl in [0.15, 0.2, 0.25, 0.3]:
                    try:
                        trades = strategy_ema_crossover(df, fast_ema=fast, slow_ema=slow,
                                                        target_pct=target, stop_loss_pct=sl)
                        metrics = calculate_metrics(trades, f"EMA_Cross_{fast}_{slow}")
                        if metrics and metrics['total_trades'] >= 20:
                            metrics['params'] = {'fast_ema': fast, 'slow_ema': slow,
                                                'target_pct': target, 'stop_loss_pct': sl}
                            all_results.append(metrics)
                    except:
                        continue
    
    # Strategy 4: Stochastic Extreme
    print("4. Testing Stochastic Extreme Strategy...")
    for stoch_period in [5, 9, 14]:
        for stoch_os in [10, 15, 20]:
            for target in [0.08, 0.1, 0.12]:
                for sl in [0.25, 0.3, 0.4]:
                    try:
                        trades = strategy_stoch_extreme(df, stoch_period=stoch_period, 
                                                        stoch_oversold=stoch_os, stoch_overbought=100-stoch_os,
                                                        target_pct=target, stop_loss_pct=sl)
                        metrics = calculate_metrics(trades, f"Stoch_Extreme_{stoch_period}")
                        if metrics and metrics['total_trades'] >= 20:
                            metrics['params'] = {'stoch_period': stoch_period, 'stoch_oversold': stoch_os,
                                                'target_pct': target, 'stop_loss_pct': sl}
                            all_results.append(metrics)
                    except:
                        continue
    
    # Strategy 5: MACD Momentum
    print("5. Testing MACD Momentum Strategy...")
    for target in [0.1, 0.12, 0.15, 0.2]:
        for sl in [0.2, 0.25, 0.3]:
            try:
                trades = strategy_macd_momentum(df, target_pct=target, stop_loss_pct=sl)
                metrics = calculate_metrics(trades, "MACD_Momentum")
                if metrics and metrics['total_trades'] >= 20:
                    metrics['params'] = {'target_pct': target, 'stop_loss_pct': sl}
                    all_results.append(metrics)
            except:
                continue
    
    # Strategy 6: Engulfing
    print("6. Testing Engulfing Pattern Strategy...")
    for target in [0.08, 0.1, 0.12]:
        for sl in [0.15, 0.2, 0.25]:
            try:
                trades = strategy_engulfing(df, target_pct=target, stop_loss_pct=sl)
                metrics = calculate_metrics(trades, "Engulfing")
                if metrics and metrics['total_trades'] >= 20:
                    metrics['params'] = {'target_pct': target, 'stop_loss_pct': sl}
                    all_results.append(metrics)
            except:
                continue
    
    # Strategy 7: Multi Confluence
    print("7. Testing Multi-Indicator Confluence Strategy...")
    for target in [0.05, 0.08, 0.1]:
        for sl in [0.3, 0.4, 0.5]:
            try:
                trades = strategy_multi_confluence(df, target_pct=target, stop_loss_pct=sl)
                metrics = calculate_metrics(trades, "Multi_Confluence")
                if metrics and metrics['total_trades'] >= 10:
                    metrics['params'] = {'target_pct': target, 'stop_loss_pct': sl}
                    all_results.append(metrics)
            except:
                continue
    
    # Strategy 8: Ultra Conservative
    print("8. Testing Ultra Conservative Strategy...")
    for target in [0.03, 0.05, 0.08]:
        for sl in [0.4, 0.5, 0.6, 0.8]:
            try:
                trades = strategy_ultra_conservative_5min(df, target_pct=target, stop_loss_pct=sl)
                metrics = calculate_metrics(trades, "Ultra_Conservative")
                if metrics and metrics['total_trades'] >= 5:
                    metrics['params'] = {'target_pct': target, 'stop_loss_pct': sl}
                    all_results.append(metrics)
            except:
                continue
    
    # Strategy 9: Quick Scalp
    print("9. Testing Quick Scalp Strategy...")
    for target in [0.02, 0.03, 0.05]:
        for sl in [0.1, 0.15, 0.2]:
            try:
                trades = strategy_quick_scalp(df, target_pct=target, stop_loss_pct=sl)
                metrics = calculate_metrics(trades, "Quick_Scalp")
                if metrics and metrics['total_trades'] >= 30:
                    metrics['params'] = {'target_pct': target, 'stop_loss_pct': sl}
                    all_results.append(metrics)
            except:
                continue
    
    return all_results


def main():
    print("="*70)
    print("BITCOIN 5-MINUTE SCALPING STRATEGY FINDER")
    print("="*70)
    
    # Fetch data
    df = fetch_bitcoin_5min_data()
    
    if df is None or len(df) < 500:
        print("Error: Not enough data")
        return
    
    # Calculate indicators
    print("\nCalculating technical indicators...")
    df = calculate_indicators(df)
    
    # Run all strategies
    all_results = run_all_strategies(df)
    
    if not all_results:
        print("No valid results found!")
        return
    
    # Sort by win rate
    sorted_results = sorted(all_results, key=lambda x: x['win_rate'], reverse=True)
    
    print("\n" + "="*70)
    print("TOP 15 STRATEGIES BY WIN RATE (5-MINUTE TIMEFRAME)")
    print("="*70)
    
    for i, result in enumerate(sorted_results[:15], 1):
        print(f"\n#{i} {result['strategy']}")
        print(f"   Win Rate: {result['win_rate']}%")
        print(f"   Total Trades: {result['total_trades']}")
        print(f"   Profit Factor: {result['profit_factor']}")
        print(f"   Total P&L: {result['total_pnl_pct']}%")
        print(f"   Avg Win: {result['avg_win_pct']}% | Avg Loss: {result['avg_loss_pct']}%")
        print(f"   Risk/Reward: {result['risk_reward']}")
        print(f"   Avg Bars Held: {result['avg_bars_held']}")
        print(f"   Parameters: {result['params']}")
    
    # Best strategy
    best = sorted_results[0]
    print("\n" + "="*70)
    print("🏆 BEST 5-MINUTE SCALPING STRATEGY")
    print("="*70)
    print(f"\nStrategy: {best['strategy']}")
    print(f"Win Rate: {best['win_rate']}%")
    print(f"Parameters: {best['params']}")
    print(f"Total Trades: {best['total_trades']}")
    print(f"Profit Factor: {best['profit_factor']}")
    print(f"Total P&L: {best['total_pnl_pct']}%")
    
    # Find 80%+ win rate strategies
    high_wr = [r for r in sorted_results if r['win_rate'] >= 80]
    print(f"\n\n📊 Strategies with 80%+ Win Rate: {len(high_wr)}")
    
    if high_wr:
        for r in high_wr[:5]:
            print(f"  - {r['strategy']}: {r['win_rate']}% ({r['total_trades']} trades)")
    
    # Save results
    results_data = {
        'best_strategy': best,
        'top_15': sorted_results[:15],
        'high_winrate': high_wr[:10] if high_wr else [],
        'data_info': {
            'symbol': 'BTC-USD',
            'timeframe': '5m',
            'start': str(df['Date'].min()),
            'end': str(df['Date'].max()),
            'total_candles': len(df)
        }
    }
    
    with open('btc_5min_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print("\nResults saved to btc_5min_results.json")
    
    return results_data


if __name__ == "__main__":
    results = main()
