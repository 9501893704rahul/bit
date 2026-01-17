"""
Bitcoin High Win Rate Scalping Strategy Backtester
===================================================
Testing multiple scalping strategies to find configurations with 90%+ win rate

Scalping Strategies Tested:
1. RSI Oversold/Overbought Mean Reversion
2. Bollinger Band Squeeze Breakout
3. EMA Pullback Strategy
4. VWAP Mean Reversion
5. Stochastic Oversold Bounce
6. Price Action Scalping (Engulfing Patterns)
7. Multi-Timeframe Momentum
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')


def fetch_bitcoin_data(period="2y", interval="1h"):
    """Fetch Bitcoin historical data from Yahoo Finance"""
    print(f"Fetching Bitcoin data ({interval} timeframe)...")
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period=period, interval=interval)
    df = df.reset_index()
    
    if 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'Date'})
    
    df.columns = [col if col in ['Date', 'Datetime'] else col for col in df.columns]
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    
    print(f"Fetched {len(df)} candles from {df['Date'].min()} to {df['Date'].max()}")
    return df


def calculate_indicators(df):
    """Calculate all technical indicators needed for strategies"""
    df = df.copy()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # RSI with different periods
    for period in [7, 9, 21]:
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # EMAs
    for period in [5, 8, 13, 21, 50, 100, 200]:
        df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    
    # SMAs
    for period in [10, 20, 50]:
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
    
    # Stochastic
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
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['ATR_Pct'] = df['ATR'] / df['Close'] * 100
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['ROC'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
    
    # Candle patterns
    df['Body'] = df['Close'] - df['Open']
    df['Body_Pct'] = abs(df['Body']) / df['Open'] * 100
    df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Is_Bullish'] = df['Close'] > df['Open']
    df['Is_Bearish'] = df['Close'] < df['Open']
    
    # Previous candle info
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


# ============== SCALPING STRATEGIES ==============

def strategy_rsi_extreme_reversal(df, rsi_oversold=20, rsi_overbought=80, 
                                   target_pct=0.3, stop_loss_pct=0.5, rsi_period=14):
    """
    RSI Extreme Mean Reversion - Buy when RSI is extremely oversold
    High win rate due to mean reversion tendency at extremes
    """
    trades = []
    rsi_col = f'RSI_{rsi_period}' if f'RSI_{rsi_period}' in df.columns else 'RSI'
    
    i = 50
    while i < len(df) - 10:
        row = df.iloc[i]
        
        # Long signal - RSI extremely oversold
        if row[rsi_col] < rsi_oversold and row['Close'] > row['EMA_200']:
            entry_price = row['Close']
            target = entry_price * (1 + target_pct / 100)
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'LONG', entry_price, target, stop_loss, max_bars=20)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        # Short signal - RSI extremely overbought
        elif row[rsi_col] > rsi_overbought and row['Close'] < row['EMA_200']:
            entry_price = row['Close']
            target = entry_price * (1 - target_pct / 100)
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'SHORT', entry_price, target, stop_loss, max_bars=20)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        i += 1
    
    return trades


def strategy_bollinger_bounce(df, bb_threshold=0.05, target_pct=0.4, stop_loss_pct=0.6):
    """
    Bollinger Band Bounce - Buy at lower band, sell at upper band
    High win rate in ranging markets
    """
    trades = []
    
    i = 50
    while i < len(df) - 10:
        row = df.iloc[i]
        
        # Long signal - Price near lower Bollinger Band
        if row['BB_Position'] < bb_threshold and row['RSI'] < 35:
            entry_price = row['Close']
            target = entry_price * (1 + target_pct / 100)
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'LONG', entry_price, target, stop_loss, max_bars=30)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        # Short signal - Price near upper Bollinger Band
        elif row['BB_Position'] > (1 - bb_threshold) and row['RSI'] > 65:
            entry_price = row['Close']
            target = entry_price * (1 - target_pct / 100)
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'SHORT', entry_price, target, stop_loss, max_bars=30)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        i += 1
    
    return trades


def strategy_ema_pullback(df, target_pct=0.25, stop_loss_pct=0.4):
    """
    EMA Pullback Strategy - Enter on pullback to EMA in trending market
    """
    trades = []
    
    i = 50
    while i < len(df) - 10:
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Uptrend: EMA8 > EMA21 > EMA50
        uptrend = row['EMA_8'] > row['EMA_21'] > row['EMA_50']
        # Downtrend: EMA8 < EMA21 < EMA50
        downtrend = row['EMA_8'] < row['EMA_21'] < row['EMA_50']
        
        # Long - Pullback to EMA21 in uptrend
        if uptrend and prev['Low'] > prev['EMA_21'] and row['Low'] <= row['EMA_21'] and row['Close'] > row['EMA_21']:
            entry_price = row['Close']
            target = entry_price * (1 + target_pct / 100)
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'LONG', entry_price, target, stop_loss, max_bars=15)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        # Short - Pullback to EMA21 in downtrend
        elif downtrend and prev['High'] < prev['EMA_21'] and row['High'] >= row['EMA_21'] and row['Close'] < row['EMA_21']:
            entry_price = row['Close']
            target = entry_price * (1 - target_pct / 100)
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'SHORT', entry_price, target, stop_loss, max_bars=15)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        i += 1
    
    return trades


def strategy_stochastic_oversold(df, stoch_oversold=15, stoch_overbought=85, 
                                  target_pct=0.35, stop_loss_pct=0.5):
    """
    Stochastic Oversold/Overbought with confirmation
    """
    trades = []
    
    i = 50
    while i < len(df) - 10:
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Long - Stochastic oversold with bullish crossover
        if (prev['Stoch_K'] < stoch_oversold and row['Stoch_K'] > prev['Stoch_K'] and 
            row['Stoch_K'] > row['Stoch_D'] and row['RSI'] < 40):
            entry_price = row['Close']
            target = entry_price * (1 + target_pct / 100)
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'LONG', entry_price, target, stop_loss, max_bars=20)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        # Short - Stochastic overbought with bearish crossover
        elif (prev['Stoch_K'] > stoch_overbought and row['Stoch_K'] < prev['Stoch_K'] and 
              row['Stoch_K'] < row['Stoch_D'] and row['RSI'] > 60):
            entry_price = row['Close']
            target = entry_price * (1 - target_pct / 100)
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'SHORT', entry_price, target, stop_loss, max_bars=20)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        i += 1
    
    return trades


def strategy_engulfing_scalp(df, target_pct=0.3, stop_loss_pct=0.45):
    """
    Engulfing Pattern Scalping with trend filter
    """
    trades = []
    
    i = 50
    while i < len(df) - 10:
        row = df.iloc[i]
        
        # Bullish engulfing in uptrend
        if row['Bullish_Engulfing'] and row['Close'] > row['EMA_50'] and row['RSI'] < 60:
            entry_price = row['Close']
            target = entry_price * (1 + target_pct / 100)
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'LONG', entry_price, target, stop_loss, max_bars=15)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        # Bearish engulfing in downtrend
        elif row['Bearish_Engulfing'] and row['Close'] < row['EMA_50'] and row['RSI'] > 40:
            entry_price = row['Close']
            target = entry_price * (1 - target_pct / 100)
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'SHORT', entry_price, target, stop_loss, max_bars=15)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        i += 1
    
    return trades


def strategy_micro_scalp(df, target_pct=0.15, stop_loss_pct=0.3):
    """
    Micro Scalping - Very small targets with tight conditions
    Designed for high win rate with small profits
    """
    trades = []
    
    i = 50
    while i < len(df) - 10:
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Multiple confirmations for long
        long_conditions = (
            row['RSI'] < 45 and row['RSI'] > 30 and  # Not extreme but oversold
            row['Stoch_K'] < 40 and  # Stochastic oversold
            row['Close'] > row['EMA_50'] and  # Above major EMA
            row['MACD_Hist'] > prev['MACD_Hist'] and  # MACD improving
            row['Close'] > row['BB_Lower']  # Above lower BB
        )
        
        if long_conditions:
            entry_price = row['Close']
            target = entry_price * (1 + target_pct / 100)
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'LONG', entry_price, target, stop_loss, max_bars=10)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        # Multiple confirmations for short
        short_conditions = (
            row['RSI'] > 55 and row['RSI'] < 70 and
            row['Stoch_K'] > 60 and
            row['Close'] < row['EMA_50'] and
            row['MACD_Hist'] < prev['MACD_Hist'] and
            row['Close'] < row['BB_Upper']
        )
        
        if short_conditions:
            entry_price = row['Close']
            target = entry_price * (1 - target_pct / 100)
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'SHORT', entry_price, target, stop_loss, max_bars=10)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        i += 1
    
    return trades


def strategy_momentum_scalp(df, target_pct=0.2, stop_loss_pct=0.35):
    """
    Momentum Scalping - Trade with strong momentum
    """
    trades = []
    
    i = 50
    while i < len(df) - 10:
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Strong bullish momentum
        if (row['MACD'] > row['MACD_Signal'] and 
            row['MACD_Hist'] > 0 and row['MACD_Hist'] > prev['MACD_Hist'] and
            row['RSI'] > 50 and row['RSI'] < 70 and
            row['Volume_Ratio'] > 1.2):
            
            entry_price = row['Close']
            target = entry_price * (1 + target_pct / 100)
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'LONG', entry_price, target, stop_loss, max_bars=12)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        # Strong bearish momentum
        elif (row['MACD'] < row['MACD_Signal'] and 
              row['MACD_Hist'] < 0 and row['MACD_Hist'] < prev['MACD_Hist'] and
              row['RSI'] < 50 and row['RSI'] > 30 and
              row['Volume_Ratio'] > 1.2):
            
            entry_price = row['Close']
            target = entry_price * (1 - target_pct / 100)
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'SHORT', entry_price, target, stop_loss, max_bars=12)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        i += 1
    
    return trades


def strategy_ultra_conservative(df, target_pct=0.1, stop_loss_pct=0.5):
    """
    Ultra Conservative - Very small target, wide stop loss
    Designed for maximum win rate (but poor risk/reward)
    """
    trades = []
    
    i = 50
    while i < len(df) - 10:
        row = df.iloc[i]
        
        # Very strict conditions for long
        if (row['RSI'] < 35 and 
            row['Stoch_K'] < 25 and
            row['BB_Position'] < 0.2 and
            row['Close'] > row['EMA_200'] and
            row['MACD_Hist'] > df.iloc[i-1]['MACD_Hist']):
            
            entry_price = row['Close']
            target = entry_price * (1 + target_pct / 100)
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            
            trade = execute_trade(df, i, 'LONG', entry_price, target, stop_loss, max_bars=50)
            if trade:
                trades.append(trade)
                i = trade['exit_idx']
        
        i += 1
    
    return trades


def execute_trade(df, entry_idx, trade_type, entry_price, target, stop_loss, max_bars=20):
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
                    'target': target,
                    'stop_loss': stop_loss,
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
                    'target': target,
                    'stop_loss': stop_loss,
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
                    'target': target,
                    'stop_loss': stop_loss,
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
                    'target': target,
                    'stop_loss': stop_loss,
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
        'target': target,
        'stop_loss': stop_loss,
        'pnl_pct': pnl,
        'exit_reason': 'TIME_EXIT',
        'is_winner': pnl > 0,
        'bars_held': exit_idx - entry_idx
    }


def calculate_metrics(trades, strategy_name=""):
    """Calculate comprehensive trading metrics"""
    if not trades:
        return {'strategy': strategy_name, 'total_trades': 0, 'win_rate': 0}
    
    trades_df = pd.DataFrame(trades)
    
    total_trades = len(trades)
    winning_trades = trades_df['is_winner'].sum()
    losing_trades = total_trades - winning_trades
    
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    avg_win = trades_df[trades_df['is_winner']]['pnl_pct'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[~trades_df['is_winner']]['pnl_pct'].mean() if losing_trades > 0 else 0
    
    total_pnl = trades_df['pnl_pct'].sum()
    avg_pnl = trades_df['pnl_pct'].mean()
    
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_bars_held = trades_df['bars_held'].mean()
    
    return {
        'strategy': strategy_name,
        'total_trades': int(total_trades),
        'winning_trades': int(winning_trades),
        'losing_trades': int(losing_trades),
        'win_rate': round(win_rate, 2),
        'avg_win_pct': round(avg_win, 3),
        'avg_loss_pct': round(avg_loss, 3),
        'total_pnl_pct': round(total_pnl, 2),
        'avg_pnl_pct': round(avg_pnl, 3),
        'risk_reward_ratio': round(risk_reward, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999,
        'avg_bars_held': round(avg_bars_held, 1)
    }


def optimize_strategy(df, strategy_func, param_grid, strategy_name):
    """Optimize strategy parameters to find highest win rate"""
    results = []
    
    for params in param_grid:
        try:
            trades = strategy_func(df, **params)
            metrics = calculate_metrics(trades, strategy_name)
            metrics['params'] = params
            results.append(metrics)
        except Exception as e:
            continue
    
    return results


def run_all_strategies(df):
    """Run all strategies and find the best ones"""
    all_results = []
    
    print("\n" + "="*70)
    print("TESTING SCALPING STRATEGIES FOR HIGHEST WIN RATE")
    print("="*70)
    
    # Strategy 1: RSI Extreme Reversal - Optimize
    print("\n1. Testing RSI Extreme Reversal Strategy...")
    param_grid = []
    for rsi_os in [15, 20, 25]:
        for target in [0.15, 0.2, 0.25, 0.3, 0.4]:
            for sl in [0.4, 0.5, 0.6, 0.8, 1.0]:
                param_grid.append({'rsi_oversold': rsi_os, 'rsi_overbought': 100-rsi_os, 
                                  'target_pct': target, 'stop_loss_pct': sl})
    
    results = optimize_strategy(df, strategy_rsi_extreme_reversal, param_grid, "RSI Extreme Reversal")
    all_results.extend(results)
    best = max(results, key=lambda x: x['win_rate']) if results else None
    if best:
        print(f"   Best: Win Rate {best['win_rate']}% | {best['total_trades']} trades | Params: {best['params']}")
    
    # Strategy 2: Bollinger Bounce
    print("\n2. Testing Bollinger Band Bounce Strategy...")
    param_grid = []
    for bb_thresh in [0.05, 0.1, 0.15]:
        for target in [0.2, 0.3, 0.4, 0.5]:
            for sl in [0.5, 0.6, 0.8, 1.0]:
                param_grid.append({'bb_threshold': bb_thresh, 'target_pct': target, 'stop_loss_pct': sl})
    
    results = optimize_strategy(df, strategy_bollinger_bounce, param_grid, "Bollinger Bounce")
    all_results.extend(results)
    best = max(results, key=lambda x: x['win_rate']) if results else None
    if best:
        print(f"   Best: Win Rate {best['win_rate']}% | {best['total_trades']} trades | Params: {best['params']}")
    
    # Strategy 3: EMA Pullback
    print("\n3. Testing EMA Pullback Strategy...")
    param_grid = []
    for target in [0.15, 0.2, 0.25, 0.3]:
        for sl in [0.3, 0.4, 0.5, 0.6]:
            param_grid.append({'target_pct': target, 'stop_loss_pct': sl})
    
    results = optimize_strategy(df, strategy_ema_pullback, param_grid, "EMA Pullback")
    all_results.extend(results)
    best = max(results, key=lambda x: x['win_rate']) if results else None
    if best:
        print(f"   Best: Win Rate {best['win_rate']}% | {best['total_trades']} trades | Params: {best['params']}")
    
    # Strategy 4: Stochastic Oversold
    print("\n4. Testing Stochastic Oversold Strategy...")
    param_grid = []
    for stoch_os in [10, 15, 20]:
        for target in [0.2, 0.3, 0.4]:
            for sl in [0.4, 0.5, 0.6, 0.8]:
                param_grid.append({'stoch_oversold': stoch_os, 'stoch_overbought': 100-stoch_os,
                                  'target_pct': target, 'stop_loss_pct': sl})
    
    results = optimize_strategy(df, strategy_stochastic_oversold, param_grid, "Stochastic Oversold")
    all_results.extend(results)
    best = max(results, key=lambda x: x['win_rate']) if results else None
    if best:
        print(f"   Best: Win Rate {best['win_rate']}% | {best['total_trades']} trades | Params: {best['params']}")
    
    # Strategy 5: Engulfing Scalp
    print("\n5. Testing Engulfing Pattern Scalp Strategy...")
    param_grid = []
    for target in [0.2, 0.3, 0.4]:
        for sl in [0.4, 0.5, 0.6]:
            param_grid.append({'target_pct': target, 'stop_loss_pct': sl})
    
    results = optimize_strategy(df, strategy_engulfing_scalp, param_grid, "Engulfing Scalp")
    all_results.extend(results)
    best = max(results, key=lambda x: x['win_rate']) if results else None
    if best:
        print(f"   Best: Win Rate {best['win_rate']}% | {best['total_trades']} trades | Params: {best['params']}")
    
    # Strategy 6: Micro Scalp
    print("\n6. Testing Micro Scalp Strategy...")
    param_grid = []
    for target in [0.1, 0.15, 0.2]:
        for sl in [0.25, 0.3, 0.4, 0.5]:
            param_grid.append({'target_pct': target, 'stop_loss_pct': sl})
    
    results = optimize_strategy(df, strategy_micro_scalp, param_grid, "Micro Scalp")
    all_results.extend(results)
    best = max(results, key=lambda x: x['win_rate']) if results else None
    if best:
        print(f"   Best: Win Rate {best['win_rate']}% | {best['total_trades']} trades | Params: {best['params']}")
    
    # Strategy 7: Momentum Scalp
    print("\n7. Testing Momentum Scalp Strategy...")
    param_grid = []
    for target in [0.15, 0.2, 0.25]:
        for sl in [0.3, 0.4, 0.5]:
            param_grid.append({'target_pct': target, 'stop_loss_pct': sl})
    
    results = optimize_strategy(df, strategy_momentum_scalp, param_grid, "Momentum Scalp")
    all_results.extend(results)
    best = max(results, key=lambda x: x['win_rate']) if results else None
    if best:
        print(f"   Best: Win Rate {best['win_rate']}% | {best['total_trades']} trades | Params: {best['params']}")
    
    # Strategy 8: Ultra Conservative
    print("\n8. Testing Ultra Conservative Strategy...")
    param_grid = []
    for target in [0.08, 0.1, 0.12, 0.15]:
        for sl in [0.4, 0.5, 0.6, 0.8, 1.0]:
            param_grid.append({'target_pct': target, 'stop_loss_pct': sl})
    
    results = optimize_strategy(df, strategy_ultra_conservative, param_grid, "Ultra Conservative")
    all_results.extend(results)
    best = max(results, key=lambda x: x['win_rate']) if results else None
    if best:
        print(f"   Best: Win Rate {best['win_rate']}% | {best['total_trades']} trades | Params: {best['params']}")
    
    return all_results


def generate_trade_details(trades):
    """Convert trades to JSON-serializable format"""
    trade_list = []
    for trade in trades:
        trade_list.append({
            'type': trade['type'],
            'entry_date': str(trade['entry_date']),
            'exit_date': str(trade['exit_date']),
            'entry_price': round(float(trade['entry_price']), 2),
            'exit_price': round(float(trade['exit_price']), 2),
            'pnl_pct': round(float(trade['pnl_pct']), 3),
            'exit_reason': trade['exit_reason'],
            'is_winner': bool(trade['is_winner']),
            'bars_held': int(trade['bars_held'])
        })
    return trade_list


def main():
    print("="*70)
    print("BITCOIN HIGH WIN RATE SCALPING STRATEGY FINDER")
    print("="*70)
    
    # Fetch data
    df = fetch_bitcoin_data(period="2y", interval="1h")
    
    # Calculate indicators
    print("\nCalculating technical indicators...")
    df = calculate_indicators(df)
    
    # Run all strategies
    all_results = run_all_strategies(df)
    
    # Filter results with minimum trades
    valid_results = [r for r in all_results if r['total_trades'] >= 20]
    
    # Sort by win rate
    sorted_results = sorted(valid_results, key=lambda x: x['win_rate'], reverse=True)
    
    print("\n" + "="*70)
    print("TOP 10 STRATEGIES BY WIN RATE (min 20 trades)")
    print("="*70)
    
    top_strategies = sorted_results[:10]
    for i, result in enumerate(top_strategies, 1):
        print(f"\n#{i} {result['strategy']}")
        print(f"   Win Rate: {result['win_rate']}%")
        print(f"   Total Trades: {result['total_trades']}")
        print(f"   Profit Factor: {result['profit_factor']}")
        print(f"   Total P&L: {result['total_pnl_pct']}%")
        print(f"   Avg Win: {result['avg_win_pct']}% | Avg Loss: {result['avg_loss_pct']}%")
        print(f"   Risk/Reward: {result['risk_reward_ratio']}")
        print(f"   Parameters: {result['params']}")
    
    # Find best overall strategy
    best_strategy = top_strategies[0] if top_strategies else None
    
    if best_strategy:
        print("\n" + "="*70)
        print("🏆 BEST STRATEGY FOUND")
        print("="*70)
        print(f"\nStrategy: {best_strategy['strategy']}")
        print(f"Win Rate: {best_strategy['win_rate']}%")
        print(f"Parameters: {best_strategy['params']}")
        
        # Re-run best strategy to get trade details
        if best_strategy['strategy'] == "RSI Extreme Reversal":
            best_trades = strategy_rsi_extreme_reversal(df, **best_strategy['params'])
        elif best_strategy['strategy'] == "Bollinger Bounce":
            best_trades = strategy_bollinger_bounce(df, **best_strategy['params'])
        elif best_strategy['strategy'] == "EMA Pullback":
            best_trades = strategy_ema_pullback(df, **best_strategy['params'])
        elif best_strategy['strategy'] == "Stochastic Oversold":
            best_trades = strategy_stochastic_oversold(df, **best_strategy['params'])
        elif best_strategy['strategy'] == "Engulfing Scalp":
            best_trades = strategy_engulfing_scalp(df, **best_strategy['params'])
        elif best_strategy['strategy'] == "Micro Scalp":
            best_trades = strategy_micro_scalp(df, **best_strategy['params'])
        elif best_strategy['strategy'] == "Momentum Scalp":
            best_trades = strategy_momentum_scalp(df, **best_strategy['params'])
        elif best_strategy['strategy'] == "Ultra Conservative":
            best_trades = strategy_ultra_conservative(df, **best_strategy['params'])
        else:
            best_trades = []
        
        # Save results
        results_data = {
            'best_strategy': best_strategy,
            'top_10_strategies': top_strategies,
            'all_results': sorted_results[:50],
            'best_trades': generate_trade_details(best_trades),
            'data_info': {
                'symbol': 'BTC-USD',
                'timeframe': '1h',
                'start_date': str(df['Date'].min()),
                'end_date': str(df['Date'].max()),
                'total_candles': len(df)
            }
        }
        
        with open('bitcoin_scalping_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print("\nResults saved to bitcoin_scalping_results.json")
    
    # Check if any strategy achieved 90%+ win rate
    high_wr_strategies = [r for r in sorted_results if r['win_rate'] >= 90]
    if high_wr_strategies:
        print("\n" + "="*70)
        print("🎯 STRATEGIES WITH 90%+ WIN RATE")
        print("="*70)
        for r in high_wr_strategies:
            print(f"\n{r['strategy']}: {r['win_rate']}% win rate ({r['total_trades']} trades)")
            print(f"   Params: {r['params']}")
    else:
        print("\n⚠️  No strategy achieved 90% win rate with minimum 20 trades.")
        print("   Highest win rate found:", sorted_results[0]['win_rate'] if sorted_results else "N/A")
        print("\n   Note: 90% win rate is extremely difficult to achieve consistently.")
        print("   Consider the trade-off: higher win rate often means smaller profits per trade.")
    
    return results_data


if __name__ == "__main__":
    results = main()
