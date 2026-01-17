"""
Bitcoin High Win Rate Strategy Finder
=====================================
This script tests multiple trading strategies on Bitcoin to find the highest win rate configurations.

Strategies Tested:
1. RSI Mean Reversion - Buy oversold, sell overbought
2. Bollinger Band Mean Reversion - Buy at lower band, sell at middle/upper
3. Moving Average Pullback - Buy dips in uptrend
4. Support/Resistance Bounce - Buy at support levels
5. MACD Divergence - Trade divergences
6. Scalping with Tight Targets - Small profits, wide stops
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')


def fetch_bitcoin_data(period="5y", interval="1d"):
    """Fetch Bitcoin historical data from Yahoo Finance"""
    print("Fetching Bitcoin data...")
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period=period, interval=interval)
    df = df.reset_index()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    print(f"Fetched {len(df)} data points from {df['Date'].min().date()} to {df['Date'].max().date()}")
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
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    df['ATR_Pct'] = df['ATR'] / df['Close'] * 100
    
    # Stochastic
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # Price change percentages
    df['Pct_Change'] = df['Close'].pct_change() * 100
    df['Pct_Change_5'] = df['Close'].pct_change(5) * 100
    
    # Trend detection
    df['Uptrend'] = df['Close'] > df['SMA_50']
    df['Strong_Uptrend'] = (df['Close'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_200'])
    
    return df


def backtest_strategy(df, signals, target_pct, stop_loss_pct, max_holding_days=10):
    """
    Generic backtesting function
    signals: DataFrame with 'Signal' column (1 for buy, -1 for sell, 0 for hold)
    """
    trades = []
    position = None
    
    for i in range(len(df)):
        if position is None and i < len(signals) and signals.iloc[i] == 1:
            # Enter long position
            entry_price = df.iloc[i]['Close']
            entry_date = df.iloc[i]['Date']
            target = entry_price * (1 + target_pct / 100)
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            position = {
                'entry_price': entry_price,
                'entry_date': entry_date,
                'entry_idx': i,
                'target': target,
                'stop_loss': stop_loss
            }
        
        elif position is not None:
            current_price = df.iloc[i]['Close']
            current_high = df.iloc[i]['High']
            current_low = df.iloc[i]['Low']
            days_held = i - position['entry_idx']
            
            exit_price = None
            exit_reason = None
            
            # Check target hit
            if current_high >= position['target']:
                exit_price = position['target']
                exit_reason = 'TARGET'
            # Check stop loss hit
            elif current_low <= position['stop_loss']:
                exit_price = position['stop_loss']
                exit_reason = 'STOP_LOSS'
            # Check max holding days
            elif days_held >= max_holding_days:
                exit_price = current_price
                exit_reason = 'TIME_EXIT'
            
            if exit_price:
                pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': df.iloc[i]['Date'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'is_winner': pnl_pct > 0,
                    'days_held': days_held
                })
                position = None
    
    return trades


def strategy_rsi_oversold(df, rsi_threshold=30, target_pct=2, stop_loss_pct=5):
    """Buy when RSI is oversold"""
    signals = pd.Series(0, index=df.index)
    signals[df['RSI'] < rsi_threshold] = 1
    return backtest_strategy(df, signals, target_pct, stop_loss_pct)


def strategy_bollinger_bounce(df, target_pct=1.5, stop_loss_pct=4):
    """Buy when price touches lower Bollinger Band"""
    signals = pd.Series(0, index=df.index)
    signals[df['Close'] <= df['BB_Lower']] = 1
    return backtest_strategy(df, signals, target_pct, stop_loss_pct)


def strategy_ma_pullback(df, target_pct=2, stop_loss_pct=3):
    """Buy pullbacks to moving average in uptrend"""
    signals = pd.Series(0, index=df.index)
    # Price pulls back to SMA20 while in uptrend (above SMA50)
    condition = (df['Close'] <= df['SMA_20'] * 1.01) & (df['Close'] >= df['SMA_20'] * 0.99) & df['Uptrend']
    signals[condition] = 1
    return backtest_strategy(df, signals, target_pct, stop_loss_pct)


def strategy_stochastic_oversold(df, stoch_threshold=20, target_pct=1.5, stop_loss_pct=4):
    """Buy when Stochastic is oversold"""
    signals = pd.Series(0, index=df.index)
    signals[(df['Stoch_K'] < stoch_threshold) & (df['Stoch_D'] < stoch_threshold)] = 1
    return backtest_strategy(df, signals, target_pct, stop_loss_pct)


def strategy_macd_crossover(df, target_pct=3, stop_loss_pct=2):
    """Buy on MACD bullish crossover"""
    signals = pd.Series(0, index=df.index)
    # MACD crosses above signal line
    signals[(df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))] = 1
    return backtest_strategy(df, signals, target_pct, stop_loss_pct)


def strategy_double_bottom_rsi(df, rsi_threshold=35, target_pct=1, stop_loss_pct=5):
    """Buy when RSI shows double bottom pattern"""
    signals = pd.Series(0, index=df.index)
    # RSI was oversold, recovered, and is oversold again
    for i in range(20, len(df)):
        recent_rsi = df['RSI'].iloc[i-20:i]
        if df['RSI'].iloc[i] < rsi_threshold:
            # Check if there was a previous oversold condition that recovered
            oversold_count = (recent_rsi < rsi_threshold).sum()
            recovered = (recent_rsi > 50).any()
            if oversold_count >= 2 and recovered:
                signals.iloc[i] = 1
    return backtest_strategy(df, signals, target_pct, stop_loss_pct)


def strategy_scalping_tight_target(df, target_pct=0.5, stop_loss_pct=3):
    """Scalping with very tight targets and wider stops - high win rate but low R:R"""
    signals = pd.Series(0, index=df.index)
    # Enter on any small dip (negative day after positive days)
    condition = (df['Pct_Change'] < -0.5) & (df['Pct_Change'].shift(1) > 0) & df['Uptrend']
    signals[condition] = 1
    return backtest_strategy(df, signals, target_pct, stop_loss_pct, max_holding_days=3)


def strategy_extreme_oversold(df, rsi_threshold=25, target_pct=0.75, stop_loss_pct=5):
    """Buy only at extreme oversold conditions with tight target"""
    signals = pd.Series(0, index=df.index)
    # Very oversold RSI + price below lower BB
    condition = (df['RSI'] < rsi_threshold) & (df['Close'] < df['BB_Lower'])
    signals[condition] = 1
    return backtest_strategy(df, signals, target_pct, stop_loss_pct, max_holding_days=5)


def strategy_mean_reversion_combo(df, target_pct=0.5, stop_loss_pct=4):
    """Combined mean reversion signals for higher probability"""
    signals = pd.Series(0, index=df.index)
    # Multiple oversold conditions must align
    condition = (
        (df['RSI'] < 35) & 
        (df['Stoch_K'] < 25) & 
        (df['Close'] <= df['BB_Lower'] * 1.02)
    )
    signals[condition] = 1
    return backtest_strategy(df, signals, target_pct, stop_loss_pct, max_holding_days=5)


def strategy_dip_buying_strong_trend(df, target_pct=0.5, stop_loss_pct=3):
    """Buy small dips in strong uptrend"""
    signals = pd.Series(0, index=df.index)
    # Strong uptrend + small pullback
    condition = (
        df['Strong_Uptrend'] & 
        (df['Pct_Change'] < -1) & 
        (df['Pct_Change'] > -5) &
        (df['RSI'] > 40) & (df['RSI'] < 60)
    )
    signals[condition] = 1
    return backtest_strategy(df, signals, target_pct, stop_loss_pct, max_holding_days=3)


def calculate_metrics(trades):
    """Calculate comprehensive trading metrics"""
    if not trades:
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
    
    gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        'total_trades': int(total_trades),
        'winning_trades': int(winning_trades),
        'losing_trades': int(losing_trades),
        'win_rate': round(win_rate, 2),
        'avg_win_pct': round(float(avg_win), 2),
        'avg_loss_pct': round(float(avg_loss), 2),
        'total_pnl_pct': round(float(total_pnl), 2),
        'avg_pnl_pct': round(float(avg_pnl), 2),
        'profit_factor': round(float(profit_factor), 2) if profit_factor != float('inf') else 999.99
    }


def optimize_strategy(df, strategy_func, strategy_name, target_range, sl_range):
    """Optimize a strategy to find highest win rate parameters"""
    results = []
    
    for target in target_range:
        for sl in sl_range:
            try:
                trades = strategy_func(df, target_pct=target, stop_loss_pct=sl)
                metrics = calculate_metrics(trades)
                if metrics and metrics['total_trades'] >= 10:  # Minimum trades for statistical significance
                    metrics['strategy'] = strategy_name
                    metrics['target_pct'] = target
                    metrics['stop_loss_pct'] = sl
                    results.append(metrics)
            except Exception as e:
                continue
    
    return results


def run_all_strategies(df):
    """Run all strategies and find the best ones"""
    all_results = []
    
    # Define parameter ranges
    target_range = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    sl_range = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    
    print("\nTesting strategies...")
    
    # Strategy 1: RSI Oversold
    print("  Testing RSI Oversold strategy...")
    for rsi in [20, 25, 30, 35]:
        for target in target_range:
            for sl in sl_range:
                try:
                    trades = strategy_rsi_oversold(df, rsi_threshold=rsi, target_pct=target, stop_loss_pct=sl)
                    metrics = calculate_metrics(trades)
                    if metrics and metrics['total_trades'] >= 10:
                        metrics['strategy'] = f'RSI_Oversold_{rsi}'
                        metrics['target_pct'] = target
                        metrics['stop_loss_pct'] = sl
                        all_results.append(metrics)
                except:
                    continue
    
    # Strategy 2: Bollinger Bounce
    print("  Testing Bollinger Bounce strategy...")
    for target in target_range:
        for sl in sl_range:
            try:
                trades = strategy_bollinger_bounce(df, target_pct=target, stop_loss_pct=sl)
                metrics = calculate_metrics(trades)
                if metrics and metrics['total_trades'] >= 10:
                    metrics['strategy'] = 'Bollinger_Bounce'
                    metrics['target_pct'] = target
                    metrics['stop_loss_pct'] = sl
                    all_results.append(metrics)
            except:
                continue
    
    # Strategy 3: MA Pullback
    print("  Testing MA Pullback strategy...")
    for target in target_range:
        for sl in sl_range:
            try:
                trades = strategy_ma_pullback(df, target_pct=target, stop_loss_pct=sl)
                metrics = calculate_metrics(trades)
                if metrics and metrics['total_trades'] >= 10:
                    metrics['strategy'] = 'MA_Pullback'
                    metrics['target_pct'] = target
                    metrics['stop_loss_pct'] = sl
                    all_results.append(metrics)
            except:
                continue
    
    # Strategy 4: Stochastic Oversold
    print("  Testing Stochastic Oversold strategy...")
    for stoch in [15, 20, 25]:
        for target in target_range:
            for sl in sl_range:
                try:
                    trades = strategy_stochastic_oversold(df, stoch_threshold=stoch, target_pct=target, stop_loss_pct=sl)
                    metrics = calculate_metrics(trades)
                    if metrics and metrics['total_trades'] >= 10:
                        metrics['strategy'] = f'Stochastic_Oversold_{stoch}'
                        metrics['target_pct'] = target
                        metrics['stop_loss_pct'] = sl
                        all_results.append(metrics)
                except:
                    continue
    
    # Strategy 5: Extreme Oversold
    print("  Testing Extreme Oversold strategy...")
    for rsi in [20, 25, 30]:
        for target in target_range:
            for sl in sl_range:
                try:
                    trades = strategy_extreme_oversold(df, rsi_threshold=rsi, target_pct=target, stop_loss_pct=sl)
                    metrics = calculate_metrics(trades)
                    if metrics and metrics['total_trades'] >= 5:  # Lower threshold for extreme conditions
                        metrics['strategy'] = f'Extreme_Oversold_{rsi}'
                        metrics['target_pct'] = target
                        metrics['stop_loss_pct'] = sl
                        all_results.append(metrics)
                except:
                    continue
    
    # Strategy 6: Mean Reversion Combo
    print("  Testing Mean Reversion Combo strategy...")
    for target in target_range:
        for sl in sl_range:
            try:
                trades = strategy_mean_reversion_combo(df, target_pct=target, stop_loss_pct=sl)
                metrics = calculate_metrics(trades)
                if metrics and metrics['total_trades'] >= 5:
                    metrics['strategy'] = 'Mean_Reversion_Combo'
                    metrics['target_pct'] = target
                    metrics['stop_loss_pct'] = sl
                    all_results.append(metrics)
            except:
                continue
    
    # Strategy 7: Scalping Tight Target
    print("  Testing Scalping strategy...")
    for target in [0.25, 0.5, 0.75, 1.0]:
        for sl in sl_range:
            try:
                trades = strategy_scalping_tight_target(df, target_pct=target, stop_loss_pct=sl)
                metrics = calculate_metrics(trades)
                if metrics and metrics['total_trades'] >= 10:
                    metrics['strategy'] = 'Scalping_Tight_Target'
                    metrics['target_pct'] = target
                    metrics['stop_loss_pct'] = sl
                    all_results.append(metrics)
            except:
                continue
    
    # Strategy 8: Dip Buying Strong Trend
    print("  Testing Dip Buying strategy...")
    for target in target_range:
        for sl in sl_range:
            try:
                trades = strategy_dip_buying_strong_trend(df, target_pct=target, stop_loss_pct=sl)
                metrics = calculate_metrics(trades)
                if metrics and metrics['total_trades'] >= 10:
                    metrics['strategy'] = 'Dip_Buying_Strong_Trend'
                    metrics['target_pct'] = target
                    metrics['stop_loss_pct'] = sl
                    all_results.append(metrics)
            except:
                continue
    
    return all_results


def get_best_trade_details(df, strategy_name, target_pct, stop_loss_pct):
    """Get detailed trades for the best strategy"""
    if 'RSI_Oversold' in strategy_name:
        rsi = int(strategy_name.split('_')[-1])
        trades = strategy_rsi_oversold(df, rsi_threshold=rsi, target_pct=target_pct, stop_loss_pct=stop_loss_pct)
    elif 'Bollinger_Bounce' in strategy_name:
        trades = strategy_bollinger_bounce(df, target_pct=target_pct, stop_loss_pct=stop_loss_pct)
    elif 'MA_Pullback' in strategy_name:
        trades = strategy_ma_pullback(df, target_pct=target_pct, stop_loss_pct=stop_loss_pct)
    elif 'Stochastic_Oversold' in strategy_name:
        stoch = int(strategy_name.split('_')[-1])
        trades = strategy_stochastic_oversold(df, stoch_threshold=stoch, target_pct=target_pct, stop_loss_pct=stop_loss_pct)
    elif 'Extreme_Oversold' in strategy_name:
        rsi = int(strategy_name.split('_')[-1])
        trades = strategy_extreme_oversold(df, rsi_threshold=rsi, target_pct=target_pct, stop_loss_pct=stop_loss_pct)
    elif 'Mean_Reversion_Combo' in strategy_name:
        trades = strategy_mean_reversion_combo(df, target_pct=target_pct, stop_loss_pct=stop_loss_pct)
    elif 'Scalping' in strategy_name:
        trades = strategy_scalping_tight_target(df, target_pct=target_pct, stop_loss_pct=stop_loss_pct)
    elif 'Dip_Buying' in strategy_name:
        trades = strategy_dip_buying_strong_trend(df, target_pct=target_pct, stop_loss_pct=stop_loss_pct)
    else:
        trades = []
    
    # Convert to JSON-serializable format
    trade_list = []
    for t in trades:
        trade_list.append({
            'entry_date': t['entry_date'].strftime('%Y-%m-%d'),
            'exit_date': t['exit_date'].strftime('%Y-%m-%d'),
            'entry_price': round(float(t['entry_price']), 2),
            'exit_price': round(float(t['exit_price']), 2),
            'pnl_pct': round(float(t['pnl_pct']), 2),
            'exit_reason': t['exit_reason'],
            'is_winner': bool(t['is_winner']),
            'days_held': int(t['days_held'])
        })
    return trade_list


def main():
    """Main function to find highest win rate Bitcoin strategy"""
    print("=" * 70)
    print("BITCOIN HIGH WIN RATE STRATEGY FINDER")
    print("=" * 70)
    
    # Fetch data
    df = fetch_bitcoin_data(period="5y")
    
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
    
    print("\n" + "=" * 70)
    print("TOP 20 STRATEGIES BY WIN RATE")
    print("=" * 70)
    
    top_20 = sorted_results[:20]
    for i, result in enumerate(top_20, 1):
        print(f"\n{i}. {result['strategy']}")
        print(f"   Win Rate: {result['win_rate']}% | Target: {result['target_pct']}% | SL: {result['stop_loss_pct']}%")
        print(f"   Trades: {result['total_trades']} | Total P&L: {result['total_pnl_pct']}% | Profit Factor: {result['profit_factor']}")
    
    # Find strategies with 80%+ win rate
    high_winrate = [r for r in sorted_results if r['win_rate'] >= 80]
    
    print("\n" + "=" * 70)
    print(f"STRATEGIES WITH 80%+ WIN RATE: {len(high_winrate)} found")
    print("=" * 70)
    
    if high_winrate:
        for result in high_winrate[:10]:
            print(f"\n🏆 {result['strategy']}")
            print(f"   WIN RATE: {result['win_rate']}%")
            print(f"   Target: {result['target_pct']}% | Stop Loss: {result['stop_loss_pct']}%")
            print(f"   Total Trades: {result['total_trades']}")
            print(f"   Avg Win: {result['avg_win_pct']}% | Avg Loss: {result['avg_loss_pct']}%")
            print(f"   Total P&L: {result['total_pnl_pct']}%")
            print(f"   Profit Factor: {result['profit_factor']}")
    
    # Best overall strategy
    best = sorted_results[0]
    print("\n" + "=" * 70)
    print("🥇 BEST STRATEGY (HIGHEST WIN RATE)")
    print("=" * 70)
    print(f"\nStrategy: {best['strategy']}")
    print(f"WIN RATE: {best['win_rate']}%")
    print(f"Target: {best['target_pct']}% | Stop Loss: {best['stop_loss_pct']}%")
    print(f"Total Trades: {best['total_trades']}")
    print(f"Winning Trades: {best['winning_trades']} | Losing Trades: {best['losing_trades']}")
    print(f"Average Win: {best['avg_win_pct']}%")
    print(f"Average Loss: {best['avg_loss_pct']}%")
    print(f"Total P&L: {best['total_pnl_pct']}%")
    print(f"Profit Factor: {best['profit_factor']}")
    
    # Get detailed trades for best strategy
    best_trades = get_best_trade_details(df, best['strategy'], best['target_pct'], best['stop_loss_pct'])
    
    # Save results
    results_data = {
        'best_strategy': best,
        'top_20_strategies': top_20,
        'high_winrate_strategies': high_winrate[:20] if high_winrate else [],
        'all_results_count': len(all_results),
        'best_trades': best_trades,
        'data_range': {
            'start': df['Date'].min().strftime('%Y-%m-%d'),
            'end': df['Date'].max().strftime('%Y-%m-%d'),
            'total_days': len(df)
        }
    }
    
    with open('bitcoin_strategy_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print("\nResults saved to bitcoin_strategy_results.json")
    
    return results_data


if __name__ == "__main__":
    results = main()
