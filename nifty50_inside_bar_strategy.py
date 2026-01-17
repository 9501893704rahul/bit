"""
Nifty 50 Inside Bar Trading Strategy Backtester
================================================
This script implements and backtests the Inside Bar trading strategy on Nifty 50 index.

Inside Bar Pattern:
- An Inside Bar is a candlestick pattern where the current bar's high is lower than 
  the previous bar's high AND the current bar's low is higher than the previous bar's low.
- It indicates consolidation and potential breakout.

Trading Rules:
- BUY Signal: When price breaks above the Inside Bar's high (mother bar's high)
- SELL Signal: When price breaks below the Inside Bar's low (mother bar's low)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')


def fetch_nifty50_data(period="5y"):
    """Fetch Nifty 50 historical data from Yahoo Finance"""
    print("Fetching Nifty 50 data...")
    # ^NSEI is the Yahoo Finance ticker for Nifty 50
    nifty = yf.Ticker("^NSEI")
    df = nifty.history(period=period)
    df = df.reset_index()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    print(f"Fetched {len(df)} days of data from {df['Date'].min().date()} to {df['Date'].max().date()}")
    return df


def identify_inside_bars(df):
    """
    Identify Inside Bar patterns in the data.
    An Inside Bar has:
    - High < Previous High (Mother Bar High)
    - Low > Previous Low (Mother Bar Low)
    """
    df = df.copy()
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    
    # Inside Bar condition
    df['Is_Inside_Bar'] = (df['High'] < df['Prev_High']) & (df['Low'] > df['Prev_Low'])
    
    # Mother Bar levels (for breakout detection)
    df['Mother_Bar_High'] = df['Prev_High']
    df['Mother_Bar_Low'] = df['Prev_Low']
    
    inside_bar_count = df['Is_Inside_Bar'].sum()
    print(f"Found {inside_bar_count} Inside Bar patterns")
    
    return df


def backtest_inside_bar_strategy(df, stop_loss_pct=1.0, target_pct=2.0, holding_days=5):
    """
    Backtest the Inside Bar breakout strategy.
    
    Strategy Rules:
    1. When an Inside Bar is identified, wait for breakout
    2. BUY if next day opens/trades above Mother Bar High
    3. SELL/SHORT if next day opens/trades below Mother Bar Low
    4. Use stop loss and target based on percentage
    5. Exit after holding_days if neither SL nor target hit
    
    Parameters:
    - stop_loss_pct: Stop loss percentage from entry
    - target_pct: Target profit percentage from entry
    - holding_days: Maximum days to hold the position
    """
    df = df.copy()
    trades = []
    
    i = 1
    while i < len(df) - holding_days:
        if df.iloc[i]['Is_Inside_Bar']:
            mother_high = df.iloc[i]['Mother_Bar_High']
            mother_low = df.iloc[i]['Mother_Bar_Low']
            inside_bar_date = df.iloc[i]['Date']
            
            # Check next day for breakout
            next_day = df.iloc[i + 1]
            
            trade = None
            
            # Bullish breakout - price goes above mother bar high
            if next_day['High'] > mother_high:
                entry_price = mother_high
                stop_loss = entry_price * (1 - stop_loss_pct / 100)
                target = entry_price * (1 + target_pct / 100)
                trade = {
                    'type': 'LONG',
                    'entry_date': next_day['Date'],
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'inside_bar_date': inside_bar_date,
                    'mother_high': mother_high,
                    'mother_low': mother_low
                }
            
            # Bearish breakout - price goes below mother bar low
            elif next_day['Low'] < mother_low:
                entry_price = mother_low
                stop_loss = entry_price * (1 + stop_loss_pct / 100)
                target = entry_price * (1 - target_pct / 100)
                trade = {
                    'type': 'SHORT',
                    'entry_date': next_day['Date'],
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'inside_bar_date': inside_bar_date,
                    'mother_high': mother_high,
                    'mother_low': mother_low
                }
            
            if trade:
                # Simulate trade outcome
                exit_price = None
                exit_date = None
                exit_reason = None
                
                for j in range(i + 1, min(i + 1 + holding_days, len(df))):
                    day = df.iloc[j]
                    
                    if trade['type'] == 'LONG':
                        # Check if target hit
                        if day['High'] >= trade['target']:
                            exit_price = trade['target']
                            exit_date = day['Date']
                            exit_reason = 'TARGET'
                            break
                        # Check if stop loss hit
                        elif day['Low'] <= trade['stop_loss']:
                            exit_price = trade['stop_loss']
                            exit_date = day['Date']
                            exit_reason = 'STOP_LOSS'
                            break
                    else:  # SHORT
                        # Check if target hit
                        if day['Low'] <= trade['target']:
                            exit_price = trade['target']
                            exit_date = day['Date']
                            exit_reason = 'TARGET'
                            break
                        # Check if stop loss hit
                        elif day['High'] >= trade['stop_loss']:
                            exit_price = trade['stop_loss']
                            exit_date = day['Date']
                            exit_reason = 'STOP_LOSS'
                            break
                
                # If neither target nor stop loss hit, exit at close of last holding day
                if exit_price is None:
                    last_day_idx = min(i + holding_days, len(df) - 1)
                    exit_price = df.iloc[last_day_idx]['Close']
                    exit_date = df.iloc[last_day_idx]['Date']
                    exit_reason = 'TIME_EXIT'
                
                # Calculate P&L
                if trade['type'] == 'LONG':
                    pnl_pct = ((exit_price - trade['entry_price']) / trade['entry_price']) * 100
                else:
                    pnl_pct = ((trade['entry_price'] - exit_price) / trade['entry_price']) * 100
                
                trade['exit_price'] = exit_price
                trade['exit_date'] = exit_date
                trade['exit_reason'] = exit_reason
                trade['pnl_pct'] = pnl_pct
                trade['is_winner'] = pnl_pct > 0
                
                trades.append(trade)
                
                # Skip to after exit date to avoid overlapping trades
                while i < len(df) - 1 and df.iloc[i]['Date'] < exit_date:
                    i += 1
        i += 1
    
    return trades


def calculate_metrics(trades):
    """Calculate comprehensive trading metrics"""
    if not trades:
        return {}
    
    trades_df = pd.DataFrame(trades)
    
    total_trades = len(trades)
    winning_trades = trades_df['is_winner'].sum()
    losing_trades = total_trades - winning_trades
    
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    avg_win = trades_df[trades_df['is_winner']]['pnl_pct'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[~trades_df['is_winner']]['pnl_pct'].mean() if losing_trades > 0 else 0
    
    total_pnl = trades_df['pnl_pct'].sum()
    avg_pnl = trades_df['pnl_pct'].mean()
    
    # Risk-Reward Ratio
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # Profit Factor
    gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Max Drawdown
    cumulative_pnl = trades_df['pnl_pct'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max()
    
    # Consecutive wins/losses
    trades_df['win_streak'] = (trades_df['is_winner'] != trades_df['is_winner'].shift()).cumsum()
    win_streaks = trades_df[trades_df['is_winner']].groupby('win_streak').size()
    loss_streaks = trades_df[~trades_df['is_winner']].groupby('win_streak').size()
    max_consecutive_wins = win_streaks.max() if len(win_streaks) > 0 else 0
    max_consecutive_losses = loss_streaks.max() if len(loss_streaks) > 0 else 0
    
    # By trade type
    long_trades = trades_df[trades_df['type'] == 'LONG']
    short_trades = trades_df[trades_df['type'] == 'SHORT']
    
    long_win_rate = (long_trades['is_winner'].sum() / len(long_trades) * 100) if len(long_trades) > 0 else 0
    short_win_rate = (short_trades['is_winner'].sum() / len(short_trades) * 100) if len(short_trades) > 0 else 0
    
    # By exit reason
    exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
    
    metrics = {
        'total_trades': int(total_trades),
        'winning_trades': int(winning_trades),
        'losing_trades': int(losing_trades),
        'win_rate': round(win_rate, 2),
        'avg_win_pct': round(avg_win, 2),
        'avg_loss_pct': round(avg_loss, 2),
        'total_pnl_pct': round(total_pnl, 2),
        'avg_pnl_pct': round(avg_pnl, 2),
        'risk_reward_ratio': round(risk_reward, 2),
        'profit_factor': round(profit_factor, 2),
        'max_drawdown_pct': round(max_drawdown, 2),
        'max_consecutive_wins': int(max_consecutive_wins),
        'max_consecutive_losses': int(max_consecutive_losses),
        'long_trades': int(len(long_trades)),
        'short_trades': int(len(short_trades)),
        'long_win_rate': round(long_win_rate, 2),
        'short_win_rate': round(short_win_rate, 2),
        'exit_by_target': int(exit_reasons.get('TARGET', 0)),
        'exit_by_stop_loss': int(exit_reasons.get('STOP_LOSS', 0)),
        'exit_by_time': int(exit_reasons.get('TIME_EXIT', 0))
    }
    
    return metrics


def run_parameter_optimization(df, sl_range=(0.5, 2.5, 0.5), target_range=(1.0, 4.0, 0.5)):
    """Run backtests with different parameters to find optimal settings"""
    results = []
    
    for sl in np.arange(*sl_range):
        for target in np.arange(*target_range):
            trades = backtest_inside_bar_strategy(df, stop_loss_pct=sl, target_pct=target)
            metrics = calculate_metrics(trades)
            if metrics:
                metrics['stop_loss_pct'] = round(sl, 1)
                metrics['target_pct'] = round(target, 1)
                results.append(metrics)
    
    return results


def generate_trade_details(trades):
    """Convert trades to JSON-serializable format"""
    trade_list = []
    for trade in trades:
        trade_list.append({
            'type': trade['type'],
            'entry_date': trade['entry_date'].strftime('%Y-%m-%d'),
            'exit_date': trade['exit_date'].strftime('%Y-%m-%d'),
            'entry_price': round(float(trade['entry_price']), 2),
            'exit_price': round(float(trade['exit_price']), 2),
            'stop_loss': round(float(trade['stop_loss']), 2),
            'target': round(float(trade['target']), 2),
            'pnl_pct': round(float(trade['pnl_pct']), 2),
            'exit_reason': trade['exit_reason'],
            'is_winner': bool(trade['is_winner'])
        })
    return trade_list


def main():
    """Main function to run the backtest"""
    print("=" * 60)
    print("NIFTY 50 INSIDE BAR TRADING STRATEGY BACKTEST")
    print("=" * 60)
    
    # Fetch data
    df = fetch_nifty50_data(period="5y")
    
    # Identify Inside Bars
    df = identify_inside_bars(df)
    
    # Run backtest with default parameters
    print("\nRunning backtest with default parameters...")
    print("Stop Loss: 1.0%, Target: 2.0%, Max Holding: 5 days")
    trades = backtest_inside_bar_strategy(df, stop_loss_pct=1.0, target_pct=2.0, holding_days=5)
    
    # Calculate metrics
    metrics = calculate_metrics(trades)
    
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"\nTotal Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Losing Trades: {metrics['losing_trades']}")
    print(f"\n*** WIN RATE: {metrics['win_rate']}% ***")
    print(f"\nAverage Win: {metrics['avg_win_pct']}%")
    print(f"Average Loss: {metrics['avg_loss_pct']}%")
    print(f"Risk-Reward Ratio: {metrics['risk_reward_ratio']}")
    print(f"\nTotal P&L: {metrics['total_pnl_pct']}%")
    print(f"Average P&L per Trade: {metrics['avg_pnl_pct']}%")
    print(f"Profit Factor: {metrics['profit_factor']}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']}%")
    print(f"\nLong Trades: {metrics['long_trades']} (Win Rate: {metrics['long_win_rate']}%)")
    print(f"Short Trades: {metrics['short_trades']} (Win Rate: {metrics['short_win_rate']}%)")
    print(f"\nExit by Target: {metrics['exit_by_target']}")
    print(f"Exit by Stop Loss: {metrics['exit_by_stop_loss']}")
    print(f"Exit by Time: {metrics['exit_by_time']}")
    
    # Run parameter optimization
    print("\n" + "=" * 60)
    print("PARAMETER OPTIMIZATION")
    print("=" * 60)
    optimization_results = run_parameter_optimization(df)
    
    # Find best parameters by win rate
    best_by_win_rate = max(optimization_results, key=lambda x: x['win_rate'])
    print(f"\nBest by Win Rate:")
    print(f"  SL: {best_by_win_rate['stop_loss_pct']}%, Target: {best_by_win_rate['target_pct']}%")
    print(f"  Win Rate: {best_by_win_rate['win_rate']}%, Profit Factor: {best_by_win_rate['profit_factor']}")
    
    # Find best parameters by profit factor
    best_by_pf = max(optimization_results, key=lambda x: x['profit_factor'] if x['profit_factor'] != float('inf') else 0)
    print(f"\nBest by Profit Factor:")
    print(f"  SL: {best_by_pf['stop_loss_pct']}%, Target: {best_by_pf['target_pct']}%")
    print(f"  Win Rate: {best_by_pf['win_rate']}%, Profit Factor: {best_by_pf['profit_factor']}")
    
    # Save results to JSON for dashboard
    results_data = {
        'default_metrics': metrics,
        'trades': generate_trade_details(trades),
        'optimization_results': optimization_results,
        'best_by_win_rate': best_by_win_rate,
        'best_by_profit_factor': best_by_pf,
        'data_range': {
            'start': df['Date'].min().strftime('%Y-%m-%d'),
            'end': df['Date'].max().strftime('%Y-%m-%d'),
            'total_days': len(df),
            'inside_bars_found': int(df['Is_Inside_Bar'].sum())
        }
    }
    
    with open('backtest_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print("\nResults saved to backtest_results.json")
    
    return results_data


if __name__ == "__main__":
    results = main()
