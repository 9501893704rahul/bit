# Bitcoin Ultra Conservative Scalping Strategy

## 🎯 92.31% Win Rate Strategy

Real-time Bitcoin scalping signals based on a backtested strategy with 92.31% win rate.

## 📊 Strategy Parameters

| Parameter | Value |
|-----------|-------|
| Target Profit | 0.08% |
| Stop Loss | 1.0% |
| Backtest Win Rate | 92.31% |
| Timeframe | 1 Hour |

## ✅ Entry Conditions (ALL must be met)

1. **RSI < 35** - Price is oversold
2. **Stochastic K < 25** - Additional oversold confirmation
3. **BB Position < 0.2** - Price near lower Bollinger Band
4. **Price > EMA 200** - Uptrend filter
5. **MACD Improving** - Momentum turning positive

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Streamlit App
```bash
streamlit run streamlit_app.py
```

### Run Flask Live Bot
```bash
python live_scalping_bot.py
```

## ⚠️ Disclaimer

**PAPER TRADING ONLY** - This is for educational purposes. Do NOT use real money without proper risk management and understanding of the risks involved.

- High win rate comes with low risk/reward ratio (0.08:1)
- One losing trade can wipe out ~12 winning trades
- Past performance does not guarantee future results

## 📁 Files

- `streamlit_app.py` - Streamlit dashboard for live signals
- `live_scalping_bot.py` - Flask-based live trading bot
- `bitcoin_scalping_strategy.py` - Backtesting script
- `nifty50_inside_bar_strategy.py` - Nifty 50 Inside Bar strategy

## 📈 Backtest Results

- **Total Trades**: 65
- **Winning Trades**: 60
- **Losing Trades**: 5
- **Win Rate**: 92.31%
- **Profit Factor**: 0.96
- **Average Win**: 0.08%
- **Average Loss**: -1.0%
