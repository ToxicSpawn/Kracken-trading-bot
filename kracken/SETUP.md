# Setup Guide

## Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

Note: TA-Lib may require additional system dependencies:
- Ubuntu/Debian: `sudo apt-get install ta-lib`
- macOS: `brew install ta-lib`
- Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

2. **Configure the Bot**
   - Copy `config.json` and update with your exchange API keys
   - Set up Telegram bot (optional):
     - Create a bot with @BotFather on Telegram
     - Get your chat ID
     - Add token and chat_id to config.json

3. **Set up Database (Optional)**
   - Install PostgreSQL
   - Create database: `CREATE DATABASE kracken;`
   - Update database credentials in config.json

4. **Run the Bot**
```bash
python main.py
```

## Docker Setup

1. **Build and Run**
```bash
cd docker
docker-compose up --build
```

2. **Environment Variables**
   - Set `POSTGRES_PASSWORD` in docker-compose.yml or as environment variable

## Strategy Configuration

### Enable/Disable Strategies

Edit `config.json`:

```json
{
  "strategies": {
    "ml": {
      "enabled": true,
      "lstm": {
        "enabled": true,
        "symbols": ["BTC/USDT", "ETH/USDT"]
      }
    },
    "arbitrage": {
      "enabled": true,
      "triangular": {
        "enabled": true,
        "min_profit_pct": 0.3
      }
    }
  }
}
```

## Risk Management

Configure risk parameters in `config.json`:

- **Position Sizing**: Choose "kelly" or "volatility"
- **Stop Loss**: ATR multiplier and trailing percentage
- **Black Swan**: Max drawdown and volatility thresholds

## Backtesting

To run backtests, use the BacktestEngine:

```python
from backtesting.engine import BacktestEngine
from strategies.ml_strategies import LSTMStrategy
import pandas as pd

# Load historical data
data = {"BTC/USDT": pd.read_csv("historical_data.csv")}

# Create engine
engine = BacktestEngine(config, data)

# Run backtest
strategy = LSTMStrategy(config)
results = engine.run_backtest(strategy, "BTC/USDT")
```

## Troubleshooting

### Import Errors
- Ensure all dependencies are installed
- Check Python version (3.10+)
- Verify all modules are in the correct directories

### Exchange Connection Issues
- Verify API keys are correct
- Check exchange API status
- Ensure rate limits are not exceeded

### Database Connection Issues
- Verify PostgreSQL is running
- Check database credentials
- Ensure database exists

## Support

For issues or questions, check the README.md or open an issue.

