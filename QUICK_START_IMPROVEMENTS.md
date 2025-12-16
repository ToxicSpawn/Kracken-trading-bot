# Quick Start: Using the New Improvements

This guide shows you how to quickly use the new improvements made to the trading bot.

## 1. Encrypt Your API Keys (Recommended)

```python
from utils.security import store_encrypted_env

# Encrypt your API key
encrypted_key = store_encrypted_env("KRAKEN_API_KEY", "your_actual_api_key")
encrypted_secret = store_encrypted_env("KRAKEN_API_SECRET", "your_actual_secret")

print(f"Add to .env:")
print(f"KRAKEN_API_KEY={encrypted_key}")
print(f"KRAKEN_API_SECRET={encrypted_secret}")
```

## 2. Use the Improved Kraken Client

```python
from exchange.kraken_client_improved import KrakenClient

# Client automatically handles encrypted keys and retries
client = KrakenClient()

# All methods now have retry logic and better error handling
ticker = await client.fetch_ticker("BTC/USD")
balance = await client.fetch_balance()
```

## 3. Run a Backtest

```python
# See examples/backtest_example.py for full example
python examples/backtest_example.py
```

## 4. Use WebSocket for Real-Time Data

```python
# See examples/websocket_example.py for full example
python examples/websocket_example.py
```

## 5. Test Your Strategy with Paper Trading

```python
# See examples/paper_trading_example.py for full example
python examples/paper_trading_example.py
```

## 6. Optimize Strategy Parameters

```python
# See examples/optimization_example.py for full example
python examples/optimization_example.py
```

## Key Improvements Summary

✅ **Retry Logic**: All API calls automatically retry on failure  
✅ **Type Hints**: Full type annotations for better IDE support  
✅ **Security**: API key encryption support  
✅ **Backtesting**: Test strategies before risking real money  
✅ **WebSocket**: Real-time market data streaming  
✅ **Caching**: Reduce API calls with Redis caching  
✅ **Paper Trading**: Enhanced simulation with slippage and fees  
✅ **Optimization**: Automated parameter tuning with Optuna  
✅ **Tests**: Comprehensive test suite  
✅ **CI/CD**: Automated testing and Docker builds  

## Installation

```bash
# Install new dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run examples
python examples/backtest_example.py
```

## Migration from Old Code

### Old Way:
```python
from exchange.kraken_client import KrakenClient
client = KrakenClient()
ticker = await client.fetch_ticker("BTC/USD")  # No retries, no type hints
```

### New Way:
```python
from exchange.kraken_client_improved import KrakenClient
client = KrakenClient()  # Auto-handles encrypted keys
ticker = await client.fetch_ticker("BTC/USD")  # Automatic retries, full type hints
```

## Next Steps

1. Read `IMPROVEMENTS_GUIDE.md` for detailed documentation
2. Run the examples to see improvements in action
3. Integrate new features into your strategies
4. Set up CI/CD for automated testing

