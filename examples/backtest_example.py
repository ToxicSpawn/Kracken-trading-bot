"""Example: Running a backtest with the backtesting engine."""

import asyncio
import pandas as pd
from datetime import datetime, timedelta

from analytics.backtesting_engine import BacktestEngine, SMACrossStrategy
from exchange.kraken_client_improved import KrakenClient


async def fetch_historical_data(symbol: str, timeframe: str = "1h", days: int = 30) -> pd.DataFrame:
    """
    Fetch historical data for backtesting.

    Args:
        symbol: Trading pair
        timeframe: Timeframe (1h, 1d, etc.)
        days: Number of days of data

    Returns:
        DataFrame with OHLCV data
    """
    client = KrakenClient()
    limit = days * 24 if timeframe == "1h" else days  # Approximate

    ohlcv = await client.fetch_ohlcv(symbol, timeframe, limit=limit)

    # Convert to DataFrame
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    return df


async def run_backtest_example():
    """Run a simple backtest example."""
    print("Fetching historical data...")
    data = await fetch_historical_data("BTC/USD", timeframe="1h", days=30)

    print(f"Loaded {len(data)} candles")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Create backtest engine
    engine = BacktestEngine(
        initial_cash=10000.0,
        commission=0.001,  # 0.1%
        slippage=0.0005,  # 0.05%
    )

    # Run backtest with SMA crossover strategy
    print("\nRunning backtest with SMA crossover strategy...")
    results = engine.run_backtest(
        data=data,
        strategy=SMACrossStrategy,
        strategy_params={"fast_period": 10, "slow_period": 30},
    )

    # Print results
    print("\n=== Backtest Results ===")
    print(f"Initial Cash: ${results['initial_cash']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']*100:.2f}%")
    print(f"Average Win: ${results['average_win']:.2f}")
    print(f"Average Loss: ${results['average_loss']:.2f}")


if __name__ == "__main__":
    asyncio.run(run_backtest_example())

