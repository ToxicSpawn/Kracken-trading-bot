"""Example: Optimizing strategy parameters with Optuna."""

import asyncio
import pandas as pd

from analytics.optimization import StrategyOptimizer, create_sma_optimization_objective
from analytics.backtesting_engine import BacktestEngine, SMACrossStrategy
from exchange.kraken_client_improved import KrakenClient


async def fetch_data_for_optimization(symbol: str = "BTC/USD") -> pd.DataFrame:
    """Fetch data for optimization."""
    client = KrakenClient()
    ohlcv = await client.fetch_ohlcv(symbol, "1h", limit=720)  # 30 days

    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    return df


async def run_optimization_example():
    """Run parameter optimization example."""
    print("Fetching data for optimization...")
    data = await fetch_data_for_optimization()

    # Create backtest engine
    engine = BacktestEngine(initial_cash=10000.0)

    def backtest_with_params(params: dict) -> float:
        """Run backtest with given parameters and return total return."""
        results = engine.run_backtest(
            data=data,
            strategy=SMACrossStrategy,
            strategy_params=params,
        )
        return results["total_return"]

    # Create optimization objective
    objective = create_sma_optimization_objective(backtest_with_params)

    # Run optimization
    optimizer = StrategyOptimizer(direction="maximize", n_trials=50)
    results = optimizer.optimize(objective, study_name="sma_crossover_optimization")

    print("\n=== Optimization Results ===")
    print(f"Best Parameters: {results['best_params']}")
    print(f"Best Return: {results['best_value']*100:.2f}%")
    print(f"Trials: {results['n_trials']}")


if __name__ == "__main__":
    asyncio.run(run_optimization_example())

