"""Backtesting framework."""

from neural_kraken.backtesting.advanced_engine import AdvancedBacktestingEngine, BacktestResult, Trade
from neural_kraken.backtesting.strategies import (
    mean_reversion_strategy,
    momentum_strategy,
    rsi_strategy,
    combined_strategy,
)

__all__ = [
    "AdvancedBacktestingEngine",
    "BacktestResult",
    "Trade",
    "mean_reversion_strategy",
    "momentum_strategy",
    "rsi_strategy",
    "combined_strategy",
]

