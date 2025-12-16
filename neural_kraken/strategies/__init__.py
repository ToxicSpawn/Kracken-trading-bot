"""Trading strategies."""

from neural_kraken.strategies.mean_reversion import MeanReversionStrategy
from neural_kraken.strategies.arbitrage import TriangularArbitrage
from neural_kraken.strategies.market_making import MarketMaker
from neural_kraken.strategies.arbitrage_advanced import AdvancedArbitrageStrategy, ArbitrageOpportunity

__all__ = [
    "MeanReversionStrategy",
    "TriangularArbitrage",
    "MarketMaker",
    "AdvancedArbitrageStrategy",
    "ArbitrageOpportunity",
]

