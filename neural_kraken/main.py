"""Main entry point for Neural Kraken trading system."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from neural_kraken.data.kafka_consumer import MarketDataConsumer
from neural_kraken.models.lstm_momentum import MomentumLSTM
from neural_kraken.strategies.mean_reversion import MeanReversionStrategy
from neural_kraken.strategies.arbitrage import TriangularArbitrage
from neural_kraken.strategies.market_making import MarketMaker

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


class NeuralKrakenSystem:
    """Main trading system."""

    def __init__(
        self,
        kafka_brokers: str = "localhost:9092",
        use_lstm: bool = True,
        use_mean_reversion: bool = True,
        use_arbitrage: bool = True,
        use_market_making: bool = False,
    ) -> None:
        """
        Initialize Neural Kraken system.

        Args:
            kafka_brokers: Kafka broker addresses
            use_lstm: Enable LSTM momentum strategy
            use_mean_reversion: Enable mean reversion strategy
            use_arbitrage: Enable arbitrage strategy
            use_market_making: Enable market making strategy
        """
        self.kafka_brokers = kafka_brokers
        self.consumer: Optional[MarketDataConsumer] = None

        # Initialize strategies
        self.strategies = {}

        if use_lstm:
            self.strategies["lstm"] = MomentumLSTM()
            logger.info("LSTM momentum strategy enabled")

        if use_mean_reversion:
            self.strategies["mean_reversion"] = MeanReversionStrategy()
            logger.info("Mean reversion strategy enabled")

        if use_arbitrage:
            self.strategies["arbitrage"] = TriangularArbitrage()
            logger.info("Arbitrage strategy enabled")

        if use_market_making:
            self.strategies["market_making"] = MarketMaker()
            logger.info("Market making strategy enabled")

    async def start(self) -> None:
        """Start the trading system."""
        logger.info("Starting Neural Kraken system...")

        # Initialize Kafka consumer
        self.consumer = MarketDataConsumer(
            bootstrap_servers=self.kafka_brokers,
            topics=["market.ticker", "market.trade", "market.orderbook"],
        )

        # Register callbacks
        self.consumer.register_callback("market.ticker", self._handle_ticker)
        self.consumer.register_callback("market.trade", self._handle_trade)
        self.consumer.register_callback("market.orderbook", self._handle_orderbook)

        # Start consumer
        self.consumer.start()

        # Process messages
        logger.info("Neural Kraken system started. Processing market data...")
        self.consumer.process_messages()

    def _handle_ticker(self, data: dict) -> None:
        """Handle ticker updates."""
        try:
            # Process with all strategies
            for name, strategy in self.strategies.items():
                if hasattr(strategy, "process_ticker"):
                    signal = strategy.process_ticker(data)
                    logger.info(f"{name} signal: {signal}")
        except Exception as e:
            logger.error(f"Error handling ticker: {e}")

    def _handle_trade(self, data: dict) -> None:
        """Handle trade updates."""
        logger.debug(f"Trade update: {data}")

    def _handle_orderbook(self, data: dict) -> None:
        """Handle order book updates."""
        try:
            # Update arbitrage order books
            if "arbitrage" in self.strategies:
                symbol = data.get("symbol")
                if symbol:
                    self.strategies["arbitrage"].update_order_book(symbol, data)

            # Update market maker
            if "market_making" in self.strategies:
                symbol = data.get("symbol")
                if symbol:
                    quotes = self.strategies["market_making"].calculate_quotes(data)
                    logger.info(f"Market maker quotes for {symbol}: {quotes}")
        except Exception as e:
            logger.error(f"Error handling orderbook: {e}")

    def stop(self) -> None:
        """Stop the trading system."""
        if self.consumer:
            self.consumer.stop()
        logger.info("Neural Kraken system stopped")


async def main() -> None:
    """Main entry point."""
    system = NeuralKrakenSystem(
        kafka_brokers=os.getenv("KAFKA_BROKERS", "localhost:9092"),
        use_lstm=True,
        use_mean_reversion=True,
        use_arbitrage=True,
    )

    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        system.stop()


if __name__ == "__main__":
    asyncio.run(main())

