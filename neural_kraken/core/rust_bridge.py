"""Python bridge to Rust core for low-latency operations."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

try:
    # This will be available after building the Rust extension
    import neural_kraken_rust as rust_core
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    logging.warning("Rust core not available. Using Python fallback.")

logger = logging.getLogger(__name__)


class RustBridge:
    """Bridge to Rust core for ultra-low latency operations."""

    def __init__(self) -> None:
        """Initialize Rust bridge."""
        self.available = RUST_AVAILABLE
        if not self.available:
            logger.warning("Rust core not available. Performance may be degraded.")

    def process_market_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process market data in Rust for low-latency.

        Args:
            data: Market data dictionary

        Returns:
            Processed data dictionary
        """
        if not self.available:
            # Python fallback
            return self._python_process(data)

        try:
            json_data = json.dumps(data)
            result = rust_core.process_market_data(json_data)
            return json.loads(result)
        except Exception as e:
            logger.error(f"Rust processing error: {e}. Falling back to Python.")
            return self._python_process(data)

    def calculate_momentum(self, prices: list[float], window: int = 20) -> float:
        """
        Calculate momentum in Rust.

        Args:
            prices: List of prices
            window: Window size

        Returns:
            Momentum value
        """
        if not self.available:
            return self._python_momentum(prices, window)

        try:
            result = rust_core.calculate_momentum(prices, window)
            return result
        except Exception as e:
            logger.error(f"Rust momentum error: {e}. Falling back to Python.")
            return self._python_momentum(prices, window)

    def calculate_zscore(self, prices: list[float], window: int = 20) -> float:
        """
        Calculate z-score in Rust.

        Args:
            prices: List of prices
            window: Window size

        Returns:
            Z-score value
        """
        if not self.available:
            return self._python_zscore(prices, window)

        try:
            result = rust_core.calculate_zscore(prices, window)
            return result
        except Exception as e:
            logger.error(f"Rust z-score error: {e}. Falling back to Python.")
            return self._python_zscore(prices, window)

    def _python_process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Python fallback for processing."""
        # Simple pass-through for now
        return data

    def _python_momentum(self, prices: list[float], window: int) -> float:
        """Python fallback for momentum calculation."""
        if len(prices) < window + 1:
            return 0.0
        return (prices[-1] - prices[-window-1]) / prices[-window-1]

    def _python_zscore(self, prices: list[float], window: int) -> float:
        """Python fallback for z-score calculation."""
        import numpy as np
        if len(prices) < window:
            return 0.0
        window_prices = prices[-window:]
        mean = np.mean(window_prices)
        std = np.std(window_prices)
        if std == 0:
            return 0.0
        return (prices[-1] - mean) / std

