"""Mean reversion trading strategy."""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

from neural_kraken.core.rust_bridge import RustBridge

logger = logging.getLogger(__name__)


class MeanReversionStrategy:
    """Mean reversion strategy using z-score."""

    def __init__(
        self,
        window: int = 20,
        z_threshold: float = 1.5,
        use_rust: bool = True,
    ) -> None:
        """
        Initialize mean reversion strategy.

        Args:
            window: Rolling window size
            z_threshold: Z-score threshold for signals
            use_rust: Use Rust for calculations
        """
        self.window = window
        self.z_threshold = z_threshold
        self.use_rust = use_rust
        self.rust_bridge = RustBridge() if use_rust else None
        self.data_buffer: Dict[str, pd.DataFrame] = {}

    def process_ticker(self, ticker: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process ticker and generate signal.

        Args:
            ticker: Ticker data dictionary

        Returns:
            Signal dictionary
        """
        symbol = ticker.get("symbol", "UNKNOWN")

        # Initialize buffer
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = pd.DataFrame(columns=["price", "timestamp"])

        # Add new data
        new_row = pd.DataFrame({
            "price": [ticker["price"]],
            "timestamp": [ticker.get("timestamp", pd.Timestamp.now())],
        })
        self.data_buffer[symbol] = pd.concat([self.data_buffer[symbol], new_row], ignore_index=True)

        # Keep only last N records
        if len(self.data_buffer[symbol]) > self.window * 2:
            self.data_buffer[symbol] = self.data_buffer[symbol].iloc[-self.window * 2 :]

        # Check if we have enough data
        if len(self.data_buffer[symbol]) < self.window:
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "z_score": 0.0,
                "confidence": 0.0,
            }

        # Calculate z-score
        prices = self.data_buffer[symbol]["price"].values.tolist()

        if self.use_rust and self.rust_bridge and self.rust_bridge.available:
            z_score = self.rust_bridge.calculate_zscore(prices, self.window)
        else:
            z_score = self._calculate_zscore_python(prices)

        # Generate signal
        if z_score > self.z_threshold:
            signal = "SELL"  # Overbought
            confidence = min(abs(z_score) / self.z_threshold, 1.0)
        elif z_score < -self.z_threshold:
            signal = "BUY"  # Oversold
            confidence = min(abs(z_score) / self.z_threshold, 1.0)
        else:
            signal = "HOLD"
            confidence = 0.0

        return {
            "symbol": symbol,
            "signal": signal,
            "z_score": float(z_score),
            "confidence": confidence,
            "price": ticker["price"],
            "timestamp": ticker.get("timestamp", pd.Timestamp.now()),
        }

    def _calculate_zscore_python(self, prices: list[float]) -> float:
        """Calculate z-score in Python."""
        if len(prices) < self.window:
            return 0.0

        window_prices = prices[-self.window :]
        mean = np.mean(window_prices)
        std = np.std(window_prices)

        if std == 0:
            return 0.0

        return (prices[-1] - mean) / std

