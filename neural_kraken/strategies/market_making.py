"""Market making strategy using Avellaneda-Stoikov model."""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MarketMaker:
    """Market making strategy."""

    def __init__(
        self,
        gamma: float = 0.1,
        k: float = 1.5,
        sigma: float = 0.01,
    ) -> None:
        """
        Initialize market maker.

        Args:
            gamma: Risk aversion parameter
            k: Order book depth parameter
            sigma: Volatility estimate
        """
        self.gamma = gamma
        self.k = k
        self.sigma = sigma
        self.inventory = 0.0
        self.mid_price = 0.0

    def update_mid_price(self, price: float) -> None:
        """Update mid price."""
        self.mid_price = price

    def update_inventory(self, amount: float) -> None:
        """Update inventory."""
        self.inventory += amount

    def calculate_quotes(self, order_book: Dict) -> Tuple[float, float]:
        """
        Calculate bid and ask quotes.

        Args:
            order_book: Order book dictionary

        Returns:
            (bid_price, ask_price)
        """
        # Get current spread
        best_bid = order_book.get("bids", [[self.mid_price * 0.999]])[0][0] if order_book.get("bids") else self.mid_price * 0.999
        best_ask = order_book.get("asks", [[self.mid_price * 1.001]])[0][0] if order_book.get("asks") else self.mid_price * 1.001

        # Calculate reservation price (Avellaneda-Stoikov)
        reservation_price = self.mid_price - self.inventory * self.gamma * self.sigma**2

        # Calculate optimal spread
        optimal_spread = (
            self.gamma * self.sigma**2 * 0.5
            + (2 / self.gamma) * np.log(1 + self.gamma / self.k)
        )

        # Calculate quotes
        bid_price = reservation_price - optimal_spread / 2
        ask_price = reservation_price + optimal_spread / 2

        # Adjust for order book depth (1 pip below/above best)
        bid_price = min(bid_price, best_bid - 0.0001)
        ask_price = max(ask_price, best_ask + 0.0001)

        return float(bid_price), float(ask_price)

    def calculate_order_sizes(self, bid_price: float, ask_price: float) -> Tuple[float, float]:
        """
        Calculate order sizes based on inventory.

        Args:
            bid_price: Bid price
            ask_price: Ask price

        Returns:
            (bid_size, ask_size)
        """
        base_size = 1.0

        if self.inventory > 0:
            # Positive inventory: want to sell more
            ask_size = base_size + self.inventory * 0.1
            bid_size = base_size - self.inventory * 0.05
        elif self.inventory < 0:
            # Negative inventory: want to buy more
            bid_size = base_size - self.inventory * 0.1
            ask_size = base_size + self.inventory * 0.05
        else:
            bid_size = base_size
            ask_size = base_size

        return max(bid_size, 0.1), max(ask_size, 0.1)

