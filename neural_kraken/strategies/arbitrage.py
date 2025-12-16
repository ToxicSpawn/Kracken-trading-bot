"""Triangular arbitrage strategy."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TriangularArbitrage:
    """Triangular arbitrage opportunity finder."""

    def __init__(self, min_profit_pct: float = 0.1) -> None:
        """
        Initialize arbitrage finder.

        Args:
            min_profit_pct: Minimum profit percentage (0.1 = 0.1%)
        """
        self.min_profit_pct = min_profit_pct
        self.order_books: Dict[str, Dict] = {}

    def update_order_book(self, symbol: str, order_book: Dict) -> None:
        """
        Update order book for a symbol.

        Args:
            symbol: Trading pair symbol
            order_book: Order book with 'bids' and 'asks'
        """
        self.order_books[symbol] = order_book

    def find_opportunities(self, pairs: List[Tuple[str, str, str]]) -> List[Dict]:
        """
        Find arbitrage opportunities.

        Args:
            pairs: List of (pair1, pair2, pair3) tuples

        Returns:
            List of opportunity dictionaries
        """
        opportunities = []

        for pair1, pair2, pair3 in pairs:
            # Get order books
            ob1 = self.order_books.get(pair1)
            ob2 = self.order_books.get(pair2)
            ob3 = self.order_books.get(pair3)

            if not all([ob1, ob2, ob3]):
                continue

            # Calculate both paths
            path1 = self._calculate_path(pair1, pair2, pair3, ob1, ob2, ob3)
            path2 = self._calculate_path(pair1, pair3, pair2, ob1, ob3, ob2)

            for path in [path1, path2]:
                if path and path.get("profit_pct", 0) > self.min_profit_pct:
                    opportunities.append(path)

        # Sort by profit
        opportunities.sort(key=lambda x: x.get("profit_pct", 0), reverse=True)
        return opportunities

    def _calculate_path(
        self,
        pair1: str,
        pair2: str,
        pair3: str,
        ob1: Dict,
        ob2: Dict,
        ob3: Dict,
    ) -> Optional[Dict]:
        """
        Calculate arbitrage path.

        Example: BTC/USD -> ETH/BTC -> ETH/USD
        """
        try:
            # Start with 1 unit of base currency from pair1
            # Get best prices
            if not ob1.get("asks") or not ob2.get("asks") or not ob3.get("bids"):
                return None

            price1 = float(ob1["asks"][0][0])  # Buy price for pair1
            price2 = float(ob2["asks"][0][0])  # Buy price for pair2
            price3 = float(ob3["bids"][0][0])  # Sell price for pair3

            # Calculate amounts
            initial_amount = 1.0
            amount_after_1 = initial_amount / price1
            amount_after_2 = amount_after_1 / price2
            final_amount = amount_after_2 * price3

            profit = final_amount - initial_amount
            profit_pct = (profit / initial_amount) * 100

            return {
                "path": [pair1, pair2, pair3],
                "initial_amount": initial_amount,
                "final_amount": final_amount,
                "profit": profit,
                "profit_pct": profit_pct,
            }
        except (IndexError, ValueError, ZeroDivisionError) as e:
            logger.warning(f"Error calculating arbitrage path: {e}")
            return None

