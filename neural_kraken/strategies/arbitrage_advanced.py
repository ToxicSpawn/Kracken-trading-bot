"""Advanced multi-exchange arbitrage strategy with comprehensive opportunity detection."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity representation."""

    path: List[str]
    exchanges: List[str]
    sides: List[str]  # 'buy' or 'sell'
    initial_amount: float
    final_amount: float
    profit: float
    profit_pct: float
    prices: List[float]
    liquidity: List[float]
    latency: List[float]


class AdvancedArbitrageStrategy:
    """Advanced arbitrage strategy with multi-exchange and triangular arbitrage."""

    def __init__(
        self,
        min_profit_pct: float = 0.1,
        max_order_size: float = 1.0,
        max_position_size: float = 10.0,
    ) -> None:
        """
        Initialize advanced arbitrage strategy.

        Args:
            min_profit_pct: Minimum profit percentage (0.1 = 0.1%)
            max_order_size: Maximum order size per leg
            max_position_size: Maximum position size
        """
        self.min_profit_pct = min_profit_pct
        self.max_order_size = max_order_size
        self.max_position_size = max_position_size
        self.supported_pairs = {
            "BTC/USD",
            "ETH/USD",
            "ETH/BTC",
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USD",
            "SOL/BTC",
        }
        self.order_books: Dict[str, Dict[str, Dict]] = defaultdict(dict)  # symbol -> exchange -> order_book
        self.execution_latency: Dict[str, List[float]] = defaultdict(list)

    def update_order_book(self, symbol: str, exchange: str, order_book: Dict) -> None:
        """Update order book for a symbol on an exchange."""
        self.order_books[symbol][exchange] = order_book

    def find_opportunities(self) -> List[ArbitrageOpportunity]:
        """Find all arbitrage opportunities."""
        opportunities = []

        # Check triangular arbitrage
        opportunities.extend(self._find_triangular_opportunities())

        # Check direct arbitrage between exchanges
        opportunities.extend(self._find_direct_opportunities())

        # Sort by profit
        opportunities.sort(key=lambda x: x.profit_pct, reverse=True)
        return opportunities

    def _find_triangular_opportunities(self) -> List[ArbitrageOpportunity]:
        """Find triangular arbitrage opportunities."""
        opportunities = []

        # Generate all possible triangular paths
        pairs_list = list(self.supported_pairs)

        for i, pair1 in enumerate(pairs_list):
            for j, pair2 in enumerate(pairs_list):
                if i == j:
                    continue

                # Try to find a third pair that completes the triangle
                base1, quote1 = self._split_pair(pair1)
                base2, quote2 = self._split_pair(pair2)

                if not base1 or not quote1 or not base2 or not quote2:
                    continue

                # Check for triangular paths
                if quote1 == base2:
                    # pair1 -> pair2 -> pair3 (e.g., BTC/USD -> ETH/BTC -> ETH/USD)
                    pair3 = f"{base1}/{quote2}"
                    if pair3 in self.supported_pairs:
                        opp = self._check_triangular_path(pair1, pair2, pair3)
                        if opp:
                            opportunities.append(opp)

                elif base1 == base2:
                    # pair1 -> pair2 -> pair3 (e.g., BTC/USD -> BTC/USDT -> USD/USDT)
                    pair3 = f"{quote1}/{quote2}"
                    if pair3 in self.supported_pairs:
                        opp = self._check_triangular_path(pair1, pair2, pair3)
                        if opp:
                            opportunities.append(opp)

        return opportunities

    def _find_direct_opportunities(self) -> List[ArbitrageOpportunity]:
        """Find direct arbitrage opportunities between exchanges."""
        opportunities = []

        for symbol in self.supported_pairs:
            if symbol not in self.order_books:
                continue

            exchange_books = self.order_books[symbol]
            if len(exchange_books) < 2:
                continue

            # Find best bid and ask across exchanges
            best_bid = (0.0, "")
            best_ask = (float("inf"), "")

            for exchange, book in exchange_books.items():
                bids = book.get("bids", [])
                asks = book.get("asks", [])

                if bids and bids[0][0] > best_bid[0]:
                    best_bid = (bids[0][0], exchange)

                if asks and asks[0][0] < best_ask[0]:
                    best_ask = (asks[0][0], exchange)

            # Check if arbitrage exists
            if best_bid[0] > best_ask[0] and best_bid[1] != best_ask[1]:
                profit_pct = ((best_bid[0] - best_ask[0]) / best_ask[0]) * 100.0

                if profit_pct > self.min_profit_pct:
                    opportunities.append(
                        ArbitrageOpportunity(
                            path=[symbol, symbol],
                            exchanges=[best_ask[1], best_bid[1]],
                            sides=["buy", "sell"],
                            initial_amount=1.0,
                            final_amount=best_bid[0] / best_ask[0],
                            profit=best_bid[0] - best_ask[0],
                            profit_pct=profit_pct,
                            prices=[best_ask[0], best_bid[0]],
                            liquidity=[
                                self._calculate_liquidity(exchange_books[best_ask[1]], "buy"),
                                self._calculate_liquidity(exchange_books[best_bid[1]], "sell"),
                            ],
                            latency=[
                                self._calculate_exchange_latency(best_ask[1]),
                                self._calculate_exchange_latency(best_bid[1]),
                            ],
                        )
                    )

        return opportunities

    def _check_triangular_path(
        self,
        pair1: str,
        pair2: str,
        pair3: str,
    ) -> Optional[ArbitrageOpportunity]:
        """Check a specific triangular arbitrage path."""
        # Get best order books for each pair
        book1 = self._get_best_order_book(pair1)
        book2 = self._get_best_order_book(pair2)
        book3 = self._get_best_order_book(pair3)

        if not all([book1, book2, book3]):
            return None

        # Try both directions
        for direction in ["forward", "reverse"]:
            if direction == "forward":
                # Path: pair1 -> pair2 -> pair3
                result = self._calculate_triangular_arbitrage(
                    pair1, pair2, pair3, book1, book2, book3, ["buy", "buy", "sell"]
                )
            else:
                # Path: pair3 -> pair2 -> pair1
                result = self._calculate_triangular_arbitrage(
                    pair3, pair2, pair1, book3, book2, book1, ["buy", "sell", "sell"]
                )

            if result and result.profit_pct > self.min_profit_pct:
                return result

        return None

    def _calculate_triangular_arbitrage(
        self,
        pair1: str,
        pair2: str,
        pair3: str,
        book1: Dict,
        book2: Dict,
        book3: Dict,
        sides: List[str],
    ) -> Optional[ArbitrageOpportunity]:
        """Calculate triangular arbitrage for a specific path."""
        try:
            # Get prices
            price1 = self._get_price(book1, sides[0])
            price2 = self._get_price(book2, sides[1])
            price3 = self._get_price(book3, sides[2])

            if not all([price1, price2, price3]):
                return None

            # Calculate arbitrage
            # Start with 1 unit of base currency from pair1
            initial_amount = 1.0

            # Step 1: Convert through pair1
            amount_after_1 = initial_amount / price1 if sides[0] == "buy" else initial_amount * price1

            # Step 2: Convert through pair2
            amount_after_2 = amount_after_1 / price2 if sides[1] == "buy" else amount_after_1 * price2

            # Step 3: Convert through pair3
            final_amount = amount_after_2 * price3 if sides[2] == "sell" else amount_after_2 / price3

            # Calculate profit
            profit = final_amount - initial_amount
            profit_pct = (profit / initial_amount) * 100.0

            if profit_pct > self.min_profit_pct:
                return ArbitrageOpportunity(
                    path=[pair1, pair2, pair3],
                    exchanges=[book1.get("exchange", ""), book2.get("exchange", ""), book3.get("exchange", "")],
                    sides=sides,
                    initial_amount=initial_amount,
                    final_amount=final_amount,
                    profit=profit,
                    profit_pct=profit_pct,
                    prices=[price1, price2, price3],
                    liquidity=[
                        self._calculate_liquidity(book1, sides[0]),
                        self._calculate_liquidity(book2, sides[1]),
                        self._calculate_liquidity(book3, sides[2]),
                    ],
                    latency=[
                        self._calculate_exchange_latency(book1.get("exchange", "")),
                        self._calculate_exchange_latency(book2.get("exchange", "")),
                        self._calculate_exchange_latency(book3.get("exchange", "")),
                    ],
                )

        except Exception as e:
            logger.warning(f"Error calculating triangular arbitrage: {e}")

        return None

    def _get_best_order_book(self, symbol: str) -> Optional[Dict]:
        """Get best order book for a symbol (lowest spread)."""
        if symbol not in self.order_books:
            return None

        exchange_books = self.order_books[symbol]
        if not exchange_books:
            return None

        best_book = None
        best_spread = float("inf")

        for exchange, book in exchange_books.items():
            bids = book.get("bids", [])
            asks = book.get("asks", [])

            if bids and asks:
                spread = asks[0][0] - bids[0][0]
                if spread < best_spread:
                    best_spread = spread
                    best_book = book.copy()
                    best_book["exchange"] = exchange

        return best_book

    def _get_price(self, book: Dict, side: str) -> Optional[float]:
        """Get best price from order book."""
        if side == "buy":
            asks = book.get("asks", [])
            return asks[0][0] if asks else None
        else:
            bids = book.get("bids", [])
            return bids[0][0] if bids else None

    def _split_pair(self, pair: str) -> Tuple[Optional[str], Optional[str]]:
        """Split trading pair into base and quote."""
        parts = pair.split("/")
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, None

    def _calculate_liquidity(self, book: Dict, side: str) -> float:
        """Calculate liquidity for a side of the order book."""
        if side == "buy":
            asks = book.get("asks", [])
            return sum(ask[1] for ask in asks[:5]) if asks else 0.0
        else:
            bids = book.get("bids", [])
            return sum(bid[1] for bid in bids[:5]) if bids else 0.0

    def _calculate_exchange_latency(self, exchange: str) -> float:
        """Calculate average exchange latency."""
        latencies = self.execution_latency.get(exchange, [])
        if latencies:
            return sum(latencies) / len(latencies)
        return 100.0  # Default latency in ms

    def calculate_optimal_order_size(self, opportunity: ArbitrageOpportunity) -> float:
        """Calculate optimal order size considering liquidity and risk."""
        # Minimum liquidity across the path
        min_liquidity = min(opportunity.liquidity) if opportunity.liquidity else 0.0

        # Maximum size considering liquidity
        max_size = min(min_liquidity, self.max_order_size, self.max_position_size)

        # Consider latency - reduce size if latency is high
        avg_latency = sum(opportunity.latency) / len(opportunity.latency) if opportunity.latency else 100.0
        if avg_latency > 500:  # High latency
            max_size *= 0.5

        return max_size

