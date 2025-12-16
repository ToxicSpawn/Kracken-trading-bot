"""Smart order routing across multiple exchanges."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExchangeInfo:
    """Exchange information."""

    name: str
    latency_ms: float
    liquidity: float  # 0-1 scale
    fees: float  # percentage
    supported_pairs: List[str]


@dataclass
class Order:
    """Order representation."""

    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    order_type: str  # 'market' or 'limit'
    price: Optional[float] = None


class SmartOrderRouter:
    """Smart order router for multi-exchange execution."""

    def __init__(self) -> None:
        """Initialize router."""
        self.exchanges: Dict[str, ExchangeInfo] = {}
        self.current_prices: Dict[str, float] = {}
        self._initialize_exchanges()

    def _initialize_exchanges(self) -> None:
        """Initialize exchange information."""
        self.exchanges["kraken"] = ExchangeInfo(
            name="kraken",
            latency_ms=500.0,
            liquidity=0.9,
            fees=0.0016,  # 0.16%
            supported_pairs=["BTC/USD", "ETH/USD", "SOL/USD"],
        )

        self.exchanges["binance"] = ExchangeInfo(
            name="binance",
            latency_ms=300.0,
            liquidity=0.95,
            fees=0.001,  # 0.1%
            supported_pairs=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        )

        self.exchanges["coinbase"] = ExchangeInfo(
            name="coinbase",
            latency_ms=400.0,
            liquidity=0.85,
            fees=0.0025,  # 0.25%
            supported_pairs=["BTC/USD", "ETH/USD", "SOL/USD"],
        )

    def update_price(self, symbol: str, price: float) -> None:
        """Update current price for a symbol."""
        self.current_prices[symbol] = price

    def route_order(self, order: Order) -> List[Tuple[str, Order]]:
        """
        Route order to best exchange(s).

        Args:
            order: Order to route

        Returns:
            List of (exchange_name, order) tuples
        """
        # Find available exchanges
        available = self._get_available_exchanges(order.symbol)

        if not available:
            logger.warning(f"No exchanges available for {order.symbol}")
            return []

        # Route based on order type
        if order.order_type == "market":
            return self._route_market_order(order, available)
        elif order.order_type == "limit":
            return self._route_limit_order(order, available)
        else:
            return self._route_market_order(order, available)

    def _get_available_exchanges(self, symbol: str) -> List[Tuple[str, ExchangeInfo]]:
        """Get exchanges that support the symbol."""
        available = []
        normalized = self._normalize_symbol(symbol)

        for name, exchange in self.exchanges.items():
            if normalized in exchange.supported_pairs or symbol in exchange.supported_pairs:
                available.append((name, exchange))

        return available

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for exchange compatibility."""
        # Convert USD to USDT for Binance
        return symbol.replace("USD", "USDT")

    def _route_market_order(self, order: Order, available: List[Tuple[str, ExchangeInfo]]) -> List[Tuple[str, Order]]:
        """Route market order to best exchange."""
        # Sort by liquidity and fees
        sorted_exchanges = sorted(
            available,
            key=lambda x: x[1].liquidity * (1.0 - x[1].fees),
            reverse=True,
        )

        if not sorted_exchanges:
            return []

        best_exchange = sorted_exchanges[0][0]
        return [(best_exchange, order)]

    def _route_limit_order(self, order: Order, available: List[Tuple[str, ExchangeInfo]]) -> List[Tuple[str, Order]]:
        """Route limit order to best exchange."""
        if order.price is None:
            # Fallback to market routing
            return self._route_market_order(order, available)

        # Sort by price proximity and liquidity
        sorted_exchanges = sorted(
            available,
            key=lambda x: (
                abs(self.current_prices.get(f"{x[0]}_{order.symbol}", order.price) - order.price),
                -x[1].liquidity,
            ),
        )

        if not sorted_exchanges:
            return []

        best_exchange = sorted_exchanges[0][0]
        return [(best_exchange, order)]

    def calculate_execution_cost(self, exchange: str, order: Order) -> float:
        """
        Calculate execution cost for an order.

        Args:
            exchange: Exchange name
            order: Order

        Returns:
            Total cost including fees
        """
        if exchange not in self.exchanges:
            return 0.0

        exchange_info = self.exchanges[exchange]
        price = order.price or self.current_prices.get(order.symbol, 0.0)
        notional = order.amount * price

        # Fee cost
        fee_cost = notional * exchange_info.fees

        # Slippage estimate (simplified)
        slippage_cost = notional * 0.0005  # 0.05% slippage

        return fee_cost + slippage_cost

