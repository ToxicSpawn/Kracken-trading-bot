"""Enhanced paper trading simulation with realistic order execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PaperOrder:
    """Paper trading order representation."""

    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: float
    order_type: str  # 'market' or 'limit'
    status: str  # 'pending', 'filled', 'cancelled'
    filled_amount: float = 0.0
    filled_price: float = 0.0
    created_at: datetime = None
    filled_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Initialize timestamps."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class PaperBalance:
    """Paper trading account balance."""

    base_currency: str  # e.g., 'USD'
    quote_currency: str  # e.g., 'BTC'
    base_balance: float
    quote_balance: float
    total_value_usd: float = 0.0


class PaperTradingSimulator:
    """
    Enhanced paper trading simulator with:
    - Realistic order execution
    - Slippage simulation
    - Fee calculation
    - Balance tracking
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        base_currency: str = "USD",
        commission_rate: float = 0.001,  # 0.1%
        slippage_rate: float = 0.0005,  # 0.05%
    ) -> None:
        """
        Initialize paper trading simulator.

        Args:
            initial_balance: Starting balance in base currency
            base_currency: Base currency (e.g., 'USD')
            commission_rate: Trading commission rate
            slippage_rate: Slippage rate for market orders
        """
        self.initial_balance = initial_balance
        self.base_currency = base_currency
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        # Track balances per symbol
        self.balances: dict[str, PaperBalance] = {}
        self.orders: dict[str, PaperOrder] = {}
        self.order_counter = 0

    def get_balance(self, symbol: str) -> PaperBalance:
        """
        Get balance for a trading pair.

        Args:
            symbol: Trading pair (e.g., 'BTC/USD')

        Returns:
            PaperBalance object
        """
        if symbol not in self.balances:
            # Parse symbol to get quote currency
            quote = symbol.split("/")[0] if "/" in symbol else "BTC"
            self.balances[symbol] = PaperBalance(
                base_currency=self.base_currency,
                quote_currency=quote,
                base_balance=self.initial_balance if symbol not in self.balances else self.balances[symbol].base_balance,
                quote_balance=0.0,
            )
        return self.balances[symbol]

    def create_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "market",
        price: Optional[float] = None,
        current_price: Optional[float] = None,
    ) -> PaperOrder:
        """
        Create a paper trading order.

        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Order amount
            order_type: 'market' or 'limit'
            price: Limit price (required for limit orders)
            current_price: Current market price (for market orders)

        Returns:
            PaperOrder object
        """
        self.order_counter += 1
        order_id = f"paper_{self.order_counter}"

        # For market orders, use current price with slippage
        if order_type == "market":
            if current_price is None:
                raise ValueError("current_price required for market orders")
            execution_price = current_price * (1 + self.slippage_rate) if side == "buy" else current_price * (1 - self.slippage_rate)
        else:
            if price is None:
                raise ValueError("price required for limit orders")
            execution_price = price

        order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            amount=amount,
            price=execution_price,
            order_type=order_type,
            status="pending",
        )

        # Execute immediately for market orders
        if order_type == "market":
            order = self._execute_order(order, execution_price)

        self.orders[order_id] = order
        return order

    def _execute_order(self, order: PaperOrder, execution_price: float) -> PaperOrder:
        """
        Execute a paper order.

        Args:
            order: Order to execute
            execution_price: Execution price

        Returns:
            Executed order
        """
        balance = self.get_balance(order.symbol)

        if order.side == "buy":
            # Calculate cost including commission
            cost = order.amount * execution_price
            commission = cost * self.commission_rate
            total_cost = cost + commission

            # Check if we have enough balance
            if balance.base_balance < total_cost:
                logger.warning(f"Insufficient balance for order {order.order_id}")
                order.status = "cancelled"
                return order

            # Update balances
            balance.base_balance -= total_cost
            balance.quote_balance += order.amount

        else:  # sell
            # Check if we have enough quote currency
            if balance.quote_balance < order.amount:
                logger.warning(f"Insufficient quote balance for order {order.order_id}")
                order.status = "cancelled"
                return order

            # Calculate proceeds including commission
            proceeds = order.amount * execution_price
            commission = proceeds * self.commission_rate
            net_proceeds = proceeds - commission

            # Update balances
            balance.quote_balance -= order.amount
            balance.base_balance += net_proceeds

        # Mark order as filled
        order.status = "filled"
        order.filled_amount = order.amount
        order.filled_price = execution_price
        order.filled_at = datetime.utcnow()

        logger.info(
            f"Paper order {order.order_id} filled: {order.side} {order.amount} {order.symbol} @ {execution_price:.2f}"
        )

        return order

    def get_total_value(self, symbol: str, current_price: float) -> float:
        """
        Calculate total portfolio value for a symbol.

        Args:
            symbol: Trading pair
            current_price: Current market price

        Returns:
            Total value in base currency
        """
        balance = self.get_balance(symbol)
        quote_value = balance.quote_balance * current_price
        total = balance.base_balance + quote_value
        balance.total_value_usd = total
        return total

    def get_pnl(self, symbol: str, current_price: float) -> float:
        """
        Calculate profit/loss for a symbol.

        Args:
            symbol: Trading pair
            current_price: Current market price

        Returns:
            PnL in base currency
        """
        total_value = self.get_total_value(symbol, current_price)
        return total_value - self.initial_balance

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled, False if not found or already filled
        """
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if order.status == "filled":
            return False

        order.status = "cancelled"
        logger.info(f"Paper order {order_id} cancelled")
        return True

