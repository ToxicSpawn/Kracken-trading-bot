"""Prometheus metrics for monitoring."""

from __future__ import annotations

import logging
from typing import Optional

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not available. Install with: pip install prometheus-client")

logger = logging.getLogger(__name__)


class TradingMetrics:
    """Prometheus metrics for trading system."""

    def __init__(self) -> None:
        """Initialize metrics."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available. Metrics disabled.")
            return

        # Market data metrics
        self.ticker_messages = Counter("ticker_messages_total", "Total ticker messages received")
        self.trade_messages = Counter("trade_messages_total", "Total trade messages received")
        self.order_book_updates = Counter("order_book_updates_total", "Total order book updates")

        # Order metrics
        self.orders_placed = Counter("orders_placed_total", "Total orders placed")
        self.orders_filled = Counter("orders_filled_total", "Total orders filled")
        self.order_errors = Counter("order_errors_total", "Total order errors")
        self.active_orders = Gauge("active_orders", "Number of active orders")

        # Strategy metrics
        self.strategy_signals = Counter("strategy_signals_total", "Total strategy signals")
        self.buy_signals = Counter("buy_signals_total", "Total buy signals")
        self.sell_signals = Counter("sell_signals_total", "Total sell signals")

        # Performance metrics
        self.processing_latency = Histogram(
            "processing_latency_seconds",
            "Processing latency",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
        )
        self.order_execution_latency = Histogram(
            "order_execution_latency_seconds",
            "Order execution latency",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        )

        # Exchange metrics
        self.exchange_connections = Gauge("exchange_connections", "Number of exchange connections")
        self.exchange_latency = Gauge("exchange_latency_seconds", "Exchange API latency")

        # Portfolio metrics
        self.portfolio_value = Gauge("portfolio_value", "Current portfolio value (USD)")
        self.portfolio_return = Gauge("portfolio_return_pct", "Current portfolio return (%)")

        # Risk metrics
        self.portfolio_var = Gauge("portfolio_var", "Current portfolio VaR")
        self.position_size = Gauge("position_size", "Current position size")

        # Arbitrage metrics
        self.arbitrage_opportunities = Counter("arbitrage_opportunities_total", "Total arbitrage opportunities", ["type"])
        self.arbitrage_profit = Gauge("arbitrage_profit_pct", "Arbitrage profit percentage")

        logger.info("Trading metrics initialized")

    def start_server(self, port: int = 8001) -> None:
        """Start Prometheus metrics server."""
        if not PROMETHEUS_AVAILABLE:
            return

        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")

    def record_ticker(self) -> None:
        """Record ticker message."""
        if PROMETHEUS_AVAILABLE:
            self.ticker_messages.inc()

    def record_trade(self) -> None:
        """Record trade message."""
        if PROMETHEUS_AVAILABLE:
            self.trade_messages.inc()

    def record_order_book_update(self) -> None:
        """Record order book update."""
        if PROMETHEUS_AVAILABLE:
            self.order_book_updates.inc()

    def record_order_placed(self) -> None:
        """Record order placed."""
        if PROMETHEUS_AVAILABLE:
            self.orders_placed.inc()
            self.active_orders.inc()

    def record_order_filled(self) -> None:
        """Record order filled."""
        if PROMETHEUS_AVAILABLE:
            self.orders_filled.inc()
            self.active_orders.dec()

    def record_order_error(self) -> None:
        """Record order error."""
        if PROMETHEUS_AVAILABLE:
            self.order_errors.inc()

    def record_signal(self, signal_type: str) -> None:
        """Record trading signal."""
        if PROMETHEUS_AVAILABLE:
            self.strategy_signals.inc()
            if signal_type == "BUY":
                self.buy_signals.inc()
            elif signal_type == "SELL":
                self.sell_signals.inc()

    def record_processing_latency(self, latency: float) -> None:
        """Record processing latency."""
        if PROMETHEUS_AVAILABLE:
            self.processing_latency.observe(latency)

    def record_execution_latency(self, latency: float) -> None:
        """Record execution latency."""
        if PROMETHEUS_AVAILABLE:
            self.order_execution_latency.observe(latency)

    def update_exchange_latency(self, exchange: str, latency: float) -> None:
        """Update exchange latency."""
        if PROMETHEUS_AVAILABLE:
            self.exchange_latency.set(latency)

    def update_portfolio_metrics(self, value: float, return_pct: float, var: float) -> None:
        """Update portfolio metrics."""
        if PROMETHEUS_AVAILABLE:
            self.portfolio_value.set(value)
            self.portfolio_return.set(return_pct)
            self.portfolio_var.set(var)

    def update_position_size(self, size: float) -> None:
        """Update position size."""
        if PROMETHEUS_AVAILABLE:
            self.position_size.set(size)

    def record_arbitrage_opportunity(self, opp_type: str, profit_pct: float) -> None:
        """Record arbitrage opportunity."""
        if PROMETHEUS_AVAILABLE:
            self.arbitrage_opportunities.labels(type=opp_type).inc()
            self.arbitrage_profit.set(profit_pct)


# Global metrics instance
_metrics: Optional[TradingMetrics] = None


def get_metrics() -> TradingMetrics:
    """Get or create global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = TradingMetrics()
    return _metrics

