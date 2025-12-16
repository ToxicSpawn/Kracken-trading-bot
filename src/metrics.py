"""Prometheus metrics."""
from prometheus_client import Counter, Gauge, start_http_server
from .config import settings

orders_placed = Counter("bot_orders_placed_total", "Orders placed")
pnl_usd = Gauge("bot_pnl_usd", "Running PnL")

def start_metrics():
    """Start Prometheus metrics server."""
    start_http_server(settings.metrics_port)
    print(f"Metrics server started on port {settings.metrics_port}")

