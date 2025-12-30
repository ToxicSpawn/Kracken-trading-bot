"""Live metrics (Prometheus compatible) with safe fallback.

This module is designed to be imported even when `prometheus_client` is not installed.
If `prometheus_client` is missing, all functions become no-ops.

Env:
  ENABLE_METRICS=true|false
  METRICS_HOST=0.0.0.0
  METRICS_PORT=9109
"""
from __future__ import annotations

import os
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ENABLED = False
_PROM = None  # lazy import container


def _truthy(x: Optional[str]) -> bool:
    return (x or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def enabled() -> bool:
    return _ENABLED


def _lazy_import() -> bool:
    global _PROM
    if _PROM is not None:
        return bool(_PROM)
    try:
        from prometheus_client import Counter, Gauge, Histogram, start_http_server  # type: ignore
        _PROM = {
            "Counter": Counter,
            "Gauge": Gauge,
            "Histogram": Histogram,
            "start_http_server": start_http_server,
        }
        return True
    except Exception as e:
        logger.warning("prometheus_client not available; metrics disabled (%r)", e)
        _PROM = False
        return False


_m = {}  # metric objects


def start_metrics_server() -> None:
    """Start Prometheus HTTP metrics endpoint if enabled."""
    global _ENABLED, _m

    if not _truthy(os.getenv("ENABLE_METRICS")):
        _ENABLED = False
        return

    if not _lazy_import():
        _ENABLED = False
        return

    host = os.getenv("METRICS_HOST", "0.0.0.0")
    port = int(os.getenv("METRICS_PORT", "9109"))

    Counter = _PROM["Counter"]
    Gauge = _PROM["Gauge"]
    Histogram = _PROM["Histogram"]

    _m = {
        "intents_total": Counter("bot_intents_total", "Total intents observed", ["source", "symbol", "side"]),
        "intents_dropped_total": Counter("bot_intents_dropped_total", "Total intents dropped", ["reason"]),
        "orders_submitted_total": Counter("bot_orders_submitted_total", "Total orders submitted", ["symbol", "side"]),
        "orders_errors_total": Counter("bot_orders_errors_total", "Total order errors", ["symbol"]),
        "allocator_mult": Gauge("bot_allocator_multiplier", "Latest allocator multiplier", ["source"]),
        "eligibility_mult": Gauge("bot_eligibility_multiplier", "Latest eligibility multiplier", ["source"]),
        "system_mode": Gauge("bot_system_mode", "System mode (NORMAL=1, PAUSED=2, MANAGE_ONLY=3, DEGRADED=4)"),
        "trading_enabled": Gauge("bot_trading_enabled", "Trading enabled (1/0)"),
        "loss_cluster_count": Gauge("bot_loss_cluster_count_window", "Loss cluster loss count in window"),
        "order_latency_s": Histogram("bot_order_latency_seconds", "Order submit->ack latency seconds"),
    }

    _PROM["start_http_server"](port, addr=host)
    _ENABLED = True
    logger.info("Metrics enabled on http://%s:%d/metrics", host, port)


def _mode_to_num(mode: str) -> int:
    m = (mode or "").upper()
    if m == "PAUSED":
        return 2
    if m == "MANAGE_ONLY":
        return 3
    if m == "DEGRADED":
        return 4
    return 1


def set_system_state(state) -> None:
    if not _ENABLED:
        return
    try:
        _m["system_mode"].set(_mode_to_num(getattr(state, "system_mode", "NORMAL")))
        _m["trading_enabled"].set(1.0 if getattr(state, "trading_enabled", True) else 0.0)

        lc = (getattr(state, "meta", {}) or {}).get("loss_cluster")
        if lc is not None:
            try:
                snap = lc.snapshot()
            except Exception:
                snap = None
            if snap and "loss_count_window" in snap:
                _m["loss_cluster_count"].set(float(snap["loss_count_window"]))
    except Exception:
        return


def on_intent(intent: Dict[str, Any]) -> None:
    if not _ENABLED:
        return
    try:
        src = str(intent.get("source") or intent.get("meta", {}).get("strategy") or "unknown")
        sym = str(intent.get("symbol") or "unknown")
        side = str(intent.get("side") or "unknown")
        _m["intents_total"].labels(src, sym, side).inc()
    except Exception:
        return


def on_intent_dropped(reason: str) -> None:
    if not _ENABLED:
        return
    try:
        _m["intents_dropped_total"].labels(str(reason or "UNKNOWN")).inc()
    except Exception:
        return


def on_allocator(intent: Dict[str, Any]) -> None:
    if not _ENABLED:
        return
    try:
        src = str(intent.get("source") or intent.get("meta", {}).get("strategy") or "unknown")
        a = (intent.get("meta", {}) or {}).get("allocator", {})
        fm = a.get("final_multiplier")
        if fm is not None:
            _m["allocator_mult"].labels(src).set(float(fm))
    except Exception:
        return


def on_eligibility(intent: Dict[str, Any], mult: float) -> None:
    if not _ENABLED:
        return
    try:
        src = str(intent.get("source") or intent.get("meta", {}).get("strategy") or "unknown")
        _m["eligibility_mult"].labels(src).set(float(mult))
    except Exception:
        return


def on_order_submitted(intent: Dict[str, Any]) -> None:
    if not _ENABLED:
        return
    try:
        sym = str(intent.get("symbol") or "unknown")
        side = str(intent.get("side") or "unknown")
        _m["orders_submitted_total"].labels(sym, side).inc()
    except Exception:
        return


def on_order_error(intent: Dict[str, Any]) -> None:
    if not _ENABLED:
        return
    try:
        sym = str(intent.get("symbol") or "unknown")
        _m["orders_errors_total"].labels(sym).inc()
    except Exception:
        return


class _LatencyTimer:
    def __init__(self) -> None:
        import time
        self._t0 = time.time()

    def observe(self) -> None:
        if not _ENABLED:
            return
        import time
        dt = max(0.0, time.time() - self._t0)
        try:
            _m["order_latency_s"].observe(dt)
        except Exception:
            pass


def order_latency_timer() -> _LatencyTimer:
    return _LatencyTimer()

