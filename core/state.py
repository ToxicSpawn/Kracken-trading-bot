from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from collections import deque

from core.loss_cluster import LossClusterTracker


@dataclass
class PositionSnapshot:
    symbol: str
    size: float = 0.0
    entry_price: float | None = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccountState:
    name: str
    equity: float = 0.0
    balance: float = 0.0
    positions: Dict[str, PositionSnapshot] = field(default_factory=dict)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_drawdown_pct: float = -20.0
    risk_multiplier: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairSnapshot:
    symbol: str
    last_price: float | None = None
    last_signal_short: str | None = None
    last_signal_mid: str | None = None
    last_signal_long: str | None = None
    trend_score: float = 0.0
    vol_score: float = 0.0
    regime: str = "sideways"
    liquidity_score: float = 0.0
    slippage_bps: float = 0.0
    stress_multiplier: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalState:
    accounts: Dict[str, AccountState] = field(default_factory=dict)
    pairs: Dict[str, PairSnapshot] = field(default_factory=dict)
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    trading_enabled: bool = True
    mode: str = "paper"
    meta: Dict[str, Any] = field(default_factory=dict)
    _intent_queue: deque = field(default_factory=deque, init=False)

    def loss_cluster(self) -> LossClusterTracker:
        lc = self.meta.get("loss_cluster")
        if lc is None:
            lc = LossClusterTracker()
            self.meta["loss_cluster"] = lc
        return lc

    def get_loss_cluster(self) -> LossClusterTracker:
        """Alias for loss_cluster() for compatibility."""
        return self.loss_cluster()

    def record_trade_outcome(self, pnl: float, ts_iso: Optional[str] = None) -> None:
        """Research-friendly helper: also stores last outcome in state.meta."""
        self.meta["last_trade_pnl"] = float(pnl)
        if ts_iso:
            self.meta["last_trade_ts"] = ts_iso
        self.get_loss_cluster().record_outcome(float(pnl))

    def add_intent(self, intent: Dict[str, Any]) -> None:
        """Add a trading intent to the queue."""
        self._intent_queue.append(intent)

    def drain_intents(self, max_n: int = 100) -> List[Dict[str, Any]]:
        """Drain up to max_n intents from the queue."""
        intents = []
        for _ in range(min(max_n, len(self._intent_queue))):
            if self._intent_queue:
                intents.append(self._intent_queue.popleft())
        return intents