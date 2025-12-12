from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AllocationDecision:
    multiplier: float
    reason: str


@dataclass
class AllocationConfig:
    # How aggressively to shrink a strategy after underperformance
    min_mult: float = 0.0
    max_mult: float = 1.0
    default_mult: float = 1.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def allocate_risk_multiplier(state, intent: Dict[str, Any], cfg: Optional[AllocationConfig] = None) -> AllocationDecision:
    """Capital allocation layer (strategy + symbol risk budget).

    This is intentionally lightweight: it looks for performance signals that your system already emits
    (scorecard/metrics/meta). If missing, it defaults to 1.0 and does nothing.

    Expected optional inputs:
      - intent['source'] : agent/strategy name
      - state.meta['scorecard'][source]['risk_weight'] : 0..1
      - state.meta['symbol_caps'][symbol] : 0..1
    """
    cfg = cfg or AllocationConfig()

    src = intent.get("source") or intent.get("meta", {}).get("strategy") or "unknown"
    symbol = intent.get("symbol")

    scorecard = (state.meta or {}).get("scorecard", {}) if hasattr(state, "meta") else {}
    src_obj = scorecard.get(src, {})
    rw = src_obj.get("risk_weight")

    symbol_caps = (state.meta or {}).get("symbol_caps", {}) if hasattr(state, "meta") else {}
    sw = symbol_caps.get(symbol)

    mult = cfg.default_mult
    reason = "DEFAULT"

    if rw is not None:
        mult *= float(rw)
        reason = "SCORECARD_WEIGHT"
    if sw is not None:
        mult *= float(sw)
        reason = reason + "+SYMBOL_CAP"

    mult = _clamp(float(mult), cfg.min_mult, cfg.max_mult)
    return AllocationDecision(multiplier=mult, reason=reason)

