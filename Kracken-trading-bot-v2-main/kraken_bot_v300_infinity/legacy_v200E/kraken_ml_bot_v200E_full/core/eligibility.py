from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class EligibilityDecision:
    allow: bool
    mode: str  # ALLOW | DROP | THROTTLE
    reason: str
    multiplier: float = 1.0


@dataclass
class EligibilityConfig:
    # Conservative defaults; adjust in config/policies.yaml if desired
    min_signal_confidence: float = 0.55
    max_spread_bps: float = 80.0
    max_slippage_bps: float = 120.0
    block_modes: Tuple[str, ...] = ("OFF", "SHOCK")
    throttle_modes: Tuple[str, ...] = ("SAFE", "UNCERTAIN")
    throttle_multiplier: float = 0.25


def evaluate_trade_eligibility(state, intent: Dict[str, Any], cfg: Optional[EligibilityConfig] = None) -> EligibilityDecision:
    cfg = cfg or EligibilityConfig()

    # System mode hard gates
    if getattr(state, "system_mode", "NORMAL") == "PAUSED":
        return EligibilityDecision(False, "DROP", "SYSTEM_PAUSED", 0.0)

    # Omega mode gates (from ReasoningLayerAgent)
    omega = (state.meta or {}).get("omega", {}) if hasattr(state, "meta") else {}
    omega_mode = omega.get("mode", "NORMAL")
    if omega_mode in cfg.block_modes:
        return EligibilityDecision(False, "DROP", f"OMEGA_{omega_mode}", 0.0)
    if omega_mode in cfg.throttle_modes:
        return EligibilityDecision(True, "THROTTLE", f"OMEGA_{omega_mode}", cfg.throttle_multiplier)

    # Signal confidence gate (if provided by upstream)
    conf = intent.get("meta", {}).get("confidence")
    if conf is not None and float(conf) < cfg.min_signal_confidence:
        return EligibilityDecision(False, "DROP", "LOW_CONFIDENCE", 0.0)

    symbol = intent.get("symbol")
    ps = (state.pairs or {}).get(symbol) if hasattr(state, "pairs") else None
    if ps:
        spread_bps = float(ps.meta.get("spread_bps", 0.0) if hasattr(ps, "meta") and isinstance(ps.meta, dict) else 0.0)
        slippage_bps = float(getattr(ps, "slippage_bps", 0.0) or 0.0)
        if spread_bps and spread_bps > cfg.max_spread_bps:
            return EligibilityDecision(True, "THROTTLE", "WIDE_SPREAD", 0.25)
        if slippage_bps and slippage_bps > cfg.max_slippage_bps:
            return EligibilityDecision(False, "DROP", "HIGH_SLIPPAGE", 0.0)

    # Disagreement gate (if blending meta exists)
    disagree = intent.get("meta", {}).get("disagreement")
    if disagree is not None and float(disagree) > 0.65:
        return EligibilityDecision(True, "THROTTLE", "HIGH_DISAGREEMENT", 0.25)

    return EligibilityDecision(True, "ALLOW", "OK", 1.0)

