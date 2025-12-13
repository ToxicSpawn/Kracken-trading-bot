from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class AllocatorConfig:
    # Floor prevents "tiny dust trades" unless you explicitly want them.
    min_multiplier: float = 0.0
    max_multiplier: float = 1.0

    # Regime factors
    regime_confident: float = 1.0
    regime_uncertain: float = 0.5
    regime_transition: float = 0.25

    # Exposure factor thresholds (fraction of cap used)
    exposure_t1: float = 0.50
    exposure_t2: float = 0.75
    exposure_f1: float = 1.0
    exposure_f2: float = 0.5
    exposure_f3: float = 0.25

    # Signal quality factor clamp
    signal_floor: float = 0.2
    signal_ceil: float = 1.0

    # Optional scorecard weight support
    default_scorecard_weight: float = 1.0


@dataclass
class AllocationResult:
    multiplier: float
    reasons: Dict[str, Any]


def compute_risk_multiplier(state, intent: Dict[str, Any], cfg: Optional[AllocatorConfig] = None) -> AllocationResult:
    """Global capital allocator applied at the execution choke point.

    Multiplier = LOSS_CLUSTER * REGIME * SIGNAL_QUALITY * EXPOSURE * SCORECARD

    Expected optional state/meta inputs:
      - state.meta['loss_cluster'] : LossClusterTracker with throttle_multiplier()/should_pause()
      - state.meta['omega']['mode'] : NORMAL/SAFE/UNCERTAIN/OFF/SHOCK
      - state.meta['omega']['transition'] : bool (if your reasoning layer emits this)
      - state.meta['exposure_utilization'] : float in [0,1] (optional)
      - state.meta['scorecard'][source]['risk_weight'] : float in [0,1] (optional)

    Expected optional intent meta:
      - intent['meta']['confidence'] : float in [0,1]
      - intent['meta']['disagreement'] : float in [0,1]
      - intent['source'] : strategy/agent name
    """
    cfg = cfg or AllocatorConfig()
    reasons: Dict[str, Any] = {}

    mult = 1.0

    # --- LOSS CLUSTER ---
    lc = (getattr(state, "meta", {}) or {}).get("loss_cluster")
    if lc is not None:
        try:
            if lc.should_pause():
                return AllocationResult(multiplier=0.0, reasons={"loss_cluster": "PAUSED"})
            lcm = float(lc.throttle_multiplier())
            mult *= lcm
            reasons["loss_cluster_mult"] = lcm
        except Exception:
            # If tracker is misconfigured, fail open in research mode (mult unchanged)
            reasons["loss_cluster_mult"] = "ERR"

    # --- REGIME ---
    omega = (getattr(state, "meta", {}) or {}).get("omega", {})
    omega_mode = (omega or {}).get("mode", "NORMAL")
    is_transition = bool((omega or {}).get("transition", False))

    if omega_mode in ("OFF", "SHOCK"):
        return AllocationResult(multiplier=0.0, reasons={"regime": f"OMEGA_{omega_mode}"})
    if is_transition:
        mult *= cfg.regime_transition
        reasons["regime"] = "TRANSITION"
    elif omega_mode in ("UNCERTAIN", "SAFE"):
        mult *= cfg.regime_uncertain
        reasons["regime"] = omega_mode
    else:
        mult *= cfg.regime_confident
        reasons["regime"] = "CONFIDENT"

    # --- SIGNAL QUALITY ---
    meta = intent.get("meta", {}) or {}
    conf = meta.get("confidence")
    disagree = meta.get("disagreement")
    if conf is not None:
        c = float(conf)
        d = float(disagree) if disagree is not None else 0.0
        quality = c * (1.0 - _clamp(d, 0.0, 1.0))
        quality = _clamp(quality, cfg.signal_floor, cfg.signal_ceil)
        mult *= quality
        reasons["signal_quality"] = {"confidence": c, "disagreement": d, "quality": quality}

    # --- EXPOSURE ---
    util = (getattr(state, "meta", {}) or {}).get("exposure_utilization")
    if util is not None:
        u = _clamp(float(util), 0.0, 1.0)
        if u <= cfg.exposure_t1:
            ef = cfg.exposure_f1
        elif u <= cfg.exposure_t2:
            ef = cfg.exposure_f2
        else:
            ef = cfg.exposure_f3
        mult *= ef
        reasons["exposure_utilization"] = {"u": u, "factor": ef}

    # --- SCORECARD ---
    src = intent.get("source") or meta.get("strategy") or "unknown"
    scorecard = (getattr(state, "meta", {}) or {}).get("scorecard", {})
    rw = None
    try:
        rw = scorecard.get(src, {}).get("risk_weight")
    except Exception:
        rw = None
    if rw is None:
        rw = cfg.default_scorecard_weight
    else:
        rw = float(rw)
    rw = _clamp(rw, 0.0, 1.0)
    mult *= rw
    reasons["scorecard_weight"] = {"source": src, "risk_weight": rw}

    mult = _clamp(float(mult), cfg.min_multiplier, cfg.max_multiplier)
    reasons["final_multiplier"] = mult
    return AllocationResult(multiplier=mult, reasons=reasons)

