from core.eligibility import evaluate_trade_eligibility, EligibilityConfig
from core.capital_allocator import allocate_risk_multiplier
from core.state import GlobalState, PairSnapshot, AccountState


def test_eligibility_blocks_paused():
    s = GlobalState()
    s.system_mode = "PAUSED"
    d = evaluate_trade_eligibility(s, {"symbol": "BTC/AUD", "side": "BUY", "qty": 1.0})
    assert d.allow is False


def test_eligibility_throttles_safe_mode():
    s = GlobalState()
    s.meta["omega"] = {"mode": "SAFE"}
    d = evaluate_trade_eligibility(s, {"symbol": "BTC/AUD", "side": "BUY", "qty": 1.0}, EligibilityConfig())
    assert d.allow is True
    assert d.multiplier < 1.0


def test_allocator_uses_scorecard_weight():
    s = GlobalState()
    s.meta["scorecard"] = {"execution_intent": {"risk_weight": 0.5}}
    it = {"source": "execution_intent", "symbol": "BTC/AUD", "side": "BUY", "qty": 2.0}
    a = allocate_risk_multiplier(s, it)
    assert abs(a.multiplier - 0.5) < 1e-9

