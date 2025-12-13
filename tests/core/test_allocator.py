from core.allocator import compute_risk_multiplier
from core.state import GlobalState


def test_allocator_defaults_to_1():
    s = GlobalState()
    it = {"symbol": "BTC/AUD", "side": "BUY", "qty": 1.0, "meta": {"confidence": 1.0, "disagreement": 0.0}, "source": "x"}
    res = compute_risk_multiplier(s, it)
    assert 0.0 <= res.multiplier <= 1.0


def test_allocator_blocks_off_mode():
    s = GlobalState()
    s.meta["omega"] = {"mode": "OFF"}
    it = {"symbol": "BTC/AUD", "side": "BUY", "qty": 1.0, "meta": {"confidence": 1.0}}
    res = compute_risk_multiplier(s, it)
    assert res.multiplier == 0.0

