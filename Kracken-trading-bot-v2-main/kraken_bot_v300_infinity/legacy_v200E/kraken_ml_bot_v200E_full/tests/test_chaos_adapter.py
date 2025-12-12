import asyncio
import pytest

from core.chaos_adapter import ChaosAdapter


class DummyAdapter:
    async def place_order(self, intent):
        return {"ok": True, "intent": intent}


@pytest.mark.asyncio
async def test_chaos_failure_injection():
    inner = DummyAdapter()
    chaos = ChaosAdapter(inner, fail_rate=1.0, delay_ms=0)
    with pytest.raises(RuntimeError):
        await chaos.place_order({"symbol": "BTC/AUD", "side": "BUY", "qty": 1.0})

