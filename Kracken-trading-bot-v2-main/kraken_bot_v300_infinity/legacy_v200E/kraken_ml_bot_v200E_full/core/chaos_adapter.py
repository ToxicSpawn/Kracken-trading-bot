from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional


class ChaosAdapter:
    """Wrapper adapter that can inject failures for chaos testing."""

    def __init__(self, inner, fail_rate: float = 0.0, delay_ms: int = 0) -> None:
        self.inner = inner
        self.fail_rate = float(fail_rate)
        self.delay_ms = int(delay_ms)

    async def place_order(self, intent: Dict[str, Any]) -> Any:
        import random
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000.0)
        if self.fail_rate > 0 and random.random() < self.fail_rate:
            raise RuntimeError("CHAOS_INJECTED_FAILURE")
        return await self.inner.place_order(intent)

    # passthrough for reconciler interfaces if needed
    def __getattr__(self, name: str):
        return getattr(self.inner, name)

