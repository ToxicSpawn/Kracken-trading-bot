# Level-15 Patch: Capital Allocator + Trade Eligibility Gate + Chaos Testing

This package adds the remaining "prop firm" control layers:

## 1) Trade Eligibility Gate
- core/eligibility.py
Hard gates and throttles trading when:
- Omega mode is OFF/SHOCK (drop)
- Omega mode is SAFE/UNCERTAIN (throttle)
- signal confidence is low (drop)
- spread is wide (throttle)
- slippage is high (drop)
- strategy disagreement is high (throttle)

## 2) Capital Allocation Layer
- core/capital_allocator.py
Applies a risk multiplier based on:
- state.meta['scorecard'][source]['risk_weight']
- state.meta['symbol_caps'][symbol]

Defaults to 1.0 if not present.

## 3) Chaos Adapter + Tests
- core/chaos_adapter.py
Allows injecting failures and delay for robustness testing.

## How to integrate
1) Add files in /core and /tests.
2) Insert the snippet in README_SNIPPET_RUNTIME.md into core/runtime.py
   inside the governed execution loop, before RiskGovernor evaluation.
3) Run:
   pytest -q

## Not financial advice
This code is for engineering/testing and does not constitute financial advice.

