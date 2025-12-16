"""Example strategies for backtesting."""

from __future__ import annotations

from typing import Dict, Optional


def mean_reversion_strategy(data: Dict, window: int = 20, z_threshold: float = 1.5) -> Optional[Dict]:
    """Mean reversion strategy."""
    prices = data.get("close_history", [])

    if len(prices) < window:
        return None

    # Calculate mean and std
    window_prices = prices[-window:]
    mean = sum(window_prices) / len(window_prices)
    variance = sum((x - mean) ** 2 for x in window_prices) / len(window_prices)
    std = variance ** 0.5

    if std == 0:
        return None

    # Calculate z-score
    z_score = (data["price"] - mean) / std

    # Generate signal
    if z_score > z_threshold:
        return {
            "signal": "SELL",
            "confidence": abs(z_score),
            "details": {
                "strategy": "mean_reversion",
                "mean": mean,
                "std": std,
                "z_score": z_score,
                "window": window,
            },
        }
    elif z_score < -z_threshold:
        return {
            "signal": "BUY",
            "confidence": abs(z_score),
            "details": {
                "strategy": "mean_reversion",
                "mean": mean,
                "std": std,
                "z_score": z_score,
                "window": window,
            },
        }

    return None


def momentum_strategy(data: Dict, window: int = 5) -> Optional[Dict]:
    """Momentum strategy."""
    prices = data.get("close_history", [])

    if len(prices) < window + 1:
        return None

    # Calculate returns
    recent_return = (prices[-1] - prices[-window-1]) / prices[-window-1]

    # Generate signal
    if recent_return > 0.01:  # 1% threshold
        return {
            "signal": "BUY",
            "confidence": recent_return,
            "details": {
                "strategy": "momentum",
                "return": recent_return,
                "window": window,
            },
        }
    elif recent_return < -0.01:
        return {
            "signal": "SELL",
            "confidence": abs(recent_return),
            "details": {
                "strategy": "momentum",
                "return": recent_return,
                "window": window,
            },
        }

    return None


def rsi_strategy(data: Dict, overbought: int = 70, oversold: int = 30) -> Optional[Dict]:
    """RSI strategy."""
    rsi = data.get("rsi", 50)

    if rsi > overbought:
        return {
            "signal": "SELL",
            "confidence": (rsi - overbought) / (100 - overbought),
            "details": {
                "strategy": "rsi",
                "rsi": rsi,
                "overbought": overbought,
                "oversold": oversold,
            },
        }
    elif rsi < oversold:
        return {
            "signal": "BUY",
            "confidence": (oversold - rsi) / oversold,
            "details": {
                "strategy": "rsi",
                "rsi": rsi,
                "overbought": overbought,
                "oversold": oversold,
            },
        }

    return None


def combined_strategy(data: Dict) -> Optional[Dict]:
    """Combine multiple strategies with voting."""
    strategies = [
        mean_reversion_strategy(data, window=20, z_threshold=1.5),
        momentum_strategy(data, window=5),
        rsi_strategy(data, overbought=70, oversold=30),
    ]

    # Filter out None signals
    signals = [s for s in strategies if s is not None]

    if not signals:
        return None

    # Count signals
    buy_signals = sum(1 for s in signals if s["signal"] == "BUY")
    sell_signals = sum(1 for s in signals if s["signal"] == "SELL")

    # Calculate confidence
    total_signals = len(signals)
    confidence = (buy_signals - sell_signals) / total_signals

    # Determine final signal
    if confidence > 0.3:  # More than 65% buy signals
        return {
            "signal": "BUY",
            "confidence": confidence,
            "details": {
                "strategy": "combined",
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "total_strategies": total_signals,
            },
        }
    elif confidence < -0.3:  # More than 65% sell signals
        return {
            "signal": "SELL",
            "confidence": -confidence,
            "details": {
                "strategy": "combined",
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "total_strategies": total_signals,
            },
        }

    return None

