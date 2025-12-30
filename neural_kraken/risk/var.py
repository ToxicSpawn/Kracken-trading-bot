"""Value at Risk (VaR) and risk management."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


class ValueAtRisk:
    """Value at Risk calculator."""

    def __init__(self, confidence_level: float = 0.95, window: int = 100) -> None:
        """
        Initialize VaR calculator.

        Args:
            confidence_level: Confidence level (0.95 = 95%)
            window: Rolling window size
        """
        self.confidence_level = confidence_level
        self.window = window
        self.returns: List[float] = []

    def update_returns(self, returns: pd.Series) -> None:
        """Update returns history."""
        self.returns = returns.tail(self.window).tolist()

    def calculate_var(self, portfolio_value: float) -> float:
        """
        Calculate parametric VaR.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            VaR value
        """
        if len(self.returns) < 2:
            return 0.0

        mean = np.mean(self.returns)
        std = np.std(self.returns)

        z_score = norm.ppf(1 - self.confidence_level)
        var = portfolio_value * (mean + z_score * std)

        return max(0.0, -var)  # Return positive value

    def calculate_historical_var(self, portfolio_value: float) -> float:
        """
        Calculate historical VaR.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            VaR value
        """
        if len(self.returns) < 2:
            return 0.0

        portfolio_returns = portfolio_value * np.array(self.returns)
        var = np.percentile(portfolio_returns, 100 * (1 - self.confidence_level))

        return max(0.0, -var)

    def calculate_cvar(self, portfolio_value: float) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).

        Args:
            portfolio_value: Current portfolio value

        Returns:
            CVaR value
        """
        if len(self.returns) < 2:
            return 0.0

        var = self.calculate_historical_var(portfolio_value)
        portfolio_returns = portfolio_value * np.array(self.returns)
        cvar = portfolio_returns[portfolio_returns <= -var].mean()

        return max(0.0, -cvar)


class StressTester:
    """Stress testing for portfolios."""

    def __init__(self, initial_portfolio: dict) -> None:
        """
        Initialize stress tester.

        Args:
            initial_portfolio: Initial portfolio dictionary
        """
        self.initial_portfolio = initial_portfolio
        self.scenarios = self._generate_scenarios()

    def _generate_scenarios(self) -> List[dict]:
        """Generate stress test scenarios."""
        return [
            {
                "name": "2008 Financial Crisis",
                "price_changes": {
                    "BTC/USD": -0.5,
                    "ETH/USD": -0.6,
                },
                "volatility_increase": 3.0,
            },
            {
                "name": "COVID-19 Crash",
                "price_changes": {
                    "BTC/USD": -0.4,
                    "ETH/USD": -0.5,
                },
                "volatility_increase": 4.0,
            },
            {
                "name": "Regulatory Crackdown",
                "price_changes": {
                    "BTC/USD": -0.7,
                    "ETH/USD": -0.8,
                },
                "volatility_increase": 5.0,
            },
        ]

    def run_test(self, scenario: dict) -> dict:
        """
        Run stress test for a scenario.

        Args:
            scenario: Scenario dictionary

        Returns:
            Test results
        """
        portfolio = self.initial_portfolio.copy()
        initial_value = self._calculate_portfolio_value(portfolio)

        # Apply price changes
        for asset, change in scenario["price_changes"].items():
            if asset in portfolio:
                portfolio[asset]["price"] *= (1 + change)

        final_value = self._calculate_portfolio_value(portfolio)
        max_drawdown = (initial_value - final_value) / initial_value

        return {
            "scenario": scenario["name"],
            "initial_value": initial_value,
            "final_value": final_value,
            "max_drawdown": max_drawdown,
            "liquidation": final_value < 0,
        }

    def _calculate_portfolio_value(self, portfolio: dict) -> float:
        """Calculate total portfolio value."""
        total = 0.0
        for asset, data in portfolio.items():
            total += data["amount"] * data["price"]
        return total

    def run_all_tests(self) -> List[dict]:
        """Run all stress tests."""
        return [self.run_test(scenario) for scenario in self.scenarios]

