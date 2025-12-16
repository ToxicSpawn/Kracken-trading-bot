"""Advanced risk management with stress testing and portfolio optimization."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Position representation."""

    symbol: str
    amount: float
    entry_price: float
    current_price: float
    entry_time: datetime


@dataclass
class StressTestResult:
    """Stress test result."""

    scenario: str
    initial_value: float
    final_value: float
    max_drawdown: float
    var_95: float
    var_99: float
    liquidation: bool
    metrics: Dict = field(default_factory=dict)


@dataclass
class StressScenario:
    """Stress test scenario."""

    name: str
    price_changes: Dict[str, float]  # symbol -> percentage change
    volatility_increase: float
    liquidity_decrease: float


class AdvancedRiskManager:
    """Advanced risk manager with stress testing and portfolio optimization."""

    def __init__(
        self,
        var_confidence: float = 0.95,
        var_window: int = 100,
        max_position_size: float = 10.0,
        max_portfolio_var: float = 1000.0,
        max_portfolio_drawdown: float = 20.0,
    ) -> None:
        """
        Initialize risk manager.

        Args:
            var_confidence: VaR confidence level
            var_window: Rolling window for VaR
            max_position_size: Maximum position size
            max_portfolio_var: Maximum portfolio VaR
            max_portfolio_drawdown: Maximum portfolio drawdown (%)
        """
        self.var_confidence = var_confidence
        self.var_window = var_window
        self.max_position_size = max_position_size
        self.max_portfolio_var = max_portfolio_var
        self.max_portfolio_drawdown = max_portfolio_drawdown
        self.portfolio: Dict[str, Position] = {}
        self.returns_history: Dict[str, List[float]] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.stress_test_results: List[StressTestResult] = []

    def update_position(self, symbol: str, amount: float, price: float) -> None:
        """Update position."""
        if symbol not in self.portfolio:
            self.portfolio[symbol] = Position(
                symbol=symbol,
                amount=0.0,
                entry_price=price,
                current_price=price,
                entry_time=datetime.utcnow(),
            )

        position = self.portfolio[symbol]
        position.amount += amount
        position.current_price = price

        if position.amount == 0.0:
            del self.portfolio[symbol]

    def update_price(self, symbol: str, price: float) -> None:
        """Update price for a symbol."""
        if symbol in self.portfolio:
            self.portfolio[symbol].current_price = price

        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(price)

        # Keep only last N prices
        if len(self.price_history[symbol]) > self.var_window * 2:
            self.price_history[symbol] = self.price_history[symbol][-self.var_window * 2 :]

    def record_return(self, symbol: str, price: float) -> None:
        """Record return for a symbol."""
        if symbol in self.portfolio:
            position = self.portfolio[symbol]
            prev_price = position.current_price
            if prev_price > 0.0:
                return_pct = (price - prev_price) / prev_price
                if symbol not in self.returns_history:
                    self.returns_history[symbol] = []
                self.returns_history[symbol].append(return_pct)

                # Keep only last N returns
                if len(self.returns_history[symbol]) > self.var_window:
                    self.returns_history[symbol] = self.returns_history[symbol][-self.var_window :]

    def calculate_var(self, symbol: str) -> Optional[float]:
        """Calculate VaR for a symbol."""
        if symbol not in self.returns_history or len(self.returns_history[symbol]) < 2:
            return None

        returns = self.returns_history[symbol]
        mean = np.mean(returns)
        std = np.std(returns)

        z_score = norm.ppf(1 - self.var_confidence)
        var = mean + z_score * std

        # Get position value
        if symbol in self.portfolio:
            position = self.portfolio[symbol]
            position_value = position.amount * position.current_price
            return position_value * abs(var)

        return None

    def calculate_portfolio_var(self) -> float:
        """Calculate portfolio VaR."""
        total_var = 0.0
        for symbol in self.portfolio.keys():
            var = self.calculate_var(symbol)
            if var:
                total_var += var
        return total_var

    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        return sum(pos.amount * pos.current_price for pos in self.portfolio.values())

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.price_history:
            return 0.0

        # Calculate equity curve
        equity_curve = []
        for symbol, prices in self.price_history.items():
            if symbol in self.portfolio:
                position = self.portfolio[symbol]
                for price in prices:
                    equity_curve.append(position.amount * price)

        if not equity_curve:
            return 0.0

        equity_series = pd.Series(equity_curve)
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max
        return drawdown.min() * 100

    def check_order_risk(self, symbol: str, side: str, amount: float, price: float) -> Tuple[bool, Optional[str]]:
        """
        Check if an order passes risk checks.

        Returns:
            (allowed, error_message)
        """
        # Check position size
        current_position = self.portfolio.get(symbol, Position(symbol, 0.0, price, price, datetime.utcnow())).amount
        new_position = current_position + (amount if side == "buy" else -amount)

        if abs(new_position) > self.max_position_size:
            return False, f"Position size {abs(new_position)} exceeds limit {self.max_position_size}"

        # Check portfolio VaR
        current_var = self.calculate_portfolio_var()
        if current_var > self.max_portfolio_var:
            return False, f"Portfolio VaR {current_var} exceeds limit {self.max_portfolio_var}"

        # Check drawdown
        current_drawdown = self.calculate_max_drawdown()
        if current_drawdown > self.max_portfolio_drawdown:
            return False, f"Portfolio drawdown {current_drawdown}% exceeds limit {self.max_portfolio_drawdown}%"

        return True, None

    def run_stress_test(self, scenario: StressScenario) -> StressTestResult:
        """Run stress test for a scenario."""
        initial_value = self.calculate_portfolio_value()

        # Create temporary portfolio with scenario applied
        temp_portfolio = {}
        for symbol, position in self.portfolio.items():
            price_change = scenario.price_changes.get(symbol, 0.0)
            new_price = position.current_price * (1 + price_change)
            temp_portfolio[symbol] = Position(
                symbol=symbol,
                amount=position.amount,
                entry_price=position.entry_price,
                current_price=new_price,
                entry_time=position.entry_time,
            )

        # Calculate metrics
        final_value = sum(pos.amount * pos.current_price for pos in temp_portfolio.values())
        max_drawdown = abs((final_value - initial_value) / initial_value * 100) if initial_value > 0 else 0.0

        # Calculate VaR (simplified)
        var_95 = self.calculate_portfolio_var() * (1 + scenario.volatility_increase)
        var_99 = var_95 * 1.5

        liquidation = final_value <= 0.0

        result = StressTestResult(
            scenario=scenario.name,
            initial_value=initial_value,
            final_value=final_value,
            max_drawdown=max_drawdown,
            var_95=var_95,
            var_99=var_99,
            liquidation=liquidation,
            metrics={
                "portfolio_value": final_value,
                "portfolio_return": (final_value - initial_value) / initial_value * 100 if initial_value > 0 else 0.0,
            },
        )

        self.stress_test_results.append(result)
        return result

    def generate_stress_scenarios(self) -> List[StressScenario]:
        """Generate standard stress test scenarios."""
        scenarios = []

        # Historical scenarios
        scenarios.append(
            StressScenario(
                name="2008 Financial Crisis",
                price_changes={"BTC/USD": -0.5, "ETH/USD": -0.6},
                volatility_increase=3.0,
                liquidity_decrease=0.5,
            )
        )

        scenarios.append(
            StressScenario(
                name="COVID-19 Crash",
                price_changes={"BTC/USD": -0.4, "ETH/USD": -0.5},
                volatility_increase=4.0,
                liquidity_decrease=0.6,
            )
        )

        scenarios.append(
            StressScenario(
                name="Regulatory Crackdown",
                price_changes={"BTC/USD": -0.7, "ETH/USD": -0.8},
                volatility_increase=5.0,
                liquidity_decrease=0.7,
            )
        )

        # Symbol-specific scenarios
        for symbol in self.portfolio.keys():
            scenarios.append(
                StressScenario(
                    name=f"{symbol} 50% Drop",
                    price_changes={symbol: -0.5},
                    volatility_increase=3.0,
                    liquidity_decrease=0.5,
                )
            )

        return scenarios

    def run_stress_test_suite(self) -> List[StressTestResult]:
        """Run all stress tests."""
        scenarios = self.generate_stress_scenarios()
        results = []
        for scenario in scenarios:
            results.append(self.run_stress_test(scenario))
        return results


class PortfolioOptimizer:
    """Portfolio optimizer using Modern Portfolio Theory."""

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        """
        Initialize portfolio optimizer.

        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio
        """
        self.risk_free_rate = risk_free_rate

    def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        method: str = "mean_variance",
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights.

        Args:
            returns: DataFrame of returns (columns = assets, rows = time)
            method: Optimization method ('mean_variance', 'risk_parity')

        Returns:
            Dictionary of optimal weights
        """
        if method == "mean_variance":
            return self._mean_variance_optimization(returns)
        elif method == "risk_parity":
            return self._risk_parity_optimization(returns)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _mean_variance_optimization(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Mean-variance optimization."""
        # Calculate expected returns and covariance
        expected_returns = returns.mean()
        cov_matrix = returns.cov()

        n = len(expected_returns)

        # Objective: minimize portfolio variance
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        # Bounds: weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(n))

        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n] * n)

        # Optimize
        result = minimize(objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)

        if result.success:
            weights = result.x
            return dict(zip(returns.columns, weights))
        else:
            # Fallback to equal weights
            return {asset: 1.0 / n for asset in returns.columns}

    def _risk_parity_optimization(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Risk parity optimization."""
        cov_matrix = returns.cov()
        n = len(cov_matrix)

        # Risk parity: equal risk contribution
        # Simplified version - use inverse volatility weighting
        volatilities = np.sqrt(np.diag(cov_matrix))
        inv_vol = 1.0 / volatilities
        weights = inv_vol / inv_vol.sum()

        return dict(zip(returns.columns, weights))

    def calculate_efficient_frontier(
        self,
        returns: pd.DataFrame,
        num_points: int = 50,
    ) -> List[Tuple[float, float]]:
        """
        Calculate efficient frontier.

        Args:
            returns: DataFrame of returns
            num_points: Number of points on frontier

        Returns:
            List of (volatility, return) tuples
        """
        expected_returns = returns.mean()
        cov_matrix = returns.cov()

        # Get min and max returns
        min_return = expected_returns.min()
        max_return = expected_returns.max()

        frontier = []
        for target_return in np.linspace(min_return, max_return, num_points):
            # Optimize for target return
            n = len(expected_returns)

            def objective(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w: np.dot(w, expected_returns) - target_return},
            ]

            bounds = tuple((0, 1) for _ in range(n))
            initial_weights = np.array([1.0 / n] * n)

            result = minimize(objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)

            if result.success:
                volatility = result.fun
                frontier.append((volatility, target_return))

        return frontier

