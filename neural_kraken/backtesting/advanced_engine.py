"""Advanced backtesting engine with walk-forward analysis and comprehensive metrics."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot  # noqa: F401
    import seaborn  # noqa: F401
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available for plotting")

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Trade representation."""

    symbol: str
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    amount: float
    side: str
    pnl: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    tags: Dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Backtest results."""

    initial_balance: float
    final_balance: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    expectancy: float
    trades: List[Trade]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    metrics: Dict = field(default_factory=dict)


class AdvancedBacktestingEngine:
    """Advanced backtesting engine with walk-forward analysis."""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        fee_rate: float = 0.001,
        slippage: float = 0.0005,
        risk_free_rate: float = 0.02,
    ) -> None:
        """
        Initialize backtesting engine.

        Args:
            initial_balance: Starting balance
            fee_rate: Trading fee rate
            slippage: Slippage rate
            risk_free_rate: Risk-free rate for Sharpe ratio
        """
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.strategies: Dict[str, Callable] = {}
        self.data: Optional[pd.DataFrame] = None
        self.results: Dict[str, BacktestResult] = {}

    def add_strategy(self, name: str, strategy_func: Callable) -> None:
        """Add a strategy to backtest."""
        self.strategies[name] = strategy_func

    def load_data(self, data: pd.DataFrame) -> None:
        """Load historical data."""
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")

        data = data.copy()
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data = data.set_index("timestamp").sort_index()
        else:
            data.index = pd.to_datetime(data.index)

        # Add technical indicators
        data = self._add_features(data)
        self.data = data

    def run_backtest(
        self,
        strategy_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, BacktestResult]:
        """Run backtest for strategies."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Filter data
        data = self.data.copy()
        if start_date:
            data = data[data.index >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data.index <= pd.to_datetime(end_date)]

        strategies_to_test = [strategy_name] if strategy_name else list(self.strategies.keys())

        for name in strategies_to_test:
            if name not in self.strategies:
                logger.warning(f"Strategy {name} not found")
                continue

            logger.info(f"Running backtest for strategy: {name}")
            result = self._run_single_backtest(name, data)
            self.results[name] = result

        return self.results

    def _run_single_backtest(self, strategy_name: str, data: pd.DataFrame) -> BacktestResult:
        """Run backtest for a single strategy."""
        strategy = self.strategies[strategy_name]

        # Initialize state
        balance = self.initial_balance
        position = 0.0
        trades: List[Trade] = []
        equity_curve = []
        current_trade: Optional[Trade] = None
        price_history = []

        # Run backtest
        for i, (timestamp, row) in enumerate(data.iterrows()):
            price_history.append(row["close"])

            # Get signal from strategy
            signal_data = {
                "price": row["close"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "volume": row["volume"],
                "returns": row.get("returns", 0.0),
                "ma_5": row.get("ma_5", row["close"]),
                "ma_20": row.get("ma_20", row["close"]),
                "rsi": row.get("rsi", 50.0),
                "volatility": row.get("volatility", 0.0),
                "close_history": price_history[-60:] if len(price_history) > 60 else price_history,
            }

            try:
                signal = strategy(signal_data)
            except Exception as e:
                logger.warning(f"Strategy error at {timestamp}: {e}")
                signal = None

            # Execute trades
            if signal and signal.get("signal") != "HOLD":
                current_price = row["close"]

                # Calculate order price with slippage
                if signal["signal"] == "BUY":
                    order_price = current_price * (1 + self.slippage)
                else:
                    order_price = current_price * (1 - self.slippage)

                # Calculate order size
                order_size = (balance * 0.1) / order_price if signal["signal"] == "BUY" else position * 0.1

                # Execute trade
                if signal["signal"] == "BUY" and position == 0:
                    # Open long position
                    entry_price = order_price
                    position = order_size
                    fees = order_size * entry_price * self.fee_rate
                    current_trade = Trade(
                        symbol="BTC/USD",
                        entry_time=timestamp,
                        exit_time=None,
                        entry_price=entry_price,
                        exit_price=None,
                        amount=order_size,
                        side="BUY",
                        fees=fees,
                        slippage=self.slippage,
                        tags=signal.get("details", {}),
                    )
                    balance -= order_size * entry_price * (1 + self.fee_rate)

                elif signal["signal"] == "SELL" and position > 0 and current_trade:
                    # Close long position
                    exit_price = order_price
                    pnl = (exit_price - current_trade.entry_price) * current_trade.amount
                    fees = current_trade.amount * exit_price * self.fee_rate
                    current_trade.exit_time = timestamp
                    current_trade.exit_price = exit_price
                    current_trade.pnl = pnl - fees - current_trade.fees
                    current_trade.fees += fees
                    trades.append(current_trade)
                    balance += current_trade.amount * exit_price * (1 - self.fee_rate)
                    position = 0.0
                    current_trade = None

            # Update equity curve
            equity = balance + (position * row["close"] if position > 0 else 0)
            equity_curve.append((timestamp, equity))

        # Close any open position
        if position > 0 and current_trade:
            last_row = data.iloc[-1]
            exit_price = last_row["close"] * (1 - self.slippage)
            pnl = (exit_price - current_trade.entry_price) * current_trade.amount
            fees = current_trade.amount * exit_price * self.fee_rate
            current_trade.exit_time = data.index[-1]
            current_trade.exit_price = exit_price
            current_trade.pnl = pnl - fees - current_trade.fees
            current_trade.fees += fees
            trades.append(current_trade)

        # Calculate metrics
        equity_series = pd.Series(
            [x[1] for x in equity_curve],
            index=[x[0] for x in equity_curve],
        )

        return self._calculate_metrics(equity_series, trades)

    def _add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        # Returns
        data["returns"] = data["close"].pct_change()

        # Moving averages
        data["ma_5"] = data["close"].rolling(5).mean()
        data["ma_20"] = data["close"].rolling(20).mean()

        # RSI
        data["rsi"] = self._compute_rsi(data["close"])

        # Volatility
        data["volatility"] = data["returns"].rolling(20).std()

        return data.dropna()

    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_metrics(self, equity_curve: pd.Series, trades: List[Trade]) -> BacktestResult:
        """Calculate comprehensive performance metrics."""
        if len(equity_curve) == 0:
            return BacktestResult(
                initial_balance=self.initial_balance,
                final_balance=self.initial_balance,
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                calmar_ratio=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                expectancy=0.0,
                trades=trades,
                equity_curve=equity_curve,
                drawdown_curve=pd.Series(),
                metrics={},
            )

        # Returns
        returns = equity_curve.pct_change().dropna()

        # Total return
        total_return = (equity_curve.iloc[-1] / self.initial_balance - 1) * 100

        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        annualized_return = ((1 + total_return / 100) ** (365.25 / days) - 1) * 100 if days > 0 else 0.0

        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0.0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else float("inf")

        # Max drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        drawdown_curve = drawdown

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else float("inf")

        # Win rate
        winning_trades = [t for t in trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0.0

        # Profit factor
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Expectancy
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        losing_trades = [t for t in trades if t.pnl < 0]
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss)) if trades else 0.0

        # Additional metrics
        metrics = {
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "avg_trade_pnl": sum(t.pnl for t in trades) / len(trades) if trades else 0.0,
            "avg_win_pnl": avg_win,
            "avg_loss_pnl": avg_loss,
            "largest_win": max(t.pnl for t in trades) if trades else 0.0,
            "largest_loss": min(t.pnl for t in trades) if trades else 0.0,
        }

        return BacktestResult(
            initial_balance=self.initial_balance,
            final_balance=equity_curve.iloc[-1],
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            trades=trades,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            metrics=metrics,
        )

    def run_walk_forward_analysis(
        self,
        strategy_name: str,
        window_size: str = "30D",
        step_size: str = "7D",
        lookahead: str = "1D",
    ) -> Dict:
        """Run walk-forward analysis."""
        if self.data is None:
            raise ValueError("No data loaded.")

        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not found")

        # Convert to timedelta
        window_delta = pd.to_timedelta(window_size)
        step_delta = pd.to_timedelta(step_size)
        lookahead_delta = pd.to_timedelta(lookahead)

        # Generate windows
        start_date = self.data.index[0]
        end_date = self.data.index[-1]

        windows = []
        current_start = start_date
        while current_start + window_delta + lookahead_delta <= end_date:
            current_end = current_start + window_delta
            test_start = current_end
            test_end = current_end + lookahead_delta

            windows.append({
                "train": (current_start, current_end),
                "test": (test_start, test_end),
            })

            current_start += step_delta

        # Run walk-forward analysis
        results = []
        for i, window in enumerate(windows):
            train_start, train_end = window["train"]
            test_start, test_end = window["test"]

            # Filter data
            test_data = self.data[(self.data.index >= test_start) & (self.data.index < test_end)]

            if len(test_data) == 0:
                continue

            # Run backtest
            result = self._run_single_backtest(strategy_name, test_data)
            results.append({
                "window": i + 1,
                "train_period": (train_start, train_end),
                "test_period": (test_start, test_end),
                "result": result,
            })

        # Aggregate results
        aggregate = self._aggregate_walk_forward_results(results)

        return {
            "windows": results,
            "aggregate": aggregate,
        }

    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict:
        """Aggregate walk-forward results."""
        if not results:
            return {}

        metrics_list = []
        for result in results:
            res = result["result"]
            metrics_list.append({
                "total_return": res.total_return,
                "annualized_return": res.annualized_return,
                "sharpe_ratio": res.sharpe_ratio,
                "max_drawdown": res.max_drawdown,
                "win_rate": res.win_rate,
                "profit_factor": res.profit_factor,
            })

        metrics_df = pd.DataFrame(metrics_list)

        return {
            "avg_total_return": metrics_df["total_return"].mean(),
            "std_total_return": metrics_df["total_return"].std(),
            "avg_sharpe_ratio": metrics_df["sharpe_ratio"].mean(),
            "avg_max_drawdown": metrics_df["max_drawdown"].mean(),
            "avg_win_rate": metrics_df["win_rate"].mean(),
            "robustness_score": self._calculate_robustness_score(metrics_df),
        }

    def _calculate_robustness_score(self, metrics_df: pd.DataFrame) -> float:
        """Calculate robustness score (0-100)."""
        cv_total_return = metrics_df["total_return"].std() / metrics_df["total_return"].mean() if metrics_df["total_return"].mean() != 0 else 1.0
        cv_sharpe = metrics_df["sharpe_ratio"].std() / metrics_df["sharpe_ratio"].mean() if metrics_df["sharpe_ratio"].mean() != 0 else 1.0

        consistency_scores = [
            1 - min(cv_total_return, 1.0),
            1 - min(cv_sharpe, 1.0),
            metrics_df["win_rate"].mean(),
            metrics_df["profit_factor"].mean() / (metrics_df["profit_factor"].mean() + 1) if metrics_df["profit_factor"].mean() > 0 else 0.0,
        ]

        return np.mean(consistency_scores) * 100

