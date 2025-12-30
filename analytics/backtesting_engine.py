"""Backtesting engine using backtrader for strategy testing."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

import backtrader as bt
import pandas as pd

from strategies.base import SignalType

logger = logging.getLogger(__name__)


class StrategyBacktester(bt.Strategy):
    """
    Generic backtesting strategy wrapper that can be used with any signal generator.
    """

    params = (
        ("initial_cash", 10000.0),
        ("commission", 0.001),  # 0.1% commission
        ("slippage", 0.0005),  # 0.05% slippage
    )

    def __init__(self, signal_generator: Optional[Any] = None) -> None:
        """
        Initialize strategy.

        Args:
            signal_generator: Optional function/object that generates trading signals
        """
        self.signal_generator = signal_generator
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def log(self, txt: str, dt: Optional[datetime] = None) -> None:
        """Log trading actions."""
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f"{dt.isoformat()}: {txt}")

    def notify_order(self, order: bt.Order) -> None:
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}",
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(
                    f"SELL EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}",
                )

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        self.order = None

    def notify_trade(self, trade: bt.Trade) -> None:
        """Handle trade notifications."""
        if not trade.isclosed:
            return

        self.log(f"OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}")

    def next(self) -> None:
        """Execute strategy logic on each bar."""
        # Get signal from generator if available
        if self.signal_generator:
            try:
                signal = self.signal_generator.get_signal(
                    self.datas[0].datetime.datetime(0),
                    self.datas[0].close[0],
                )
            except Exception as e:
                logger.warning(f"Signal generator error: {e}")
                signal = None
        else:
            signal = None

        # Default: simple buy and hold (override in subclasses)
        if signal is None:
            return

        # Execute based on signal
        if signal.signal == SignalType.BUY and not self.position:
            self.order = self.buy()
        elif signal.signal == SignalType.SELL and self.position:
            self.order = self.close()


class BacktestEngine:
    """
    Backtesting engine for testing trading strategies.
    """

    def __init__(
        self,
        initial_cash: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ) -> None:
        """
        Initialize backtest engine.

        Args:
            initial_cash: Starting capital
            commission: Commission rate (e.g., 0.001 = 0.1%)
            slippage: Slippage rate (e.g., 0.0005 = 0.05%)
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy: type[bt.Strategy],
        strategy_params: Optional[dict[str, Any]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Run a backtest.

        Args:
            data: DataFrame with columns [timestamp, open, high, low, close, volume]
            strategy: Strategy class to test
            strategy_params: Strategy parameters
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            Dictionary with backtest results
        """
        # Create Cerebro engine
        cerebro = bt.Cerebro()

        # Add strategy
        if strategy_params:
            cerebro.addstrategy(strategy, **strategy_params)
        else:
            cerebro.addstrategy(strategy)

        # Prepare data
        data.index = pd.to_datetime(data.index)
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        # Convert to backtrader format
        feed = bt.feeds.PandasData(dataname=data)

        # Add data feed
        cerebro.adddata(feed)

        # Set initial cash
        cerebro.broker.setcash(self.initial_cash)

        # Set commission
        cerebro.broker.setcommission(commission=self.commission)

        # Add slippage
        if self.slippage > 0:
            cerebro.broker.set_slippage_perc(perc=self.slippage)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        # Run backtest
        logger.info("Starting backtest...")
        results = cerebro.run()
        result = results[0]

        # Extract results
        final_value = cerebro.broker.getvalue()
        sharpe = result.analyzers.sharpe.get_analysis().get("sharperatio", 0.0)
        drawdown = result.analyzers.drawdown.get_analysis()
        # returns = result.analyzers.returns.get_analysis()  # Not currently used
        trades = result.analyzers.trades.get_analysis()

        return {
            "initial_cash": self.initial_cash,
            "final_value": final_value,
            "total_return": (final_value - self.initial_cash) / self.initial_cash,
            "sharpe_ratio": sharpe if sharpe else 0.0,
            "max_drawdown": drawdown.get("max", {}).get("drawdown", 0.0),
            "total_trades": trades.get("total", {}).get("total", 0),
            "winning_trades": trades.get("won", {}).get("total", 0),
            "losing_trades": trades.get("lost", {}).get("total", 0),
            "win_rate": (
                trades.get("won", {}).get("total", 0)
                / max(trades.get("total", {}).get("total", 1), 1)
            ),
            "average_win": trades.get("won", {}).get("pnl", {}).get("average", 0.0),
            "average_loss": trades.get("lost", {}).get("pnl", {}).get("average", 0.0),
        }

    def plot_results(self, cerebro: bt.Cerebro, filename: Optional[str] = None) -> None:
        """
        Plot backtest results.

        Args:
            cerebro: Cerebro engine instance
            filename: Optional filename to save plot
        """
        try:
            cerebro.plot(style="candlestick", barup="green", bardown="red")
            if filename:
                import matplotlib.pyplot as plt

                plt.savefig(filename)
                logger.info(f"Plot saved to {filename}")
        except Exception as e:
            logger.warning(f"Failed to plot results: {e}")


# Example: Simple Moving Average Crossover Strategy
class SMACrossStrategy(bt.Strategy):
    """Simple Moving Average Crossover Strategy."""

    params = (
        ("fast_period", 10),
        ("slow_period", 30),
        ("printlog", False),
    )

    def __init__(self) -> None:
        """Initialize indicators."""
        self.sma_fast = bt.indicators.SMA(period=self.p.fast_period)
        self.sma_slow = bt.indicators.SMA(period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

    def next(self) -> None:
        """Execute strategy on each bar."""
        if not self.position:
            if self.crossover > 0:  # Fast crosses above slow
                self.buy()
        else:
            if self.crossover < 0:  # Fast crosses below slow
                self.close()

