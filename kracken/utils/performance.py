"""Performance metrics calculation and tracking."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def calculate_performance_metrics(trades: List[Dict], equity_curve: pd.DataFrame, 
                                  initial_balance: float) -> Dict:
    """Calculate performance metrics from trades and equity curve."""
    if not trades:
        return {
            "Total Trades": 0,
            "Return [%]": 0,
            "Sharpe Ratio": 0,
            "Max Drawdown [%]": 0,
            "Win Rate [%]": 0,
            "Profit Factor": 0,
            "Avg Win [%]": 0,
            "Avg Loss [%]": 0,
            "Expectancy": 0
        }
    
    # Calculate basic metrics
    total_trades = len(trades)
    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0) <= 0]
    
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    
    total_win = sum(t.get("pnl", 0) for t in winning_trades) if winning_trades else 0
    total_loss = abs(sum(t.get("pnl", 0) for t in losing_trades)) if losing_trades else 0
    profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
    
    avg_win = sum(t.get("pnl_pct", 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t.get("pnl_pct", 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
    expectancy = (win_rate * avg_win - (100 - win_rate) * abs(avg_loss)) / 100 if total_trades > 0 else 0
    
    # Calculate returns
    final_balance = trades[-1].get("balance", initial_balance) if trades else initial_balance
    total_return = (final_balance - initial_balance) / initial_balance * 100
    
    # Calculate drawdown
    if not equity_curve.empty and "balance" in equity_curve.columns:
        equity = equity_curve["balance"].values
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max * 100
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    else:
        max_drawdown = 0
    
    # Calculate Sharpe ratio (assuming risk-free rate = 0)
    if not equity_curve.empty and "balance" in equity_curve.columns:
        daily_returns = equity_curve["balance"].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 1 and daily_returns.std() > 0 else 0
    else:
        sharpe_ratio = 0
    
    return {
        "Total Trades": total_trades,
        "Return [%]": total_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown [%]": max_drawdown,
        "Win Rate [%]": win_rate,
        "Profit Factor": profit_factor,
        "Avg Win [%]": avg_win,
        "Avg Loss [%]": avg_loss,
        "Expectancy": expectancy,
        "Final Balance": final_balance
    }


class PerformanceTracker:
    """Tracks performance for individual strategies."""
    
    def __init__(self, config: Dict, strategy_name: str):
        self.config = config
        self.strategy_name = strategy_name
        self.metrics = {}
    
    def update(self, metrics: Dict):
        """Update performance metrics."""
        self.metrics = metrics
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        return self.metrics
