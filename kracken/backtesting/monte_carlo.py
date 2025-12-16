"""Monte Carlo simulations."""
import numpy as np
import pandas as pd
from typing import Dict
from utils.logger import logger

class MonteCarloSimulator:
    """Monte Carlo simulation for future returns."""
    
    def __init__(self, historical_returns: pd.Series, config: Dict):
        self.historical_returns = historical_returns
        backtest_config = config.get("backtesting", {})
        mc_config = backtest_config.get("monte_carlo", {})
        self.n_simulations = mc_config.get("n_simulations", 1000)
        self.n_days = mc_config.get("n_days", 365)
        self.initial_balance = backtest_config.get("initial_balance", 10000)
    
    def run_simulation(self) -> Dict:
        """Run Monte Carlo simulation of future returns."""
        mean_return = self.historical_returns.mean()
        std_return = self.historical_returns.std()
        
        simulations = np.zeros((self.n_simulations, self.n_days))
        final_balances = np.zeros(self.n_simulations)
        
        for i in range(self.n_simulations):
            # Generate random daily returns
            daily_returns = np.random.normal(mean_return, std_return, self.n_days)
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + daily_returns) - 1
            simulations[i] = cumulative_returns
            
            # Calculate final balance
            final_balances[i] = self.initial_balance * (1 + cumulative_returns[-1])
        
        # Calculate key metrics
        cagr = (1 + simulations) ** (252 / self.n_days) - 1  # Annualized return
        max_drawdown = (simulations.max(axis=1) - simulations.min(axis=1)) / (1 + simulations.max(axis=1))
        
        results = {
            "simulations": simulations,
            "final_balances": final_balances,
            "metrics": {
                "median_final_balance": float(np.median(final_balances)),
                "mean_final_balance": float(np.mean(final_balances)),
                "min_final_balance": float(np.min(final_balances)),
                "max_final_balance": float(np.max(final_balances)),
                "median_cagr": float(np.median(cagr, axis=1).mean() * 100),
                "worst_cagr": float(np.min(cagr, axis=1).mean() * 100),
                "median_max_drawdown": float(np.median(max_drawdown) * 100),
                "worst_max_drawdown": float(np.max(max_drawdown) * 100),
                "probability_of_loss": float(np.mean(final_balances < self.initial_balance) * 100)
            }
        }
        
        logger.info(f"Monte Carlo Results:")
        logger.info(f"- Median final balance: ${results['metrics']['median_final_balance']:,.2f}")
        logger.info(f"- Probability of loss: {results['metrics']['probability_of_loss']:.2f}%")
        logger.info(f"- Median CAGR: {results['metrics']['median_cagr']:.2f}%")
        logger.info(f"- Worst-case drawdown: {results['metrics']['worst_max_drawdown']:.2f}%")
        
        return results

