"""Parameter optimization using Optuna for hyperparameter tuning."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import optuna
from optuna import Trial

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """
    Optimizer for trading strategy parameters using Optuna.
    """

    def __init__(
        self,
        direction: str = "maximize",
        n_trials: int = 100,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Initialize optimizer.

        Args:
            direction: 'maximize' or 'minimize' the objective
            n_trials: Number of optimization trials
            timeout: Optional timeout in seconds
        """
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.study: Optional[optuna.Study] = None

    def optimize(
        self,
        objective_func: Callable[[Trial], float],
        study_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run optimization.

        Args:
            objective_func: Function that takes a Trial and returns a score
            study_name: Optional study name for persistence

        Returns:
            Dictionary with best parameters and value
        """
        self.study = optuna.create_study(
            direction=self.direction,
            study_name=study_name,
        )

        logger.info(f"Starting optimization with {self.n_trials} trials...")
        self.study.optimize(
            objective_func,
            n_trials=self.n_trials,
            timeout=self.timeout,
        )

        best_params = self.study.best_params
        best_value = self.study.best_value

        logger.info(f"Optimization complete. Best value: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(self.study.trials),
        }

    def suggest_int(
        self,
        trial: Trial,
        name: str,
        low: int,
        high: int,
        step: int = 1,
        log: bool = False,
    ) -> int:
        """Suggest integer parameter."""
        return trial.suggest_int(name, low, high, step=step, log=log)

    def suggest_float(
        self,
        trial: Trial,
        name: str,
        low: float,
        high: float,
        step: Optional[float] = None,
        log: bool = False,
    ) -> float:
        """Suggest float parameter."""
        return trial.suggest_float(name, low, high, step=step, log=log)

    def suggest_categorical(
        self,
        trial: Trial,
        name: str,
        choices: list[Any],
    ) -> Any:
        """Suggest categorical parameter."""
        return trial.suggest_categorical(name, choices)


# Example: Optimize SMA crossover strategy
def create_sma_optimization_objective(
    backtest_func: Callable[[dict[str, Any]], float],
) -> Callable[[Trial], float]:
    """
    Create an optimization objective for SMA crossover strategy.

    Args:
        backtest_func: Function that runs backtest and returns total return

    Returns:
        Objective function for Optuna
    """
    def objective(trial: Trial) -> float:
        fast_period = trial.suggest_int("fast_period", 5, 50, step=1)
        slow_period = trial.suggest_int("slow_period", 20, 200, step=1)

        # Ensure fast < slow
        if fast_period >= slow_period:
            return -999.0  # Penalty for invalid params

        params = {
            "fast_period": fast_period,
            "slow_period": slow_period,
        }

        try:
            result = backtest_func(params)
            return result
        except Exception as e:
            logger.warning(f"Backtest failed with params {params}: {e}")
            return -999.0

    return objective

