"""Base strategy class."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from utils.logger import logger

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, config: Dict, name: str):
        self.config = config
        self.name = name
        self.risk_manager = None
        self.performance_tracker = None
    
    @abstractmethod
    async def run(self, data: Optional[Dict[str, List]] = None) -> Optional[Dict]:
        """Execute the strategy logic."""
        pass
    
    def log(self, message: str):
        """Log strategy-specific messages."""
        logger.info(f"[{self.name}] {message}")
    
    def set_risk_manager(self, risk_manager):
        """Set the risk manager for this strategy."""
        self.risk_manager = risk_manager
    
    def set_performance_tracker(self, performance_tracker):
        """Set the performance tracker for this strategy."""
        self.performance_tracker = performance_tracker

