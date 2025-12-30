"""Correlation management for risk control."""
import pandas as pd
from typing import Dict, List
from utils.logger import logger

class CorrelationManager:
    """Manages correlation between positions."""
    
    def __init__(self, config: Dict):
        self.config = config
        risk_config = config.get("risk", {})
        corr_config = risk_config.get("correlation", {})
        self.threshold = corr_config.get("threshold", 0.8)
        self.update_frequency = corr_config.get("update_frequency", 3600)
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.last_update = 0
        import time
        self.last_update = time.time()
    
    def update_correlation_matrix(self, data: Dict[str, List]):
        """Update correlation matrix from market data."""
        import time
        current_time = time.time()
        
        if current_time - self.last_update < self.update_frequency:
            return
        
        try:
            # Calculate correlations between symbols
            symbols = list(data.keys())
            if len(symbols) < 2:
                return
            
            returns = {}
            for symbol, ohlcv_data in data.items():
                if len(ohlcv_data) < 100:
                    continue
                
                df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                returns[symbol] = df['close'].pct_change().dropna()
            
            if len(returns) < 2:
                return
            
            # Calculate pairwise correlations
            for symbol1 in returns:
                if symbol1 not in self.correlation_matrix:
                    self.correlation_matrix[symbol1] = {}
                
                for symbol2 in returns:
                    if symbol1 != symbol2:
                        try:
                            corr = returns[symbol1].corr(returns[symbol2])
                            self.correlation_matrix[symbol1][symbol2] = corr if not pd.isna(corr) else 0
                        except Exception:
                            self.correlation_matrix[symbol1][symbol2] = 0
            
            self.last_update = current_time
            logger.debug("Correlation matrix updated")
        except Exception as e:
            logger.warning(f"Error updating correlation matrix: {e}")
    
    def check_correlation(self, symbol: str, active_positions: Dict) -> bool:
        """Check if adding a position would create high correlation risk."""
        # Get all active symbols
        active_symbols = []
        for strategy_positions in active_positions.values():
            for position in strategy_positions:
                if position.get("active", False):
                    active_symbols.append(position["symbol"])
        
        if not active_symbols:
            return False
        
        # Check correlation with active positions
        for active_symbol in active_symbols:
            if symbol in self.correlation_matrix and active_symbol in self.correlation_matrix[symbol]:
                corr = abs(self.correlation_matrix[symbol][active_symbol])
                if corr > self.threshold:
                    logger.warning(f"High correlation detected: {symbol} <-> {active_symbol} ({corr:.2f})")
                    return True
        
        return False

