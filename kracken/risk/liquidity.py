"""Liquidity management for risk control."""
from typing import Dict, List
from utils.logger import logger

class LiquidityManager:
    """Manages liquidity checks for positions."""
    
    def __init__(self, config: Dict):
        self.config = config
        risk_config = config.get("risk", {})
        liq_config = risk_config.get("liquidity", {})
        self.max_spread_pct = liq_config.get("max_spread_pct", 0.5)
        self.min_liquidity_score = liq_config.get("min_liquidity_score", 1000)
        self.update_frequency = liq_config.get("update_frequency", 60)
        self.liquidity_scores: Dict[str, float] = {}
        self.last_update = 0
        import time
        self.last_update = time.time()
    
    def update_liquidity_data(self, data_buffers: Dict[str, List]):
        """Update liquidity scores from market data."""
        import time
        current_time = time.time()
        
        if current_time - self.last_update < self.update_frequency:
            return
        
        try:
            for symbol, ohlcv_data in data_buffers.items():
                if len(ohlcv_data) < 10:
                    continue
                
                # Calculate liquidity score based on volume and spread
                recent_data = ohlcv_data[-10:]
                volumes = [d[5] for d in recent_data]  # volume
                prices = [d[4] for d in recent_data]  # close
                
                avg_volume = sum(volumes) / len(volumes)
                price_volatility = max(prices) / min(prices) - 1 if min(prices) > 0 else 0
                
                # Liquidity score: higher volume and lower volatility = better liquidity
                liquidity_score = avg_volume / (1 + price_volatility * 100)
                self.liquidity_scores[symbol] = liquidity_score
            
            self.last_update = current_time
            logger.debug("Liquidity scores updated")
        except Exception as e:
            logger.warning(f"Error updating liquidity scores: {e}")
    
    def check_liquidity(self, symbol: str, price: float) -> bool:
        """Check if symbol has sufficient liquidity."""
        if symbol not in self.liquidity_scores:
            # If we don't have data, assume it's liquid (conservative approach)
            return True
        
        liquidity_score = self.liquidity_scores[symbol]
        
        if liquidity_score < self.min_liquidity_score:
            logger.warning(f"Insufficient liquidity for {symbol}: score={liquidity_score:.2f}")
            return False
        
        return True

