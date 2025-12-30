"""Position sizing using Kelly Criterion and volatility-based methods."""
from typing import Dict
from utils.logger import logger

class KellyCriterion:
    """Kelly Criterion position sizing."""
    
    def __init__(self, config: Dict):
        risk_config = config.get("risk", {})
        kelly_config = risk_config.get("kelly", {})
        self.win_rate = kelly_config.get("win_rate", 0.55)
        self.risk_reward_ratio = kelly_config.get("risk_reward_ratio", 2)
        self.max_position_pct = kelly_config.get("max_position_pct", 0.02)
    
    def calculate_position_size(self, account_balance: float, entry_price: float, stop_loss: float) -> float:
        """Calculate optimal position size using Kelly Criterion."""
        win_prob = self.win_rate
        loss_prob = 1 - win_prob
        win_payout = self.risk_reward_ratio
        loss_payout = 1
        
        # Kelly formula: f* = (bp - q) / b
        b = win_payout / loss_payout
        q = loss_prob
        f_star = (b * win_prob - q) / b
        
        # Adjust for risk (max position percentage)
        max_risk = self.max_position_pct * account_balance
        position_size = (f_star * account_balance) / (entry_price - stop_loss)
        
        # Cap at max_risk
        position_size = min(position_size, max_risk / (entry_price - stop_loss))
        
        logger.info(f"Kelly position size: {position_size:.4f} units ({(position_size * entry_price / account_balance * 100):.2f}% of account)")
        return position_size


class VolatilityBasedSizing:
    """Volatility-based position sizing."""
    
    def __init__(self, config: Dict):
        risk_config = config.get("risk", {})
        vol_config = risk_config.get("volatility", {})
        self.atr_multiplier = vol_config.get("atr_multiplier", 3)
        self.max_position_pct = vol_config.get("max_position_pct", 0.02)
    
    def calculate_position_size(self, account_balance: float, entry_price: float, atr: float) -> float:
        """Calculate position size based on volatility (ATR)."""
        risk_per_unit = self.atr_multiplier * atr
        max_risk = self.max_position_pct * account_balance
        position_size = max_risk / risk_per_unit
        
        logger.info(f"Volatility-based position size: {position_size:.4f} units ({(position_size * entry_price / account_balance * 100):.2f}% of account)")
        return position_size

