"""Advanced risk manager with correlation and liquidity checks."""
from typing import Dict
from risk.risk_manager import RiskManager
from risk.correlation import CorrelationManager
from risk.liquidity import LiquidityManager
from utils.logger import logger

class AdvancedRiskManager(RiskManager):
    """Advanced risk manager with correlation and liquidity management."""
    
    def __init__(self, config: Dict, alert_system):
        super().__init__(config)
        self.alert_system = alert_system
        self.correlation = CorrelationManager(config)
        self.liquidity = LiquidityManager(config)
        self.account_balance = config.get("risk", {}).get("initial_balance", 10000)
        self.max_total_risk_pct = config.get("risk", {}).get("max_total_risk_pct", 5)
        self.max_daily_loss = config.get("risk", {}).get("max_daily_loss", 50)
        self.daily_pnl = 0
        self.last_reset_date = None
        import datetime
        self.last_reset_date = datetime.date.today()
    
    def calculate_position_size(self, strategy_name: str, symbol: str, account_balance: float,
                                entry_price: float, stop_loss: float, atr: float = None) -> float:
        """Calculate position size with advanced risk checks."""
        # Check liquidity
        if not self.liquidity.check_liquidity(symbol, entry_price):
            logger.warning(f"Insufficient liquidity for {symbol}")
            return 0
        
        # Check correlation
        if self.correlation.check_correlation(symbol, self.active_positions):
            logger.warning(f"High correlation risk for {symbol}")
            return 0
        
        # Use base position sizing
        base_size = super().calculate_position_size(
            strategy_name, account_balance, entry_price, stop_loss, atr
        )
        
        # Apply total risk limit
        total_risk = sum(
            abs(p["entry_price"] - p["stop_loss"]) * p["size"] 
            for strategy in self.active_positions.values() 
            for p in strategy if p["active"]
        )
        
        max_risk = self.account_balance * (self.max_total_risk_pct / 100)
        if total_risk + abs(entry_price - stop_loss) * base_size > max_risk:
            base_size = (max_risk - total_risk) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
        
        return max(0, base_size)
    
    def check_risk_limits(self) -> bool:
        """Check if risk limits are breached."""
        import datetime
        current_date = datetime.date.today()
        
        # Reset daily PnL if new day
        if current_date != self.last_reset_date:
            self.daily_pnl = 0
            self.last_reset_date = current_date
        
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            logger.error(f"Daily loss limit breached: {self.daily_pnl}")
            self.alert_system.send_error_alert(f"Daily loss limit breached: ${self.daily_pnl:.2f}")
            return True
        
        # Check max drawdown
        if self.account_balance < self.account_balance * (1 - self.config.get("risk", {}).get("max_drawdown", 0.1)):
            logger.error("Max drawdown limit breached")
            self.alert_system.send_error_alert("Max drawdown limit breached")
            return True
        
        return False
    
    def update_daily_pnl(self, pnl: float):
        """Update daily PnL."""
        self.daily_pnl += pnl
        self.account_balance += pnl

