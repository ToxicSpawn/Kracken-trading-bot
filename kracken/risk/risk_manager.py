"""Risk manager integration."""
from typing import Dict
from risk.position_sizing import KellyCriterion, VolatilityBasedSizing
from risk.stop_loss import DynamicStopLoss, TimeBasedExit
from risk.black_swan import BlackSwanProtector
from utils.logger import logger

class RiskManager:
    """Comprehensive risk management system."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.kelly = KellyCriterion(config)
        self.volatility_sizing = VolatilityBasedSizing(config)
        self.stop_loss = DynamicStopLoss(config)
        self.time_exit = TimeBasedExit(config)
        self.black_swan = BlackSwanProtector(config)
        self.active_positions = {}
    
    def calculate_position_size(self, strategy_name: str, account_balance: float, 
                                entry_price: float, stop_loss: float, atr: float = None) -> float:
        """Calculate position size based on risk parameters."""
        risk_config = self.config.get("risk", {})
        position_sizing_method = risk_config.get("position_sizing", "kelly")
        
        if position_sizing_method == "kelly":
            return self.kelly.calculate_position_size(account_balance, entry_price, stop_loss)
        elif position_sizing_method == "volatility" and atr is not None:
            return self.volatility_sizing.calculate_position_size(account_balance, entry_price, atr)
        else:
            # Fallback to a simple percentage-based sizing
            fallback_config = risk_config.get("fallback", {})
            max_risk = fallback_config.get("max_position_pct", 0.01) * account_balance
            return max_risk / (entry_price - stop_loss) if (entry_price - stop_loss) > 0 else 0
    
    def calculate_stop_loss(self, strategy_name: str, ohlcv_data: list, entry_price: float) -> float:
        """Calculate initial stop-loss."""
        return self.stop_loss.calculate_initial_stop(ohlcv_data, entry_price)
    
    def should_exit_position(self, strategy_name: str, current_price: float, stop_loss: float, 
                            entry_price: float, entry_time: float) -> bool:
        """Check if position should be exited."""
        # Check time-based exit
        if self.time_exit.should_exit(entry_time):
            return True
        
        # Check trailing stop-loss
        new_stop = self.stop_loss.trail_stop(current_price, stop_loss, entry_price)
        if (current_price <= new_stop and entry_price > stop_loss) or \
           (current_price >= new_stop and entry_price < stop_loss):
            logger.info(f"Stop-loss triggered at {current_price:.8f} (stop: {new_stop:.8f})")
            return True
        
        return False
    
    def check_black_swan(self, current_balance: float, ohlcv_data: list = None, 
                        exchange_status: Dict = None) -> bool:
        """Check for black swan events."""
        if self.black_swan.check_drawdown(current_balance):
            return True
        if ohlcv_data and self.black_swan.check_volatility(ohlcv_data):
            return True
        if exchange_status and self.black_swan.check_exchange_health(exchange_status):
            return True
        return False
    
    def add_position(self, strategy_name: str, symbol: str, entry_price: float, 
                    stop_loss: float, entry_time: float, size: float, action: str = "buy"):
        """Add a new position to the risk manager."""
        if strategy_name not in self.active_positions:
            self.active_positions[strategy_name] = []
        
        self.active_positions[strategy_name].append({
            "symbol": symbol,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "entry_time": entry_time,
            "size": size,
            "action": action,
            "active": True
        })
        logger.info(f"Added position: {symbol} @ {entry_price:.8f} (size: {size:.4f})")
    
    def update_position(self, strategy_name: str, symbol: str, current_price: float) -> bool:
        """Update position status."""
        if strategy_name not in self.active_positions:
            return False
        
        for position in self.active_positions[strategy_name]:
            if position["symbol"] == symbol and position["active"]:
                if self.should_exit_position(
                    strategy_name,
                    current_price,
                    position["stop_loss"],
                    position["entry_price"],
                    position["entry_time"]
                ):
                    position["active"] = False
                    logger.info(f"Closed position: {symbol} @ {current_price:.8f} (PnL: {(current_price - position['entry_price']) / position['entry_price'] * 100:.2f}%)")
                    return True
        return False
    
    def get_active_positions(self, strategy_name: str = None) -> list:
        """Get active positions for a strategy or all strategies."""
        if strategy_name:
            return [p for p in self.active_positions.get(strategy_name, []) if p["active"]]
        else:
            return [p for strategy in self.active_positions.values() for p in strategy if p["active"]]

