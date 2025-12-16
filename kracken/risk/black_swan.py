"""Black swan protection and circuit breakers."""
import numpy as np
import pandas as pd
import time
from typing import Dict, List
from utils.logger import logger
from utils.alerts import send_telegram_alert

class BlackSwanProtector:
    """Black swan event detector and protector."""
    
    def __init__(self, config: Dict):
        risk_config = config.get("risk", {})
        bs_config = risk_config.get("black_swan", {})
        self.max_drawdown_pct = bs_config.get("max_drawdown_pct", 0.1)
        self.max_volatility_multiplier = bs_config.get("max_volatility_multiplier", 3)
        self.initial_balance = None
        self.last_alert_time = 0
        self.alert_cooldown = 3600  # 1 hour cooldown between alerts
    
    def check_drawdown(self, current_balance: float) -> bool:
        """Pause trading if drawdown exceeds threshold."""
        if not self.initial_balance:
            self.initial_balance = current_balance
            return False
        
        drawdown = (self.initial_balance - current_balance) / self.initial_balance
        if drawdown > self.max_drawdown_pct:
            current_time = time.time()
            if current_time - self.last_alert_time > self.alert_cooldown:
                logger.error(f"🚨 BLACK SWAN: Drawdown {drawdown*100:.2f}% > {self.max_drawdown_pct*100}%")
                send_telegram_alert(f"🚨 BLACK SWAN ALERT: Drawdown {drawdown*100:.2f}%")
                self.last_alert_time = current_time
            return True
        return False
    
    def check_volatility(self, ohlcv_data: List) -> bool:
        """Pause trading if volatility is extreme."""
        if len(ohlcv_data) < 100:
            return False
        
        try:
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            avg_volatility = returns.rolling(100).std().mean()
            
            if volatility > self.max_volatility_multiplier * avg_volatility:
                current_time = time.time()
                if current_time - self.last_alert_time > self.alert_cooldown:
                    logger.error(f"🚨 BLACK SWAN: Volatility spike {volatility:.4f} > {self.max_volatility_multiplier}x avg")
                    send_telegram_alert(f"🚨 BLACK SWAN ALERT: Volatility spike detected!")
                    self.last_alert_time = current_time
                return True
        except Exception as e:
            logger.warning(f"Error checking volatility: {e}")
        
        return False
    
    def check_exchange_health(self, exchange_status: Dict) -> bool:
        """Check if exchange is experiencing issues."""
        if not exchange_status.get("healthy", True):
            current_time = time.time()
            if current_time - self.last_alert_time > self.alert_cooldown:
                logger.error(f"🚨 BLACK SWAN: Exchange {exchange_status.get('name', 'unknown')} is unhealthy")
                send_telegram_alert(f"🚨 BLACK SWAN ALERT: Exchange {exchange_status.get('name', 'unknown')} is unhealthy")
                self.last_alert_time = current_time
            return True
        return False

