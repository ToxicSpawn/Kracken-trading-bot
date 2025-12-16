"""Dynamic stop-loss management."""
import numpy as np
import pandas as pd
import time
from typing import Dict, List
from utils.logger import logger

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available, using fallback ATR calculation")


class DynamicStopLoss:
    """Dynamic stop-loss calculator."""
    
    def __init__(self, config: Dict):
        risk_config = config.get("risk", {})
        sl_config = risk_config.get("stop_loss", {})
        self.atr_multiplier = sl_config.get("atr_multiplier", 3)
        self.trail_pct = sl_config.get("trail_pct", 0.01)
        self.min_stop_distance = sl_config.get("min_stop_distance", 0.0001)
    
    def calculate_initial_stop(self, ohlcv_data: List, entry_price: float) -> float:
        """Calculate initial ATR-based stop-loss."""
        if len(ohlcv_data) < 14:
            return entry_price * (1 - self.min_stop_distance)
        
        try:
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            if TALIB_AVAILABLE:
                atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)[-1]
            else:
                # Fallback ATR calculation
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
            
            stop_loss = entry_price - (self.atr_multiplier * atr)
            
            # Ensure stop is not too close to entry
            if abs(entry_price - stop_loss) < self.min_stop_distance:
                stop_loss = entry_price - self.min_stop_distance if entry_price > stop_loss else entry_price + self.min_stop_distance
            
            logger.info(f"Initial stop-loss: {stop_loss:.8f} (ATR: {atr:.8f})")
            return float(stop_loss)
        except Exception as e:
            logger.warning(f"Error calculating stop-loss: {e}, using default")
            return entry_price * (1 - self.min_stop_distance)
    
    def trail_stop(self, current_price: float, stop_loss: float, entry_price: float) -> float:
        """Trail stop-loss as price moves favorably."""
        if current_price > entry_price * (1 + self.trail_pct):
            new_stop = current_price * (1 - self.trail_pct)
            if new_stop > stop_loss:
                logger.info(f"Trailing stop-loss updated: {new_stop:.8f}")
                return new_stop
        return stop_loss


class TimeBasedExit:
    """Time-based exit strategy."""
    
    def __init__(self, config: Dict):
        risk_config = config.get("risk", {})
        time_config = risk_config.get("time_exit", {})
        self.max_hold_time = time_config.get("max_hold_time", 86400)  # in seconds
    
    def should_exit(self, entry_time: float) -> bool:
        """Check if position should be exited based on time."""
        current_time = time.time()
        if current_time - entry_time > self.max_hold_time:
            logger.info(f"Time-based exit triggered after {self.max_hold_time} seconds")
            return True
        return False

