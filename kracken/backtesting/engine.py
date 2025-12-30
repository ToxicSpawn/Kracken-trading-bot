"""Backtesting engine."""
import pandas as pd
from typing import Dict
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base_strategy import BaseStrategy
from risk.risk_manager import RiskManager
from utils.performance import calculate_performance_metrics
from utils.logger import logger

class BacktestEngine:
    """Backtesting engine for strategy validation."""
    
    def __init__(self, config: Dict, data: Dict[str, pd.DataFrame]):
        self.config = config
        self.data = data
        backtest_config = config.get("backtesting", {})
        self.initial_balance = backtest_config.get("initial_balance", 10000)
        self.commission = backtest_config.get("commission", 0.001)
        self.slippage = backtest_config.get("slippage", 0.0005)
        self.risk_manager = RiskManager(config)
        self.results = {}
    
    def run_backtest(self, strategy: BaseStrategy, symbol: str) -> Dict:
        """Run backtest for a single strategy and symbol."""
        if symbol not in self.data:
            logger.error(f"No data available for {symbol}")
            return {}
        
        df = self.data[symbol].copy()
        strategy_name = strategy.name
        balance = self.initial_balance
        positions = []
        trades = []
        current_position = None
        
        # Initialize strategy
        strategy.set_risk_manager(self.risk_manager)
        
        for i in range(len(df)):
            row = df.iloc[i]
            current_time = row.name if hasattr(row.name, 'timestamp') else pd.Timestamp.now()
            current_price = row['close']
            
            # Prepare data for strategy
            lookback = min(100, i)  # Last 100 data points
            ohlcv_data = df.iloc[max(0, i-lookback):i+1][['open', 'high', 'low', 'close', 'volume']].values.tolist()
            
            # Add timestamp column
            ohlcv_data_with_timestamp = [[i] + row for i, row in enumerate(ohlcv_data)]
            
            # Run strategy (synchronous for backtesting)
            try:
                import asyncio
                # Create a new event loop for this call
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                signal = loop.run_until_complete(strategy.run({symbol: ohlcv_data_with_timestamp}))
                loop.close()
            except Exception as e:
                logger.warning(f"Strategy error at step {i}: {e}")
                signal = None
            
            # Process signal
            if signal and symbol in signal:
                signal = signal[symbol]
                
                # Close existing position if needed
                if current_position:
                    if (current_position["action"] == "buy" and signal["action"] == "sell") or \
                       (current_position["action"] == "sell" and signal["action"] == "buy"):
                        # Calculate exit price with slippage
                        exit_price = current_price * (1 - self.slippage) if signal["action"] == "sell" else current_price * (1 + self.slippage)
                        
                        # Calculate PnL
                        if current_position["action"] == "buy":
                            pnl = (exit_price - current_position["entry_price"]) * current_position["size"]
                        else:
                            pnl = (current_position["entry_price"] - exit_price) * current_position["size"]
                        
                        pnl_pct = pnl / (current_position["entry_price"] * current_position["size"]) * 100 if current_position["entry_price"] * current_position["size"] > 0 else 0
                        
                        # Update balance
                        balance += pnl
                        balance *= (1 - self.commission)  # Apply commission
                        
                        # Record trade
                        trades.append({
                            "symbol": symbol,
                            "entry_time": current_position["entry_time"],
                            "exit_time": current_time,
                            "entry_price": current_position["entry_price"],
                            "exit_price": exit_price,
                            "size": current_position["size"],
                            "action": current_position["action"],
                            "pnl": pnl,
                            "pnl_pct": pnl_pct,
                            "balance": balance
                        })
                        
                        logger.info(f"Closed {current_position['action']} position in {symbol} @ {exit_price:.8f} (PnL: {pnl:.2f}, {pnl_pct:.2f}%)")
                        current_position = None
                
                # Open new position if no current position
                if not current_position and signal.get("confidence", 0) > 0.5:
                    # Calculate position size
                    atr = df.iloc[max(0, i-14):i]['close'].diff().abs().mean() if i >= 14 else 0.01
                    stop_loss = self.risk_manager.calculate_stop_loss(strategy_name, ohlcv_data_with_timestamp, current_price)
                    position_size = self.risk_manager.calculate_position_size(
                        strategy_name, balance, current_price, stop_loss, atr
                    )
                    
                    # Calculate entry price with slippage
                    entry_price = current_price * (1 + self.slippage) if signal["action"] == "buy" else current_price * (1 - self.slippage)
                    
                    # Add position
                    self.risk_manager.add_position(
                        strategy_name, symbol, entry_price, stop_loss, current_time.timestamp() if hasattr(current_time, 'timestamp') else pd.Timestamp.now().timestamp(), position_size, signal["action"]
                    )
                    current_position = {
                        "action": signal["action"],
                        "entry_price": entry_price,
                        "entry_time": current_time,
                        "size": position_size
                    }
                    
                    logger.info(f"Opened {signal['action']} position in {symbol} @ {entry_price:.8f} (size: {position_size:.4f})")
            
            # Check for stop-loss or time-based exit
            if current_position:
                self.risk_manager.update_position(strategy_name, symbol, current_price)
                
                # If position was closed by risk manager, update current_position
                active_positions = self.risk_manager.get_active_positions(strategy_name)
                if not any(p["symbol"] == symbol for p in active_positions):
                    current_position = None
            
            # Update equity curve
            if i % 100 == 0:  # Update every 100 candles to save memory
                positions.append({
                    "time": current_time,
                    "balance": balance,
                    "price": current_price
                })
        
        # Close any open position at the end
        if current_position:
            exit_price = df.iloc[-1]['close']
            if current_position["action"] == "buy":
                pnl = (exit_price - current_position["entry_price"]) * current_position["size"]
            else:
                pnl = (current_position["entry_price"] - exit_price) * current_position["size"]
            
            pnl_pct = pnl / (current_position["entry_price"] * current_position["size"]) * 100 if current_position["entry_price"] * current_position["size"] > 0 else 0
            
            balance += pnl
            balance *= (1 - self.commission)  # Apply commission
            
            trades.append({
                "symbol": symbol,
                "entry_time": current_position["entry_time"],
                "exit_time": df.index[-1],
                "entry_price": current_position["entry_price"],
                "exit_price": exit_price,
                "size": current_position["size"],
                "action": current_position["action"],
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "balance": balance
            })
            
            logger.info(f"Closed final {current_position['action']} position in {symbol} @ {exit_price:.8f} (PnL: {pnl:.2f}, {pnl_pct:.2f}%)")
        
        # Calculate performance metrics
        equity_curve = pd.DataFrame(positions) if positions else pd.DataFrame()
        performance = calculate_performance_metrics(trades, equity_curve, self.initial_balance)
        
        self.results[strategy_name] = {
            "trades": trades,
            "equity_curve": equity_curve,
            "performance": performance,
            "final_balance": balance
        }
        
        return self.results[strategy_name]

