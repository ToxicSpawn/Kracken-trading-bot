"""Main trading engine with async WebSocket support."""
import asyncio
import ccxt.pro
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base_strategy import BaseStrategy
from data.feed import DataFeed
from risk.risk_manager import RiskManager
from utils.logger import logger
from utils.config import load_config
from utils.alerts import AlertSystem
from data.database import DatabaseManager
from utils.performance import calculate_performance_metrics

class TradingEngine:
    """Main trading engine."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = load_config(config_path)
        self.exchanges: Dict[str, ccxt.pro.Exchange] = {}
        self.strategies: List[BaseStrategy] = []
        self.data_feeds: Dict[str, DataFeed] = {}
        self.running = False
        self.risk_manager = RiskManager(self.config)
        self.alert_system = AlertSystem(config_path)
        self.database = DatabaseManager(config_path)
        self.performance_trackers = {}
        
        # Initialize database
        try:
            self.database.create_tables()
        except Exception as e:
            logger.warning(f"Could not initialize database: {e}")
    
    async def initialize_exchanges(self):
        """Initialize WebSocket connections for all exchanges."""
        for exchange_id, exchange_config in self.config.get("exchanges", {}).items():
            if not exchange_config.get("enabled", True):
                continue
            
            try:
                exchange_class = getattr(ccxt.pro, exchange_id)
                exchange = exchange_class({
                    'apiKey': exchange_config.get("api_key", ""),
                    'secret': exchange_config.get("secret", ""),
                    'enableRateLimit': True,
                    'options': {'adjustForTimeDifference': True}
                })
                await exchange.load_markets()
                self.exchanges[exchange_id] = exchange
                logger.info(f"✅ Initialized {exchange_id} WebSocket")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {exchange_id}: {e}")
                self.alert_system.send_error_alert(f"Failed to initialize {exchange_id}: {e}")
    
    async def initialize_data_feeds(self):
        """Initialize data feeds for all symbols."""
        for strategy in self.strategies:
            if hasattr(strategy, 'symbols'):
                for symbol in strategy.symbols:
                    if symbol not in self.data_feeds:
                        # Find an exchange that supports this symbol
                        for exchange_id, exchange in self.exchanges.items():
                            if symbol in exchange.symbols:
                                self.data_feeds[symbol] = DataFeed(exchange, [symbol])
                                logger.info(f"📡 Initialized data feed for {symbol} on {exchange_id}")
                                break
    
    def initialize_strategies(self):
        """Initialize all trading strategies."""
        # Machine Learning Strategies
        if self.config.get("strategies", {}).get("ml", {}).get("enabled", False):
            from strategies.ml_strategies import LSTMStrategy, RandomForestStrategy
            
            ml_config = self.config.get("strategies", {}).get("ml", {})
            
            if ml_config.get("lstm", {}).get("enabled", False):
                lstm_strategy = LSTMStrategy(self.config)
                lstm_strategy.set_risk_manager(self.risk_manager)
                self.strategies.append(lstm_strategy)
            
            if ml_config.get("random_forest", {}).get("enabled", False):
                rf_strategy = RandomForestStrategy(self.config)
                rf_strategy.set_risk_manager(self.risk_manager)
                self.strategies.append(rf_strategy)
        
        # Arbitrage Strategies
        if self.config.get("strategies", {}).get("arbitrage", {}).get("enabled", False):
            from strategies.arbitrage import TriangularArbitrageStrategy, MarketMakingStrategy
            
            arb_config = self.config.get("strategies", {}).get("arbitrage", {})
            
            if arb_config.get("triangular", {}).get("enabled", False):
                arb_strategy = TriangularArbitrageStrategy(self.config, self.exchanges)
                arb_strategy.set_risk_manager(self.risk_manager)
                self.strategies.append(arb_strategy)
            
            if arb_config.get("market_making", {}).get("enabled", False):
                mm_config = arb_config.get("market_making", {})
                for exchange_id in mm_config.get("exchanges", []):
                    if exchange_id in self.exchanges:
                        mm_strategy = MarketMakingStrategy(self.config, self.exchanges[exchange_id])
                        mm_strategy.set_risk_manager(self.risk_manager)
                        self.strategies.append(mm_strategy)
        
        # Quantum Strategies (optional)
        if self.config.get("strategies", {}).get("quantum", {}).get("enabled", False):
            from strategies.quantum_strategies import QuantumPortfolioOptimizer
            
            # Example: Initialize with dummy data - real implementation would get this from market data
            assets = ["BTC", "ETH", "SOL", "ADA"]
            expected_returns = [0.05, 0.07, 0.10, 0.08]
            covariance_matrix = [
                [0.04, 0.02, 0.01, 0.01],
                [0.02, 0.09, 0.02, 0.02],
                [0.01, 0.02, 0.16, 0.03],
                [0.01, 0.02, 0.03, 0.25]
            ]
            
            quantum_strategy = QuantumPortfolioOptimizer(
                self.config, assets, expected_returns, covariance_matrix
            )
            quantum_strategy.set_risk_manager(self.risk_manager)
            self.strategies.append(quantum_strategy)
    
    async def execute_trade(self, strategy: BaseStrategy, signal: Dict):
        """Execute a trade based on a strategy signal."""
        try:
            symbol = signal.get("symbol")
            action = signal.get("action")
            price = signal.get("current_price", signal.get("price"))
            confidence = signal.get("confidence", 1.0)
            
            if not all([symbol, action, price]):
                logger.error(f"Invalid trade signal: {signal}")
                return False
            
            # Get the exchange for this symbol
            exchange = None
            for ex_id, ex in self.exchanges.items():
                if symbol in ex.symbols:
                    exchange = ex
                    break
            
            if not exchange:
                logger.error(f"No exchange found for symbol {symbol}")
                return False
            
            # Calculate position size
            account_balance = 10000  # In a real implementation, get this from exchange
            ohlcv_data = self.data_feeds.get(symbol, DataFeed(None, [])).get_data(symbol, 100)
            atr = self._calculate_atr(ohlcv_data) if ohlcv_data else 0.01
            stop_loss = self.risk_manager.calculate_stop_loss(strategy.name, ohlcv_data, price)
            position_size = self.risk_manager.calculate_position_size(
                strategy.name, account_balance, price, stop_loss, atr
            )
            
            # Execute the trade
            if action == "buy":
                order = await exchange.create_market_buy_order(symbol, position_size)
            elif action == "sell":
                order = await exchange.create_market_sell_order(symbol, position_size)
            else:
                logger.error(f"Unknown action: {action}")
                return False
            
            # Log the trade
            entry_time = pd.Timestamp.now()
            self.risk_manager.add_position(
                strategy.name, symbol, price, stop_loss, entry_time.timestamp(), position_size
            )
            
            # Log to database
            try:
                self.database.log_signal(
                    strategy.name, symbol, action, price, confidence, entry_time
                )
            except Exception as e:
                logger.warning(f"Could not log signal to database: {e}")
            
            # Send alert
            self.alert_system.send_trade_alert(
                strategy.name, symbol, action, price, position_size
            )
            
            logger.info(f"📈 Executed {action} order for {symbol} @ {price:.8f} (size: {position_size:.4f})")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error executing trade: {e}")
            self.alert_system.send_error_alert(f"Error executing trade: {e}")
            return False
    
    def _calculate_atr(self, ohlcv_data: List) -> float:
        """Calculate Average True Range (ATR)."""
        if len(ohlcv_data) < 14:
            return 0.01
        
        try:
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            return float(atr) if not pd.isna(atr) else 0.01
        except Exception:
            return 0.01
    
    async def monitor_positions(self):
        """Monitor open positions and check for exits."""
        while self.running:
            try:
                for strategy in self.strategies:
                    active_positions = self.risk_manager.get_active_positions(strategy.name)
                    for position in active_positions:
                        symbol = position["symbol"]
                        feed = self.data_feeds.get(symbol)
                        if not feed:
                            continue
                        
                        data = feed.get_data(symbol, 1)
                        if not data:
                            continue
                        
                        current_price = data[-1][4]  # Latest close price
                        
                        if self.risk_manager.update_position(
                            strategy.name, symbol, current_price
                        ):
                            # Position was closed
                            exit_price = current_price
                            pnl = (exit_price - position["entry_price"]) * position["size"] if position.get("action") == "buy" else \
                                  (position["entry_price"] - exit_price) * position["size"]
                            pnl_pct = pnl / (position["entry_price"] * position["size"]) * 100 if position["entry_price"] * position["size"] > 0 else 0
                            
                            # Log to database
                            try:
                                self.database.log_trade({
                                    "strategy_name": strategy.name,
                                    "symbol": symbol,
                                    "action": position.get("action", "buy"),
                                    "entry_time": pd.Timestamp.fromtimestamp(position["entry_time"]),
                                    "exit_time": pd.Timestamp.now(),
                                    "entry_price": position["entry_price"],
                                    "exit_price": exit_price,
                                    "size": position["size"],
                                    "pnl": pnl,
                                    "pnl_pct": pnl_pct,
                                    "balance": 10000 + pnl
                                })
                            except Exception as e:
                                logger.warning(f"Could not log trade to database: {e}")
                            
                            # Send alert
                            self.alert_system.send_trade_alert(
                                strategy.name, symbol, "close", exit_price, position["size"], pnl
                            )
            
            except Exception as e:
                logger.error(f"⚠️ Error monitoring positions: {e}")
            
            await asyncio.sleep(10)  # Check every 10 seconds
    
    async def track_performance(self):
        """Track and log performance metrics periodically."""
        while self.running:
            try:
                for strategy in self.strategies:
                    try:
                        trades = self.database.get_trades(strategy.name, 1000)
                        if not trades:
                            continue
                        
                        # Calculate performance metrics
                        equity_curve = pd.DataFrame({
                            "time": [pd.Timestamp(t["entry_time"]) if isinstance(t["entry_time"], str) else pd.Timestamp.fromtimestamp(t["entry_time"].timestamp()) if hasattr(t["entry_time"], "timestamp") else pd.Timestamp.now() for t in trades],
                            "balance": [t.get("balance", 10000) for t in trades]
                        })
                        metrics = calculate_performance_metrics(trades, equity_curve, 10000)
                        
                        # Log to database
                        if not equity_curve.empty:
                            try:
                                self.database.log_performance(
                                    strategy.name,
                                    equity_curve["time"].min(),
                                    equity_curve["time"].max(),
                                    metrics
                                )
                            except Exception as e:
                                logger.warning(f"Could not log performance: {e}")
                        
                        # Send alert
                        self.alert_system.send_performance_alert(strategy.name, metrics)
                        
                        logger.info(f"📊 Performance update for {strategy.name}:")
                        for k, v in metrics.items():
                            logger.info(f"  {k}: {v}")
                    except Exception as e:
                        logger.warning(f"Error tracking performance for {strategy.name}: {e}")
            
            except Exception as e:
                logger.error(f"⚠️ Error tracking performance: {e}")
            
            await asyncio.sleep(3600)  # Update every hour
    
    async def run_strategies(self):
        """Run all strategies in parallel."""
        while self.running:
            try:
                # Update all data feeds
                feed_tasks = [feed.update() for feed in self.data_feeds.values()]
                await asyncio.gather(*feed_tasks, return_exceptions=True)
                
                # Run all strategies
                strategy_tasks = []
                for strategy in self.strategies:
                    if hasattr(strategy, 'symbols'):
                        # Get data for all symbols the strategy needs
                        data = {symbol: self.data_feeds[symbol].get_data(symbol)
                               for symbol in strategy.symbols if symbol in self.data_feeds}
                        strategy_tasks.append(strategy.run(data))
                    else:
                        strategy_tasks.append(strategy.run())
                
                results = await asyncio.gather(*strategy_tasks, return_exceptions=True)
                
                # Process strategy results
                for strategy, result in zip(self.strategies, results):
                    if isinstance(result, Exception):
                        logger.error(f"Strategy {strategy.name} raised exception: {result}")
                        continue
                    
                    if result:
                        if isinstance(result, dict):
                            for symbol, signal in result.items():
                                logger.info(f"📊 {strategy.name} generated signal for {symbol}: {signal}")
                                # Execute the trade
                                await self.execute_trade(strategy, {**signal, "symbol": symbol})
            
            except Exception as e:
                logger.error(f"⚠️ Error in strategy execution: {e}")
                self.alert_system.send_error_alert(f"Error in strategy execution: {e}")
            
            await asyncio.sleep(0.1)  # Prevent CPU overload
    
    async def check_black_swan(self):
        """Check for black swan events periodically."""
        while self.running:
            try:
                # Check account balance (in a real implementation, get from exchange)
                account_balance = 10000
                
                # Check each data feed for volatility
                for symbol, feed in self.data_feeds.items():
                    ohlcv_data = feed.get_data(symbol, 100)
                    if ohlcv_data and self.risk_manager.check_black_swan(account_balance, ohlcv_data):
                        logger.warning(f"🚨 Black swan event detected for {symbol}, pausing trading")
                        self.alert_system.send_error_alert(f"Black swan event detected for {symbol}, pausing trading")
            
            except Exception as e:
                logger.error(f"⚠️ Error checking black swan events: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def start(self):
        """Start the trading engine."""
        self.running = True
        await self.initialize_exchanges()
        self.initialize_strategies()
        await self.initialize_data_feeds()
        
        # Start all background tasks
        tasks = [
            self.run_strategies(),
            self.monitor_positions(),
            self.track_performance(),
            self.check_black_swan()
        ]
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop(self):
        """Stop the trading engine."""
        self.running = False
        for exchange in self.exchanges.values():
            try:
                await exchange.close()
            except Exception:
                pass
        try:
            self.database.disconnect()
        except Exception:
            pass
        logger.info("🛑 Trading engine stopped")

