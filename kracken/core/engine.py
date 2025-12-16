"""Ultra-Low Latency Trading Engine."""
import asyncio
import ccxt.pro
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base_strategy import BaseStrategy
from data.fpga_feed import FPGADataFeed
from risk.advanced_risk_manager import AdvancedRiskManager
from utils.logger import logger
from utils.config import load_config
from utils.alerts import AlertSystem
from data.database import DatabaseManager
from utils.performance import calculate_performance_metrics

class UltraLowLatencyEngine:
    """Ultra-low latency trading engine with FPGA acceleration and colocation optimization."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = load_config(config_path)
        self.exchanges: Dict[str, ccxt.pro.Exchange] = {}
        self.strategies: List[BaseStrategy] = []
        self.data_feed: Optional[FPGADataFeed] = None
        self.risk_manager: Optional[AdvancedRiskManager] = None
        self.alert_system: Optional[AlertSystem] = None
        self.database: Optional[DatabaseManager] = None
        self.running = False
        self.performance_trackers = {}
        self.last_heartbeat = 0
        self.heartbeat_interval = 60  # seconds
        self.latency_metrics = {}
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components."""
        # Initialize database first (needed for other components)
        try:
            self.database = DatabaseManager("config.json")
            self.database.create_tables()
        except Exception as e:
            logger.warning(f"Could not initialize database: {e}")
            self.database = None
        
        # Initialize alert system
        self.alert_system = AlertSystem("config.json")
        
        # Initialize risk manager
        self.risk_manager = AdvancedRiskManager(self.config, self.alert_system)
        
        # Initialize data feed
        self.data_feed = FPGADataFeed(self)
        
        # Initialize performance trackers
        self._initialize_performance_trackers()
    
    def _initialize_performance_trackers(self):
        """Initialize performance trackers for all strategies."""
        from utils.performance import PerformanceTracker
        strategies_config = self.config.get("strategies", {})
        
        for strategy_type, strategy_config in strategies_config.items():
            if isinstance(strategy_config, dict) and strategy_config.get("enabled", False):
                for strategy_name, sub_config in strategy_config.items():
                    if strategy_name != "enabled" and isinstance(sub_config, dict) and sub_config.get("enabled", False):
                        self.performance_trackers[strategy_name] = PerformanceTracker(
                            self.config, strategy_name
                        )
    
    async def initialize_exchanges(self):
        """Initialize WebSocket connections for all exchanges with colocation optimization."""
        for exchange_id, exchange_config in self.config.get("exchanges", {}).items():
            if not exchange_config.get("enabled", True):
                continue
            
            try:
                exchange_class = getattr(ccxt.pro, exchange_id)
                exchange_options = {
                    'adjustForTimeDifference': True,
                    'fetchTickers': False,
                    'fetchOHLCV': False,
                    'fetchOrderBook': True,
                    'fetchTrades': False,
                    'fetchBalance': False,
                    'defaultType': 'spot',
                }
                
                # Add colocation URL if available
                colocation_url = exchange_config.get("colocation_url")
                if colocation_url:
                    exchange_options['urls'] = {'api': colocation_url}
                
                exchange = exchange_class({
                    'apiKey': exchange_config.get("api_key", ""),
                    'secret': exchange_config.get("secret", ""),
                    'enableRateLimit': True,
                    'options': exchange_options
                })
                
                # Load markets and warm up connection
                await exchange.load_markets()
                await self._warm_up_connection(exchange, exchange_id)
                
                self.exchanges[exchange_id] = exchange
                logger.info(f"⚡ Initialized {exchange_id} with colocation (Latency: {self.latency_metrics.get(exchange_id, 0):.2f}ms)")
            
            except Exception as e:
                logger.error(f"❌ Failed to initialize {exchange_id}: {e}")
                self.alert_system.send_error_alert(f"Failed to initialize {exchange_id}: {e}")
    
    async def _warm_up_connection(self, exchange, exchange_id: str):
        """Warm up the connection with latency testing."""
        if exchange_id not in self.latency_metrics:
            self.latency_metrics[exchange_id] = {}
        
        try:
            start_time = time.time()
            await exchange.fetch_order_book("BTC/USDT", limit=5)
            latency_ms = (time.time() - start_time) * 1000
            self.latency_metrics[exchange_id] = latency_ms
        except Exception as e:
            logger.warning(f"Could not warm up connection for {exchange_id}: {e}")
            self.latency_metrics[exchange_id] = 0
    
    async def initialize_strategies(self):
        """Initialize all trading strategies."""
        from strategies.ml_strategies import LSTMStrategy, RandomForestStrategy
        
        # Machine Learning Strategies
        if self.config.get("strategies", {}).get("ml", {}).get("enabled", False):
            ml_config = self.config.get("strategies", {}).get("ml", {})
            
            if ml_config.get("lstm", {}).get("enabled", False):
                lstm_strategy = LSTMStrategy(self.config)
                lstm_strategy.set_risk_manager(self.risk_manager)
                lstm_strategy.set_performance_tracker(self.performance_trackers.get("lstm"))
                self.strategies.append(lstm_strategy)
                # Add symbols to data feed
                for symbol in ml_config.get("lstm", {}).get("symbols", []):
                    self.data_feed.add_symbol(symbol)
            
            if ml_config.get("random_forest", {}).get("enabled", False):
                rf_strategy = RandomForestStrategy(self.config)
                rf_strategy.set_risk_manager(self.risk_manager)
                rf_strategy.set_performance_tracker(self.performance_trackers.get("random_forest"))
                self.strategies.append(rf_strategy)
                # Add symbols to data feed
                for symbol in ml_config.get("random_forest", {}).get("symbols", []):
                    self.data_feed.add_symbol(symbol)
        
        # Arbitrage Strategies
        if self.config.get("strategies", {}).get("arbitrage", {}).get("enabled", False):
            from strategies.arbitrage import TriangularArbitrageStrategy, MarketMakingStrategy
            
            arb_config = self.config.get("strategies", {}).get("arbitrage", {})
            
            if arb_config.get("triangular", {}).get("enabled", False):
                arb_strategy = TriangularArbitrageStrategy(self.config, self.exchanges)
                arb_strategy.set_risk_manager(self.risk_manager)
                arb_strategy.set_performance_tracker(self.performance_trackers.get("triangular"))
                self.strategies.append(arb_strategy)
            
            if arb_config.get("market_making", {}).get("enabled", False):
                mm_config = arb_config.get("market_making", {})
                for exchange_id in mm_config.get("exchanges", []):
                    if exchange_id in self.exchanges:
                        mm_strategy = MarketMakingStrategy(self.config, self.exchanges[exchange_id])
                        mm_strategy.set_risk_manager(self.risk_manager)
                        mm_strategy.set_performance_tracker(self.performance_trackers.get("market_making"))
                        self.strategies.append(mm_strategy)
        
        # Quantum Strategies (optional)
        if self.config.get("strategies", {}).get("quantum", {}).get("enabled", False):
            from strategies.quantum_strategies import QuantumPortfolioOptimizer
            
            quantum_config = self.config.get("strategies", {}).get("quantum", {})
            assets = quantum_config.get("assets", ["BTC", "ETH", "SOL", "ADA"])
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
            quantum_strategy.set_performance_tracker(self.performance_trackers.get("quantum"))
            self.strategies.append(quantum_strategy)
    
    async def execute_trade(self, strategy: BaseStrategy, signal: Dict):
        """Execute a trade with full risk management."""
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
            exchange_id = None
            for ex_id, ex in self.exchanges.items():
                if symbol in ex.symbols:
                    exchange = ex
                    exchange_id = ex_id
                    break
            
            if not exchange:
                logger.error(f"No exchange found for symbol {symbol}")
                return False
            
            # Get current market data
            ohlcv_data = self.data_feed.get_data(symbol, 100)
            if not ohlcv_data:
                logger.error(f"No market data available for {symbol}")
                return False
            
            # Calculate position size with risk management
            atr = self._calculate_atr(ohlcv_data)
            stop_loss = self.risk_manager.calculate_stop_loss(strategy.name, ohlcv_data, price)
            position_size = self.risk_manager.calculate_position_size(
                strategy.name, symbol, self.risk_manager.account_balance, price, stop_loss, atr
            )
            
            if position_size <= 0:
                logger.warning(f"Position size calculation failed for {symbol}")
                return False
            
            # Check risk limits
            if self.risk_manager.check_risk_limits():
                logger.warning("Risk limits breached, trade cancelled")
                return False
            
            # Execute the trade
            if action == "buy":
                order = await exchange.create_market_buy_order(symbol, position_size)
            elif action == "sell":
                order = await exchange.create_market_sell_order(symbol, position_size)
            elif action == "long":
                # For statistical arbitrage
                order1 = await exchange.create_market_buy_order(signal.get("symbol1", symbol), position_size)
                order2 = await exchange.create_market_sell_order(signal.get("symbol2", symbol), position_size * price)
                order = {"order1": order1, "order2": order2}
            elif action == "short":
                # For statistical arbitrage
                order1 = await exchange.create_market_sell_order(signal.get("symbol1", symbol), position_size)
                order2 = await exchange.create_market_buy_order(signal.get("symbol2", symbol), position_size * price)
                order = {"order1": order1, "order2": order2}
            else:
                logger.error(f"Unknown action: {action}")
                return False
            
            # Record the trade
            entry_time = pd.Timestamp.now()
            self.risk_manager.add_position(
                strategy.name, symbol, price, stop_loss, entry_time.timestamp(), position_size, action
            )
            
            # Log to database
            if self.database:
                try:
                    self.database.log_signal(
                        strategy.name, symbol, action, price, confidence, entry_time
                    )
                except Exception as e:
                    logger.warning(f"Could not log signal: {e}")
            
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
                        data = self.data_feed.get_data(symbol, 1)
                        if not data:
                            continue
                        
                        current_price = data[-1][4]  # Latest close price
                        
                        if self.risk_manager.update_position(
                            strategy.name, symbol, current_price
                        ):
                            # Position was closed
                            exit_price = current_price
                            if position.get("action") == "buy":
                                pnl = (exit_price - position["entry_price"]) * position["size"]
                            else:
                                pnl = (position["entry_price"] - exit_price) * position["size"]
                            
                            pnl_pct = pnl / (position["entry_price"] * position["size"]) * 100 if position["entry_price"] * position["size"] > 0 else 0
                            
                            # Update daily PnL
                            self.risk_manager.update_daily_pnl(pnl)
                            
                            # Log to database
                            if self.database:
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
                                        "balance": self.risk_manager.account_balance
                                    })
                                except Exception as e:
                                    logger.warning(f"Could not log trade: {e}")
                            
                            # Send alert
                            self.alert_system.send_trade_alert(
                                strategy.name, symbol, "close", exit_price, position["size"], pnl
                            )
            
            except Exception as e:
                logger.error(f"⚠️ Error monitoring positions: {e}")
                self.alert_system.send_error_alert(f"Error monitoring positions: {e}")
            
            await asyncio.sleep(10)  # Check every 10 seconds
    
    async def track_performance(self):
        """Track and log performance metrics periodically."""
        while self.running:
            try:
                if not self.database:
                    await asyncio.sleep(3600)
                    continue
                
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
                        metrics = calculate_performance_metrics(trades, equity_curve, self.config.get("backtesting", {}).get("initial_balance", 10000))
                        
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
                self.alert_system.send_error_alert(f"Error tracking performance: {e}")
            
            await asyncio.sleep(3600)  # Update every hour
    
    async def check_system_health(self):
        """Check system health and send heartbeat."""
        while self.running:
            try:
                current_time = time.time()
                
                # Send heartbeat
                if current_time - self.last_heartbeat > self.heartbeat_interval:
                    self.alert_system.send_telegram_alert("💓 Kracken Trading Bot is running")
                    self.last_heartbeat = current_time
                
                # Check exchange connections
                for exchange_id, exchange in self.exchanges.items():
                    try:
                        await exchange.fetch_ticker("BTC/USDT")
                    except Exception as e:
                        logger.error(f"❌ Exchange {exchange_id} connection lost: {e}")
                        self.alert_system.send_error_alert(f"Exchange {exchange_id} connection lost: {e}")
                
                # Check data feed
                if not self.data_feed.data_buffers:
                    logger.error("❌ Data feed is not receiving data")
                    self.alert_system.send_error_alert("Data feed is not receiving data")
                
                # Check risk limits
                if self.risk_manager.check_risk_limits():
                    logger.warning("⚠️ Risk limits breached")
                    self.alert_system.send_error_alert("Risk limits breached")
            
            except Exception as e:
                logger.error(f"⚠️ Error checking system health: {e}")
                self.alert_system.send_error_alert(f"Error checking system health: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def run_strategies(self):
        """Run all strategies in parallel with ultra-low latency."""
        while self.running:
            try:
                # Update data feed
                await self.data_feed.update()
                
                # Get current market data for all symbols
                current_data = {}
                for symbol in self.data_feed.symbols:
                    current_data[symbol] = self.data_feed.get_data(symbol)
                
                # Update liquidity data for risk management
                self.risk_manager.liquidity.update_liquidity_data(self.data_feed.data_buffers)
                
                # Update correlation matrix for risk management
                self.risk_manager.correlation.update_correlation_matrix(current_data)
                
                # Run all strategies
                strategy_tasks = []
                for strategy in self.strategies:
                    if hasattr(strategy, 'symbols'):
                        # Get data for all symbols the strategy needs
                        data = {symbol: current_data[symbol]
                               for symbol in strategy.symbols if symbol in current_data}
                        strategy_tasks.append(strategy.run(data))
                    else:
                        strategy_tasks.append(strategy.run(current_data))
                
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
            
            # Ultra-low latency sleep (10ms)
            await asyncio.sleep(0.01)
    
    async def start(self):
        """Start the trading engine."""
        self.running = True
        logger.info("🚀 Starting Kracken Trading Bot - Ultimate Edition")
        
        # Initialize components
        await self.initialize_exchanges()
        await self.initialize_strategies()
        
        # Start all background tasks
        tasks = [
            self.run_strategies(),
            self.monitor_positions(),
            self.track_performance(),
            self.check_system_health()
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
        if self.database:
            try:
                self.database.disconnect()
            except Exception:
                pass
        logger.info("🛑 Kracken Trading Bot stopped")
