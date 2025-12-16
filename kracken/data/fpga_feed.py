"""FPGA-accelerated data feed for ultra-low latency."""
import asyncio
import numpy as np
from typing import Dict, List, Optional
from utils.logger import logger

class FPGADataFeed:
    """FPGA-accelerated data feed (simulated - replace with actual FPGA implementation)."""
    
    def __init__(self, engine):
        self.engine = engine
        self.data_buffers: Dict[str, List] = {}
        self.symbols: List[str] = []
        self.max_data_points = 1000
        self.latency_optimized = True
    
    async def update(self):
        """Update data for all symbols via WebSocket with latency optimization."""
        update_tasks = []
        
        for exchange_id, exchange in self.engine.exchanges.items():
            for symbol in self.symbols:
                if symbol in exchange.symbols:
                    update_tasks.append(self._update_symbol(exchange, symbol))
        
        if update_tasks:
            await asyncio.gather(*update_tasks, return_exceptions=True)
    
    async def _update_symbol(self, exchange, symbol: str):
        """Update data for a single symbol."""
        try:
            ohlcv = await exchange.watch_ohlcv(symbol, "1m")
            if ohlcv:
                if symbol not in self.data_buffers:
                    self.data_buffers[symbol] = []
                
                self.data_buffers[symbol].append(ohlcv)
                
                # Keep only the most recent data points
                if len(self.data_buffers[symbol]) > self.max_data_points:
                    self.data_buffers[symbol] = self.data_buffers[symbol][-self.max_data_points:]
        except Exception as e:
            logger.debug(f"Error updating {symbol}: {e}")
    
    def get_data(self, symbol: str, lookback: int = 100) -> List:
        """Get recent data for a symbol."""
        if symbol not in self.data_buffers:
            return []
        
        data = self.data_buffers[symbol]
        if lookback:
            return data[-lookback:] if len(data) > lookback else data
        return data
    
    def add_symbol(self, symbol: str):
        """Add a symbol to monitor."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            logger.info(f"📡 Added {symbol} to data feed")

