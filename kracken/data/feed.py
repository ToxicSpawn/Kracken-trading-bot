"""Real-time data feed."""
import asyncio
from typing import Dict, List
from utils.logger import logger

class DataFeed:
    """Real-time data feed for trading symbols."""
    
    def __init__(self, exchange, symbols: List[str]):
        self.exchange = exchange
        self.symbols = symbols
        self.data: Dict[str, List] = {symbol: [] for symbol in symbols}
        self.max_data_points = 1000  # Maximum number of data points to store
    
    async def update(self):
        """Update data for all symbols via WebSocket."""
        for symbol in self.symbols:
            try:
                ohlcv = await self.exchange.watch_ohlcv(symbol, "1m")
                if ohlcv:
                    self.data[symbol].append(ohlcv)
                    
                    # Keep only the most recent data points
                    if len(self.data[symbol]) > self.max_data_points:
                        self.data[symbol] = self.data[symbol][-self.max_data_points:]
                    
                    logger.debug(f"📈 Updated {symbol}: {ohlcv}")
            except Exception as e:
                logger.error(f"⚠️ Error updating {symbol}: {e}")
    
    def get_data(self, symbol: str, lookback: int = 100) -> List:
        """Get recent data for a symbol."""
        if symbol not in self.data:
            return []
        
        if lookback:
            return self.data[symbol][-lookback:] if len(self.data[symbol]) > lookback else self.data[symbol]
        return self.data[symbol]

