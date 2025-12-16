"""Historical data fetcher."""
import asyncio
from typing import Dict, List, Optional
import pandas as pd
from utils.logger import logger

class HistoricalDataFetcher:
    """Fetch historical market data."""
    
    def __init__(self, exchange):
        self.exchange = exchange
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1d", 
                         since: Optional[int] = None, limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            if not ohlcv:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"✅ Fetched {len(df)} candles for {symbol}")
            return df
        except Exception as e:
            logger.error(f"❌ Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_multiple_symbols(self, symbols: List[str], timeframe: str = "1d",
                                    since: Optional[int] = None, limit: int = 1000) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols."""
        results = {}
        
        tasks = [self.fetch_ohlcv(symbol, timeframe, since, limit) for symbol in symbols]
        dataframes = await asyncio.gather(*tasks)
        
        for symbol, df in zip(symbols, dataframes):
            results[symbol] = df
        
        return results

