"""Unified exchange interface."""
import ccxt.pro
from typing import Dict
from utils.logger import logger

class ExchangeManager:
    """Manages multiple exchange connections."""
    
    def __init__(self, exchanges_config: Dict):
        self.exchanges_config = exchanges_config
        self.exchanges: Dict[str, ccxt.pro.Exchange] = {}
    
    async def initialize(self):
        """Initialize all enabled exchanges."""
        for exchange_id, exchange_config in self.exchanges_config.items():
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
                logger.info(f"✅ Initialized {exchange_id}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {exchange_id}: {e}")
    
    async def close_all(self):
        """Close all exchange connections."""
        for exchange in self.exchanges.values():
            try:
                await exchange.close()
            except Exception:
                pass

