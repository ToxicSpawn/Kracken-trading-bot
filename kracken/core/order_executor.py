"""Order execution logic."""
from typing import Dict, Optional
from utils.logger import logger

class OrderExecutor:
    """Handles order execution."""
    
    def __init__(self, exchange):
        self.exchange = exchange
    
    async def execute_market_order(self, symbol: str, side: str, amount: float) -> Optional[Dict]:
        """Execute a market order."""
        try:
            if side.lower() == "buy":
                order = await self.exchange.create_market_buy_order(symbol, amount)
            elif side.lower() == "sell":
                order = await self.exchange.create_market_sell_order(symbol, amount)
            else:
                logger.error(f"Unknown order side: {side}")
                return None
            
            logger.info(f"✅ Executed {side} order for {symbol}: {order.get('id', 'N/A')}")
            return order
        except Exception as e:
            logger.error(f"❌ Error executing {side} order for {symbol}: {e}")
            return None
    
    async def execute_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Optional[Dict]:
        """Execute a limit order."""
        try:
            if side.lower() == "buy":
                order = await self.exchange.create_limit_buy_order(symbol, amount, price)
            elif side.lower() == "sell":
                order = await self.exchange.create_limit_sell_order(symbol, amount, price)
            else:
                logger.error(f"Unknown order side: {side}")
                return None
            
            logger.info(f"✅ Placed {side} limit order for {symbol} @ {price:.8f}: {order.get('id', 'N/A')}")
            return order
        except Exception as e:
            logger.error(f"❌ Error placing {side} limit order for {symbol}: {e}")
            return None
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        try:
            await self.exchange.cancel_order(order_id, symbol)
            logger.info(f"✅ Cancelled order {order_id} for {symbol}")
            return True
        except Exception as e:
            logger.error(f"❌ Error cancelling order {order_id}: {e}")
            return False
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> list:
        """Get open orders."""
        try:
            if symbol:
                orders = await self.exchange.fetch_open_orders(symbol)
            else:
                orders = await self.exchange.fetch_open_orders()
            return orders
        except Exception as e:
            logger.error(f"❌ Error fetching open orders: {e}")
            return []

