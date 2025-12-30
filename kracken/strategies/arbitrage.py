"""Arbitrage & market-making strategies."""
import time
from typing import Dict, List, Optional
from strategies.base_strategy import BaseStrategy

class TriangularArbitrageStrategy(BaseStrategy):
    """Triangular arbitrage strategy."""
    
    def __init__(self, config: Dict, exchanges: Dict):
        super().__init__(config, "Triangular_Arbitrage")
        self.exchanges = exchanges
        arb_config = config.get("strategies", {}).get("arbitrage", {})
        triangular_config = arb_config.get("triangular", {})
        self.min_profit_pct = triangular_config.get("min_profit_pct", 0.3)
        self.arbitrage_pairs = self._initialize_arbitrage_pairs()
    
    def _initialize_arbitrage_pairs(self) -> List[Dict]:
        """Initialize potential arbitrage pairs."""
        return [
            {
                "name": "BTC_ETH_USDT",
                "path": ["BTC/USDT", "ETH/BTC", "ETH/USDT"],
                "min_profit_pct": self.min_profit_pct
            },
            {
                "name": "BTC_BNB_USDT",
                "path": ["BTC/USDT", "BNB/BTC", "BNB/USDT"],
                "min_profit_pct": self.min_profit_pct
            },
            {
                "name": "ETH_BNB_USDT",
                "path": ["ETH/USDT", "BNB/ETH", "BNB/USDT"],
                "min_profit_pct": self.min_profit_pct
            }
        ]
    
    async def _get_prices(self, exchange_id: str, symbols: List[str]) -> Optional[Dict[str, float]]:
        """Get current prices for symbols on a specific exchange."""
        if exchange_id not in self.exchanges:
            return None
        
        prices = {}
        for symbol in symbols:
            try:
                ticker = await self.exchanges[exchange_id].fetch_ticker(symbol)
                prices[symbol] = ticker.get("last", 0)
            except Exception as e:
                self.log(f"Error fetching {symbol} on {exchange_id}: {e}")
                return None
        return prices
    
    async def _calculate_arbitrage(self, prices: Dict[str, float], path: List[str]) -> Optional[Dict]:
        """Calculate arbitrage opportunity for a given path."""
        if len(path) != 3:
            return None
        
        # Check if the path is valid
        first_pair = path[0]
        second_pair = path[1]
        third_pair = path[2]
        
        try:
            first_base, first_quote = first_pair.split('/')
            second_base, second_quote = second_pair.split('/')
            third_base, third_quote = third_pair.split('/')
            
            if first_quote != second_base:
                return None
            if second_quote != third_quote:
                return None
            if third_base != first_base:
                return None
            
            # Calculate the arbitrage
            start_amount = 1  # Start with 1 unit of the first asset
            first_price = prices.get(first_pair, 0)
            second_price = prices.get(second_pair, 0)
            third_price = prices.get(third_pair, 0)
            
            if not all([first_price, second_price, third_price]):
                return None
            
            # First trade: start_amount of first_pair[0] -> second_pair[0]
            amount_after_first = start_amount / first_price
            
            # Second trade: amount_after_first of second_pair[0] -> third_pair[1]
            amount_after_second = amount_after_first * second_price
            
            # Third trade: amount_after_second of third_pair[1] -> first_pair[0]
            final_amount = amount_after_second / third_price
            
            # Calculate profit percentage
            profit_pct = (final_amount - start_amount) / start_amount * 100
            
            return {
                "path": path,
                "profit_pct": profit_pct,
                "start_amount": start_amount,
                "final_amount": final_amount,
                "prices": {
                    first_pair: first_price,
                    second_pair: second_price,
                    third_pair: third_price
                }
            }
        except Exception as e:
            self.log(f"Error calculating arbitrage: {e}")
            return None
    
    async def run(self, _=None) -> Optional[Dict]:
        """Run the triangular arbitrage strategy."""
        for pair_config in self.arbitrage_pairs:
            for exchange_id in self.exchanges:
                try:
                    prices = await self._get_prices(exchange_id, pair_config["path"])
                    if not prices:
                        continue
                    
                    arbitrage = await self._calculate_arbitrage(prices, pair_config["path"])
                    if arbitrage and arbitrage["profit_pct"] > pair_config["min_profit_pct"]:
                        self.log(f"Arbitrage opportunity on {exchange_id}: {arbitrage['profit_pct']:.4f}%")
                        return {
                            "action": "arbitrage",
                            "exchange": exchange_id,
                            "path": pair_config["path"],
                            "profit_pct": arbitrage["profit_pct"],
                            "confidence": min(arbitrage["profit_pct"] / 2, 1.0),
                            "details": arbitrage
                        }
                except Exception as e:
                    self.log(f"Error in arbitrage calculation for {exchange_id}: {e}")
        
        return None


class MarketMakingStrategy(BaseStrategy):
    """Market-making strategy."""
    
    def __init__(self, config: Dict, exchange):
        super().__init__(config, "Market_Making")
        self.exchange = exchange
        arb_config = config.get("strategies", {}).get("arbitrage", {})
        mm_config = arb_config.get("market_making", {})
        self.symbol = mm_config.get("symbol", "BTC/USDT")
        self.spread_pct = mm_config.get("spread_pct", 0.1)
        self.order_size = mm_config.get("order_size", 0.01)
        self.order_refresh_time = mm_config.get("order_refresh_time", 30)
        self.last_order_time = 0
        self.open_orders = []
    
    async def _get_order_book(self):
        """Get the current order book for the symbol."""
        try:
            order_book = await self.exchange.fetch_order_book(self.symbol)
            return order_book
        except Exception as e:
            self.log(f"Error fetching order book: {e}")
            return None
    
    async def _cancel_all_orders(self):
        """Cancel all open orders for the symbol."""
        try:
            orders = await self.exchange.fetch_open_orders(self.symbol)
            for order in orders:
                await self.exchange.cancel_order(order["id"], self.symbol)
                self.log(f"Cancelled order {order['id']}")
            self.open_orders = []
        except Exception as e:
            self.log(f"Error cancelling orders: {e}")
    
    async def _place_orders(self, bid_price: float, ask_price: float):
        """Place bid and ask orders."""
        try:
            # Place bid order (buy)
            bid_order = await self.exchange.create_limit_buy_order(
                self.symbol,
                self.order_size,
                bid_price
            )
            self.open_orders.append(bid_order.get("id", ""))
            
            # Place ask order (sell)
            ask_order = await self.exchange.create_limit_sell_order(
                self.symbol,
                self.order_size,
                ask_price
            )
            self.open_orders.append(ask_order.get("id", ""))
            
            self.log(f"Placed orders - Bid: {bid_price:.8f}, Ask: {ask_price:.8f}")
            return bid_order, ask_order
        except Exception as e:
            self.log(f"Error placing orders: {e}")
            return None, None
    
    async def run(self, _=None) -> Optional[Dict]:
        """Run the market-making strategy."""
        current_time = time.time()
        
        if current_time - self.last_order_time < self.order_refresh_time:
            return None  # Don't refresh orders too frequently
        
        self.last_order_time = current_time
        
        try:
            # Cancel existing orders
            await self._cancel_all_orders()
            
            # Get current order book
            order_book = await self._get_order_book()
            if not order_book:
                return None
            
            # Calculate mid price
            best_bid = order_book.get("bids", [[0]])[0][0] if len(order_book.get("bids", [])) > 0 else None
            best_ask = order_book.get("asks", [[0]])[0][0] if len(order_book.get("asks", [])) > 0 else None
            
            if not best_bid or not best_ask:
                return None
            
            mid_price = (best_bid + best_ask) / 2
            
            # Calculate bid and ask prices with spread
            bid_price = mid_price * (1 - self.spread_pct / 200)
            ask_price = mid_price * (1 + self.spread_pct / 200)
            
            # Place new orders
            await self._place_orders(bid_price, ask_price)
            
            return {
                "action": "market_make",
                "symbol": self.symbol,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "spread_pct": self.spread_pct,
                "confidence": 1.0
            }
        except Exception as e:
            self.log(f"Error in market making: {e}")
            return None

