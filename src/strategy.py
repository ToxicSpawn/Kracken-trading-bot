"""Grid/DCA strategy with limit spread."""
from .kraken_api import request
from .db import insert_order, open_positions_count
from .logger import log
from .metrics import orders_placed, pnl_usd
from .config import settings

def place_dca():
    """Place DCA order."""
    if open_positions_count() >= settings.max_open_orders:
        log.info("max orders reached")
        return
    
    ticker = request("/public/Ticker", {"pair": settings.kraken_pair})
    pair_key = list(ticker.keys())[0]  # Get first key (e.g., "XXBTZUSD")
    ask = float(ticker[pair_key]["a"][0])
    bid = float(ticker[pair_key]["b"][0])
    
    if settings.order_type == "limit":
        price = bid * (1 - settings.spread_pct / 100)
    else:
        price = ask
    
    vol = round(settings.quote_amount / price, 8)
    log.info("placing buy %.8f @ %.2f", vol, price)
    
    if settings.dry_run:
        log.warning("DRY-RUN – no order sent")
        return
    
    txid = request("/private/AddOrder", {
        "pair": settings.kraken_pair,
        "type": "buy",
        "ordertype": settings.order_type,
        "price": str(price),
        "volume": str(vol),
    })["txid"][0]
    
    insert_order(txid, "buy", vol, price)
    orders_placed.inc()
    log.info("order placed: %s", txid)

