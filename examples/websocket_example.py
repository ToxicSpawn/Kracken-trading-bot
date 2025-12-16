"""Example: Using WebSocket for real-time market data."""

import asyncio
from exchange.websocket_client import KrakenWebSocketClient


def on_ticker_update(data: dict) -> None:
    """Handle ticker updates."""
    if isinstance(data, list) and len(data) > 1:
        # Check if it's ticker data
        if isinstance(data[1], dict) and "c" in data[1]:
            ticker_data = data[1]
            last_price = ticker_data["c"][0]  # Last price
            volume = ticker_data["v"][1]  # 24h volume
            print(f"Ticker: Price=${last_price}, Volume={volume}")


def on_orderbook_update(data: dict) -> None:
    """Handle order book updates."""
    if isinstance(data, list) and len(data) > 1:
        if isinstance(data[1], dict) and "bids" in data[1]:
            book_data = data[1]
            best_bid = book_data["bids"][0][0] if book_data["bids"] else None
            best_ask = book_data["asks"][0][0] if book_data["asks"] else None
            if best_bid and best_ask:
                spread = float(best_ask) - float(best_bid)
                print(f"Order Book: Bid=${best_bid}, Ask=${best_ask}, Spread=${spread:.2f}")


def on_trade_update(data: dict) -> None:
    """Handle trade updates."""
    if isinstance(data, list) and len(data) > 1:
        if isinstance(data[1], list):
            trades = data[1]
            for trade in trades[:3]:  # Show first 3 trades
                price = trade[0]
                volume = trade[1]
                side = "BUY" if trade[3] == "b" else "SELL"
                print(f"Trade: {side} {volume} @ ${price}")


def handle_message(data: dict) -> None:
    """Handle all WebSocket messages and route to appropriate handler."""
    if isinstance(data, list) and len(data) > 2:
        data_type = data[2] if len(data) > 2 else None
        if data_type == "ticker":
            on_ticker_update(data)
        elif data_type == "book":
            on_orderbook_update(data)
        elif data_type == "trade":
            on_trade_update(data)


async def run_websocket_example():
    """Run WebSocket example."""
    print("Connecting to Kraken WebSocket...")

    # Create client with callbacks
    client = KrakenWebSocketClient(on_message=handle_message)

    # Subscribe to multiple data streams
    await client.subscribe_ticker(["XBT/USD", "ETH/USD"])
    await client.subscribe_orderbook(["XBT/USD"])
    await client.subscribe_trades(["XBT/USD"])

    print("Subscribed to market data. Listening for updates...")
    print("Press Ctrl+C to stop.\n")

    # Run for 60 seconds
    try:
        await asyncio.wait_for(client.listen(), timeout=60.0)
    except asyncio.TimeoutError:
        print("\nStopping after 60 seconds...")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(run_websocket_example())

