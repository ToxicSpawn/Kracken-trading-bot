"""WebSocket client for real-time market data streaming."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Optional

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = logging.getLogger(__name__)


class KrakenWebSocketClient:
    """
    WebSocket client for Kraken real-time market data.

    Supports:
    - Ticker updates
    - Order book updates
    - Trade updates
    - OHLCV updates
    """

    WS_URL = "wss://ws.kraken.com"

    def __init__(
        self,
        on_message: Optional[Callable[[dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """
        Initialize WebSocket client.

        Args:
            on_message: Callback function for received messages
            on_error: Callback function for errors
        """
        self.on_message = on_message
        self.on_error = on_error
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0

    async def connect(self) -> None:
        """Connect to Kraken WebSocket."""
        try:
            self.ws = await websockets.connect(self.WS_URL)
            self._running = True
            self._reconnect_delay = 1.0
            logger.info("Connected to Kraken WebSocket")
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            if self.on_error:
                self.on_error(e)
            raise

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._running = False
        if self.ws:
            await self.ws.close()
            logger.info("Disconnected from Kraken WebSocket")

    async def subscribe_ticker(self, pairs: list[str]) -> None:
        """
        Subscribe to ticker updates.

        Args:
            pairs: List of trading pairs (e.g., ['XBT/USD', 'ETH/USD'])
        """
        if not self.ws:
            await self.connect()

        message = {
            "event": "subscribe",
            "pair": pairs,
            "subscription": {"name": "ticker"},
        }
        await self._send(message)
        logger.info(f"Subscribed to ticker for {pairs}")

    async def subscribe_orderbook(self, pairs: list[str], depth: int = 10) -> None:
        """
        Subscribe to order book updates.

        Args:
            pairs: List of trading pairs
            depth: Order book depth
        """
        if not self.ws:
            await self.connect()

        message = {
            "event": "subscribe",
            "pair": pairs,
            "subscription": {"name": "book", "depth": depth},
        }
        await self._send(message)
        logger.info(f"Subscribed to order book for {pairs}")

    async def subscribe_trades(self, pairs: list[str]) -> None:
        """
        Subscribe to trade updates.

        Args:
            pairs: List of trading pairs
        """
        if not self.ws:
            await self.connect()

        message = {
            "event": "subscribe",
            "pair": pairs,
            "subscription": {"name": "trade"},
        }
        await self._send(message)
        logger.info(f"Subscribed to trades for {pairs}")

    async def subscribe_ohlc(self, pairs: list[str], interval: int = 60) -> None:
        """
        Subscribe to OHLC (candlestick) updates.

        Args:
            pairs: List of trading pairs
            interval: Interval in seconds (60, 300, 900, 3600, etc.)
        """
        if not self.ws:
            await self.connect()

        message = {
            "event": "subscribe",
            "pair": pairs,
            "subscription": {"name": "ohlc", "interval": interval},
        }
        await self._send(message)
        logger.info(f"Subscribed to OHLC for {pairs} with interval {interval}s")

    async def _send(self, message: dict[str, Any]) -> None:
        """Send message to WebSocket."""
        if not self.ws:
            raise RuntimeError("WebSocket not connected")
        await self.ws.send(json.dumps(message))

    async def listen(self) -> None:
        """Listen for messages and handle reconnection."""
        while self._running:
            try:
                if not self.ws:
                    await self.connect()

                async for message in self.ws:
                    try:
                        data = json.loads(message)
                        if self.on_message:
                            self.on_message(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        if self.on_error:
                            self.on_error(e)

            except (ConnectionClosed, WebSocketException) as e:
                if self._running:
                    logger.warning(f"WebSocket connection lost: {e}. Reconnecting...")
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2,
                        self._max_reconnect_delay,
                    )
                    self.ws = None
                else:
                    break
            except Exception as e:
                logger.error(f"Unexpected error in WebSocket listener: {e}")
                if self.on_error:
                    self.on_error(e)
                if self._running:
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2,
                        self._max_reconnect_delay,
                    )
                    self.ws = None

    async def run(self) -> None:
        """Run the WebSocket client (connect and listen)."""
        await self.connect()
        await self.listen()


# Example usage
async def example_usage() -> None:
    """Example of using the WebSocket client."""

    def on_ticker_update(data: dict[str, Any]) -> None:
        """Handle ticker updates."""
        if isinstance(data, list) and len(data) > 1:
            # Ticker data format: [channel_id, ticker_data, channel_name, pair]
            ticker_data = data[1]
            if isinstance(ticker_data, dict) and "c" in ticker_data:
                price = ticker_data["c"][0]  # Last price
                print(f"Ticker update: {price}")

    client = KrakenWebSocketClient(on_message=on_ticker_update)
    await client.subscribe_ticker(["XBT/USD", "ETH/USD"])
    await client.run()


if __name__ == "__main__":
    asyncio.run(example_usage())

