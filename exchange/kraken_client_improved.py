"""Improved Kraken client with type hints, retries, and better error handling."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import ccxt  # type: ignore

from utils.retry import async_retry_with_backoff
from utils.security import get_encrypted_env

logger = logging.getLogger(__name__)


class KrakenClient:
    """
    Improved Kraken spot client via ccxt with:
    - Type hints
    - Retry logic with exponential backoff
    - Encrypted API key support
    - Better error handling
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        sandbox: bool = False,
    ) -> None:
        """
        Initialize Kraken client.

        Args:
            api_key: API key (if None, reads from KRAKEN_API_KEY env var)
            api_secret: API secret (if None, reads from KRAKEN_API_SECRET env var)
            sandbox: Use sandbox/testnet mode
        """
        # Try to get encrypted keys first, fallback to plain env vars
        key = api_key or get_encrypted_env("KRAKEN_API_KEY") or os.getenv("KRAKEN_API_KEY")
        secret = api_secret or get_encrypted_env("KRAKEN_API_SECRET") or os.getenv("KRAKEN_API_SECRET")

        if not key or not secret:
            logger.warning("Kraken API credentials not found. Some operations will fail.")

        self._client = ccxt.kraken({
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "sandbox": sandbox,
        })
        self.sandbox = sandbox

    @async_retry_with_backoff(
        max_attempts=3,
        initial_wait=1.0,
        max_wait=30.0,
        retry_on=(ccxt.NetworkError, ccxt.ExchangeError),
    )
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 200,
        since: Optional[int] = None,
    ) -> list[list[float]]:
        """
        Fetch OHLCV (candlestick) data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            timeframe: Timeframe (1m, 5m, 1h, 1d, etc.)
            limit: Number of candles to fetch
            since: Timestamp in milliseconds (optional)

        Returns:
            List of [timestamp, open, high, low, close, volume] lists

        Raises:
            ccxt.ExchangeError: If API call fails after retries
        """
        try:
            result = await self._client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
            return result
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            raise

    @async_retry_with_backoff(
        max_attempts=3,
        initial_wait=1.0,
        max_wait=30.0,
        retry_on=(ccxt.NetworkError, ccxt.ExchangeError),
    )
    async def fetch_order_book(
        self,
        symbol: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Fetch order book.

        Args:
            symbol: Trading pair symbol
            limit: Number of orders per side

        Returns:
            Order book dictionary with 'bids' and 'asks'

        Raises:
            ccxt.ExchangeError: If API call fails after retries
        """
        try:
            result = await self._client.fetch_order_book(symbol, limit=limit)
            return result
        except Exception as e:
            logger.error(f"Failed to fetch order book for {symbol}: {e}")
            raise

    @async_retry_with_backoff(
        max_attempts=3,
        initial_wait=1.0,
        max_wait=30.0,
        retry_on=(ccxt.NetworkError, ccxt.ExchangeError),
    )
    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """
        Fetch ticker data.

        Args:
            symbol: Trading pair symbol

        Returns:
            Ticker dictionary with price, volume, etc.

        Raises:
            ccxt.ExchangeError: If API call fails after retries
        """
        try:
            result = await self._client.fetch_ticker(symbol)
            return result
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            raise

    @async_retry_with_backoff(
        max_attempts=3,
        initial_wait=1.0,
        max_wait=30.0,
        retry_on=(ccxt.NetworkError, ccxt.ExchangeError),
    )
    async def create_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "market",
        price: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Create an order.

        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            amount: Order amount
            order_type: 'market' or 'limit'
            price: Limit price (required for limit orders)

        Returns:
            Order information dictionary

        Raises:
            ccxt.ExchangeError: If API call fails after retries
            ValueError: If invalid parameters provided
        """
        if order_type == "limit" and price is None:
            raise ValueError("Price is required for limit orders")

        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")

        try:
            if order_type == "market":
                result = await self._client.create_order(symbol, "market", side, amount)
            else:
                result = await self._client.create_order(symbol, "limit", side, amount, price)
            return result
        except Exception as e:
            logger.error(f"Failed to create {side} order for {symbol}: {e}")
            raise

    @async_retry_with_backoff(
        max_attempts=3,
        initial_wait=1.0,
        max_wait=30.0,
        retry_on=(ccxt.NetworkError, ccxt.ExchangeError),
    )
    async def fetch_balance(self) -> dict[str, Any]:
        """
        Fetch account balance.

        Returns:
            Balance dictionary

        Raises:
            ccxt.ExchangeError: If API call fails after retries
        """
        try:
            result = await self._client.fetch_balance()
            return result
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            raise

    @async_retry_with_backoff(
        max_attempts=3,
        initial_wait=1.0,
        max_wait=30.0,
        retry_on=(ccxt.NetworkError, ccxt.ExchangeError),
    )
    async def fetch_order(self, order_id: str, symbol: Optional[str] = None) -> dict[str, Any]:
        """
        Fetch order status.

        Args:
            order_id: Order ID
            symbol: Trading pair symbol (optional, some exchanges require it)

        Returns:
            Order information dictionary

        Raises:
            ccxt.ExchangeError: If API call fails after retries
        """
        try:
            result = await self._client.fetch_order(order_id, symbol)
            return result
        except Exception as e:
            logger.error(f"Failed to fetch order {order_id}: {e}")
            raise

    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> dict[str, Any]:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading pair symbol (optional)

        Returns:
            Cancellation result

        Raises:
            ccxt.ExchangeError: If API call fails after retries
        """
        try:
            result = await self._client.cancel_order(order_id, symbol)
            return result
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise

