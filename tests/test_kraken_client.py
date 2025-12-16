"""Tests for improved Kraken client."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from exchange.kraken_client_improved import KrakenClient


@pytest.mark.asyncio
async def test_fetch_ticker():
    """Test fetching ticker."""
    mock_client = Mock()
    mock_client.fetch_ticker = AsyncMock(return_value={"last": 50000.0, "volume": 100.0})

    with patch("exchange.kraken_client_improved.ccxt") as mock_ccxt:
        mock_ccxt.kraken.return_value = mock_client
        client = KrakenClient(api_key="test", api_secret="test")

        result = await client.fetch_ticker("BTC/USD")
        assert result["last"] == 50000.0
        mock_client.fetch_ticker.assert_called_once_with("BTC/USD")


@pytest.mark.asyncio
async def test_fetch_ticker_retry():
    """Test that fetch_ticker retries on failure."""
    import ccxt

    mock_client = Mock()
    call_count = 0

    async def mock_fetch_ticker(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ccxt.NetworkError("Network error")
        return {"last": 50000.0}

    mock_client.fetch_ticker = mock_fetch_ticker

    with patch("exchange.kraken_client_improved.ccxt") as mock_ccxt:
        mock_ccxt.kraken.return_value = mock_client
        mock_ccxt.NetworkError = ccxt.NetworkError
        client = KrakenClient(api_key="test", api_secret="test")

        result = await client.fetch_ticker("BTC/USD")
        assert result["last"] == 50000.0
        assert call_count == 2


@pytest.mark.asyncio
async def test_create_order_market():
    """Test creating market order."""
    mock_client = Mock()
    mock_client.create_order = AsyncMock(return_value={"id": "12345", "status": "closed"})

    with patch("exchange.kraken_client_improved.ccxt") as mock_ccxt:
        mock_ccxt.kraken.return_value = mock_client
        client = KrakenClient(api_key="test", api_secret="test")

        result = await client.create_order("BTC/USD", "buy", 0.1)
        assert result["id"] == "12345"
        mock_client.create_order.assert_called_once_with("BTC/USD", "market", "buy", 0.1)


@pytest.mark.asyncio
async def test_create_order_limit():
    """Test creating limit order."""
    mock_client = Mock()
    mock_client.create_order = AsyncMock(return_value={"id": "12345", "status": "open"})

    with patch("exchange.kraken_client_improved.ccxt") as mock_ccxt:
        mock_ccxt.kraken.return_value = mock_client
        client = KrakenClient(api_key="test", api_secret="test")

        result = await client.create_order("BTC/USD", "sell", 0.1, order_type="limit", price=51000.0)
        assert result["id"] == "12345"
        mock_client.create_order.assert_called_once_with("BTC/USD", "limit", "sell", 0.1, 51000.0)


def test_create_order_invalid_side():
    """Test that invalid side raises ValueError."""
    with patch("exchange.kraken_client_improved.ccxt") as mock_ccxt:
        mock_ccxt.kraken.return_value = Mock()
        client = KrakenClient(api_key="test", api_secret="test")

        with pytest.raises(ValueError, match="Invalid side"):
            import asyncio
            asyncio.run(client.create_order("BTC/USD", "invalid", 0.1))

