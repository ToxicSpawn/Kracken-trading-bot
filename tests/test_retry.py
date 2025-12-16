"""Tests for retry utilities."""

import pytest
from unittest.mock import Mock, patch
from utils.retry import retry_with_backoff, async_retry_with_backoff


def test_retry_success():
    """Test that successful function doesn't retry."""
    @retry_with_backoff(max_attempts=3)
    def success_func():
        return "success"

    assert success_func() == "success"


def test_retry_failure():
    """Test that function retries on failure."""
    call_count = 0

    @retry_with_backoff(max_attempts=3, initial_wait=0.1)
    def failing_func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Temporary failure")
        return "success"

    result = failing_func()
    assert result == "success"
    assert call_count == 3


def test_retry_max_attempts():
    """Test that retry stops after max attempts."""
    call_count = 0

    @retry_with_backoff(max_attempts=2, initial_wait=0.1)
    def always_failing_func():
        nonlocal call_count
        call_count += 1
        raise ValueError("Always fails")

    with pytest.raises(ValueError):
        always_failing_func()

    assert call_count == 2


@pytest.mark.asyncio
async def test_async_retry_success():
    """Test async retry with success."""
    @async_retry_with_backoff(max_attempts=3)
    async def async_success_func():
        return "success"

    result = await async_success_func()
    assert result == "success"


@pytest.mark.asyncio
async def test_async_retry_failure():
    """Test async retry with eventual success."""
    call_count = 0

    @async_retry_with_backoff(max_attempts=3, initial_wait=0.1)
    async def async_failing_func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Temporary failure")
        return "success"

    result = await async_failing_func()
    assert result == "success"
    assert call_count == 2

