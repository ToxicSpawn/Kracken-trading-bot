"""Tests for cache utilities."""

import pytest
from unittest.mock import Mock, patch
from utils.cache import CacheManager


def test_cache_manager_no_redis():
    """Test cache manager when Redis is not available."""
    with patch("utils.cache.REDIS_AVAILABLE", False):
        cache = CacheManager()
        assert not cache.is_available()
        assert cache.get("test") is None
        assert cache.set("test", "value") is False


def test_cache_manager_with_redis():
    """Test cache manager with Redis."""
    mock_redis = Mock()
    mock_redis.ping.return_value = True
    mock_redis.get.return_value = '{"key": "value"}'
    mock_redis.setex.return_value = True
    mock_redis.delete.return_value = 1
    mock_redis.keys.return_value = ["key1", "key2"]

    with patch("utils.cache.redis") as mock_redis_module:
        mock_redis_module.Redis.return_value = mock_redis
        with patch("utils.cache.REDIS_AVAILABLE", True):
            cache = CacheManager()

            # Test get
            result = cache.get("test")
            assert result == {"key": "value"}

            # Test set
            assert cache.set("test", {"data": "value"}) is True

            # Test delete
            assert cache.delete("test") is True

            # Test clear_pattern
            count = cache.clear_pattern("test:*")
            assert count == 2

