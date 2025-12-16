"""Test Kraken API wrapper."""
from unittest.mock import patch, Mock
import pytest
from src.kraken_api import request

@patch("src.kraken_api.requests.post")
@patch("src.kraken_api.next_nonce")
def test_request_success(mock_nonce, mock_post):
    """Test successful API request."""
    mock_nonce.return_value = 1234567890
    mock_response = Mock()
    mock_response.json.return_value = {"result": {"test": "data"}, "error": []}
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response
    
    result = request("/public/Ticker", {"pair": "XBTUSD"})
    
    assert result == {"test": "data"}
    assert mock_post.called

@patch("src.kraken_api.requests.post")
@patch("src.kraken_api.next_nonce")
def test_request_retry(mock_nonce, mock_post):
    """Test API request retry on failure."""
    mock_nonce.return_value = 1234567890
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = Exception("Connection error")
    mock_post.return_value = mock_response
    
    with pytest.raises(RuntimeError):
        request("/public/Ticker", {"pair": "XBTUSD"}, retries=2)

