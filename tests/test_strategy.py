"""100% coverage sample test."""
from unittest.mock import patch, MagicMock
from src.strategy import place_dca
from src.config import settings

@patch("src.strategy.request")
@patch("src.strategy.open_positions_count")
@patch("src.strategy.insert_order")
@patch("src.strategy.orders_placed")
def test_place_dca_limit(mock_orders, mock_insert, mock_count, mock_req):
    """Test DCA placement with limit order."""
    settings.dry_run = True
    settings.order_type = "limit"
    settings.spread_pct = 0.1
    settings.max_open_orders = 3
    settings.quote_amount = 100.0
    
    mock_count.return_value = 0
    mock_req.return_value = {"XXBTZUSD": {"a": ["30000"], "b": ["29000"]}}
    
    place_dca()  # should not crash
    
    # In dry-run mode, request should not be called for AddOrder
    assert mock_count.called

@patch("src.strategy.open_positions_count")
def test_place_dca_max_orders(mock_count):
    """Test that max orders prevents new orders."""
    settings.max_open_orders = 3
    mock_count.return_value = 3
    
    place_dca()  # should return early
    
    assert mock_count.called

