"""
Test mocks and utilities for trading bot testing.

This module contains all mock classes previously embedded in production code,
now properly isolated for testing purposes only.
"""


class MockTradingClient:
    """Mock Alpaca TradingClient for testing."""
    
    def __init__(self, *args, **kwargs): 
        pass
        
    def get_account(self): 
        return type('Account', (), {'equity': '100000'})()
        
    def submit_order(self, *args, **kwargs): 
        return {'status': 'filled'}


class MockMarketOrderRequest:
    """Mock MarketOrderRequest for testing."""
    
    def __init__(self, *args, **kwargs):
        pass


class MockLimitOrderRequest:
    """Mock LimitOrderRequest for testing."""
    
    def __init__(self, *args, **kwargs):
        pass


class MockGetOrdersRequest:
    """Mock GetOrdersRequest for testing."""
    
    def __init__(self, *args, **kwargs):
        pass


class MockOrderSide:
    """Mock OrderSide enum for testing."""
    
    BUY = 'buy'
    SELL = 'sell'
    
    def __init__(self, *args, **kwargs):
        pass


class MockTimeInForce:
    """Mock TimeInForce enum for testing."""
    
    DAY = 'day'
    
    def __init__(self, *args, **kwargs):
        pass


class MockOrderStatus:
    """Mock OrderStatus enum for testing."""
    
    FILLED = 'filled'
    OPEN = 'open'
    
    def __init__(self, *args, **kwargs):
        pass


class MockQueryOrderStatus:
    """Mock QueryOrderStatus enum for testing."""
    
    FILLED = 'filled'
    OPEN = 'open'
    
    def __init__(self, *args, **kwargs):
        pass


class MockOrder:
    """Mock Order model for testing."""
    
    def __init__(self, *args, **kwargs):
        pass


class MockTradingStream:
    """Mock TradingStream for testing."""
    
    def __init__(self, *args, **kwargs):
        pass
        
    def subscribe_trades(self, *args, **kwargs):
        pass
        
    def subscribe_quotes(self, *args, **kwargs):
        pass
        
    def subscribe_trade_updates(self, *args, **kwargs):
        pass
        
    def run(self, *args, **kwargs):
        pass


# Mock instances for attribute access
mock_order_side = MockOrderSide()
mock_time_in_force = MockTimeInForce()
mock_order_status = MockOrderStatus()
mock_query_order_status = MockQueryOrderStatus()