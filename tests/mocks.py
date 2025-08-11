"""
Test mocks and utilities for trading bot testing.

This module contains all mock classes previously embedded in production code,
now properly isolated for testing purposes only.
"""


# Mock instances for attribute access
mock_order_side = MockOrderSide()
mock_time_in_force = MockTimeInForce()
mock_order_status = MockOrderStatus()
mock_query_order_status = MockQueryOrderStatus()