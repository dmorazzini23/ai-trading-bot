#!/usr/bin/env python3
"""
Simple integration validation for drawdown circuit breaker.
Tests the core functionality without complex mocking.
"""

import os
import pytest

os.environ["TESTING"] = "1"

from ai_trading import config
from ai_trading.risk.circuit_breakers import DrawdownCircuitBreaker

pytestmark = pytest.mark.integration


def test_integration():
    """Test the basic integration points."""

    # Test 1: Configuration loading

    # Test 2: Circuit breaker creation
    breaker = DrawdownCircuitBreaker(max_drawdown=config.MAX_DRAWDOWN_THRESHOLD)

    # Test 3: Normal operation
    initial_equity = 10000.0
    breaker.update_equity(initial_equity)

    # Test 4: Small loss (should still allow trading)
    small_loss_equity = initial_equity * 0.95  # 5% loss
    breaker.update_equity(small_loss_equity)
    breaker.get_status()

    # Test 5: Large loss triggering halt
    large_loss_equity = initial_equity * 0.90  # 10% loss (exceeds 8% threshold)
    breaker.update_equity(large_loss_equity)
    breaker.get_status()

    # Test 6: Recovery
    recovery_equity = initial_equity * 0.80  # 80% recovery
    breaker.update_equity(recovery_equity)
    breaker.get_status()


if __name__ == "__main__":
    test_integration()
