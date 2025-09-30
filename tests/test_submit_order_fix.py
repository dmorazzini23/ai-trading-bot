"""Test for the submit_order NameError fix."""
import os
from unittest.mock import Mock, patch

from ai_trading.core.enums import OrderSide

import pytest


@pytest.fixture(autouse=True)
def _ensure_env(monkeypatch):
    """Provide deterministic environment for submit_order tests."""

    monkeypatch.setenv("PYTEST_RUNNING", "1")
    overrides = {
        "ALPACA_API_KEY": "FAKE_TEST_API_KEY_NOT_REAL_123456789",
        "ALPACA_SECRET_KEY": "FAKE_TEST_SECRET_KEY_NOT_REAL_123456789",
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
        "WEBHOOK_SECRET": "fake-test-webhook-not-real",
        "FLASK_PORT": "9000",
        "TRADING_MODE": "balanced",
        "DOLLAR_RISK_LIMIT": "0.05",
        "TESTING": "1",
        "TRADE_LOG_FILE": "test_trades.csv",
        "SEED": "42",
        "RATE_LIMIT_BUDGET": "190",
        "DISABLE_DAILY_RETRAIN": "1",
        "DRY_RUN": "1",
        "SHADOW_MODE": "1",
    }
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)

    from ai_trading.config.management import reload_trading_config

    reload_trading_config()
    yield
    reload_trading_config()


def test_submit_order_with_uninitialized_exec_engine():
    """Test that submit_order raises proper error when _exec_engine is None."""
    # Import after setting environment
    from ai_trading.core import bot_engine
    from ai_trading.core.bot_engine import BotContext, submit_order

    # Ensure _exec_engine is None
    original_exec_engine = bot_engine._exec_engine
    bot_engine._exec_engine = None

    try:
        # Mock market_is_open to return True
        with patch('ai_trading.core.bot_engine.market_is_open', return_value=True):
            # Create a mock context
            mock_ctx = Mock(spec=BotContext)

            # Should raise RuntimeError for uninitialized engine
            with pytest.raises(RuntimeError, match="Execution engine not initialized"):
                submit_order(mock_ctx, "AAPL", 10, "buy")

    finally:
        # Restore original state
        bot_engine._exec_engine = original_exec_engine


def test_submit_order_with_market_closed():
    """Test that submit_order returns None when market is closed."""
    # Import after setting environment
    from unittest.mock import Mock

    from ai_trading.core.bot_engine import BotContext, submit_order

    # Mock market_is_open to return False
    with patch('ai_trading.core.bot_engine.market_is_open', return_value=False):
        mock_ctx = Mock(spec=BotContext)
        result = submit_order(mock_ctx, "AAPL", 10, "buy")
        assert result is None


def test_submit_order_successful_execution():
    """Test that submit_order works correctly when properly initialized."""
    # Import after setting environment
    from unittest.mock import Mock

    from ai_trading.core import bot_engine
    from ai_trading.core.bot_engine import BotContext, submit_order

    # Mock the execution engine
    mock_exec_engine = Mock()
    mock_order = Mock()
    mock_exec_engine.execute_order.return_value = mock_order

    # Set the global _exec_engine
    original_exec_engine = bot_engine._exec_engine
    bot_engine._exec_engine = mock_exec_engine

    try:
        # Mock market_is_open to return True and latest price lookup
        with patch('ai_trading.core.bot_engine.market_is_open', return_value=True), \
            patch('ai_trading.core.bot_engine.get_latest_price', return_value=101.0):
            mock_ctx = Mock(spec=BotContext)

            # Should successfully execute order
            result = submit_order(mock_ctx, "AAPL", 10, "buy")

            assert result == mock_order
            mock_exec_engine.execute_order.assert_called_once_with(
                "AAPL", OrderSide.BUY, 10, price=101.0
            )

    finally:
        # Restore original state
        bot_engine._exec_engine = original_exec_engine


def test_submit_order_execution_error_propagation():
    """Test that submit_order properly propagates execution errors."""
    # Import after setting environment
    from unittest.mock import Mock

    from ai_trading.core import bot_engine
    from ai_trading.core.bot_engine import BotContext, submit_order

    # Mock the execution engine to raise an exception
    mock_exec_engine = Mock()
    test_error = Exception("Test execution error")
    mock_exec_engine.execute_order.side_effect = test_error

    # Set the global _exec_engine
    original_exec_engine = bot_engine._exec_engine
    bot_engine._exec_engine = mock_exec_engine

    try:
        # Mock market_is_open to return True and latest price lookup
        with patch('ai_trading.core.bot_engine.market_is_open', return_value=True), \
            patch('ai_trading.core.bot_engine.get_latest_price', return_value=101.0):
            mock_ctx = Mock(spec=BotContext)

            # Should propagate the execution error
            with pytest.raises(Exception, match="Test execution error"):
                submit_order(mock_ctx, "AAPL", 10, "buy")

    finally:
        # Restore original state
        bot_engine._exec_engine = original_exec_engine


def test_submit_order_stub_engine_accepts_price(monkeypatch):
    """Submit order succeeds when stubbed execution engine receives a price kwarg."""

    import importlib
    import sys
    import types

    from unittest.mock import Mock

    # Keep references so we can restore the real modules after exercising the stub.
    real_execution_module = importlib.import_module("ai_trading.execution")

    # Replace the execution package with a minimal stub that lacks ExecutionEngine
    # to trigger the fallback branch inside bot_engine during import.
    sys.modules["ai_trading.execution"] = types.ModuleType("ai_trading.execution")
    for key in list(sys.modules):
        if key.startswith("ai_trading.execution."):
            sys.modules.pop(key)

    try:
        # Reload bot_engine so it picks up the stubbed execution module.
        sys.modules.pop("ai_trading.core.bot_engine", None)
        stub_bot_engine = importlib.import_module("ai_trading.core.bot_engine")
        stub_bot_engine._exec_engine = stub_bot_engine.ExecutionEngine()

        with patch("ai_trading.core.bot_engine.market_is_open", return_value=True):
            mock_ctx = Mock(spec=stub_bot_engine.BotContext)

            result = stub_bot_engine.submit_order(
                mock_ctx,
                "AAPL",
                5,
                "buy",
                price=123.45,
            )

        assert hasattr(result, "id")
        assert result.price == 123.45
    finally:
        # Restore the real execution module and reload bot_engine for subsequent tests.
        sys.modules["ai_trading.execution"] = real_execution_module
        importlib.reload(real_execution_module)
        importlib.reload(importlib.import_module("ai_trading.core.bot_engine"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
