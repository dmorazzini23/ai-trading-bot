"""Tests for the lightweight context singleton."""

from ai_trading.core.context import get_context


def test_get_context_is_singleton():
    ctx1 = get_context()
    ctx2 = get_context()
    assert ctx1 is ctx2


def test_context_has_default_attributes():
    ctx = get_context()
    assert hasattr(ctx, "alpaca_data_feed")
    assert hasattr(ctx, "log_market_fetch")
    assert hasattr(ctx, "testing")
