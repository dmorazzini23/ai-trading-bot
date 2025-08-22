import types

import pytest


@pytest.mark.unit
def test_pre_trade_health_resolves_min_rows_without_ctx_attr(dummy_data_fetcher):
    """Test that pre_trade_health_check handles missing ctx.min_rows gracefully."""
    # Try to import the function, but skip if not available
    from ai_trading.core.bot_engine import pre_trade_health_check

    ctx = types.SimpleNamespace()
    ctx.data_fetcher = dummy_data_fetcher

    # Test that it works when min_rows is explicitly passed
    result = pre_trade_health_check(ctx, ["TEST"], min_rows=10)
    assert isinstance(result, dict), "Should return a dict result"
    assert "checked" in result, "Should have checked key"

    # Test that it works when ctx doesn't have min_rows attribute
    # This should use the default value (120) instead of raising AttributeError
    result2 = pre_trade_health_check(ctx, ["TEST"], min_rows=None)
    assert isinstance(result2, dict), "Should return a dict result even without ctx.min_rows"

@pytest.mark.unit
def test_min_rows_precedence_logic():
    """Test the min_rows precedence logic in isolation."""
    # Simulate the logic from the updated function
    def resolve_min_rows(min_rows, ctx):
        if min_rows is None:
            min_rows = getattr(ctx, "min_rows", 120)
        return int(min_rows)

    # Test explicit parameter takes precedence
    ctx = types.SimpleNamespace()
    ctx.min_rows = 50
    assert resolve_min_rows(30, ctx) == 30, "Explicit parameter should take precedence"

    # Test ctx.min_rows is used when parameter is None
    assert resolve_min_rows(None, ctx) == 50, "Should use ctx.min_rows when parameter is None"

    # Test default is used when both are missing
    ctx_no_attr = types.SimpleNamespace()
    assert resolve_min_rows(None, ctx_no_attr) == 120, "Should use default when ctx.min_rows missing"
