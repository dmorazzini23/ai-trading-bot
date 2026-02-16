from __future__ import annotations

import pytest

from ai_trading.oms.orders import build_bracket, validate_order_type_support


def test_order_types_supported_and_failfast_passes() -> None:
    capabilities = {
        "limit": True,
        "market": True,
        "stop": True,
        "stop_limit": True,
        "trailing_stop": True,
        "bracket": True,
        "oco": True,
        "oto": True,
    }
    validate_order_type_support(
        configured_entry_type="limit",
        configured_exit_type="stop_limit",
        allow_bracket=True,
        allow_oco_oto=True,
        capabilities=capabilities,
    )


def test_order_types_failfast_when_unsupported() -> None:
    capabilities = {"limit": True, "market": True, "stop_limit": True, "stop": True, "trailing_stop": True}
    with pytest.raises(RuntimeError, match="capability missing"):
        validate_order_type_support(
            configured_entry_type="limit",
            configured_exit_type="stop_limit",
            allow_bracket=True,
            allow_oco_oto=False,
            capabilities=capabilities,
        )


def test_build_bracket_contains_exit_legs() -> None:
    order = build_bracket(
        symbol="AAPL",
        side="buy",
        qty=10,
        entry_limit=100.0,
        take_profit=110.0,
        stop_loss=95.0,
        client_order_id="cid-1",
    )
    assert order["type"] == "bracket"
    assert "legs" in order
