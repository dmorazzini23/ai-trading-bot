from __future__ import annotations

import pytest

from ai_trading.oms.orders import validate_order_type_support


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


