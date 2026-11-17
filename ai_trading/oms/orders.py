"""Explicit order-family builders and capability validation."""
from __future__ import annotations

from typing import Any, Mapping


def _base(
    *,
    symbol: str,
    side: str,
    qty: float,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "client_order_id": client_order_id,
    }


def validate_order_type_support(
    *,
    configured_entry_type: str,
    configured_exit_type: str,
    allow_bracket: bool,
    allow_oco_oto: bool,
    capabilities: Mapping[str, bool],
) -> None:
    supported_entry = {"limit", "market"}
    supported_exit = {"stop", "stop_limit", "trailing_stop"}
    if configured_entry_type not in supported_entry:
        raise RuntimeError(f"Unsupported configured entry order type: {configured_entry_type}")
    if configured_exit_type not in supported_exit:
        raise RuntimeError(f"Unsupported configured exit order type: {configured_exit_type}")

    required = {configured_entry_type, configured_exit_type}
    if allow_bracket:
        required.add("bracket")
    if allow_oco_oto:
        required.update({"oco", "oto"})

    for order_type in required:
        if not bool(capabilities.get(order_type, False)):
            raise RuntimeError(f"Broker capability missing for configured order type: {order_type}")
