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


def build_limit(
    *,
    symbol: str,
    side: str,
    qty: float,
    limit_price: float,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    order = _base(symbol=symbol, side=side, qty=qty, client_order_id=client_order_id)
    order.update({"type": "limit", "limit_price": float(limit_price)})
    return order


def build_market(
    *,
    symbol: str,
    side: str,
    qty: float,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    order = _base(symbol=symbol, side=side, qty=qty, client_order_id=client_order_id)
    order.update({"type": "market"})
    return order


def build_stop_limit(
    *,
    symbol: str,
    side: str,
    qty: float,
    stop_price: float,
    limit_price: float,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    order = _base(symbol=symbol, side=side, qty=qty, client_order_id=client_order_id)
    order.update(
        {
            "type": "stop_limit",
            "stop_price": float(stop_price),
            "limit_price": float(limit_price),
        }
    )
    return order


def build_stop(
    *,
    symbol: str,
    side: str,
    qty: float,
    stop_price: float,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    order = _base(symbol=symbol, side=side, qty=qty, client_order_id=client_order_id)
    order.update({"type": "stop", "stop_price": float(stop_price)})
    return order


def build_trailing_stop(
    *,
    symbol: str,
    side: str,
    qty: float,
    trail_percent: float | None = None,
    trail_price: float | None = None,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    if trail_percent is None and trail_price is None:
        raise ValueError("trail_percent or trail_price is required")
    order = _base(symbol=symbol, side=side, qty=qty, client_order_id=client_order_id)
    order.update({"type": "trailing_stop", "trail_percent": trail_percent, "trail_price": trail_price})
    return order


def build_bracket(
    *,
    symbol: str,
    side: str,
    qty: float,
    entry_limit: float,
    take_profit: float,
    stop_loss: float,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    order = build_limit(
        symbol=symbol,
        side=side,
        qty=qty,
        limit_price=entry_limit,
        client_order_id=client_order_id,
    )
    order["type"] = "bracket"
    order["legs"] = {
        "take_profit": {"limit_price": float(take_profit)},
        "stop_loss": {"stop_price": float(stop_loss)},
    }
    return order


def build_oco(
    *,
    symbol: str,
    side: str,
    qty: float,
    take_profit: float,
    stop_loss: float,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    order = _base(symbol=symbol, side=side, qty=qty, client_order_id=client_order_id)
    order["type"] = "oco"
    order["legs"] = {
        "take_profit": {"limit_price": float(take_profit)},
        "stop_loss": {"stop_price": float(stop_loss)},
    }
    return order


def build_oto(
    *,
    symbol: str,
    side: str,
    qty: float,
    entry_limit: float,
    stop_loss: float,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    order = build_limit(
        symbol=symbol,
        side=side,
        qty=qty,
        limit_price=entry_limit,
        client_order_id=client_order_id,
    )
    order["type"] = "oto"
    order["legs"] = {"stop_loss": {"stop_price": float(stop_loss)}}
    return order


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
