"""Broker reconciliation helpers."""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from dataclasses import dataclass, field
from typing import Any

from ai_trading.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReconcileResult:
    ok: bool
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def fetch_broker_positions(api: Any) -> dict[str, float]:
    positions: dict[str, float] = {}
    if api is None:
        return positions
    try:
        get_all_positions = getattr(api, "get_all_positions", None)
        list_positions = getattr(api, "list_positions", None)
        if callable(get_all_positions):
            raw_positions = get_all_positions()
        elif callable(list_positions):
            raw_positions = list_positions()
        else:
            return positions
        for pos in raw_positions or []:
            symbol = pos.get("symbol") if isinstance(pos, dict) else getattr(pos, "symbol", None)
            qty = pos.get("qty", pos.get("quantity", 0)) if isinstance(pos, dict) else getattr(pos, "qty", getattr(pos, "quantity", 0))
            side = pos.get("side") if isinstance(pos, dict) else getattr(pos, "side", None)
            side_token = str(getattr(side, "value", side) or "").strip().lower()
            if symbol:
                qty_float = float(qty)
                if qty_float > 0 and side_token in {"short", "sell_short", "sell-short", "sell short"}:
                    qty_float = -qty_float
                positions[str(symbol)] = qty_float
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        logger.warning("BROKER_POSITIONS_FETCH_FAILED", extra={"error": str(exc)})
        raise RuntimeError(f"broker_positions_fetch_failed: {exc}") from exc
    return positions


def fetch_open_orders(api: Any) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    if api is None:
        return orders
    try:
        get_orders = getattr(api, "get_orders", None)
        list_orders = getattr(api, "list_orders", None)
        if callable(get_orders):
            raw_orders = get_orders()
        elif callable(list_orders):
            raw_orders = list_orders()
        else:
            return orders
        for order in raw_orders or []:
            if isinstance(order, dict):
                orders.append(
                    {
                        "id": order.get("id"),
                        "client_order_id": order.get("client_order_id"),
                        "symbol": order.get("symbol"),
                        "qty": order.get("qty", order.get("quantity")),
                        "side": order.get("side"),
                        "status": order.get("status"),
                    }
                )
            else:
                orders.append(
                    {
                        "id": getattr(order, "id", None),
                        "client_order_id": getattr(order, "client_order_id", None),
                        "symbol": getattr(order, "symbol", None),
                        "qty": getattr(order, "qty", None),
                        "side": getattr(order, "side", None),
                        "status": getattr(order, "status", None),
                    }
                )
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        logger.warning("BROKER_OPEN_ORDERS_FETCH_FAILED", extra={"error": str(exc)})
        raise RuntimeError(f"broker_open_orders_fetch_failed: {exc}") from exc
    return orders


def reconcile(
    internal_positions: dict[str, float],
    broker_positions: dict[str, float],
    *,
    tolerance_shares: float = 0.0,
) -> ReconcileResult:
    mismatches: list[dict[str, Any]] = []
    all_symbols = set(internal_positions) | set(broker_positions)
    for symbol in sorted(all_symbols):
        internal_qty = float(internal_positions.get(symbol, 0.0))
        broker_qty = float(broker_positions.get(symbol, 0.0))
        delta = broker_qty - internal_qty
        if abs(delta) > tolerance_shares:
            mismatches.append(
                {
                    "symbol": symbol,
                    "internal_qty": internal_qty,
                    "broker_qty": broker_qty,
                    "delta": delta,
                }
            )
    return ReconcileResult(
        ok=not mismatches,
        mismatches=mismatches,
        summary={
            "symbols_checked": len(all_symbols),
            "mismatch_count": len(mismatches),
        },
    )
