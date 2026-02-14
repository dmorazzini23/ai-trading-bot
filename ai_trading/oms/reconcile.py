"""Broker reconciliation helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_trading.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ReconcileResult:
    ok: bool
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def fetch_broker_positions(api: Any) -> dict[str, float]:
    positions: dict[str, float] = {}
    if api is None:
        return positions
    try:
        if hasattr(api, "list_positions") and callable(api.list_positions):
            for pos in api.list_positions():
                symbol = getattr(pos, "symbol", None)
                qty = getattr(pos, "qty", 0)
                if symbol:
                    positions[str(symbol)] = float(qty)
    except Exception as exc:
        logger.warning("BROKER_POSITIONS_FETCH_FAILED", extra={"error": str(exc)})
    return positions


def fetch_open_orders(api: Any) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    if api is None:
        return orders
    try:
        if hasattr(api, "list_orders") and callable(api.list_orders):
            for order in api.list_orders():
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
    except Exception as exc:
        logger.warning("BROKER_OPEN_ORDERS_FETCH_FAILED", extra={"error": str(exc)})
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

