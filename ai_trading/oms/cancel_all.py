"""Utilities for cancel-all order hygiene operations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_trading.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class CancelAllResult:
    total_open: int
    cancelled: int
    failed: int
    reason_code: str
    errors: list[dict[str, Any]] = field(default_factory=list)


def _list_open_orders(api: Any) -> list[Any]:
    list_orders = getattr(api, "list_orders", None)
    if callable(list_orders):
        try:
            return list(list_orders(status="open") or [])
        except TypeError:
            return list(list_orders() or [])
    get_orders = getattr(api, "get_orders", None)
    if callable(get_orders):
        try:
            return list(get_orders(status="open") or [])
        except TypeError:
            return list(get_orders() or [])
    return []


def _cancel_order(api: Any, order: Any) -> None:
    order_id = getattr(order, "id", None) or getattr(order, "client_order_id", None)
    if not order_id:
        raise RuntimeError("Order missing id/client_order_id")
    cancel_order = getattr(api, "cancel_order", None)
    if callable(cancel_order):
        cancel_order(order_id)
        return
    cancel_by_id = getattr(api, "cancel_order_by_id", None)
    if callable(cancel_by_id):
        cancel_by_id(order_id)
        return
    cancel_orders = getattr(api, "cancel_orders", None)
    if callable(cancel_orders):
        cancel_orders()
        return
    raise RuntimeError("Broker API missing cancel capability")


def cancel_all_open_orders(ctx: Any) -> CancelAllResult:
    """Cancel all broker-open orders and return a structured summary."""

    api = getattr(ctx, "api", None)
    if api is None:
        return CancelAllResult(
            total_open=0,
            cancelled=0,
            failed=0,
            reason_code="CANCEL_ALL_TRIGGERED",
            errors=[{"error": "missing_api"}],
        )

    try:
        open_orders = _list_open_orders(api)
    except Exception as exc:
        logger.warning("CANCEL_ALL_LIST_FAILED", extra={"error": str(exc)})
        return CancelAllResult(
            total_open=0,
            cancelled=0,
            failed=1,
            reason_code="CANCEL_ALL_TRIGGERED",
            errors=[{"error": str(exc)}],
        )

    cancelled = 0
    failed = 0
    errors: list[dict[str, Any]] = []
    for order in open_orders:
        try:
            _cancel_order(api, order)
            cancelled += 1
        except Exception as exc:
            failed += 1
            errors.append(
                {
                    "order_id": getattr(order, "id", None),
                    "client_order_id": getattr(order, "client_order_id", None),
                    "error": str(exc),
                }
            )

    result = CancelAllResult(
        total_open=len(open_orders),
        cancelled=cancelled,
        failed=failed,
        reason_code="CANCEL_ALL_TRIGGERED",
        errors=errors,
    )
    logger.warning(
        "CANCEL_ALL_TRIGGERED",
        extra={
            "total_open": result.total_open,
            "cancelled": result.cancelled,
            "failed": result.failed,
        },
    )
    return result
