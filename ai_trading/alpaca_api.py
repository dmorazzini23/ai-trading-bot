"""Minimal Alpaca helpers used in tests."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

try:  # pragma: no cover - requests optional
    from requests.exceptions import HTTPError
except Exception:  # pragma: no cover

    class HTTPError(Exception):
        pass


from ai_trading.utils import clamp_timeout

DRY_RUN = False
SHADOW_MODE = False
_pending_orders_lock = threading.Lock()
_pending_orders: dict[str, Any] = {}
partial_fill_tracker: dict[str, int] = {}
partial_fills: set[str] = set()


def is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429 or "429" in str(exc)


def is_retryable(status_code: int, exc: Exception | None = None) -> bool:
    return status_code in {429, 500, 502, 503, 504} or bool(exc and is_rate_limit(exc))


def generate_client_order_id(symbol: str) -> str:
    return f"{symbol}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}


def _build_order(req: Any) -> SimpleNamespace:
    m = _as_mapping(req)
    coid = m.get("client_order_id") or generate_client_order_id(m.get("symbol", "ord"))
    return SimpleNamespace(
        symbol=m.get("symbol"),
        qty=m.get("qty") or m.get("quantity"),
        side=m.get("side"),
        time_in_force=m.get("time_in_force") or m.get("tif") or "day",
        type=m.get("type") or "market",
        limit_price=m.get("limit_price"),
        stop_price=m.get("stop_price"),
        client_order_id=coid,
        extended_hours=m.get("extended_hours", False),
    )


def submit_order(
    api: Any,
    req: Any,
    *,
    dry_run: bool = False,
    shadow: bool = False,
    timeout: float | None = None,
    max_retries: int = 3,
) -> Any:
    """Submit an order and return an object with ``.id``."""
    if dry_run or DRY_RUN:
        return {"status": "dry_run", "id": "dry-run"}
    if shadow or SHADOW_MODE:
        return {"status": "shadow", "id": "shadow"}

    order = _build_order(req)
    timeout_v = clamp_timeout(timeout)
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            try:
                resp = api.submit_order(order, timeout=timeout_v)
            except TypeError:
                resp = api.submit_order(order)
            oid = getattr(resp, "id", None) or getattr(resp, "order_id", None)
            return SimpleNamespace(id=oid or order.client_order_id, raw=resp)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            status = getattr(exc, "status_code", 0)
            if not is_retryable(status, exc) or attempt == max_retries:
                raise
            time.sleep(0.1)
    if last_exc:
        raise last_exc
    raise RuntimeError("submit_order failed")


def handle_trade_update(event: Any) -> None:
    oid = getattr(getattr(event, "order", None), "id", None)
    if not oid:
        return
    qty = int(getattr(getattr(event, "order", None), "filled_qty", 0))
    if event.event == "partial_fill":
        prev = partial_fill_tracker.get(oid)
        if prev != qty:
            partial_fill_tracker[oid] = qty
            partial_fills.add(oid)
            import logging

            logging.getLogger(__name__).debug(
                "ORDER_PARTIAL_FILL", extra={"order_id": oid}
            )
    elif event.event == "fill":
        if oid in partial_fills:
            import logging

            logging.getLogger(__name__).debug("ORDER_FILLED", extra={"order_id": oid})
        partial_fill_tracker.pop(oid, None)
        partial_fills.discard(oid)


# Compatibility exports expected by tests
HTTPError = HTTPError  # re-export for import contracts
