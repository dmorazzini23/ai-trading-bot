"""Lightweight Alpaca API helpers with legacy compatibility."""

from __future__ import annotations

import os
import sys
import time
import types
import uuid
from importlib.util import find_spec
from typing import Any

import requests

from ai_trading.utils import HTTP_TIMEOUT_S, clamp_timeout

# Legacy compatibility for tests
SHADOW_MODE: bool = False

_ALPACA_MODULE_KEYS = ("alpaca_trade_api", "alpaca", "alpaca.trading", "alpaca.data")


def _detect_alpaca_available() -> bool:
    """Detect whether Alpaca SDK modules are importable."""

    try:
        testing = os.getenv("TESTING", "").lower() in {"1", "true", "yes"}
        if testing:
            if any(
                mod in sys.modules and sys.modules.get(mod) is None for mod in _ALPACA_MODULE_KEYS
            ):
                return False
        for mod in ("alpaca_trade_api", "alpaca.trading", "alpaca.data"):
            if find_spec(mod) is None:
                return False
        return True
    except Exception:  # pragma: no cover - best effort
        return False


ALPACA_AVAILABLE = _detect_alpaca_available()

RATE_LIMIT_HTTP_CODES = {429, 408, 409, 425, 503}
RETRYABLE_STATUS = RATE_LIMIT_HTTP_CODES | {500, 502, 504}
RETRYABLE_CODES = RETRYABLE_STATUS


def _coerce(req: Any, key: str, default: Any = None) -> Any:
    if isinstance(req, dict):
        return req.get(key, default)
    return getattr(req, key, default)


def generate_client_order_id(prefix: str = "bot") -> str:
    """Return a unique client order id."""

    return f"{prefix}-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"


def submit_order(
    client: Any,
    symbol_or_req: Any,
    qty: int | None = None,
    side: str | None = None,
    type: str = "market",
    time_in_force: str = "day",
    client_order_id: str | None = None,
    dry_run: bool = False,
    shadow: bool = False,
    **kwargs: Any,
):
    """Submit an order and return a normalized response."""

    if not isinstance(symbol_or_req, str):
        req = symbol_or_req
        symbol = _coerce(req, "symbol") or ""
        qty = _coerce(req, "qty", _coerce(req, "quantity")) or 0
        side = _coerce(req, "side") or ""
        time_in_force = _coerce(req, "time_in_force", time_in_force)
        client_order_id = client_order_id or _coerce(req, "client_order_id")
    else:
        symbol = symbol_or_req
        if qty is None or side is None:
            raise TypeError("qty and side required")

    order_payload = {
        "symbol": symbol,
        "qty": int(qty),
        "side": side,
        "type": type,
        "time_in_force": time_in_force,
        **kwargs,
    }
    if client_order_id is None:
        client_order_id = generate_client_order_id()
    order_payload["client_order_id"] = client_order_id

    if SHADOW_MODE or dry_run or shadow or not getattr(client, "submit_order", None):
        return types.SimpleNamespace(
            id="dry-run", status="accepted", shadow=bool(shadow or SHADOW_MODE)
        )

    backoff = 0.2
    retries = 3
    timeout_v = clamp_timeout(None, default=HTTP_TIMEOUT_S)
    for attempt in range(retries):
        try:
            resp = client.submit_order(**order_payload, timeout=timeout_v)
            status = getattr(resp, "status_code", None)
            if (
                isinstance(status, int)
                and status in RATE_LIMIT_HTTP_CODES
                and attempt < retries - 1
            ):
                time.sleep(backoff)
                backoff *= 2
                continue
            oid = (
                getattr(resp, "id", None)
                or getattr(resp, "order_id", None)
                or (resp.get("id") if isinstance(resp, dict) else None)
                or (resp.get("order_id") if isinstance(resp, dict) else None)
            )
            oid_val = oid if oid is not None else client_order_id
            return types.SimpleNamespace(order_id=str(oid_val), id=oid_val, raw=resp)
        except requests.exceptions.HTTPError as exc:  # pragma: no cover - network mocked
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if (status in RATE_LIMIT_HTTP_CODES or status is None) and attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise


def alpaca_get(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - legacy stub
    return None


def start_trade_updates_stream(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
    return None


__all__ = [
    "SHADOW_MODE",
    "ALPACA_AVAILABLE",
    "RATE_LIMIT_HTTP_CODES",
    "RETRYABLE_STATUS",
    "RETRYABLE_CODES",
    "generate_client_order_id",
    "submit_order",
    "alpaca_get",
    "start_trade_updates_stream",
]
