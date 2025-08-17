"""Alpaca helper utilities with minimal pass-through semantics."""

from __future__ import annotations

import os
import secrets
import time
import types
from typing import Any

import requests

from ai_trading.utils import HTTP_TIMEOUT, clamp_timeout

SHADOW_MODE: bool = os.getenv("SHADOW_MODE", "").lower() in {"1", "true", "yes"}
RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}

__all__ = [
    "SHADOW_MODE",
    "RETRYABLE_HTTP_STATUSES",
    "submit_order",
    "client_order_id",
]


def _coerce(req: Any, key: str, default: Any = None) -> Any:
    if isinstance(req, dict):
        return req.get(key, default)
    return getattr(req, key, default)


def client_order_id(symbol: str, ts: float | None = None) -> str:
    """Generate a unique client order id for ``symbol``."""
    ts_ms = int((ts if ts is not None else time.time()) * 1000)
    return f"{symbol}-{ts_ms}-{secrets.token_hex(4)}"


def submit_order(api: Any, req: Any, *, timeout: float | None = None) -> Any:
    """Submit an order via ``api`` and return the provider response."""
    if SHADOW_MODE or not hasattr(api, "submit_order"):
        return types.SimpleNamespace(id="dry-run", status="accepted")
    payload = {
        "symbol": _coerce(req, "symbol"),
        "qty": _coerce(req, "qty", _coerce(req, "quantity")),
        "side": _coerce(req, "side"),
        "time_in_force": _coerce(req, "time_in_force", "day"),
    }
    coid = _coerce(req, "client_order_id")
    if coid is None:
        coid = client_order_id(payload["symbol"])
    payload["client_order_id"] = coid
    timeout_v = clamp_timeout(timeout, default=HTTP_TIMEOUT)
    backoff = 0.2
    max_tries = 3
    for attempt in range(max_tries):
        try:
            resp = api.submit_order(**payload, timeout=timeout_v)
            if isinstance(resp, dict):
                return types.SimpleNamespace(**resp)
            return resp
        except requests.exceptions.HTTPError as exc:  # pragma: no cover - network mocked
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if (status in RETRYABLE_HTTP_STATUSES or status is None) and attempt < max_tries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
        except Exception:
            if attempt < max_tries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise


def alpaca_get(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - legacy stub
    return None


def start_trade_updates_stream(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
    return None
