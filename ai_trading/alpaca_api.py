"""Alpaca helper utilities used by tests and trading code."""

from __future__ import annotations

import datetime as _dt
import time
import types
import uuid
from typing import Any

import requests

from ai_trading.utils import DEFAULT_HTTP_TIMEOUT

RETRIABLE_STATUS = (429, 500, 502, 503, 504)


def unique_client_order_id(prefix: str = "bot") -> str:
    """Return a provider-safe unique client order id."""
    day = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d")
    return f"{prefix}-{day}-{uuid.uuid4().hex[:8]}"


def _coerce(v: Any, key: str, default: Any = None) -> Any:
    if isinstance(v, dict):
        return v.get(key, default)
    return getattr(v, key, default)


def _payload_from_req(req: Any, *, include_id: bool = True) -> dict[str, Any]:
    payload = {
        "symbol": _coerce(req, "symbol"),
        "qty": _coerce(req, "qty", _coerce(req, "quantity")),
        "side": _coerce(req, "side"),
        "time_in_force": _coerce(req, "time_in_force", "day"),
        "type": _coerce(req, "type"),
        "limit_price": _coerce(req, "limit_price"),
        "stop_price": _coerce(req, "stop_price"),
    }
    coid = _coerce(req, "client_order_id")
    if include_id:
        payload["client_order_id"] = coid or unique_client_order_id()
    elif coid:
        payload["client_order_id"] = coid
    return {k: v for k, v in payload.items() if v is not None}


def _wrap(resp: Any) -> Any:
    return types.SimpleNamespace(**resp) if isinstance(resp, dict) else resp


def submit_order(
    api: Any,
    req: Any,
    *,
    shadow: bool = False,
    dry_run: bool = False,
    max_retries: int = 3,
) -> Any:
    """Submit an order to ``api`` with simple retry handling."""
    payload = _payload_from_req(req, include_id=not (shadow or dry_run))
    if shadow:
        return types.SimpleNamespace(id="SHADOW", **payload)
    if dry_run:
        return types.SimpleNamespace(id="DRY_RUN", **payload)

    submit = getattr(api, "submit_order", None)
    if submit is None:  # pragma: no cover - misconfigured API
        raise RuntimeError("api missing submit_order")

    for attempt in range(max_retries + 1):
        try:
            resp = submit(**payload, timeout=DEFAULT_HTTP_TIMEOUT)
            status = getattr(resp, "status_code", None)
            if status in RETRIABLE_STATUS and attempt < max_retries:
                time.sleep(0.2)
                continue
            return _wrap(resp)
        except requests.exceptions.HTTPError as exc:  # pragma: no cover - network mocked
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in RETRIABLE_STATUS and attempt < max_retries:
                time.sleep(0.2)
                continue
            raise
        except Exception:  # noqa: BLE001
            if attempt < max_retries:
                time.sleep(0.2)
                continue
            raise


__all__ = [
    "RETRIABLE_STATUS",
    "unique_client_order_id",
    "submit_order",
    "alpaca_get",
    "start_trade_updates_stream",
]


def alpaca_get(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - legacy stub
    return None


def start_trade_updates_stream(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
    return None
