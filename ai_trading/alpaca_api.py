"""Lightweight Alpaca API helpers with legacy compatibility."""

from __future__ import annotations

import os
import sys
import time
import uuid
from importlib.util import find_spec
from typing import Any

import requests

from ai_trading.utils import clamp_timeout

SHADOW_MODE = False
RETRY_HTTP_CODES = {408, 429, 500, 502, 503, 504}

_ALPACA_MODULE_KEYS = ("alpaca_trade_api", "alpaca", "alpaca.trading", "alpaca.data")


def _detect_alpaca_available() -> bool:
    """Detect whether Alpaca SDK modules are importable."""

    try:
        testing = os.getenv("TESTING", "").lower() in {"1", "true", "yes"}
        if testing and any(
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


def is_retryable_status(code: int) -> bool:
    return int(code) in RETRY_HTTP_CODES


def _coerce(req: Any, key: str, default: Any = None) -> Any:
    if isinstance(req, dict):
        return req.get(key, default)
    return getattr(req, key, default)


def generate_client_order_id(prefix: str = "bot") -> str:
    """Return a unique client order id."""

    return f"{prefix}-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"


def submit_order(
    api: Any,
    req: Any | None = None,
    *,
    symbol: str | None = None,
    qty: int | None = None,
    side: str | None = None,
    type: str = "market",
    time_in_force: str = "day",
    client_order_id: str | None = None,
    dry_run: bool = False,
    shadow: bool = False,
    **kwargs: Any,
) -> dict:
    """Submit an order through ``api`` with retry and shadow support."""

    if req is not None:
        if isinstance(req, str) and symbol is None:
            symbol = req
        elif not isinstance(req, str):
            symbol = symbol or _coerce(req, "symbol")
            qty = qty if qty is not None else _coerce(req, "qty", _coerce(req, "quantity"))
            side = side or _coerce(req, "side")
            time_in_force = _coerce(req, "time_in_force", time_in_force)
            client_order_id = client_order_id or _coerce(req, "client_order_id")
    if symbol is None or qty is None or side is None:
        raise TypeError("symbol, qty and side required")

    order_payload = {
        "symbol": symbol,
        "qty": int(qty),
        "side": side,
        "type": type,
        "time_in_force": time_in_force,
        **kwargs,
    }
    if client_order_id is None:
        client_order_id = f"{uuid.uuid4().hex}-{int(time.time())}"[:20]
    order_payload["client_order_id"] = client_order_id

    if SHADOW_MODE or dry_run or shadow or not getattr(api, "submit_order", None):
        return {
            "provider_order_id": "dry-run",
            "client_order_id": client_order_id,
            "status": "accepted",
        }

    backoff = 0.2
    retries = 3
    timeout_v = clamp_timeout(None)
    for attempt in range(retries):
        try:
            resp = api.submit_order(**order_payload, timeout=timeout_v)
            status = getattr(resp, "status_code", None)
            if isinstance(status, int) and is_retryable_status(status) and attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            oid = (
                getattr(resp, "id", None)
                or getattr(resp, "order_id", None)
                or (resp.get("id") if isinstance(resp, dict) else None)
                or (resp.get("order_id") if isinstance(resp, dict) else None)
            )
            provider_id = str(oid) if oid is not None else ""
            return {
                "provider_order_id": provider_id,
                "client_order_id": client_order_id,
                "status": getattr(resp, "status", "accepted"),
            }
        except requests.exceptions.HTTPError as exc:  # pragma: no cover
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and is_retryable_status(status) and attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
        except requests.exceptions.RequestException:
            if attempt < retries - 1:
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
    "RETRY_HTTP_CODES",
    "generate_client_order_id",
    "submit_order",
    "is_retryable_status",
    "alpaca_get",
    "start_trade_updates_stream",
]
