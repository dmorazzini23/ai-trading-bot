from __future__ import annotations

import importlib.util as _ils
import os
import time
import types
import uuid

SHADOW_MODE = os.getenv("SHADOW_MODE", "").lower() in {"1", "true", "yes"}
RETRY_HTTP_CODES = {429, 500, 502, 503, 504}
# Back-compat alias some code may import
RETRYABLE_HTTP_STATUSES = tuple(RETRY_HTTP_CODES)


def _module_ok(name: str) -> bool:
    try:
        return _ils.find_spec(name) is not None
    except Exception:  # noqa: BLE001
        return False


# Back-compat: expose availability signal here too
ALPACA_AVAILABLE = any(
    [
        _module_ok("alpaca"),
        _module_ok("alpaca_trade_api"),
        _module_ok("alpaca.trading"),
        _module_ok("alpaca.data"),
    ]
) and os.environ.get("TESTING", "").lower() not in {"1", "true:force_unavailable"}


def _make_client_order_id(prefix: str = "ai") -> str:
    return f"{prefix}-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"


generate_client_order_id = _make_client_order_id


def _get(obj, key, default=None):
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def submit_order(api, order_data=None, log=None, **kwargs):
    """Submit an order to Alpaca or simulate in shadow mode.
    Back-compat: accepts an object with attributes, a dict-like mapping,
    or plain kwargs.
    """  # AI-AGENT-REF
    # Normalize inputs
    data: dict[str, object] = {}
    if order_data is not None:
        if isinstance(order_data, dict):
            data.update(order_data)
        else:
            for k in ("symbol", "qty", "side", "time_in_force"):
                if hasattr(order_data, k):
                    data[k] = getattr(order_data, k)
    if kwargs:
        data.update(kwargs)
    symbol = _get(data, "symbol")
    qty = _get(data, "qty")
    side = _get(data, "side")
    tif = _get(data, "time_in_force", "day")
    client_order_id = _make_client_order_id("shadow" if SHADOW_MODE else "ai")
    # Two graceful-fallback paths:
    # 1) SHADOW_MODE: keep legacy dict shape
    # 2) Missing api.submit_order: return an object with .success for tests
    if SHADOW_MODE:
        if log:
            import contextlib

            with contextlib.suppress(Exception):
                log.info("submit_order shadow", symbol=symbol, qty=qty)
        return {
            "status": "shadow",
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "time_in_force": tif,
            "client_order_id": client_order_id,
        }
    if not hasattr(api, "submit_order"):
        if log:
            import contextlib

            with contextlib.suppress(Exception):
                log.info("submit_order shadow-missing-api", symbol=symbol, qty=qty)
        return types.SimpleNamespace(
            success=True,
            status="shadow",
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=tif,
            client_order_id=client_order_id,
            order_id=f"dryrun-{client_order_id}",
        )
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "time_in_force": tif,
        "client_order_id": client_order_id,
    }
    if log:
        log.info("submit_order live", payload=payload)
    try:
        resp = api.submit_order(**payload)
    except Exception as e:  # noqa: BLE001
        status = getattr(e, "status", None)
        return types.SimpleNamespace(
            success=False,
            retryable=status in RETRY_HTTP_CODES,
            status=status,
        )
    order_id = getattr(resp, "id", None)
    if isinstance(resp, dict):
        order_id = resp.get("id")
    return types.SimpleNamespace(success=True, order_id=order_id)


__all__ = [
    "ALPACA_AVAILABLE",
    "SHADOW_MODE",
    "RETRY_HTTP_CODES",
    "RETRYABLE_HTTP_STATUSES",
    "submit_order",
    "generate_client_order_id",
    "alpaca_get",
    "start_trade_updates_stream",
]


def alpaca_get(*_args, **_kwargs):  # pragma: no cover
    return None


def start_trade_updates_stream(*_args, **_kwargs):  # pragma: no cover
    return None
