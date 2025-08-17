from __future__ import annotations

import importlib.util as _ils
import os
import time
import types
import uuid
from contextlib import suppress

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
    """Submit an order to Alpaca or simulate in shadow/fallback mode."""  # AI-AGENT-REF
    data: dict[str, object] = {}
    if order_data is not None:
        if isinstance(order_data, dict):
            data.update(order_data)
        else:
            for key in ("symbol", "qty", "side", "time_in_force"):
                if hasattr(order_data, key):
                    data[key] = getattr(order_data, key)
    if kwargs:
        data.update(kwargs)

    symbol = _get(data, "symbol")
    qty = _get(data, "qty")
    side = _get(data, "side")
    tif = _get(data, "time_in_force", "day")
    client_order_id = _make_client_order_id("shadow" if SHADOW_MODE else "ai")
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "time_in_force": tif,
        "client_order_id": client_order_id,
    }
    submit = getattr(api, "submit_order", None)

    if SHADOW_MODE or not callable(submit):
        if log:
            with suppress(Exception):
                log.info("submit_order shadow", symbol=symbol, qty=qty)
        return types.SimpleNamespace(
            success=True,
            status="shadow",
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=tif,
            client_order_id=client_order_id,
            broker_order_id=f"shadow-{client_order_id}",
        )

    if log:
        with suppress(Exception):
            log.info("submit_order live", payload=payload)

    try:
        resp = submit(**payload)
        if isinstance(resp, dict):
            resp.setdefault("client_order_id", client_order_id)
            broker_id = resp.get("broker_order_id") or resp.get("id")
            resp.setdefault("broker_order_id", broker_id)
            resp.setdefault("id", broker_id)
            return types.SimpleNamespace(success=True, **resp)
        broker_id = getattr(resp, "broker_order_id", None) or getattr(resp, "id", None)
        resp.client_order_id = getattr(resp, "client_order_id", client_order_id)
        resp.broker_order_id = broker_id or client_order_id
        resp.success = True
        return resp
    except Exception as exc:  # noqa: BLE001
        status = getattr(exc, "status", 0)
        retryable = status in RETRY_HTTP_CODES
        if log:
            with suppress(Exception):
                log.warning("submit_order fallback: %s", exc)
        return types.SimpleNamespace(
            success=False,
            status=status,
            retryable=retryable,
            error=str(exc),
            client_order_id=client_order_id,
            broker_order_id=client_order_id,
            id=client_order_id,
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=tif,
        )


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
