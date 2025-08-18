from __future__ import annotations

import os
import time
import types
import uuid
from contextlib import suppress

from ai_trading.utils.optdeps import module_ok  # AI-AGENT-REF: optional import helper

# AI-AGENT-REF: robust optional Alpaca dependency handling
SHADOW_MODE = os.getenv("SHADOW_MODE", "").lower() in {"1", "true", "yes"}
RETRY_HTTP_CODES = {429, 500, 502, 503, 504}
RETRYABLE_HTTP_STATUSES = tuple(RETRY_HTTP_CODES)


ALPACA_AVAILABLE = any(
    [
        module_ok("alpaca"),
        module_ok("alpaca_trade_api"),
        module_ok("alpaca.trading"),
        module_ok("alpaca.data"),
    ]
) and os.getenv("TESTING", "").lower() not in {"1", "true:force_unavailable"}


def _make_client_order_id(prefix: str = "ai") -> str:
    return f"{prefix}-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"


generate_client_order_id = _make_client_order_id


def _get(obj, key, default=None):
    """Fetch ``key`` from ``obj`` by attribute or mapping lookup."""  # AI-AGENT-REF
    if obj is None:
        return default
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def submit_order(api, order_data=None, log=None, **kwargs):
    """Submit an order and return a canonical ``SimpleNamespace``."""  # AI-AGENT-REF
    data: dict[str, object] = {}
    if order_data is not None:
        if isinstance(order_data, dict):
            data.update(order_data)
        else:
            for k in ("symbol", "qty", "side", "time_in_force", "client_order_id"):
                v = _get(order_data, k)
                if v is not None:
                    data[k] = v
    if kwargs:
        data.update(kwargs)

    symbol = _get(data, "symbol")
    qty = _get(data, "qty")
    side = _get(data, "side")
    tif = _get(data, "time_in_force", "day")
    client_order_id = _get(data, "client_order_id") or _make_client_order_id()
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "time_in_force": tif,
        "client_order_id": client_order_id,
    }

    def _shadow() -> types.SimpleNamespace:
        broker_id = f"shadow-{client_order_id}"
        return types.SimpleNamespace(
            status="shadow",
            success=True,
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=tif,
            client_order_id=client_order_id,
            broker_order_id=broker_id,
            order_id=broker_id,  # back-compat alias
            message="shadow-mode OK",
        )

    if SHADOW_MODE or not callable(getattr(api, "submit_order", None)):
        if log:
            with suppress(Exception):
                log.info("submit_order shadow", payload=payload)
        return _shadow()

    if log:
        with suppress(Exception):
            log.info("submit_order live", payload=payload)

    try:
        resp = api.submit_order(**payload)
    except Exception as e:  # noqa: BLE001  # broker/client can raise many types
        status = getattr(e, "status", "error")
        return types.SimpleNamespace(
            success=False,
            status=status,
            retryable=status in RETRY_HTTP_CODES,
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=tif,
            client_order_id=client_order_id,
            broker_order_id=None,
            order_id=None,
            message=str(e),
        )

    if isinstance(resp, dict):
        broker_id = (
            resp.get("id") or resp.get("order_id") or resp.get("broker_order_id")
        )
        client_id = resp.get("client_order_id", client_order_id)
    else:
        broker_id = (
            getattr(resp, "id", None)
            or getattr(resp, "order_id", None)
            or getattr(resp, "broker_order_id", None)
        )
        client_id = getattr(resp, "client_order_id", client_order_id)
    return types.SimpleNamespace(
        status="submitted",
        success=True,
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=tif,
        client_order_id=client_id,
        broker_order_id=broker_id,
        order_id=broker_id,
        message="ok",
    )


def alpaca_get(*_a, **_k):  # legacy stub
    return None


def start_trade_updates_stream(*_a, **_k):  # legacy stub
    return None


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
