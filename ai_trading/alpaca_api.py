from __future__ import annotations

import importlib.util as _ils
import os
import time
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
    """
    Submit an order to Alpaca or simulate when shadow-mode is on or
    the provided api object lacks 'submit_order'.
    Always return a types.SimpleNamespace with a boolean .success attribute.
    Accepts dict-like payloads, attribute objects, or plain kwargs.
    """
    import types, time, uuid

    # normalize input -> data dict
    data = {}
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

    # Fallback/shadow path OR api missing submit_order -> simulate but return object
    if SHADOW_MODE or not hasattr(api, "submit_order"):
        if log:
            try:
                log.info("submit_order shadow", symbol=symbol, qty=qty)
            except Exception:
                pass
        return types.SimpleNamespace(
            success=True,
            status="shadow",
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=tif,
            client_order_id=client_order_id,
        )

    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "time_in_force": tif,
        "client_order_id": client_order_id,
    }
    if log:
        try:
            log.info("submit_order live", payload=payload)
        except Exception:
            pass

    try:
        resp = api.submit_order(**payload)
    except Exception as e:
        status = getattr(e, "status", None)
        return types.SimpleNamespace(
            success=False,
            retryable=(status in RETRY_HTTP_CODES),
            status=status,
            error=str(e),
            client_order_id=client_order_id,
        )

    order_id = getattr(resp, "id", None)
    if isinstance(resp, dict):
        order_id = resp.get("id")
    return types.SimpleNamespace(
        success=True,
        order_id=order_id,
        client_order_id=client_order_id,
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
