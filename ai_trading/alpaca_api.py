from __future__ import annotations

import os
import sys
import time
import types
import uuid

_ALPACA_MODULE_NAMES = (
    "alpaca_trade_api",
    "alpaca",
    "alpaca.trading",
    "alpaca.data",
)
ALPACA_AVAILABLE = all(
    name in sys.modules and sys.modules.get(name) is not None
    for name in _ALPACA_MODULE_NAMES
)
if os.environ.get("TESTING", "").lower() == "true" and any(
    sys.modules.get(n) is None for n in _ALPACA_MODULE_NAMES
):
    ALPACA_AVAILABLE = False

SHADOW_MODE = False

_RETRYABLE_CODES = {429, 500, 502, 503, 504}
RETRYABLE_HTTP_STATUSES = tuple(_RETRYABLE_CODES)


def _make_client_order_id(prefix: str = "ai") -> str:
    return f"{prefix}-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"


def submit_order(api, req, log: types.SimpleNamespace | None = None):
    """Submit an order or return a shadow response."""  # AI-AGENT-REF
    client_order_id = _make_client_order_id(
        os.getenv("ALPACA_CLIENT_ORDER_ID_PREFIX", "ai")
    )
    if SHADOW_MODE:
        if log:
            log.info(
                "submit_order shadow",
                symbol=getattr(req, "symbol", None),
                qty=getattr(req, "qty", None),
            )
        return {
            "status": "shadow",
            "symbol": getattr(req, "symbol", None),
            "qty": getattr(req, "qty", None),
            "side": getattr(req, "side", None),
            "time_in_force": getattr(req, "time_in_force", None),
            "client_order_id": client_order_id,
        }

    order_payload = {
        "symbol": getattr(req, "symbol", None),
        "qty": getattr(req, "qty", None),
        "side": getattr(req, "side", None),
        "time_in_force": getattr(req, "time_in_force", None),
        "client_order_id": client_order_id,
    }
    if log:
        log.info("submit_order live", payload=order_payload)
    resp = api.submit_order(**order_payload)
    if isinstance(resp, dict):
        return types.SimpleNamespace(**resp)
    return resp


# Backwards compatibility helper
generate_client_order_id = _make_client_order_id


def alpaca_get(*_args, **_kwargs):  # pragma: no cover - legacy stub
    return None


def start_trade_updates_stream(*_args, **_kwargs):  # pragma: no cover
    return None


__all__ = [
    "ALPACA_AVAILABLE",
    "SHADOW_MODE",
    "RETRYABLE_HTTP_STATUSES",
    "submit_order",
    "generate_client_order_id",
    "alpaca_get",
    "start_trade_updates_stream",
]

