from __future__ import annotations

import os
import sys
import uuid
from typing import Any

# --- availability detection (respects tests that stub sys.modules) ---------
_ALPACA_MODULE_NAMES = (
    "alpaca_trade_api",
    "alpaca",
    "alpaca.trading",
    "alpaca.data",
)
ALPACA_AVAILABLE: bool = all(
    name in sys.modules and sys.modules.get(name) is not None
    for name in _ALPACA_MODULE_NAMES
)
if os.environ.get("TESTING", "").lower() == "true" and any(
    sys.modules.get(n) is None for n in _ALPACA_MODULE_NAMES
):
    ALPACA_AVAILABLE = False

# Legacy constant kept for tests
SHADOW_MODE: bool = os.environ.get("AI_TRADING_SHADOW_MODE", "0") in (
    "1",
    "true",
    "True",
)

# Retryable statuses expected by tests (include rate-limit)
_RETRYABLE_CODES = {429, 500, 502, 503, 504}
RETRYABLE_HTTP_STATUSES = tuple(_RETRYABLE_CODES)


def _coerce_req_to_payload(req: Any) -> dict:
    d = {
        "symbol": getattr(req, "symbol", None),
        "qty": getattr(req, "qty", None),
        "side": getattr(req, "side", None),
        "type": getattr(req, "type", "market"),
        "time_in_force": getattr(req, "time_in_force", "day"),
        "client_order_id": getattr(req, "client_order_id", None),
    }
    if not d["client_order_id"]:
        d["client_order_id"] = uuid.uuid4().hex[:20]
    return d


def submit_order(
    api: Any,
    req: Any,
    logger: Any | None = None,
    *,
    shadow_mode: bool | None = None,
    dry_run: bool | None = None,
) -> dict:
    """Submit an order to an Alpaca-like API."""
    mode_shadow = SHADOW_MODE if shadow_mode is None else bool(shadow_mode)
    payload = _coerce_req_to_payload(req)

    if mode_shadow or dry_run:
        if logger:
            logger.info(
                "SHADOW_SUBMIT",
                extra={
                    "symbol": payload["symbol"],
                    "qty": payload["qty"],
                    "side": payload["side"],
                },
            )
        return {"status": "shadow", **payload}

    try:
        resp = api.submit_order(order_data=payload)
    except TypeError:
        resp = api.submit_order(payload)

    if isinstance(resp, dict):
        return resp
    oid = getattr(resp, "id", None) or getattr(resp, "order_id", None)
    return {"id": oid, "status": "submitted", **payload}


__all__ = [
    "ALPACA_AVAILABLE",
    "SHADOW_MODE",
    "RETRYABLE_HTTP_STATUSES",
    "submit_order",
    "alpaca_get",
    "start_trade_updates_stream",
]


def alpaca_get(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - legacy stub
    return None


def start_trade_updates_stream(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
    return None
