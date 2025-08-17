from __future__ import annotations

import contextlib
import os
import sys
import time
import types
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

SHADOW_MODE: bool = False  # AI-AGENT-REF: tests monkeypatch this

_RETRYABLE_CODES = {429, 500, 502, 503, 504}
RETRYABLE_HTTP_STATUSES = tuple(_RETRYABLE_CODES)


def generate_client_order_id(prefix: str = "cid") -> str:
    """Return unique client order id."""  # AI-AGENT-REF
    return f"{prefix}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"


def _coerce_req(req: Any) -> dict[str, Any]:
    if isinstance(req, dict):
        return req
    if isinstance(req, types.SimpleNamespace):
        return vars(req)
    return {
        k: getattr(req, k)
        for k in ("symbol", "qty", "side", "time_in_force")
        if hasattr(req, k)
    }


def submit_order(
    api: Any,
    req: Any,
    log: Any | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Normalize order submission with shadow mode support."""  # AI-AGENT-REF
    data = _coerce_req(req)
    coid = data.get("client_order_id") or generate_client_order_id()
    data["client_order_id"] = coid

    if SHADOW_MODE:
        if log:
            with contextlib.suppress(Exception):
                log.info("SHADOW_SUBMIT", data=data)
        return {"id": f"shadow-{coid}", "client_order_id": coid, "status": "shadow"}

    try:
        if hasattr(api, "submit_order"):
            resp = api.submit_order(order_data=data)
        else:
            resp = {"id": f"dry-{coid}", "client_order_id": coid, "status": "accepted"}
    except Exception as e:  # noqa: BLE001
        if log:
            with contextlib.suppress(Exception):
                log.warning("ORDER_SUBMIT_ERROR", err=str(e))
        return {"id": None, "client_order_id": coid, "status": "error"}

    if isinstance(resp, dict):
        rid = resp.get("id") or resp.get("order_id")
        return {
            "id": rid,
            "client_order_id": coid,
            "status": resp.get("status", "accepted"),
        }
    rid = getattr(resp, "id", None) or getattr(resp, "order_id", None)
    status = getattr(resp, "status", "accepted")
    return {"id": rid, "client_order_id": coid, "status": status}


__all__ = [
    "ALPACA_AVAILABLE",
    "SHADOW_MODE",
    "RETRYABLE_HTTP_STATUSES",
    "submit_order",
    "generate_client_order_id",
    "alpaca_get",
    "start_trade_updates_stream",
]


def alpaca_get(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - legacy stub
    return None


def start_trade_updates_stream(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
    return None
