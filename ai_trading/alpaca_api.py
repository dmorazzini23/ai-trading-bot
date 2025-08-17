"""Thin Alpaca helpers with retry logic."""

from __future__ import annotations

import time
import types
import uuid
from typing import Any

import requests

HTTP_429_TOO_MANY_REQUESTS = 429
SUBMIT_RETRY_HTTP_CODES: set[int] = {408, 409, 429, 500, 502, 503, 504}

DRY_RUN = False
SHADOW_MODE = False


def build_client_order_id(prefix: str = "bot") -> str:
    """Generate a client order id."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _extract_id(resp: Any) -> str | None:
    if resp is None:
        return None
    if isinstance(resp, dict):
        return resp.get("id") or resp.get("order_id") or resp.get("client_order_id")
    return (
        getattr(resp, "id", None)
        or getattr(resp, "order_id", None)
        or getattr(resp, "client_order_id", None)
    )


def submit_order(
    api: Any,
    req: Any,
    *,
    client_order_id: str | None = None,
    max_retries: int = 3,
    backoff_s: float = 0.2,
) -> Any:
    """Submit an order via ``api`` with optional retries."""  # AI-AGENT-REF: retry+id normalizer
    if SHADOW_MODE:
        return {"status": "shadow"}
    if DRY_RUN:
        return {"status": "dry_run"}

    coid = (
        client_order_id
        or getattr(req, "client_order_id", None)
        or (req.get("client_order_id") if isinstance(req, dict) else None)
    )
    if coid is None:
        coid = build_client_order_id()
    payload = {
        "symbol": getattr(req, "symbol", None) or req["symbol"],
        "qty": getattr(req, "qty", None) or req.get("qty") or getattr(req, "quantity", None),
        "side": getattr(req, "side", None) or req["side"],
        "time_in_force": getattr(req, "time_in_force", None) or req.get("time_in_force") or "day",
    }
    if coid:
        payload["client_order_id"] = coid

    submit = getattr(api, "submit_order", None) or getattr(api, "place_order", None)
    if submit is None:
        raise RuntimeError("API missing submit_order/place_order")

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            try:
                resp = submit(**payload)
            except TypeError as e:
                if client_order_id and "client_order_id" in str(e):
                    payload.pop("client_order_id", None)
                    resp = submit(**payload)
                else:
                    raise
            oid = _extract_id(resp)
            if isinstance(resp, dict):
                if oid is not None and "id" not in resp:
                    resp["id"] = oid
                return types.SimpleNamespace(**resp)
            if oid is not None and not hasattr(resp, "id"):
                return types.SimpleNamespace(id=oid)
            return resp
        except requests.exceptions.HTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in SUBMIT_RETRY_HTTP_CODES and attempt < max_retries:
                time.sleep(backoff_s)
                last_exc = e
                continue
            raise
        except Exception as e:  # noqa: BLE001
            if attempt < max_retries:
                time.sleep(backoff_s)
                last_exc = e
                continue
            raise
    raise last_exc or RuntimeError("submit_order failed unexpectedly")


__all__ = [
    "HTTP_429_TOO_MANY_REQUESTS",
    "SUBMIT_RETRY_HTTP_CODES",
    "build_client_order_id",
    "submit_order",
    "DRY_RUN",
    "SHADOW_MODE",
    "requests",
]
