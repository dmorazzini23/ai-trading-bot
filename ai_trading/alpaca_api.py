"""Thin Alpaca helpers with retry and DRY_RUN/SHADOW compatibility."""

from __future__ import annotations

import time
from typing import Any

from requests.exceptions import HTTPError as HTTPError  # re-export

DRY_RUN = False
SHADOW_MODE = False

RETRY_STATUS = {429, 500, 502, 503, 504}


def extract_order_id(resp: Any) -> str | None:
    """Best-effort extraction of an order identifier."""
    if resp is None:
        return None
    if isinstance(resp, dict):
        return resp.get("id") or resp.get("order_id") or resp.get("client_order_id")
    return (
        getattr(resp, "id", None)
        or getattr(resp, "order_id", None)
        or getattr(resp, "client_order_id", None)
    )


def submit_order(api: Any, req: Any, retries: int = 2, sleep_s: float = 0.2):
    """Submit an order via ``api`` and return the raw response."""
    if DRY_RUN:
        return {"id": "dry-run"}
    if SHADOW_MODE:
        return {"id": "shadow"}

    payload = {
        "symbol": getattr(req, "symbol", None) or req["symbol"],
        "qty": getattr(req, "qty", None)
        or getattr(req, "quantity", None)
        or req.get("qty"),
        "side": getattr(req, "side", None) or req["side"],
        "time_in_force": getattr(req, "time_in_force", None)
        or req.get("time_in_force")
        or "day",
    }
    coid = getattr(req, "client_order_id", None) or req.get("client_order_id")
    if coid:
        payload["client_order_id"] = coid

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            submit = getattr(api, "submit_order", None) or getattr(
                api, "place_order", None
            )
            if submit is None:
                raise RuntimeError("API has no submit_order/place_order")
            resp = submit(**payload)
            oid = extract_order_id(resp)
            if isinstance(resp, dict):
                resp.setdefault("id", oid)
            elif oid is not None:
                return resp
            return resp
        except HTTPError as e:  # pragma: no cover - network mocked
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in RETRY_STATUS and attempt < retries:
                time.sleep(sleep_s)
                last_exc = e
                continue
            raise
        except Exception as e:  # noqa: BLE001
            if attempt < retries:
                time.sleep(sleep_s)
                last_exc = e
                continue
            raise
    raise last_exc or RuntimeError("submit_order failed unexpectedly")


__all__ = [
    "HTTPError",
    "DRY_RUN",
    "SHADOW_MODE",
    "RETRY_STATUS",
    "extract_order_id",
    "submit_order",
    "alpaca_get",
    "start_trade_updates_stream",
]


def alpaca_get(*args, **kwargs):  # pragma: no cover - legacy stub
    return None


def start_trade_updates_stream(*args, **kwargs):  # pragma: no cover - legacy stub
    return None
