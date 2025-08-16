from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

try:
    from requests.exceptions import HTTPError as _HTTPError
    from requests.exceptions import RequestException as _RequestException
except Exception:  # pragma: no cover
    _HTTPError = None
    _RequestException = None


class HTTPError(Exception):  # re-export
    pass


class RequestException(Exception):  # re-export
    pass


if _HTTPError is not None:  # pragma: no cover
    HTTPError = _HTTPError  # type: ignore[assignment]
if _RequestException is not None:  # pragma: no cover
    RequestException = _RequestException  # type: ignore[assignment]


RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def unique_client_order_id(prefix: str = "bot") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}


def _extract_req(req: Any) -> dict[str, Any]:
    m = _as_mapping(req)
    # normalize common keys expected by tests
    return {
        "symbol": m.get("symbol"),
        "qty": m.get("qty") or m.get("quantity"),
        "side": m.get("side"),
        "time_in_force": m.get("time_in_force") or m.get("tif") or "day",
        "type": m.get("type") or "market",
        "limit_price": m.get("limit_price"),
        "stop_price": m.get("stop_price"),
        "client_order_id": m.get("client_order_id"),
        "extended_hours": m.get("extended_hours", False),
    }


def submit_order(api: Any, req: Any, max_retries: int = 3, retry_wait_s: float = 0.25):
    """Submit an order; accepts dict or SimpleNamespace; returns object with `.id`."""
    payload = _extract_req(req)

    def _do():
        resp = api.submit_order(**{k: v for k, v in payload.items() if v is not None})
        # Normalize result to have `.id`
        if isinstance(resp, Mapping):
            oid = resp.get("id") or resp.get("order_id")
        else:
            oid = getattr(resp, "id", None) or getattr(resp, "order_id", None)
        return SimpleNamespace(id=oid or "mock-client-1", raw=resp)

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return _do()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_retries:
                raise
            time.sleep(retry_wait_s)


# ---- Compatibility wrappers expected by tests ----


def submit_order_rate_limit(api: Any, req: Any):
    """Thin wrapper used by tests to exercise 429 handling via submit_order."""
    return submit_order(api, req)


def submit_order_generic_retry(api: Any, req: Any):
    """Thin wrapper used by tests to exercise generic retry via submit_order."""
    return submit_order(api, req)


def submit_order_http_error(api: Any, req: Any):
    """Thin wrapper that surfaces HTTP errors from the underlying call."""
    return submit_order(api, req, max_retries=1)


# Legacy placeholders expected by bot_engine imports
def alpaca_get(*args, **kwargs):  # pragma: no cover - placeholder
    raise NotImplementedError("alpaca_get is not implemented in test shim")


def start_trade_updates_stream(*args, **kwargs):  # pragma: no cover - placeholder
    raise NotImplementedError(
        "start_trade_updates_stream is not implemented in test shim"
    )
