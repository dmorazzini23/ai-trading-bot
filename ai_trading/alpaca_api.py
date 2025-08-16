from __future__ import annotations

import time
import types
import uuid
from collections.abc import Mapping
from typing import Any

# Export exceptions so tests can import them without pulling in 'requests'
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


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def unique_client_order_id(prefix: str = "bot") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _to_order_ns(req: Any, client_order_id: str | None = None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        symbol=_get(req, "symbol"),
        qty=_get(req, "qty"),
        side=_get(req, "side"),
        time_in_force=_get(req, "time_in_force", "day"),
        client_order_id=client_order_id
        or _get(req, "client_order_id")
        or unique_client_order_id(),
    )


def _call_submit(api: Any, order_ns: types.SimpleNamespace) -> Any:
    """
    Call style compatibility:
      1) submit_order(order_data=ns)
      2) submit_order(**ns.__dict__)
      3) submit_order(ns)
      4) submit_order(ns.__dict__)
    Return the provider's raw response unmodified.
    """
    for call in (
        lambda: api.submit_order(order_data=order_ns),
        lambda: api.submit_order(**order_ns.__dict__),
        lambda: api.submit_order(order_ns),
        lambda: api.submit_order(order_ns.__dict__),
    ):
        try:
            return call()
        except TypeError:
            continue
    # If nothing worked, raise a clean error for the tests
    raise TypeError("submit_order call patterns not supported by provided API")


def submit_order(
    api: Any,
    req: Any,
    *,
    client_order_id: str | None = None,
    dry_run: bool = False,
    shadow: bool = False,
) -> Any:
    """
    Core order helper used by tests:
    - Accepts req as Mapping or object with attributes.
    - In dry_run/shadow modes returns a simple object with an 'id' attribute.
    - Otherwise returns whatever the provider returns (unmodified).
    """
    if shadow or dry_run:
        return types.SimpleNamespace(id="shadow" if shadow else "dry-run")
    order_ns = _to_order_ns(req, client_order_id)
    return _call_submit(api, order_ns)


def submit_order_retryable(
    api: Any,
    req: Any,
    *,
    retries: int = 3,
    backoff_s: float = 0.05,
) -> Any:
    """
    Retry on HTTPError/RequestException with retryable status codes.
    Always returns the provider's raw response if successful.
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return submit_order(api, req)
        except (HTTPError, RequestException) as e:
            last = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status and status not in RETRYABLE_STATUS:
                raise
            if attempt == retries:
                raise
            time.sleep(backoff_s * (2**attempt))
    if last:
        raise last
