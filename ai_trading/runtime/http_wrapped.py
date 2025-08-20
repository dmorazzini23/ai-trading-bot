
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


def get_json_with_retries(
    fetch: Callable[[], Any], retries: int = 3, backoff: float = 0.05
) -> Any:
    """Return JSON-decoded payload with retry logic.

    * Attempts = retries + 1 (first try + retries)
    * Retries on network errors and JSON parse errors from .json()
    """  # AI-AGENT-REF: add robust HTTP retry wrapper
    last_exc: Exception | None = None
    attempts = int(max(0, retries)) + 1
    for attempt in range(1, attempts + 1):
        try:
            resp = fetch()
            if hasattr(resp, "json") and callable(getattr(resp, "json")):
                try:
                    return resp.json()
                except Exception as exc:  # parse error -> retry
                    last_exc = exc
                    raise
            if isinstance(resp, (bytes, bytearray)):
                return json.loads(resp.decode("utf-8"))
            if isinstance(resp, str):
                return json.loads(resp)
            return resp
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.debug("http retry %s/%s after %s", attempt, attempts - 1, exc)
            if attempt < attempts:
                time.sleep(backoff * attempt)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable")


__all__ = ["get_json_with_retries"]
