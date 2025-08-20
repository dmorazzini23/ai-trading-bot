from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


def get_json_with_retries(
    fetch: Callable[[], Any], retries: int = 3, backoff: float = 0.05
) -> Any:
    """Return JSON-decoded payload with basic retry logic."""  # AI-AGENT-REF: add HTTP retry wrapper
    attempts = max(1, int(retries))
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = fetch()
            if hasattr(resp, "json"):
                return resp.json()
            if isinstance(resp, (bytes, bytearray)):
                return json.loads(resp.decode("utf-8"))
            if isinstance(resp, str):
                return json.loads(resp)
            return resp
        except Exception as exc:  # intentionally broad at boundary
            last_exc = exc
            logger.debug("http retry %s/%s after %s", attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(backoff * attempt)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable")


__all__ = ["get_json_with_retries"]

