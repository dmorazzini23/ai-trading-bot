from __future__ import annotations

"""Helpers for working with HTTP fallback cache responses.

Some providers in :mod:`ai_trading.data.fetch` may return raw HTTP responses
that lack a ``json()`` helper (e.g. ``urllib3`` responses).  This module exposes
utility helpers that attempt to decode such responses consistently.
"""

from typing import Any
import json

from ai_trading.logging import get_logger

logger = get_logger(__name__)


def resp_json(resp: Any) -> Any:
    """Return JSON from a response-like object.

    Supports native ``.json()`` helpers, file-like ``.read()`` objects, and
    common attributes such as ``.text`` or ``.data``.  Falls back to decoding the
    string representation when it resembles JSON.  Returns ``{}`` on failure to
    keep fallback callers defensive.
    """

    # Native JSON helper
    try:
        if hasattr(resp, "json"):
            return resp.json()
    except Exception:
        logger.debug("RESPONSE_JSON_METHOD_FAILED", exc_info=True)

    raw: Any = None

    for attr in ("data", "text", "content"):
        try:
            value = getattr(resp, attr)
        except Exception:
            continue
        else:
            if value is not None:
                raw = value
                break

    if raw is None and hasattr(resp, "read"):
        try:
            raw = resp.read()
        except Exception:
            raw = None

    if raw is None and hasattr(resp, "body"):
        try:
            raw = resp.body
        except Exception:
            raw = None

    if raw is None:
        raw = resp

    try:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="ignore")
        elif not isinstance(raw, str):
            raw = str(raw)
    except Exception:
        logger.debug("RESPONSE_JSON_COERCE_TO_STRING_FAILED", exc_info=True)
        return {}

    raw = raw.strip()
    if not raw:
        return {}

    try:
        return json.loads(raw)
    except Exception:
        logger.debug("RESPONSE_JSON_PARSE_FAILED", exc_info=True)
        return {}


# Backwards compatible aliases for potential external consumers
parse_resp = resp_json
parse_json = resp_json

__all__ = ["resp_json", "parse_resp", "parse_json"]
