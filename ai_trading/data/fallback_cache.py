from __future__ import annotations

"""Helpers for working with HTTP fallback cache responses.

Some providers in :mod:`ai_trading.data.fetch` may return raw HTTP responses
that lack a ``json()`` helper (e.g. ``urllib3`` responses).  This module exposes
utility helpers that attempt to decode such responses consistently.
"""

from typing import Any
import json


def resp_json(resp: Any) -> Any:
    """Return the JSON payload from a response-like object.

    The function first tries ``resp.json()``.  If the method is missing or
    raises an exception, it falls back to parsing ``resp.data`` or ``resp.text``
    as JSON.  If no JSON payload can be decoded an empty ``dict`` is returned.
    """
    try:
        return resp.json()  # type: ignore[attr-defined]
    except Exception:
        pass

    raw = getattr(resp, "data", None)
    if raw is None:
        raw = getattr(resp, "text", None)
    if raw is None:
        return {}
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            return {}

    try:
        return json.loads(raw)
    except Exception:
        return {}


# Backwards compatible aliases for potential external consumers
parse_resp = resp_json
parse_json = resp_json

__all__ = ["resp_json", "parse_resp", "parse_json"]
