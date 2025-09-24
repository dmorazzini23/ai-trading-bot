from __future__ import annotations

"""Helpers for working with HTTP fallback cache responses.

Some providers in :mod:`ai_trading.data.fetch` may return raw HTTP responses
that lack a ``json()`` helper (e.g. ``urllib3`` responses).  This module exposes
utility helpers that attempt to decode such responses consistently.
"""

from typing import Any
import json


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
        pass

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
        return {}

    raw = raw.strip()
    if not raw:
        return {}

    try:
        return json.loads(raw)
    except Exception:
        return {}


# Backwards compatible aliases for potential external consumers
parse_resp = resp_json
parse_json = resp_json

__all__ = ["resp_json", "parse_resp", "parse_json"]
