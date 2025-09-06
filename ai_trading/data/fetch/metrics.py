from __future__ import annotations

"""Helpers for exposing data fetch metrics.

This module provides a lightweight interface to snapshot the in-memory
metrics counters used during data fetching operations.  Historically the
caller would raise ``ValueError('invalid_response')`` when a response object
could not be parsed.  For robustness we now return a best-effort metric
snapshot instead of propagating the exception.
"""

from dataclasses import asdict
from typing import Any, Mapping

from ai_trading.data.metrics import metrics


def snapshot(resp: Any | None = None) -> dict[str, int]:
    """Return metrics extracted from ``resp`` or current counters.

    Parameters
    ----------
    resp:
        Optional HTTP-like response object supplying a ``json()`` method.  If
        the response is missing or malformed the function will gracefully
        fall back to returning the current in-memory counters instead of
        raising ``ValueError('invalid_response')``.
    """
    try:
        if resp is None or not hasattr(resp, "json"):
            raise ValueError("invalid_response")
        payload = resp.json()
        if not isinstance(payload, Mapping):
            raise ValueError("invalid_response")
        return dict(payload)
    except Exception:
        # Return the current counters instead of propagating the error.
        return asdict(metrics)


__all__ = ["snapshot"]
