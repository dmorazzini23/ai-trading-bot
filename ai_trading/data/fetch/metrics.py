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

from ai_trading.data.metrics import (
    backup_provider_used as _backup_provider_used,
    metrics,
    provider_disable_total as _provider_disable_total,
    provider_disabled as _provider_disabled,
    provider_fallback as _provider_fallback,
)


def snapshot(resp: Any | None = None) -> dict[str, int]:
    """Return metrics extracted from ``resp`` or current counters.

    Parameters
    ----------
    resp:
        Optional HTTP-like response object supplying a ``json()`` method. If
        the response is missing or malformed the function will gracefully fall
        back to returning the current in-memory counters instead of raising
        ``ValueError('invalid_response')``.
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


def backup_provider_used(provider: str, symbol: str) -> int:
    """Return times ``provider`` served data for ``symbol``."""

    return int(
        _backup_provider_used.labels(provider=provider, symbol=symbol)._value.get()
    )


def provider_fallback(from_provider: str, to_provider: str) -> int:
    """Return fallback count from ``from_provider`` to ``to_provider``."""

    return int(
        _provider_fallback.labels(
            from_provider=from_provider, to_provider=to_provider
        )._value.get()
    )


def provider_disabled(provider: str) -> int:
    """Return 1 if ``provider`` is currently disabled, else 0."""

    return int(_provider_disabled.labels(provider=provider)._value.get())


def provider_disable_total(provider: str) -> int:
    """Return total times ``provider`` was disabled."""

    return int(_provider_disable_total.labels(provider=provider)._value.get())


__all__ = [
    "snapshot",
    "backup_provider_used",
    "provider_fallback",
    "provider_disabled",
    "provider_disable_total",
]
