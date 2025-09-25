"""Utilities for redacting sensitive information from log payloads."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
import re
from typing import Any

_RE_KEYS = re.compile("(key|secret|token|password)", re.IGNORECASE)
_MASK = "***REDACTED***"
_ENV_MASK = "***"

_SENSITIVE_ENV = {
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_OAUTH",
    "WEBHOOK_SECRET",
    "NEWS_API_KEY",
    "FINNHUB_API_KEY",
    "SENTIMENT_API_KEY",
    "ALTERNATIVE_SENTIMENT_API_KEY",
    "FUNDAMENTAL_API_KEY",
    "IEX_API_TOKEN",
    "DATABASE_URL",
    "REDIS_URL",
    "MASTER_ENCRYPTION_KEY",
}


def _redact_inplace(obj: Any) -> Any:
    """Recursively redact matching keys."""

    if isinstance(obj, Mapping):
        for k, v in list(obj.items()):
            if isinstance(k, str) and _RE_KEYS.search(k):
                obj[k] = _MASK
            else:
                obj[k] = _redact_inplace(v)
        return obj
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = _redact_inplace(v)
        return obj
    return obj


def redact(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a redacted copy of *payload*."""

    dup: MutableMapping[str, Any] = deepcopy(payload)
    return _redact_inplace(dup)


def redact_env(env: Mapping[str, Any], *, drop: bool = False) -> Mapping[str, Any]:
    """Return copy of *env* with known sensitive keys masked or dropped.

    Parameters
    ----------
    env:
        Mapping of environment variables to sanitize.
    drop:
        When ``True`` remove sensitive keys entirely instead of masking their
        values.  Defaults to ``False`` which keeps the keys but masks the
        values with :data:`_ENV_MASK`.
    """

    dup: MutableMapping[str, Any] = dict(env)
    for key in list(dup):
        if key in _SENSITIVE_ENV and dup[key]:
            if drop:
                dup.pop(key)
            else:
                dup[key] = _ENV_MASK
    return dup


__all__ = ["redact", "redact_env"]

