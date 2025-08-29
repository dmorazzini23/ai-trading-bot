"""Utilities for redacting sensitive information from log payloads."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
import re
from typing import Any

_RE_KEYS = re.compile("(key|secret|token|password)", re.IGNORECASE)
_MASK = "***REDACTED***"

_SENSITIVE_ENV = {"ALPACA_API_KEY", "ALPACA_SECRET_KEY", "WEBHOOK_SECRET"}


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


def redact_env(env: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return copy of *env* with known sensitive keys masked."""

    dup: MutableMapping[str, Any] = dict(env)
    for key in _SENSITIVE_ENV:
        if key in dup and dup[key]:
            dup[key] = _MASK
    return dup


__all__ = ["redact", "redact_env"]

