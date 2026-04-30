"""Utilities for redacting sensitive information from log payloads."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
import re
from typing import Any, cast

_RE_KEYS = re.compile(
    (
        r"(api[_-]?key|secret|token|password|pwd|passwd|connection[_-]?string|"
        r"database[_-]?url|dsn|authorization|bearer|credential|private[_-]?key|"
        r"access[_-]?key|session[_-]?id|oauth|webhook[_-]?url|(^|[_-])key($|[_-]))"
    ),
    re.IGNORECASE,
)
_MASK = "***REDACTED***"
_ENV_MASK = "***"
_LEGACY_ALPACA_PREFIX = "AP" "CA_"
_URL_ENV = {"ALPACA_TRADING_BASE_URL"}

_SENSITIVE_ENV = {
    "ALPACA_API_KEY",
    "ALPACA_API_KEY_ID",
    "ALPACA_API_SECRET_KEY",
    "ALPACA_KEY_ID",
    "ALPACA_SECRET_KEY",
    "ALPACA_OAUTH",
    "ALPACA_OAUTH_TOKEN",
    f"{_LEGACY_ALPACA_PREFIX}API_KEY_ID",
    f"{_LEGACY_ALPACA_PREFIX}API_SECRET_KEY",
    "WEBHOOK_SECRET",
    "AI_TRADING_SLACK_WEBHOOK_URL",
    "SLACK_WEBHOOK_URL",
    "AI_TRADING_OPENCLAW_RUNTIME_WEBHOOK_URL",
    "OPENCLAW_RUNTIME_WEBHOOK_URL",
    "AI_TRADING_OPENCLAW_HOOK_TOKEN",
    "OPENCLAW_HOOK_TOKEN",
    "AI_TRADING_JSM_OPS_API_KEY",
    "AI_TRADING_JSM_OPS_API_TOKEN",
    "AI_TRADING_JSM_OPS_BEARER_TOKEN",
    "AI_TRADING_GRAFANA_API_TOKEN",
    "AI_TRADING_PROM_REMOTE_WRITE_PASSWORD",
    "TRADIER_ACCESS_TOKEN",
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

_ALIAS_MAP = {
    "ALPACA_API_URL": "ALPACA_TRADING_BASE_URL",
    "ALPACA_BASE_URL": "ALPACA_TRADING_BASE_URL",
}


def normalize_aliases(env: Mapping[str, Any]) -> dict[str, Any]:
    """Return a mapping with environment aliases normalized to canonical keys."""

    resolved: dict[str, Any] = {}
    for key, value in env.items():
        canonical = _ALIAS_MAP.get(key, key)
        if canonical in resolved:
            existing = resolved[canonical]
            if key == canonical:
                resolved[canonical] = value
            elif (existing in (None, "")) and value not in (None, ""):
                resolved[canonical] = value
            continue
        resolved[canonical] = value
    return resolved


def _redact_inplace(obj: Any) -> Any:
    """Recursively redact matching keys."""

    if isinstance(obj, MutableMapping):
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

    dup: MutableMapping[str, Any] = deepcopy(dict(payload))
    return cast(Mapping[str, Any], _redact_inplace(dup))


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

    normalized = normalize_aliases(env)
    dup: MutableMapping[str, Any] = dict(normalized)
    for key in list(dup):
        if key in _URL_ENV and dup[key]:
            if not drop:
                dup[key] = _ENV_MASK
            continue
        if (key in _SENSITIVE_ENV or _RE_KEYS.search(key)) and dup[key]:
            if drop:
                dup.pop(key)
            else:
                dup[key] = _ENV_MASK
    return dup


__all__ = ["redact", "redact_env", "normalize_aliases"]
