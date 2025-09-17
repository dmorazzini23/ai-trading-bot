from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict

from ai_trading.logging.redact import redact_env

# Mapping of environment variable aliases to their canonical names.
_ALIAS_MAP: Dict[str, str] = {
    "ALPACA_BASE_URL": "ALPACA_API_URL",
}


def _apply_aliases(env: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a new mapping with aliases normalized to canonical keys."""
    resolved: Dict[str, Any] = {}
    for key, value in env.items():
        canonical = _ALIAS_MAP.get(key, key)
        if canonical not in resolved:
            resolved[canonical] = value
    return resolved


def redact_config_env(env: Mapping[str, Any]) -> Mapping[str, Any]:
    """Redact *env* while honoring alias mapping.

    Alias variables like ``ALPACA_BASE_URL`` are emitted under their canonical
    ``ALPACA_API_URL`` key so logs remain consistent regardless of which
    variant the user provided.
    """
    normalized = _apply_aliases(env)
    return redact_env(normalized, drop=True)


__all__ = ["redact_config_env"]
