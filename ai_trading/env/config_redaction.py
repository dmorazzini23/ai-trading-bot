from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ai_trading.logging.redact import normalize_aliases, redact_env


def redact_config_env(env: Mapping[str, Any]) -> Mapping[str, Any]:
    """Redact *env* while honoring alias mapping.

    Alias variables like ``ALPACA_BASE_URL`` are emitted under their canonical
    ``ALPACA_API_URL`` key so logs remain consistent regardless of which
    variant the user provided.
    """
    normalized = normalize_aliases(env)
    return redact_env(normalized, drop=True)


__all__ = ["redact_config_env"]
