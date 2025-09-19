"""Environment helpers exposed as attribute-style access."""

from __future__ import annotations

import os
from typing import Any


_BOOL_TRUE = {"1", "true", "yes", "on", "enable", "enabled"}
_BOOL_FALSE = {"0", "false", "no", "off", "disable", "disabled"}
_ALIASES = {
    "ALPACA_KEY": "ALPACA_API_KEY",
    "ALPACA_SECRET": "ALPACA_SECRET_KEY",
}
_BOOL_KEYS = {"ALPACA_ALLOW_SIP", "ALPACA_HAS_SIP"}


class _EnvProxy:
    """Proxy object exposing ``os.environ`` via attributes."""

    def __getattr__(self, name: str) -> Any:
        key = _ALIASES.get(name, name)
        value = os.getenv(key)
        if value is None:
            return None
        if key in _BOOL_KEYS:
            value_lower = value.strip().lower()
            if value_lower in _BOOL_TRUE:
                return True
            if value_lower in _BOOL_FALSE:
                return False
        return value


env = _EnvProxy()


__all__ = ["env"]

