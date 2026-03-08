from __future__ import annotations

"""Strategy profile loader (opt-in).

Loads JSON profiles that define per-symbol overrides for strategy parameters.
Profiles are only applied when explicitly configured via the
``STRATEGY_PROFILE`` environment variable (path) or when a path is passed to
the loader. This keeps production defaults unchanged, per AGENTS.md.
"""

import json
from functools import lru_cache
from typing import Any, cast

from ai_trading.config.management import get_env


@lru_cache(maxsize=1)
def _load_profile_cached(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if isinstance(loaded, dict):
        return cast(dict[str, Any], loaded)
    return {}


def load_strategy_profile(path_or_env: str | None = None) -> dict[str, Any] | None:
    """Load a strategy profile JSON if configured; otherwise return None."""
    path = (
        path_or_env
        or get_env("STRATEGY_PROFILE", None, cast=str, resolve_aliases=False)
        or get_env("AI_TRADING_STRATEGY_PROFILE", None, cast=str, resolve_aliases=False)
    )
    if not path:
        return None
    try:
        return _load_profile_cached(path)
    except (OSError, ValueError, TypeError):
        return None


def lookup_overrides(profile: dict[str, Any] | None, symbol: str, strategy: str) -> dict[str, Any]:
    """Return overrides for a given symbol/strategy from profile; {} if none."""
    if not profile:
        return {}
    sym = symbol.upper()
    try:
        return dict(profile.get("symbols", {}).get(sym, {}).get(strategy, {}) or {})
    except Exception:
        return {}


__all__ = ["load_strategy_profile", "lookup_overrides"]
