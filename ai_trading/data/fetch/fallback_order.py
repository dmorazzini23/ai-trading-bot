from __future__ import annotations

"""State tracking for provider fallback order.

This module exposes lightweight registries used by tests to inspect which
providers were utilized as fallbacks and for which symbols. The registries are
append-only and persist until :func:`reset` is called, allowing retries to be
observed across test assertions.
"""

import time
from typing import Dict, List, Optional

from ai_trading.logging import get_logger


logger = get_logger(__name__)

# Public dictionary tracking whether a provider was ever used as a fallback.
FALLBACK_ORDER: Dict[str, bool] = {}

# Chronological record of providers used as fallbacks.
FALLBACK_PROVIDERS: List[str] = []
"""List of fallback providers in the order they were invoked."""

# Chronological record of symbols that triggered provider fallbacks.
FALLBACK_SYMBOLS: List[str] = []
"""List of symbols corresponding to ``FALLBACK_PROVIDERS`` entries."""

_HIGH_RES_PROVIDERS = {"finnhub", "finnhub_low_latency"}
_PROMOTION_TTL_SECONDS = 900.0
_PROMOTED_PROVIDERS: Dict[str, tuple[str, float]] = {}


def _normalize_provider(provider: str | None) -> str | None:
    if provider is None:
        return None
    try:
        candidate = provider.strip().lower()
    except AttributeError:
        return None
    return candidate or None


def _symbol_key(symbol: str | None) -> str | None:
    if symbol is None:
        return None
    try:
        normalized = symbol.strip().upper()
    except AttributeError:
        return None
    return normalized or None


def register_fallback(provider: str, symbol: str) -> None:
    """Record a fallback to ``provider`` for ``symbol``.

    Appends the arguments to :data:`FALLBACK_PROVIDERS` and
    :data:`FALLBACK_SYMBOLS`. These registries persist across retries for test
    introspection and are not automatically deduplicated.
    """

    FALLBACK_PROVIDERS.append(provider)
    FALLBACK_SYMBOLS.append(symbol)
    FALLBACK_ORDER[provider] = True

    promoted = _normalize_provider(provider)
    symbol_key = _symbol_key(symbol)
    if promoted in _HIGH_RES_PROVIDERS and symbol_key:
        _PROMOTED_PROVIDERS[symbol_key] = (promoted, time.monotonic())


def resolve_promoted_provider(symbol: str, *, interval: str | None = None) -> Optional[str]:
    """Return the promoted provider for ``symbol`` when still fresh."""

    symbol_key = _symbol_key(symbol)
    if not symbol_key:
        return None
    entry = _PROMOTED_PROVIDERS.get(symbol_key)
    if not entry:
        return None
    provider, ts = entry
    ttl = _PROMOTION_TTL_SECONDS
    now = time.monotonic()
    if now - ts > ttl:
        _PROMOTED_PROVIDERS.pop(symbol_key, None)
        return None
    return provider


def promote_high_resolution(symbol: str, provider: str | None = None) -> Optional[str]:
    """Promote ``provider`` for ``symbol`` to be preferred on the next fallback."""

    symbol_key = _symbol_key(symbol)
    if not symbol_key:
        return None
    normalized = _normalize_provider(provider)
    if normalized is None:
        return None
    if normalized not in _HIGH_RES_PROVIDERS:
        return None
    _PROMOTED_PROVIDERS[symbol_key] = (normalized, time.monotonic())
    logger.info(
        "FALLBACK_PROVIDER_PROMOTED",
        extra={"symbol": symbol_key, "provider": normalized},
    )
    return normalized


def demote_provider(symbol: str, provider: str | None = None) -> None:
    """Remove any promotion for ``symbol`` when ``provider`` is unsuitable."""

    symbol_key = _symbol_key(symbol)
    if not symbol_key:
        return
    if provider is None:
        _PROMOTED_PROVIDERS.pop(symbol_key, None)
        return
    normalized = _normalize_provider(provider)
    current = _PROMOTED_PROVIDERS.get(symbol_key)
    if current and current[0] == normalized:
        _PROMOTED_PROVIDERS.pop(symbol_key, None)
        logger.info(
            "FALLBACK_PROVIDER_DEMOTED",
            extra={"symbol": symbol_key, "provider": normalized},
        )


def mark_yahoo(symbol: str | None = None) -> None:
    """Record that Yahoo was used as a fallback.

    ``symbol`` is optional for legacy callers without symbol context. When
    provided, the pair is forwarded to :func:`register_fallback`.
    """

    if symbol is None:
        symbol = ""
    register_fallback("yahoo", symbol)


def reset() -> None:
    """Reset tracked state and registries (used in tests)."""

    FALLBACK_ORDER.clear()
    FALLBACK_PROVIDERS.clear()
    FALLBACK_SYMBOLS.clear()
    _PROMOTED_PROVIDERS.clear()


__all__ = [
    "FALLBACK_ORDER",
    "FALLBACK_PROVIDERS",
    "FALLBACK_SYMBOLS",
    "register_fallback",
    "mark_yahoo",
    "resolve_promoted_provider",
    "promote_high_resolution",
    "demote_provider",
    "reset",
]
