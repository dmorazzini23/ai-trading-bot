from __future__ import annotations

"""State tracking for provider fallback order.

This module exposes lightweight registries used by tests to inspect which
providers were utilized as fallbacks and for which symbols. The registries are
append-only and persist until :func:`reset` is called, allowing retries to be
observed across test assertions.
"""

from typing import Dict, List

# Public dictionary tracking whether a provider was ever used as a fallback.
FALLBACK_ORDER: Dict[str, bool] = {}

# Chronological record of providers used as fallbacks.
FALLBACK_PROVIDERS: List[str] = []
"""List of fallback providers in the order they were invoked."""

# Chronological record of symbols that triggered provider fallbacks.
FALLBACK_SYMBOLS: List[str] = []
"""List of symbols corresponding to ``FALLBACK_PROVIDERS`` entries."""


def register_fallback(provider: str, symbol: str) -> None:
    """Record a fallback to ``provider`` for ``symbol``.

    Appends the arguments to :data:`FALLBACK_PROVIDERS` and
    :data:`FALLBACK_SYMBOLS`. These registries persist across retries for test
    introspection and are not automatically deduplicated.
    """

    FALLBACK_PROVIDERS.append(provider)
    FALLBACK_SYMBOLS.append(symbol)
    FALLBACK_ORDER[provider] = True


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


__all__ = [
    "FALLBACK_ORDER",
    "FALLBACK_PROVIDERS",
    "FALLBACK_SYMBOLS",
    "register_fallback",
    "mark_yahoo",
    "reset",
]
