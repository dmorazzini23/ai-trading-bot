from __future__ import annotations

"""Lightweight indicator factory used in analytics tests.

This module exposes a :func:`create_indicator` helper that instantiates
indicator implementations by name.  It is intentionally minimal and
re-uses the streaming and incremental indicator classes defined in
``ai_trading.indicator_manager`` to avoid code duplication.
"""

from typing import Protocol, Type, Any

from ai_trading.indicator_manager import (
    StreamingSMA,
    StreamingEMA,
    StreamingRSI,
    IncrementalSMA,
    IncrementalEMA,
    IncrementalRSI,
)


class Indicator(Protocol):
    """Protocol for indicator implementations."""

    def update(self, value: float) -> float | None:  # pragma: no cover - simple protocol
        """Update indicator with a new value."""


# Mapping of indicator names to their implementing classes.
_INDICATORS: dict[str, Type[Indicator]] = {
    "sma": StreamingSMA,
    "ema": StreamingEMA,
    "rsi": StreamingRSI,
    "incremental_sma": IncrementalSMA,
    "incremental_ema": IncrementalEMA,
    "incremental_rsi": IncrementalRSI,
}


def create_indicator(indicator_name: str, **params: Any) -> Indicator:
    """Instantiate an indicator by name.

    Parameters
    ----------
    indicator_name:
        Identifier of the indicator.  The lookup is case insensitive.
    **params:
        Keyword arguments forwarded to the indicator constructor.

    Returns
    -------
    Indicator
        An instantiated indicator object.

    Raises
    ------
    ValueError
        If ``name`` does not correspond to a known indicator.
    """

    cls = _INDICATORS.get(indicator_name.lower())
    if cls is None:
        raise ValueError(f"Unknown indicator: {indicator_name}")
    return cls(**params)


__all__ = ["Indicator", "create_indicator"]
