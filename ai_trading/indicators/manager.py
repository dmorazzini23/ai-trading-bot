from __future__ import annotations

from typing import Protocol, Type

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

    def update(self, value: float) -> float | None:
        """Update indicator with new value."""


class IndicatorManager:
    """Factory for creating indicator instances."""

    _INDICATORS: dict[str, Type[Indicator]] = {
        "sma": StreamingSMA,
        "ema": StreamingEMA,
        "rsi": StreamingRSI,
        "incremental_sma": IncrementalSMA,
        "incremental_ema": IncrementalEMA,
        "incremental_rsi": IncrementalRSI,
    }

    def create_indicator(self, indicator_name: str, **params) -> Indicator:
        """Instantiate an indicator by name.

        Args:
            indicator_name: Identifier of the indicator (case-insensitive).
            **params: Parameters passed to the indicator's constructor.

        Returns:
            Indicator: Instantiated indicator object.

        Raises:
            ValueError: If the indicator name is unknown.
        """
        cls = self._INDICATORS.get(indicator_name.lower())
        if cls is None:
            raise ValueError(f"Unknown indicator: {indicator_name}")
        return cls(**params)


__all__ = ["Indicator", "IndicatorManager"]
