"""Compat layer for Alpaca ``TimeFrame``.

This module exposes a wrapper around the Alpaca SDK's ``TimeFrame`` class
that is safe to instantiate with no arguments.  ``TimeFrame()`` will
produce a timeframe representing ``1 Day``.  The wrapper also ensures that
all instances expose ``amount`` and ``unit`` attributes, even if the
underlying SDK changes implementation details.
"""

from __future__ import annotations

from ai_trading.alpaca_api import ALPACA_AVAILABLE

if ALPACA_AVAILABLE:  # pragma: no cover - depends on alpaca-py
    from alpaca.data.timeframe import TimeFrame as _BaseTimeFrame, TimeFrameUnit  # type: ignore
else:  # pragma: no cover - exercised when SDK missing
    from ai_trading.alpaca_api import TimeFrame as _BaseTimeFrame, TimeFrameUnit  # type: ignore


class TimeFrame(_BaseTimeFrame):  # type: ignore[misc]
    """Timeframe with safe defaults and attribute accessors."""

    def __init__(self, amount: int = 1, unit=TimeFrameUnit.Day):  # type: ignore[assignment]
        super().__init__(amount, unit)
        # Guarantee ``amount`` and ``unit`` attributes for downstream code
        object.__setattr__(self, "amount", getattr(self, "amount", amount))
        object.__setattr__(self, "unit", getattr(self, "unit", unit))


__all__ = ["TimeFrame", "TimeFrameUnit"]
