"""Compat layer for Alpaca ``TimeFrame``.

This module exposes a wrapper around the Alpaca SDK's ``TimeFrame`` class
that is safe to instantiate with no arguments.  ``TimeFrame()`` will
produce a timeframe representing ``1 Day``.  The wrapper also ensures that
all instances expose ``amount`` and ``unit`` attributes, even if the
underlying SDK changes implementation details.
"""

from __future__ import annotations

from typing import Any

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


def canonicalize_timeframe(tf: Any) -> TimeFrame:
    """Return ``tf`` as a normalized :class:`TimeFrame` instance.

    Accepts existing ``TimeFrame`` objects, integers (interpreted as ``Day``
    units), strings like ``"1Day"`` or ``"minute"``, or arbitrary objects with
    ``amount`` and ``unit`` attributes.  Any malformed input falls back to
    ``1 Day``.
    """

    try:
        if isinstance(tf, TimeFrame) and tf.__class__ is TimeFrame:
            return tf
    except Exception:
        pass

    unit_cls = TimeFrameUnit

    if isinstance(tf, (int, float)):
        return TimeFrame(int(tf) or 1, unit_cls.Day)

    amount = getattr(tf, "amount", None)
    unit = getattr(tf, "unit", None)
    if amount is not None and unit is not None:
        try:
            if not isinstance(unit, unit_cls):
                name = getattr(unit, "name", str(unit)).capitalize()
                unit = getattr(unit_cls, name, unit_cls.Day)
            return TimeFrame(int(amount), unit)
        except Exception:
            pass

    try:
        s = str(tf).strip()
        if s:
            import re

            m = re.match(r"(\d+)?\s*(\w+)", s)
            if m:
                amt = int(m.group(1) or 1)
                unit_name = m.group(2).capitalize()
                unit_name = {
                    "M": "Minute",
                    "Min": "Minute",
                    "Minute": "Minute",
                    "H": "Hour",
                    "Hr": "Hour",
                    "Hour": "Hour",
                    "D": "Day",
                    "Day": "Day",
                    "W": "Week",
                    "Week": "Week",
                    "Mo": "Month",
                    "Month": "Month",
                }.get(unit_name, unit_name)
                unit = getattr(unit_cls, unit_name, unit_cls.Day)
                return TimeFrame(amt, unit)
    except Exception:
        pass

    return TimeFrame()


__all__ = ["TimeFrame", "TimeFrameUnit", "canonicalize_timeframe"]
