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


def _safe_setattr(obj: object, name: str, value: object) -> None:
    """Best-effort setattr that tolerates read-only descriptors."""

    try:
        object.__setattr__(obj, name, value)
    except (AttributeError, TypeError):
        # Some SDK versions expose ``amount``/``unit`` as read-only properties.
        # When that happens we rely on their getters without mutating state.
        return


def _resolve_timeframe_unit_cls() -> Any:
    """Return a unit enum/object exposing Minute/Hour/Day/Week/Month."""

    unit_cls = TimeFrameUnit
    if all(hasattr(unit_cls, name) for name in ("Minute", "Hour", "Day", "Week", "Month")):
        return unit_cls
    try:
        from ai_trading.alpaca_api import TimeFrameUnit as _FallbackUnit  # type: ignore

        if all(hasattr(_FallbackUnit, name) for name in ("Minute", "Hour", "Day", "Week", "Month")):
            return _FallbackUnit
    except Exception:
        pass

    class _UnitFallback:
        Minute = "Minute"
        Hour = "Hour"
        Day = "Day"
        Week = "Week"
        Month = "Month"

    return _UnitFallback


class TimeFrame(_BaseTimeFrame):  # type: ignore[misc]
    """Timeframe with safe defaults and attribute accessors."""

    def __init__(self, amount: int = 1, unit=TimeFrameUnit.Day):  # type: ignore[assignment]
        unit_cls = _resolve_timeframe_unit_cls()
        if unit is None:
            unit = getattr(unit_cls, "Day", "Day")
        try:
            super().__init__(amount, unit)
        except Exception:
            # When third-party tests monkeypatch Alpaca's enum internals,
            # base-class validation can crash. Keep a lightweight usable object.
            try:
                amount_value = int(amount)
            except Exception:
                amount_value = 1
            if amount_value <= 0:
                amount_value = 1
            _safe_setattr(self, "amount_value", amount_value)
            _safe_setattr(self, "unit_value", unit)
            _safe_setattr(self, "amount", amount_value)
            _safe_setattr(self, "unit", unit)
            return
        # Guarantee ``amount`` and ``unit`` attributes for downstream code
        try:
            current_amount = getattr(self, "amount")
        except AttributeError:
            current_amount = None
        if current_amount is None:
            current_amount = amount
        _safe_setattr(self, "amount", current_amount)

        try:
            current_unit = getattr(self, "unit")
        except AttributeError:
            current_unit = None
        if current_unit is None:
            current_unit = unit
        _safe_setattr(self, "unit", current_unit)


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

    unit_cls = _resolve_timeframe_unit_cls()

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
