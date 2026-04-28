"""Alpaca ``TimeFrame`` helpers.

This module exposes a wrapper around the Alpaca SDK's ``TimeFrame`` class
that is safe to instantiate with no arguments.  ``TimeFrame()`` will
produce a timeframe representing ``1 Day``.  The wrapper also ensures that
all instances expose ``amount`` and ``unit`` attributes, even if the
underlying SDK changes implementation details.
"""

from __future__ import annotations

from typing import Any, cast

from ai_trading.logging import get_logger

logger = get_logger(__name__)

_BaseTimeFrame: Any | None = None
TimeFrameUnit: Any


def _safe_setattr(obj: object, name: str, value: object) -> None:
    """Best-effort setattr that tolerates read-only descriptors."""

    try:
        object.__setattr__(obj, name, value)
    except (AttributeError, TypeError):
        # Some SDK versions expose ``amount``/``unit`` as read-only properties.
        # When that happens we rely on their getters without mutating state.
        return


def _load_timeframe_bindings() -> tuple[Any, Any]:
    """Return Alpaca timeframe classes without importing the SDK at module import."""

    global _BaseTimeFrame
    unit_cls = globals().get("TimeFrameUnit")
    if _BaseTimeFrame is not None and unit_cls is not None:
        return _BaseTimeFrame, unit_cls
    try:
        from alpaca.data.timeframe import TimeFrame as base_cls, TimeFrameUnit as loaded_unit_cls  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in targeted import tests
        raise RuntimeError("alpaca-py==0.42.1 is required for TimeFrame support") from exc
    _BaseTimeFrame = base_cls
    globals()["TimeFrameUnit"] = loaded_unit_cls
    return base_cls, loaded_unit_cls


def _resolve_timeframe_unit_cls() -> Any:
    """Return a unit enum/object exposing Minute/Hour/Day/Week/Month."""

    unit_cls = globals().get("TimeFrameUnit")
    if unit_cls is None:
        _, unit_cls = _load_timeframe_bindings()
    if all(hasattr(unit_cls, name) for name in ("Minute", "Hour", "Day", "Week", "Month")):
        return unit_cls
    raise RuntimeError("alpaca-py TimeFrameUnit is missing required units")


class _TimeFrameFactoryMeta(type):
    def __call__(cls, amount: int = 1, unit=None) -> Any:
        base_cls, _ = _load_timeframe_bindings()
        unit_cls = _resolve_timeframe_unit_cls()
        if unit is None:
            unit = unit_cls.Day
        instance = base_cls(amount, unit)
        try:
            current_amount = getattr(instance, "amount")
        except AttributeError:
            current_amount = None
        if current_amount is None:
            current_amount = amount
        _safe_setattr(instance, "amount", current_amount)

        try:
            current_unit = getattr(instance, "unit")
        except AttributeError:
            current_unit = None
        if current_unit is None:
            current_unit = unit
        _safe_setattr(instance, "unit", current_unit)
        return instance

    def __instancecheck__(cls, instance: object) -> bool:
        base_cls, _ = _load_timeframe_bindings()
        return isinstance(instance, base_cls)

    def __getattr__(cls, name: str) -> Any:
        base_cls, _ = _load_timeframe_bindings()
        return getattr(base_cls, name)


class TimeFrame(metaclass=_TimeFrameFactoryMeta):
    """Type-like lazy factory for Alpaca ``TimeFrame`` instances."""

    amount: int
    unit: Any

    def __init__(self, amount: int = 1, unit: Any = None) -> None:
        del amount, unit


def canonicalize_timeframe(tf: Any) -> TimeFrame:
    """Return ``tf`` as a normalized :class:`TimeFrame` instance.

    Accepts existing ``TimeFrame`` objects, integers (interpreted as ``Day``
    units), strings like ``"1Day"`` or ``"minute"``, or arbitrary objects with
    ``amount`` and ``unit`` attributes.  Any malformed input falls back to
    ``1 Day``.
    """

    base_cls, _ = _load_timeframe_bindings()
    if isinstance(tf, base_cls):
        return cast(TimeFrame, tf)

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
        except (TypeError, ValueError, AttributeError):
            logger.debug("TIMEFRAME_ATTR_COERCE_FAILED", exc_info=True)

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
    except (TypeError, ValueError, AttributeError):
        logger.debug("TIMEFRAME_STRING_PARSE_FAILED", extra={"value": tf}, exc_info=True)

    return TimeFrame()


def __getattr__(name: str) -> Any:
    if name == "TimeFrameUnit":
        _, unit_cls = _load_timeframe_bindings()
        return unit_cls
    raise AttributeError(name)


__all__ = ["TimeFrame", "TimeFrameUnit", "canonicalize_timeframe"]
