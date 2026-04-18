"""Reusable market data request models.

This module exposes light-weight wrappers around Alpaca's request models
that accept flexible ``timeframe`` inputs.  Callers may supply the
``TimeFrame`` object provided by the SDK, a string like ``"1Day"`` or even a
simple object with ``amount`` and ``unit`` attributes.  The field validator
ensures the value is converted to the active ``TimeFrame`` type before the
underlying model performs validation.
"""

from __future__ import annotations

from typing import Any, cast

from ai_trading.alpaca_api import (
    _data_classes,
    get_stock_bars_request_cls,
    get_timeframe_cls,
    get_timeframe_unit_cls,
)
from ai_trading.logging import get_logger
from ai_trading.timeframe import canonicalize_timeframe

logger = get_logger(__name__)


def _resolve_data_bindings() -> tuple[type[Any], type[Any], Any]:
    """Return request/timeframe bindings from the same active Alpaca source."""

    try:
        request_cls, timeframe_cls, timeframe_unit_cls = _data_classes()
        return (
            cast(type[Any], request_cls),
            cast(type[Any], timeframe_cls),
            timeframe_unit_cls,
        )
    except Exception:
        return (
            cast(type[Any], get_stock_bars_request_cls()),
            cast(type[Any], get_timeframe_cls()),
            get_timeframe_unit_cls(),
        )

def _resolve_timeframe_type() -> type[Any]:
    _, timeframe_cls, _ = _resolve_data_bindings()
    return timeframe_cls


def _resolve_timeframe_unit_type() -> Any:
    try:
        _, _, timeframe_unit_cls = _resolve_data_bindings()
        return timeframe_unit_cls
    except Exception:  # pragma: no cover - unit class optional
        logger.debug("TIMEFRAME_UNIT_CLASS_LOAD_FAILED", exc_info=True)
        return None


def _resolve_request_type() -> type[Any]:
    request_cls, _, _ = _resolve_data_bindings()
    return request_cls


def _coerce_timeframe(tf: Any) -> Any:
    """Return ``tf`` as an instance of the active ``TimeFrame`` class."""

    TimeFrameType = _resolve_timeframe_type()
    unit_cls = _resolve_timeframe_unit_type()
    coerced = canonicalize_timeframe(tf)
    try:
        if isinstance(coerced, TimeFrameType):
            return coerced
    except Exception:
        logger.debug("TIMEFRAME_INSTANCE_CHECK_FAILED", exc_info=True)

    amount = getattr(coerced, "amount", None)
    unit = getattr(coerced, "unit", None)
    try:
        amount_val = int(amount) if amount is not None else 1
    except Exception:
        amount_val = 1

    try:
        if unit_cls is not None and unit is not None and not isinstance(unit, unit_cls):
            name = getattr(unit, "name", str(unit)).capitalize()
            unit = getattr(unit_cls, name, getattr(unit_cls, "Day", unit))
    except Exception:
        unit = getattr(unit_cls, "Day", unit)

    try:
        return TimeFrameType(amount_val, unit if unit is not None else getattr(unit_cls, "Day", unit))
    except Exception:
        logger.debug("TIMEFRAME_CONSTRUCTION_FAILED", exc_info=True)
        return TimeFrameType()  # type: ignore[call-arg]


# When the real SDK is available the base request class derives from Pydantic's
# ``BaseModel``.  Subclass it so we can attach a validator that normalises the
# timeframe field.  If the base class is a plain dataclass (our fallback when
# Alpaca is unavailable) we fall back to a simple constructor wrapper.
class StockBarsRequest:
    symbol_or_symbols: Any
    timeframe: Any
    start: Any | None
    end: Any | None
    limit: int | None
    feed: str | None

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        args_list = list(args)
        if len(args_list) >= 2 and "timeframe" not in kwargs:
            args_list[1] = _coerce_timeframe(args_list[1])
        elif "timeframe" in kwargs:
            kwargs["timeframe"] = _coerce_timeframe(kwargs["timeframe"])
        return _resolve_request_type()(*args_list, **kwargs)


class _TimeFrameProxy:
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _resolve_timeframe_type()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(_resolve_timeframe_type(), name)


TimeFrame = _TimeFrameProxy()


__all__ = ["TimeFrame", "StockBarsRequest"]
