"""Reusable market data request models.

This module exposes light-weight wrappers around Alpaca's request models
that accept flexible ``timeframe`` inputs.  Callers may supply the
``TimeFrame`` object provided by the SDK, a string like ``"1Day"`` or even a
simple object with ``amount`` and ``unit`` attributes.  The field validator
ensures the value is converted to the active ``TimeFrame`` type before the
underlying model performs validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ai_trading.logging import get_logger
from ai_trading.timeframe import canonicalize_timeframe

from ._alpaca_guard import should_import_alpaca_sdk

logger = get_logger(__name__)

_USE_ALPACA_REQUESTS = should_import_alpaca_sdk()

if _USE_ALPACA_REQUESTS:
    try:  # pragma: no cover - exercised when SDK available
        from ai_trading.alpaca_api import (
            get_stock_bars_request_cls,
            get_timeframe_cls,
            get_timeframe_unit_cls,
        )
    except Exception:  # pragma: no cover - fall back when import fails
        _USE_ALPACA_REQUESTS = False

if _USE_ALPACA_REQUESTS:
    TimeFrameType = get_timeframe_cls()
    try:
        unit_cls = get_timeframe_unit_cls()
    except Exception:  # pragma: no cover - unit class optional
        logger.debug("TIMEFRAME_UNIT_CLASS_LOAD_FAILED", exc_info=True)
        unit_cls = None
    _BaseStockBarsRequestType = get_stock_bars_request_cls()
else:
    class TimeFrameUnit(str, Enum):
        Minute = "Min"
        Hour = "Hour"
        Day = "Day"
        Week = "Week"
        Month = "Month"

    @dataclass
    class _FallbackTimeFrame:
        amount: int = 1
        unit: TimeFrameUnit = TimeFrameUnit.Day

        def __str__(self) -> str:
            return f"{self.amount}{self.unit.value}"

    _FallbackTimeFrame.Minute = _FallbackTimeFrame(1, TimeFrameUnit.Minute)  # type: ignore[attr-defined]
    _FallbackTimeFrame.Hour = _FallbackTimeFrame(1, TimeFrameUnit.Hour)  # type: ignore[attr-defined]
    _FallbackTimeFrame.Day = _FallbackTimeFrame(1, TimeFrameUnit.Day)  # type: ignore[attr-defined]
    _FallbackTimeFrame.Week = _FallbackTimeFrame(1, TimeFrameUnit.Week)  # type: ignore[attr-defined]
    _FallbackTimeFrame.Month = _FallbackTimeFrame(1, TimeFrameUnit.Month)  # type: ignore[attr-defined]

    unit_cls = TimeFrameUnit

    class _FallbackStockBarsRequestBase:
        def __init__(
            self,
            symbol_or_symbols: Any,
            timeframe: Any,
            *,
            start: Any | None = None,
            end: Any | None = None,
            limit: int | None = None,
            adjustment: str | None = None,
            feed: str | None = None,
            sort: str | None = None,
            asof: str | None = None,
            currency: str | None = None,
            **extra: Any,
        ) -> None:
            self.symbol_or_symbols = symbol_or_symbols
            self.timeframe = timeframe
            self.start = start
            self.end = end
            self.limit = limit
            self.adjustment = adjustment
            self.feed = feed
            self.sort = sort
            self.asof = asof
            self.currency = currency
            for key, value in extra.items():
                setattr(self, key, value)

    TimeFrameType = _FallbackTimeFrame
    _BaseStockBarsRequestType = _FallbackStockBarsRequestBase

if _USE_ALPACA_REQUESTS and unit_cls is not None:  # pragma: no cover - attributes set when available
    try:
        if not hasattr(TimeFrameType, "Day"):
            setattr(TimeFrameType, "Day", TimeFrameType(1, getattr(unit_cls, "Day", "Day")))  # type: ignore[arg-type]
        if not hasattr(TimeFrameType, "Minute"):
            setattr(TimeFrameType, "Minute", TimeFrameType(1, getattr(unit_cls, "Minute", "Minute")))  # type: ignore[arg-type]
        if not hasattr(TimeFrameType, "Hour"):
            setattr(TimeFrameType, "Hour", TimeFrameType(1, getattr(unit_cls, "Hour", "Hour")))  # type: ignore[arg-type]
        if not hasattr(TimeFrameType, "Week"):
            setattr(TimeFrameType, "Week", TimeFrameType(1, getattr(unit_cls, "Week", "Week")))  # type: ignore[arg-type]
        if not hasattr(TimeFrameType, "Month"):
            setattr(TimeFrameType, "Month", TimeFrameType(1, getattr(unit_cls, "Month", "Month")))  # type: ignore[arg-type]
    except Exception:
        logger.debug("TIMEFRAME_ALIAS_ATTACH_FAILED", exc_info=True)


def _coerce_timeframe(tf: Any) -> Any:
    """Return ``tf`` as an instance of the active ``TimeFrame`` class."""

    coerced = canonicalize_timeframe(tf)
    if _USE_ALPACA_REQUESTS:
        return coerced

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
try:  # pragma: no cover - pydantic optional during some tests
    from pydantic import BaseModel
    StockBarsRequest: Any

    if issubclass(_BaseStockBarsRequestType, BaseModel):

        from pydantic import model_validator

        class _StockBarsRequest(_BaseStockBarsRequestType):  # type: ignore[misc,valid-type]
            """StockBarsRequest accepting flexible timeframe inputs."""

            @model_validator(mode="before")
            @classmethod
            def _convert_timeframe(cls, data: Any) -> Any:
                if isinstance(data, dict):
                    if "timeframe" in data:
                        data["timeframe"] = _coerce_timeframe(data["timeframe"])
                elif hasattr(data, "timeframe"):
                    try:
                        setattr(
                            data,
                            "timeframe",
                            _coerce_timeframe(getattr(data, "timeframe")),
                        )
                    except Exception:
                        logger.debug("STOCK_BARS_REQUEST_TIMEFRAME_SET_FAILED", exc_info=True)
                return data

        _StockBarsRequest.model_rebuild()  # ensure validator is applied
        StockBarsRequest = _StockBarsRequest

    else:  # pragma: no cover - base class is not a Pydantic model

        def _stock_bars_request(*args: Any, **kwargs: Any) -> Any:
            args_list = list(args)
            if len(args_list) >= 2 and "timeframe" not in kwargs:
                args_list[1] = _coerce_timeframe(args_list[1])
            elif "timeframe" in kwargs:
                kwargs["timeframe"] = _coerce_timeframe(kwargs["timeframe"])
            return _BaseStockBarsRequestType(*args_list, **kwargs)

        StockBarsRequest = _stock_bars_request

except Exception:  # pragma: no cover - pydantic missing entirely

    def _stock_bars_request(*args: Any, **kwargs: Any) -> Any:
        args_list = list(args)
        if len(args_list) >= 2 and "timeframe" not in kwargs:
            args_list[1] = _coerce_timeframe(args_list[1])
        elif "timeframe" in kwargs:
            kwargs["timeframe"] = _coerce_timeframe(kwargs["timeframe"])
        return _BaseStockBarsRequestType(*args_list, **kwargs)

    StockBarsRequest = _stock_bars_request

TimeFrame = TimeFrameType


__all__ = ["TimeFrame", "StockBarsRequest"]
