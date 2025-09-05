"""Reusable market data request models.

This module exposes light-weight wrappers around Alpaca's request models
that accept flexible ``timeframe`` inputs.  Callers may supply the
``TimeFrame`` object provided by the SDK, a string like ``"1Day"`` or even a
simple object with ``amount`` and ``unit`` attributes.  The field validator
ensures the value is converted to the active ``TimeFrame`` type before the
underlying model performs validation.
"""

from __future__ import annotations

from typing import Any

from ai_trading.alpaca_api import (
    get_stock_bars_request_cls,
    get_timeframe_cls,
    get_timeframe_unit_cls,
)

# Resolve the Alpaca classes lazily at import time while tolerating the SDK
# being unavailable during tests.
TimeFrame = get_timeframe_cls()
_BaseStockBarsRequest = get_stock_bars_request_cls()


def _coerce_timeframe(tf: Any) -> Any:
    """Return ``tf`` as an instance of the active ``TimeFrame`` class."""

    try:
        if isinstance(tf, TimeFrame):
            return tf
    except Exception:  # pragma: no cover - defensive
        return tf

    try:
        unit_cls = get_timeframe_unit_cls()
    except Exception:  # pragma: no cover - optional SDK
        unit_cls = None

    # Accept objects with ``amount`` and ``unit`` attributes
    amount = getattr(tf, "amount", None)
    unit = getattr(tf, "unit", None)
    if amount is not None and unit is not None and unit_cls is not None:
        try:
            if not isinstance(unit, unit_cls):
                name = getattr(unit, "name", str(unit)).capitalize()
                unit = getattr(unit_cls, name, getattr(unit_cls, "Day"))
            return TimeFrame(int(amount), unit)  # type: ignore[arg-type]
        except Exception:
            pass

    # Fallback: parse string representations like "1Day" or "minute"
    try:
        s = str(tf).strip()
        if s:
            import re

            m = re.match(r"(\d+)?\s*(\w+)", s)
            if m:
                amt = int(m.group(1) or 1)
                unit_name = m.group(2).capitalize()
                # Map common abbreviations like "Min" -> "Minute"
                unit_name = {
                    "Min": "Minute",
                    "Hour": "Hour",
                    "Day": "Day",
                    "Week": "Week",
                    "Month": "Month",
                }.get(unit_name, unit_name)
                if unit_cls is not None:
                    unit = getattr(unit_cls, unit_name, getattr(unit_cls, "Day"))
                    return TimeFrame(amt, unit)  # type: ignore[arg-type]
    except Exception:
        pass
    
    return tf


# When the real SDK is available the base request class derives from Pydantic's
# ``BaseModel``.  Subclass it so we can attach a validator that normalises the
# timeframe field.  If the base class is a plain dataclass (our fallback when
# Alpaca is unavailable) we fall back to a simple constructor wrapper.
try:  # pragma: no cover - pydantic optional during some tests
    from pydantic import BaseModel, field_validator

    if issubclass(_BaseStockBarsRequest, BaseModel):

        from pydantic import model_validator

        class StockBarsRequest(_BaseStockBarsRequest):  # type: ignore[misc]
            """StockBarsRequest accepting flexible timeframe inputs."""

            @model_validator(mode="before")
            @classmethod
            def _convert_timeframe(cls, data: Any) -> Any:
                if isinstance(data, dict) and "timeframe" in data:
                    data["timeframe"] = _coerce_timeframe(data["timeframe"])
                return data

        StockBarsRequest.model_rebuild()  # ensure validator is applied

    else:  # pragma: no cover - base class is not a Pydantic model

        def StockBarsRequest(*args: Any, **kwargs: Any):  # type: ignore[override]
            args = list(args)
            if len(args) >= 2 and "timeframe" not in kwargs:
                args[1] = _coerce_timeframe(args[1])
            elif "timeframe" in kwargs:
                kwargs["timeframe"] = _coerce_timeframe(kwargs["timeframe"])
            return _BaseStockBarsRequest(*args, **kwargs)

except Exception:  # pragma: no cover - pydantic missing entirely

    def StockBarsRequest(*args: Any, **kwargs: Any):  # type: ignore[override]
        args = list(args)
        if len(args) >= 2 and "timeframe" not in kwargs:
            args[1] = _coerce_timeframe(args[1])
        elif "timeframe" in kwargs:
            kwargs["timeframe"] = _coerce_timeframe(kwargs["timeframe"])
        return _BaseStockBarsRequest(*args, **kwargs)


__all__ = ["TimeFrame", "StockBarsRequest"]

