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
from ai_trading.timeframe import canonicalize_timeframe

# Resolve the Alpaca classes lazily at import time while tolerating the SDK
# being unavailable during tests.
TimeFrame = get_timeframe_cls()
# Ensure common shorthand attributes exist for tests and call-sites
try:  # best-effort: some SDK versions already provide these
    unit_cls = get_timeframe_unit_cls()
    if unit_cls is not None:
        if not hasattr(TimeFrame, "Day"):
            setattr(TimeFrame, "Day", TimeFrame(1, getattr(unit_cls, "Day", "Day")))  # type: ignore[arg-type]
        if not hasattr(TimeFrame, "Minute"):
            setattr(TimeFrame, "Minute", TimeFrame(1, getattr(unit_cls, "Minute", "Minute")))  # type: ignore[arg-type]
        if not hasattr(TimeFrame, "Hour"):
            setattr(TimeFrame, "Hour", TimeFrame(1, getattr(unit_cls, "Hour", "Hour")))  # type: ignore[arg-type]
        if not hasattr(TimeFrame, "Week"):
            setattr(TimeFrame, "Week", TimeFrame(1, getattr(unit_cls, "Week", "Week")))  # type: ignore[arg-type]
except Exception:
    pass
_BaseStockBarsRequest = get_stock_bars_request_cls()


def _coerce_timeframe(tf: Any) -> Any:
    """Return ``tf`` as an instance of the active ``TimeFrame`` class."""

    return canonicalize_timeframe(tf)


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
                        pass
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
