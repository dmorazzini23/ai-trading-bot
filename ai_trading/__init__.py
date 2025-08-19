"""Top-level package for ai_trading with light optional imports."""  # AI-AGENT-REF

from __future__ import annotations

import os as _os
import sys as _sys

__version__ = "2.0.0"

from ai_trading.utils.optdeps import module_ok as _module_ok

from .alpaca_api import ALPACA_AVAILABLE  # AI-AGENT-REF: canonical flag

FINNHUB_AVAILABLE = _module_ok("finnhub")

__all__ = [
    "__version__",
    "ALPACA_AVAILABLE",
    "FINNHUB_AVAILABLE",
    "YFIN_AVAILABLE",
    "_MINUTE_CACHE",
]


def __getattr__(name: str):  # AI-AGENT-REF: lazy data_fetcher exposure
    if name in {"_MINUTE_CACHE", "YFIN_AVAILABLE"}:
        from .data_fetcher import _MINUTE_CACHE, YFIN_AVAILABLE  # noqa: F401

        globals().update(
            {"_MINUTE_CACHE": _MINUTE_CACHE, "YFIN_AVAILABLE": YFIN_AVAILABLE}
        )
        return globals()[name]
    raise AttributeError(name)


