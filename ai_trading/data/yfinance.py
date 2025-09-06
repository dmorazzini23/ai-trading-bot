"""Utilities for interacting with the optional :mod:`yfinance` package.

The helper below ensures that when requesting data with ``auto_adjust=True``
we also configure yfinance's timezone cache location.  The function returns a
boolean indicating whether the cache update succeeded.
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import Any


def download_and_cache(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str,
    *,
    auto_adjust: bool = True,
    **kwargs: Any,
) -> bool:
    """Download data via ``yfinance`` and set the tz cache if possible.

    Parameters
    ----------
    symbol:
        Ticker symbol to download.
    start, end:
        Date range for the download.
    interval:
        Bar interval to request (e.g. ``"1d"``).
    auto_adjust:
        Passed through to ``yfinance.download``.  When ``True`` this function
        attempts to update the timezone cache location.
    **kwargs:
        Additional keyword arguments forwarded to ``yfinance.download``.

    Returns
    -------
    bool
        ``True`` if the timezone cache update succeeded, ``False`` otherwise.
    """
    try:
        import yfinance as yf  # type: ignore  # local import
    except Exception:  # pragma: no cover - optional dependency
        return False

    cache_updated = False
    if auto_adjust and hasattr(yf, "set_tz_cache_location"):
        try:
            os.makedirs("/tmp/py-yfinance", exist_ok=True)
            yf.set_tz_cache_location("/tmp/py-yfinance")
            cache_updated = True
        except OSError:
            cache_updated = False

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*auto_adjust.*", module="yfinance")
        yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            threads=False,
            progress=False,
            group_by="column",
            **kwargs,
        )

    return cache_updated


__all__ = ["download_and_cache"]
