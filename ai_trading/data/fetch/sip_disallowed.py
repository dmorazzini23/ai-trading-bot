from __future__ import annotations

"""Lightweight helpers for handling disallowed SIP feed requests.

This module mirrors a minimal subset of :mod:`ai_trading.data.fetch` to allow
isolated testing.  When the SIP feed is disabled, requests should fall back to
IEX while still recording that ``sip`` was originally requested.
"""

from datetime import datetime
from typing import Any

from ai_trading.utils.lazy_imports import load_pandas

# Re-use shared state from the package's ``__init__`` module.
from . import _HTTP_SESSION, _ALLOW_SIP

pd = load_pandas()


def _to_df(payload: dict[str, Any]):
    """Return a pandas DataFrame for ``payload`` bars."""
    bars = payload.get("bars", [])
    if pd is None:  # pragma: no cover - defensive
        return bars
    try:  # pragma: no cover - optional pandas
        return pd.DataFrame(bars)
    except Exception:
        return bars


def fetch_bars(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str,
    *,
    feed: str = "sip",
    session=None,
    feeds_used: list[str] | None = None,
) -> Any:
    """Fetch market bars honouring SIP disallow rules.

    Parameters
    ----------
    feeds_used:
        Optional list extended with the actual feed used.  When a fallback from
        SIP to IEX occurs, the originally requested ``sip`` feed is appended as
        well so callers can introspect both feeds.
    """

    session = session or _HTTP_SESSION
    tf = str(timeframe)
    requested = str(feed)
    actual = requested
    if requested == "sip" and not _ALLOW_SIP:
        actual = "iex"

    params = {
        "symbol": symbol,
        "start": start,
        "end": end,
        "feed": actual,
        "timeframe": tf,
    }
    resp = session.get("https://data.alpaca.markets/v2/stocks/bars", params=params)

    if feeds_used is not None:
        feeds_used.append(actual)
        if actual != requested:
            feeds_used.append(requested)

    return _to_df(resp.json())


__all__ = ["fetch_bars"]
