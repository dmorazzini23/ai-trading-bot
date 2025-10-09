from __future__ import annotations

"""Utility helpers for handling Alpaca IEX feed fallbacks.

This module implements a very small subset of the logic that the main
``ai_trading.data.fetch`` package provides.  It focuses on situations where the
Alpaca IEX feed returns an empty payload.  When that occurs we record the
attempt, increment the global ``_IEX_EMPTY_COUNTS`` tracker and, when allowed,
retry the request against the SIP feed.  The counter persists even when the
SIP feed succeeds so future calls can skip the IEX feed until it provides
data again.  If both feeds return empty results an error is logged so callers
can react accordingly.

The functions here are intentionally lightweight so that they can be imported in
isolation for unit tests without pulling in the entire ``fetch`` module.
"""

from datetime import datetime
from typing import Any

from ai_trading.utils.env import get_alpaca_data_base_url, get_alpaca_http_headers
from ai_trading.utils.lazy_imports import load_pandas
from ai_trading.logging import get_logger
from .metrics import inc_empty_payload, mark_skipped, inc_unauthorized_sip

# Import shared state from the package's ``__init__``.  These variables are
# defined there and are re-used across modules.
from . import (
    _HTTP_SESSION,
    _ALLOW_SIP,
    _SIP_UNAUTHORIZED,
    _IEX_EMPTY_COUNTS,
    _IEX_EMPTY_THRESHOLD,
)

pd = load_pandas()
logger = get_logger(__name__)


def _to_df(payload: dict[str, Any]):
    """Return a pandas DataFrame for ``payload`` bars.

    The helper mirrors the behaviour in :mod:`ai_trading.data.fetch` by keeping
    the import of :mod:`pandas` optional.  Tests rely on the ``empty`` attribute
    being present so an empty DataFrame is returned when no bars are found.
    """

    bars = payload.get("bars", [])
    if pd is None:  # pragma: no cover - defensive fallback
        return bars
    return pd.DataFrame(bars)


def fetch_bars(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str,
    *,
    session=None,
) -> Any:
    """Fetch market bars with automatic IEX→SIP fallback.

    Parameters
    ----------
    symbol, start, end, timeframe:
        Basic request parameters mirrored from the full fetcher.  Only the
        ``feed`` parameter is inspected in tests but keeping the signature close
        to the real function aids readability.
    session:
        HTTP session used for requests.  Falls back to the package level session
        when ``None``.
    """

    session = session or _HTTP_SESSION
    tf = str(timeframe)
    key = (symbol, tf)

    # If previous attempts yielded an empty IEX response we skip straight to SIP
    # (when allowed).
    feed = "iex"
    if _IEX_EMPTY_COUNTS.get(key, 0) >= _IEX_EMPTY_THRESHOLD and _ALLOW_SIP and not _SIP_UNAUTHORIZED:
        logger.info(
            "DATA_SOURCE_FALLBACK_ATTEMPT",
            extra={"symbol": symbol, "timeframe": tf, "from": "iex", "to": "sip", "attempts": 0},
        )
        feed = "sip"

    attempts = 0
    while True:
        params = {
            "symbol": symbol,
            "start": start,
            "end": end,
            "feed": feed,
            "timeframe": tf,
        }
        data_base = get_alpaca_data_base_url()
        headers = get_alpaca_http_headers()
        resp = session.get(f"{data_base}/v2/stocks/bars", params=params, headers=headers)
        payload = resp.json()
        bars = payload.get("bars") or []
        if bars:
            # Only reset the empty counter when IEX returns data.  Successful
            # SIP responses keep the counter intact so later calls continue to
            # bypass IEX until it provides data again.
            if feed == "iex":
                _IEX_EMPTY_COUNTS.pop(key, None)
            return _to_df(payload)

        # Empty response handling
        if feed == "iex":
            attempts += 1
            _IEX_EMPTY_COUNTS[key] = _IEX_EMPTY_COUNTS.get(key, 0) + 1
            inc_empty_payload(symbol, tf)
            mark_skipped(symbol, tf)
            if _ALLOW_SIP and not _SIP_UNAUTHORIZED:
                logger.info(
                    "DATA_SOURCE_FALLBACK_ATTEMPT",
                    extra={
                        "symbol": symbol,
                        "timeframe": tf,
                        "from": "iex",
                        "to": "sip",
                        "attempts": attempts,
                    },
                )
                feed = "sip"
                continue
            if _SIP_UNAUTHORIZED:
                inc_unauthorized_sip("alpaca")
            return _to_df({})

        # If we get here the SIP request was also empty
        logger.error(
            "IEX_EMPTY_SIP_EMPTY",
            extra={"symbol": symbol, "timeframe": tf, "attempts": attempts + 1},
        )
        return _to_df({})


__all__ = ["fetch_bars"]
