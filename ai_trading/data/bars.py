from __future__ import annotations

from typing import Any

import pandas as pd

from ai_trading.data_fetcher import (
    get_bars as http_get_bars,
)  # AI-AGENT-REF: fallback helpers
from ai_trading.data_fetcher import (
    get_minute_df,
)
from ai_trading.logging import get_logger

_log = get_logger(__name__)

# Light, local Alpaca shims so this module never needs bot_engine
try:
    from alpaca.data.requests import StockBarsRequest  # type: ignore
except Exception:  # pragma: no cover  # noqa: BLE001

    class StockBarsRequest:  # type: ignore
        pass


try:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # type: ignore
except Exception:  # pragma: no cover  # noqa: BLE001

    class TimeFrame:  # type: ignore
        def __init__(self, n: int, unit: Any) -> None:
            self.n = n
            self.unit = unit

    class TimeFrameUnit:  # type: ignore
        Day = "Day"
        Minute = "Minute"


# Keep exception scope narrow and local
COMMON_EXC = (
    ValueError,
    KeyError,
    AttributeError,
    TypeError,
    RuntimeError,
    ImportError,
    OSError,
    ConnectionError,
    TimeoutError,
)


def _ensure_df(obj: Any) -> pd.DataFrame:
    """Best-effort conversion to DataFrame, never raises."""
    try:
        if obj is None:
            return pd.DataFrame()
        if isinstance(obj, pd.DataFrame):
            return obj
        if hasattr(obj, "df"):
            df = getattr(obj, "df", None)
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        return pd.DataFrame(obj) if obj is not None else pd.DataFrame()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def empty_bars_dataframe() -> pd.DataFrame:
    cols = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    return pd.DataFrame(columns=cols)  # AI-AGENT-REF: shared empty DataFrame


def _create_empty_bars_dataframe() -> pd.DataFrame:
    return empty_bars_dataframe()


def _is_minute_timeframe(tf) -> bool:
    try:
        return str(tf).lower() in ("1min", "1m", "minute", "1 minute")
    except Exception:  # noqa: BLE001
        return False  # AI-AGENT-REF: broad but safe


def safe_get_stock_bars(
    client: Any, request: StockBarsRequest, symbol: str, context: str = ""
) -> pd.DataFrame:
    """
    Safely fetch stock bars via Alpaca client and always return a DataFrame.
    This is a faithful move of the original implementation from bot_engine,
    with identical behavior and logging fields.
    """
    try:
        response = client.get_stock_bars(request)
        df = getattr(response, "df", None)
        if df is None or df.empty:
            _log.warning("ALPACA_BARS_EMPTY", extra={"symbol": symbol, "context": context})
            if _is_minute_timeframe(getattr(request, "timeframe", "")):
                df = get_minute_df(
                    symbol,
                    getattr(request, "start", None),
                    getattr(request, "end", None),
                    feed=getattr(request, "feed", None),
                )
            else:
                df = http_get_bars(
                    symbol,
                    str(getattr(request, "timeframe", "")),
                    getattr(request, "start", None),
                    getattr(request, "end", None),
                    feed=getattr(request, "feed", None) or "sip",
                )
            return _ensure_df(df)  # AI-AGENT-REF: fallback to robust fetchers

        # If MultiIndex (symbol, ts), select the symbol level
        if isinstance(df.index, pd.MultiIndex):
            try:
                df = df.xs(symbol, level=0, drop_level=False).droplevel(0)
            except (KeyError, ValueError):
                return _create_empty_bars_dataframe()

        # Normalize column casing if needed, keep original names if present
        # (match existing expectations in portfolio/core.py)
        if not df.empty:
            return df

        _log.warning(
            "ALPACA_PARSE_EMPTY",
            extra={
                "symbol": symbol,
                "context": context,
                "feed": getattr(request, "feed", None),
                "timeframe": getattr(request, "timeframe", None),
            },
        )
        return pd.DataFrame()
    except COMMON_EXC as e:
        _log.error(
            "ALPACA_BARS_FETCH_FAILED",
            extra={"symbol": symbol, "context": context, "error": str(e)},
        )
        if _is_minute_timeframe(getattr(request, "timeframe", "")):
            return _ensure_df(
                get_minute_df(
                    symbol,
                    getattr(request, "start", None),
                    getattr(request, "end", None),
                    feed=getattr(request, "feed", None),
                )
            )
        df = http_get_bars(
            symbol,
            str(getattr(request, "timeframe", "")),
            getattr(request, "start", None),
            getattr(request, "end", None),
            feed=getattr(request, "feed", None) or "sip",
        )
        return _ensure_df(df)  # AI-AGENT-REF: HTTP/Yahoo fallback on exception
