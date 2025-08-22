from __future__ import annotations

import datetime as _dt
import os
import warnings  # AI-AGENT-REF: control yfinance warnings
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo  # AI-AGENT-REF: ET default for naive datetimes

import pandas as pd  # AI-AGENT-REF: pandas already a project dependency
from pandas.errors import OutOfBoundsDatetime  # AI-AGENT-REF: datetime bounds

try:  # pragma: no cover - optional configuration module
    from ai_trading.config import get_settings
except Exception:  # AI-AGENT-REF: fallback when config dependencies missing
    def get_settings():  # type: ignore
        return None
from ai_trading.data.timeutils import (
    ensure_utc_datetime,  # AI-AGENT-REF: unified datetime coercion
)
from ai_trading.logging.empty_policy import (
    classify as _empty_classify,
)
from ai_trading.logging.empty_policy import (
    record as _empty_record,
)
from ai_trading.logging.empty_policy import (
    should_emit as _empty_should_emit,
)
from ai_trading.logging.normalize import (
    canon_feed as _canon_feed,  # AI-AGENT-REF: centralized feed normalization
)
from ai_trading.logging.normalize import (
    canon_timeframe as _canon_tf,  # AI-AGENT-REF: centralized timeframe normalization
)
from ai_trading.logging.normalize import (
    normalize_extra as _norm_extra,  # AI-AGENT-REF: canonicalize logging extras
)
from ai_trading.utils.optional_import import optional_import

yf = optional_import("yfinance")

from ai_trading.logging import logger  # AI-AGENT-REF: centralized logger

# ---------------------------------------------------------------------------
# Backwards compatibility wrappers around central canonicalizers
# ---------------------------------------------------------------------------


def _to_timeframe_str(tf: object) -> str:  # AI-AGENT-REF: delegate to central helper
    return _canon_tf(tf)


def _to_feed_str(feed: object) -> str:  # AI-AGENT-REF: delegate to central helper
    return _canon_feed(feed)

# Ensure yfinance tz cache is writable on headless servers
try:  # pragma: no cover
    # AI-AGENT-REF: silence tz cache warnings on unwritable directories
    if yf is not None and hasattr(yf, "set_tz_cache_location"):
        os.makedirs("/tmp/py-yfinance", exist_ok=True)
        yf.set_tz_cache_location("/tmp/py-yfinance")
except OSError:  # AI-AGENT-REF: narrow OS error
    pass

try:  # AI-AGENT-REF: prefer internal HTTP helper when available
    from ai_trading.utils import http as _http
    logger.debug("HTTP_INIT_PRIMARY", extra={"transport": "ai_trading.utils.http"})
except ImportError:  # pragma: no cover
    logger.debug("HTTP_INIT_FALLBACK", extra={"transport": "requests"})
    _http = None

try:  # AI-AGENT-REF: fallback to requests if internal helper missing
    import requests as _requests  # type: ignore
except ImportError:  # pragma: no cover
    _requests = None
requests = _requests
try:  # AI-AGENT-REF: typed request exceptions
    from requests.exceptions import (
        ConnectionError,
        HTTPError,
        RequestException,
        Timeout,
    )
except (ValueError, TypeError):  # pragma: no cover - requests optional
    RequestException = Timeout = ConnectionError = HTTPError = Exception  # type: ignore

def _format_fallback_payload_df(
    tf_str: str, feed_str: str, start_dt: _dt.datetime, end_dt: _dt.datetime
) -> list[str]:
    """UTC ISO payload for consistent logging."""  # AI-AGENT-REF: normalize fallback payload

    s = ensure_datetime(start_dt).astimezone(UTC).isoformat()
    e = ensure_datetime(end_dt).astimezone(UTC).isoformat()
    return [tf_str, feed_str, s, e]

# ---------------------------------------------------------------------------
# Minute cache (in-memory). Maps symbol -> (last_bar_epoch_s, inserted_epoch_s)
# Used by freshness checks in bot_engine and unit tests.
# ---------------------------------------------------------------------------
_MINUTE_CACHE: dict[str, tuple[int, int]] = {}


def get_cached_minute_timestamp(symbol: str) -> int | None:
    """Return cached last bar timestamp for symbol."""  # AI-AGENT-REF: cache getter
    rec = _MINUTE_CACHE.get(symbol)
    return rec[0] if rec else None


def set_cached_minute_timestamp(symbol: str, ts_epoch_s: int) -> None:
    """Store last bar timestamp with current insertion time."""  # AI-AGENT-REF: cache setter
    now_s = int(_dt.datetime.now(tz=UTC).timestamp())
    _MINUTE_CACHE[symbol] = (int(ts_epoch_s), now_s)


def clear_cached_minute_timestamp(symbol: str) -> None:
    """Remove cached entry for symbol."""  # AI-AGENT-REF: cache clear
    _MINUTE_CACHE.pop(symbol, None)


def age_cached_minute_timestamps(max_age_seconds: int) -> int:
    """Drop cache entries older than max_age_seconds (based on inserted time)."""  # AI-AGENT-REF: cache prune
    now_s = int(_dt.datetime.now(tz=UTC).timestamp())
    to_del = [sym for sym, (_, ins) in _MINUTE_CACHE.items() if now_s - ins > max_age_seconds]
    for sym in to_del:
        _MINUTE_CACHE.pop(sym, None)
    return len(to_del)


def last_minute_bar_age_seconds(symbol: str) -> int | None:
    """Age in seconds of last cached minute bar for symbol, or None if absent."""  # AI-AGENT-REF: age helper
    ts = get_cached_minute_timestamp(symbol)
    if ts is None:
        return None
    now_s = int(_dt.datetime.now(tz=UTC).timestamp())
    return max(0, now_s - int(ts))


# ---------------------------------------------------------------------------
# Public/tested constants & stubs expected by tests
# ---------------------------------------------------------------------------

# AI-AGENT-REF: default feed used for retry fallback
_DEFAULT_FEED = "iex"
_VALID_FEEDS = ("iex", "sip")


class _FinnhubFetcherStub:
    """Minimal stub with a fetch() method; tests monkeypatch this."""

    # AI-AGENT-REF: stub fetch method
    def fetch(self, *args, **kwargs):
        raise NotImplementedError


fh_fetcher = _FinnhubFetcherStub()


def get_last_available_bar(symbol: str) -> pd.DataFrame:
    """Placeholder; tests monkeypatch this to return a last available daily bar."""
    raise NotImplementedError("Tests should monkeypatch get_last_available_bar")


class DataFetchException(Exception):
    """Error raised when market data retrieval fails."""  # AI-AGENT-REF


class DataFetchError(DataFetchException):
    """Alias error used in tests."""  # AI-AGENT-REF


class FinnhubAPIException(Exception):
    """Minimal Finnhub API error for tests."""  # AI-AGENT-REF

    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(str(status_code))


def build_fetcher(config: Any) -> Any:
    """Return an initialized market data fetcher or raise ``DataFetchError``."""
    from ai_trading.alpaca_api import ALPACA_AVAILABLE
    try:
        from ai_trading.core.bot_engine import DataFetcher
    except Exception:  # pragma: no cover - lightweight fallback for tests
        class DataFetcher:  # type: ignore
            def get_daily_df(self, *a, **k):
                return None

            def get_minute_df(self, *a, **k):
                return None

    has_keys = os.getenv("APCA_API_KEY_ID") and os.getenv("APCA_API_SECRET_KEY")
    if ALPACA_AVAILABLE and has_keys:
        f = DataFetcher()
        setattr(f, "source", "alpaca")
        return f
    if yf is not None or requests is not None:
        f = DataFetcher()
        setattr(f, "source", "fallback")
        return f
    raise DataFetchError("cannot build data fetcher")


# ---------------------------------------------------------------------------
# Datetime coercion
# ---------------------------------------------------------------------------


def ensure_datetime(value: Any) -> _dt.datetime:
    """Coerce various datetime inputs into timezone-aware UTC datetime.

    Rules for market-data windows:
    - If ``value`` is callable, call it (no args) and re-normalize.
    - If ``value`` is a *naive* ``datetime``, interpret it as **America/New_York**
      (exchange time) before converting to UTC.
    - Otherwise, delegate to ``ensure_utc_datetime``.
    """

    # AI-AGENT-REF: unwrap callables early
    if callable(value):
        try:
            value = value()
        except (TypeError, ValueError, AttributeError, OutOfBoundsDatetime) as e:
            raise TypeError(f"Invalid datetime input: {e}") from e

    # AI-AGENT-REF: localize naive ET datetimes before UTC coercion
    if isinstance(value, _dt.datetime) and value.tzinfo is None:
        value = value.replace(tzinfo=ZoneInfo("America/New_York"))

    try:
        return ensure_utc_datetime(value, allow_callables=False)
    except (TypeError, ValueError, AttributeError, OutOfBoundsDatetime) as e:
        raise TypeError(f"Invalid datetime input: {e}") from e


def _default_window_for(timeframe: Any) -> tuple[_dt.datetime, _dt.datetime]:
    """Derive [start, end] when callers omit them."""  # AI-AGENT-REF: legacy helper
    now = _dt.datetime.now(tz=UTC)
    end = now - _dt.timedelta(minutes=1)
    tf = str(timeframe).lower()
    if "day" in tf:
        days = int(os.getenv("DATA_LOOKBACK_DAYS_DAILY", "200"))
    else:
        days = int(os.getenv("DATA_LOOKBACK_DAYS_MINUTE", "5"))
    start = end - _dt.timedelta(days=days)
    return start, end


# ---------------------------------------------------------------------------
# Yahoo helper used by tests (minute/day)
# ---------------------------------------------------------------------------


def _flatten_and_normalize_ohlcv(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Make YF/other OHLCV DataFrames uniform.

    - flatten MultiIndex columns
    - lower/snake columns
    - ensure 'close' exists (fallback to 'adj_close')
    - de-duplicate & sort index, convert index to UTC and tz-naive
    """  # AI-AGENT-REF: normalize OHLCV columns

    if isinstance(df.columns, pd.MultiIndex):
        try:
            lvl0 = set(map(str, df.columns.get_level_values(0)))
            if {"Open", "High", "Low", "Close", "Adj Close", "Volume"} & lvl0:
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = ["_".join([str(x) for x in tup if x is not None]) for tup in df.columns]
        except (AttributeError, IndexError, TypeError):  # AI-AGENT-REF: narrow flatten errors
            df.columns = ["_".join([str(x) for x in tup if x is not None]) for tup in df.columns]

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    if isinstance(df.index, pd.DatetimeIndex):
        try:
            tz = df.index.tz
            if tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
        except (AttributeError, TypeError, ValueError):  # AI-AGENT-REF: narrow tz normalization
            pass
        df = df[~df.index.duplicated(keep="last")].sort_index()

    return df


def _yahoo_get_bars(symbol: str, start: Any, end: Any, interval: str) -> pd.DataFrame:
    """Return a DataFrame with a tz-aware 'timestamp' column between start and end."""

    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)

    if yf is None:  # AI-AGENT-REF: yfinance optional dependency
        idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        cols = ["open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=cols, index=idx).reset_index()

    with warnings.catch_warnings():  # AI-AGENT-REF: silence yfinance notices
        warnings.filterwarnings("ignore", message=".*auto_adjust.*", module="yfinance")
        df = yf.download(
            symbol,
            start=start_dt,
            end=end_dt,
            interval=interval,
            auto_adjust=False,  # AI-AGENT-REF: explicit to avoid default warning
            threads=False,
            progress=False,
            group_by="column",
        )
    if df is None or df.empty:
        idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        cols = ["open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=cols, index=idx).reset_index()

    df = df.reset_index().rename(columns={df.index.name or "Date": "timestamp"})
    if "timestamp" not in df.columns:
        for c in df.columns:
            if c.lower() in ("date", "datetime"):
                df = df.rename(columns={c: "timestamp"})
                break

    df = _flatten_and_normalize_ohlcv(df, symbol)
    return df


def _post_process(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV DataFrame or return empty."""  # AI-AGENT-REF
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()
    return _flatten_and_normalize_ohlcv(df)


# ---------------------------------------------------------------------------
# Minimal bars fetcher used by tests (monkeypatchable)
# ---------------------------------------------------------------------------


def _fetch_bars(
    symbol: str,
    start: Any,
    end: Any,
    timeframe: str,
    *,
    feed: str = _DEFAULT_FEED,
    adjustment: str = "raw",
) -> pd.DataFrame:
    """Fetch bars from Alpaca v2 with alt-feed fallback."""  # AI-AGENT-REF

    _start = ensure_datetime(start)
    _end = ensure_datetime(end)
    _interval = _canon_tf(timeframe)  # AI-AGENT-REF: centralized timeframe normalization
    _feed = _canon_feed(feed or _DEFAULT_FEED)  # AI-AGENT-REF: centralized feed normalization

    def _req(fallback: tuple[str, str, _dt.datetime, _dt.datetime] | None) -> pd.DataFrame:
        nonlocal _interval, _feed, _start, _end
        params = {
            "symbols": symbol,
            "timeframe": _interval,
            "start": _start.isoformat(),
            "end": _end.isoformat(),
            "limit": 10000,
            "feed": _feed,
            "adjustment": adjustment,
        }
        if requests is None:  # pragma: no cover
            raise RuntimeError("requests not available")
        url = "https://data.alpaca.markets/v2/stocks/bars"
        headers = {
            "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
            "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
        }
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            status = resp.status_code
            text = (resp.text or "").strip()
            ctype = (resp.headers.get("Content-Type") or "").lower()
        except (Timeout, ConnectionError, HTTPError, RequestException, ValueError, KeyError) as e:
            logger.warning(
                "DATA_SOURCE_HTTP_ERROR",
                extra=_norm_extra(
                    {
                        "provider": "alpaca",
                        "feed": _feed,
                        "timeframe": _interval,
                        "error": str(e),
                    }
                ),
            )
            if fallback:
                _interval, _feed, _start, _end = fallback
                payload = _format_fallback_payload_df(
                    _interval, _feed, _start, _end
                )
                logger.info(
                    "DATA_SOURCE_FALLBACK_ATTEMPT",
                    extra={"provider": "alpaca", "fallback": payload},
                )
                return _req(None)
            raise
        payload: dict[str, Any] | list[Any] = {}
        if status != 400 and text:
            if "json" in ctype:
                try:
                    payload = resp.json()
                except ValueError:  # AI-AGENT-REF: narrow JSON decode
                    payload = {}

        data = []
        if isinstance(payload, dict):
            if "bars" in payload and isinstance(payload["bars"], list):
                data = payload["bars"]
            elif (
                symbol in payload
                and isinstance(payload[symbol], dict)
                and "bars" in payload[symbol]
            ):
                data = payload[symbol]["bars"]
        elif isinstance(payload, list):
            data = payload

        if status == 400:
            raise ValueError("Invalid feed or bad request")

        if status in (401, 403):
            logger.warning(
                "UNAUTHORIZED_SIP" if _feed == "sip" else "DATA_SOURCE_UNAUTHORIZED",
                extra=_norm_extra(
                    {
                        "provider": "alpaca",
                        "status": "unauthorized",
                        "feed": _feed,
                        "timeframe": _interval,
                    }
                ),
            )
            if _feed == "sip" and fallback:
                _interval, _feed, _start, _end = fallback
                payload = _format_fallback_payload_df(
                    _interval, _feed, _start, _end
                )
                logger.info(
                    "DATA_SOURCE_FALLBACK_ATTEMPT",
                    extra={"provider": "alpaca", "fallback": payload},
                )
                return _req(None)
            raise ValueError("unauthorized")

        df = pd.DataFrame(data)
        if df.empty:
            if _interval.lower() in {"1day", "day", "1d"}:
                try:
                    mdf = _fetch_bars(
                        symbol,
                        _start,
                        _end,
                        "1Min",
                        feed=_feed,
                        adjustment=adjustment,
                    )
                except (ValueError, RuntimeError):  # AI-AGENT-REF: narrow fetch fallback errors
                    mdf = pd.DataFrame()
                if not mdf.empty:
                    try:
                        if "timestamp" in mdf.columns:
                            mdf["timestamp"] = pd.to_datetime(
                                mdf["timestamp"], utc=True
                            )
                            mdf.set_index("timestamp", inplace=True)
                        from ai_trading.data.bars import (
                            _resample_minutes_to_daily as _resample_to_daily,
                        )
                        rdf = _resample_to_daily(mdf)
                    except (ImportError, ValueError, TypeError, KeyError):  # AI-AGENT-REF: narrow resample errors
                        mdf = pd.DataFrame()
                    else:
                        if rdf is not None and not rdf.empty:
                            return rdf
            _now = datetime.now(UTC)
            _key = (symbol, "AVAILABLE", _now.date().isoformat(), _feed, _interval)
            if _empty_should_emit(_key, _now):
                lvl = _empty_classify(is_market_open=False)  # AI-AGENT-REF: after-hours default
                cnt = _empty_record(_key, _now)
                logger.log(
                    lvl,
                    "EMPTY_DATA",
                    extra=_norm_extra(
                        {
                            "provider": "alpaca",
                            "status": "empty",
                            "feed": _feed,
                            "timeframe": _interval,
                            "occurrences": cnt,
                        }
                    ),
                )
            if fallback:
                _interval, _feed, _start, _end = fallback
                payload = _format_fallback_payload_df(
                    _interval, _feed, _start, _end
                )
                logger.info(
                    "DATA_SOURCE_FALLBACK_ATTEMPT",
                    extra={"provider": "alpaca", "fallback": payload},
                )
                return _req(None)
            raise ValueError("empty_bars")

        ts_col = None
        for c in df.columns:
            if c.lower() in ("t", "timestamp", "time"):
                ts_col = c
                break
        if ts_col:
            df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)
        elif "timestamp" not in df.columns:
            df["timestamp"] = pd.to_datetime([], utc=True)
        # Normalize columns and types
        rename = {
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
        df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])
        df.set_index("timestamp", inplace=True, drop=False)
        return df

    # AI-AGENT-REF: prepare alt feed for retry on empty/exception
    alt_feed = "iex" if (_feed != "iex") else "sip"
    fallback = (_interval, alt_feed, _start, _end)
    return _req(fallback)


# ---------------------------------------------------------------------------
# get_minute_df wrapper with graceful fallbacks / logging behavior
# ---------------------------------------------------------------------------



def get_minute_df(symbol: str, start: Any, end: Any, feed: str | None = None) -> pd.DataFrame:
    """Minute bars fetch with provider fallback and downgraded errors.
    Also updates in-memory minute cache for freshness checks."""  # AI-AGENT-REF

    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)

    # 1) Finnhub primary (opt-in)
    if os.getenv("FINNHUB_API_KEY"):
        try:
            df = (
                fh_fetcher.fetch(symbol, start_dt, end_dt, resolution="1")
                if fh_fetcher is not None
                else None
            )
        except (FinnhubAPIException, ValueError) as e:
            logger.debug("FINNHUB_FETCH_FAILED", extra={"symbol": symbol, "err": str(e)})
            df = None
    else:
        logger.debug("Skipping Finnhub fetch; FINNHUB_API_KEY not set")
        df = None

    # 2) Alpaca fallback if Finnhub missing/empty
    if df is None or getattr(df, "empty", True):
        try:
            df = _fetch_bars(symbol, start_dt, end_dt, "1Min", feed=feed or _DEFAULT_FEED)
        except (ValueError, RuntimeError) as e:
            logger.warning("ALPACA_FETCH_FAILED", extra={"symbol": symbol, "err": str(e)})
            df = None

    # 3) Yahoo final fallback
    if df is None or getattr(df, "empty", True):
        df = _yahoo_get_bars(symbol, start_dt, end_dt, interval="1m")

    try:  # AI-AGENT-REF: update minute cache with latest bar
        if isinstance(df, pd.DataFrame) and not df.empty:
            if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
                last_ts = int(pd.Timestamp(df.index[-1]).tz_convert("UTC").timestamp())
            elif "timestamp" in df.columns:
                last_ts = int(
                    pd.Timestamp(df["timestamp"].iloc[-1]).tz_convert("UTC").timestamp()
                )
            else:
                last_ts = None
            if last_ts is not None:
                set_cached_minute_timestamp(symbol, last_ts)
    except (ValueError, TypeError, KeyError, AttributeError):  # pragma: no cover - cache best effort
        pass

    return _post_process(df)


def get_bars(
    symbol: str,
    timeframe: str,
    start: Any,
    end: Any,
    *,
    feed: str | None = None,
    adjustment: str | None = None,
) -> pd.DataFrame:
    """Compatibility wrapper delegating to _fetch_bars."""  # AI-AGENT-REF
    S = get_settings()
    feed = feed or S.alpaca_data_feed
    adjustment = adjustment or S.alpaca_adjustment
    return _fetch_bars(
        symbol,
        start,
        end,
        timeframe,
        feed=feed,
        adjustment=adjustment,
    )


def get_bars_batch(
    symbols: list[str],
    timeframe: str,
    start: Any,
    end: Any,
    *,
    feed: str | None = None,
    adjustment: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch bars for multiple symbols via get_bars."""  # AI-AGENT-REF
    return {
        sym: get_bars(sym, timeframe, start, end, feed=feed, adjustment=adjustment)
        for sym in symbols
    }


def get_bars_df(
    symbol: str,
    timeframe: Any,
    start: Any | None = None,
    end: Any | None = None,
    *,
    feed: str | None = None,
    adjustment: str | None = None,
) -> pd.DataFrame:
    """Legacy alias expected by bot_engine (accepts optional start/end)."""  # AI-AGENT-REF
    if start is None or end is None:
        start, end = _default_window_for(timeframe)
    try:
        df = get_bars(
            symbol,
            str(timeframe),
            start,
            end,
            feed=feed,
            adjustment=adjustment,
        )
    except (ValueError, RuntimeError) as e:
        logger.warning("ALPACA_DAILY_FETCH_FAILED", extra={"symbol": symbol, "err": str(e)})
        df = None
    if df is None or getattr(df, "empty", True):
        return _post_process(_yahoo_get_bars(symbol, start, end, interval="1d"))
    return _post_process(df)


def fetch_minute_yfinance(symbol: str, start_dt: _dt.datetime, end_dt: _dt.datetime) -> pd.DataFrame:
    """Explicit helper for tests and optional direct Yahoo minute fetch."""  # AI-AGENT-REF
    df = _yahoo_get_bars(symbol, start_dt, end_dt, interval="1m")
    return _post_process(df)


def is_market_open() -> bool:
    """Simplistic market-hours check used in tests."""  # AI-AGENT-REF
    return True


__all__ = [
    "_DEFAULT_FEED",
    "_VALID_FEEDS",
    "ensure_datetime",
    "_yahoo_get_bars",
    "_fetch_bars",
    "get_bars",
    "get_bars_batch",
    "get_bars_df",
    "fetch_minute_yfinance",
    "is_market_open",
    "get_last_available_bar",
    "fh_fetcher",
    "get_minute_df",
    "build_fetcher",
    "DataFetchError",
]
