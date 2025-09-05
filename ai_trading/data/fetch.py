from __future__ import annotations
import datetime as _dt
import os
import warnings
import time
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo
import importlib
from ai_trading.utils.lazy_imports import load_pandas


from ai_trading.data.timeutils import ensure_utc_datetime
from ai_trading.data.market_calendar import is_trading_day, rth_session_utc
from ai_trading.logging.empty_policy import classify as _empty_classify
from ai_trading.logging.empty_policy import record as _empty_record
from ai_trading.logging.empty_policy import should_emit as _empty_should_emit
from ai_trading.logging.normalize import canon_feed as _canon_feed
from ai_trading.logging.normalize import canon_timeframe as _canon_tf
from ai_trading.logging.normalize import normalize_extra as _norm_extra
from ai_trading.logging import (
    log_backup_provider_used,
    log_empty_retries_exhausted,
    log_fetch_attempt,
    log_finnhub_disabled,
    warn_finnhub_disabled_no_data,
    logger,
)
from ai_trading.config.management import MAX_EMPTY_RETRIES
from ai_trading.config.settings import provider_priority, max_data_fallbacks
from ai_trading.data.empty_bar_backoff import (
    _SKIPPED_SYMBOLS,
    mark_success,
    record_attempt,
)
from ai_trading.data.metrics import metrics, provider_fallback
from ai_trading.net.http import HTTPSession, get_http_session
from ai_trading.utils.http import clamp_request_timeout

# Lightweight indirection to support tests monkeypatching `data_fetcher.get_settings`
def get_settings():  # pragma: no cover - simple alias for tests
    from ai_trading.config.settings import get_settings as _get

    return _get()

# Module-level session reused across requests
_HTTP_SESSION: HTTPSession = get_http_session()


# Optional dependency placeholders
pd: Any | None = None


class _RequestsModulePlaceholder:
    get = None


requests: Any = _RequestsModulePlaceholder()


class _YFinancePlaceholder:
    download = None


yf: Any = _YFinancePlaceholder()


class RequestException(Exception):
    """Fallback request exception when ``requests`` is missing."""


class Timeout(RequestException):
    pass


class ConnectionError(RequestException):
    pass


class HTTPError(RequestException):
    pass


def _incr(metric: str, *, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
    """Increment a metric via the lightweight data.metrics hook.

    Tests monkeypatch ``ai_trading.data.fetch.metrics.incr`` directly, so route
    through the module-level import rather than the heavier monitoring stack.
    """
    try:
        metrics.incr(metric, value=value, tags=tags)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - metrics optional
        pass


def _to_timeframe_str(tf: object) -> str:
    return _canon_tf(tf)


def _to_feed_str(feed: object) -> str:
    return _canon_feed(feed)


class DataFetchError(Exception):
    """Error raised when market data retrieval fails."""  # AI-AGENT-REF: stable public symbol


# Backwards compat alias
DataFetchException = DataFetchError


class EmptyBarsError(DataFetchError, ValueError):
    """Raised when a data provider returns no bars for a request."""


class FinnhubAPIException(Exception):
    """Minimal Finnhub API error for tests."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(str(status_code))


def ensure_datetime(value: Any) -> _dt.datetime:
    """Coerce various datetime inputs into timezone-aware UTC datetime.

    Rules for market-data windows:
    - If ``value`` is callable, call it (no args) and re-normalize.
    - If ``value`` is a *naive* ``datetime``, interpret it as **America/New_York**
      (exchange time) before converting to UTC.
    - Otherwise, delegate to ``ensure_utc_datetime``.
    """
    pd_mod = _ensure_pandas()
    out_of_bounds = ()
    if pd_mod is not None:
        try:
            out_of_bounds = (pd_mod.errors.OutOfBoundsDatetime,)
        except Exception:
            out_of_bounds = ()
    if callable(value):
        try:
            value = value()
        except (*out_of_bounds, TypeError, ValueError, AttributeError) as e:  # type: ignore[misc]
            raise TypeError(f"Invalid datetime input: {e}") from e
    if isinstance(value, _dt.datetime) and value.tzinfo is None:
        value = value.replace(tzinfo=ZoneInfo("America/New_York"))
    try:
        return ensure_utc_datetime(value, allow_callables=False)
    except (*out_of_bounds, TypeError, ValueError, AttributeError) as e:  # type: ignore[misc]
        raise TypeError(f"Invalid datetime input: {e}") from e


def _format_fallback_payload_df(tf_str: str, feed_str: str, start_dt: _dt.datetime, end_dt: _dt.datetime) -> list[str]:
    """UTC ISO payload for consistent logging."""
    s = ensure_datetime(start_dt).astimezone(UTC).isoformat()
    e = ensure_datetime(end_dt).astimezone(UTC).isoformat()
    return [tf_str, feed_str, s, e]


def bars_time_window_day(days: int = 10, *, end: _dt.datetime | None = None) -> tuple[_dt.datetime, _dt.datetime]:
    """Return start/end datetimes covering ``days`` full days inclusively.

    ``end`` defaults to the current UTC time. The ``start`` is normalized to
    midnight UTC ``days`` days before ``end`` so that the entire first day is
    included in the range. The returned ``start`` and ``end`` are timezone-aware
    in UTC and satisfy ``(end - start).days == days``.
    """

    end_dt = ensure_datetime(end or _dt.datetime.now(tz=UTC))
    start_dt = (end_dt - _dt.timedelta(days=days)).astimezone(UTC)
    start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_dt, end_dt


_MINUTE_CACHE: dict[str, tuple[int, int]] = {}

# Track consecutive empty-bar responses per (symbol, timeframe) pair to avoid
# repeated fetch noise and allow skipping per interval
_EMPTY_BAR_COUNTS: dict[tuple[str, str], int] = {}
# Track consecutive error="empty" responses specifically from the IEX feed to
# allow proactive SIP fallback on subsequent requests.
_IEX_EMPTY_COUNTS: dict[tuple[str, str], int] = {}
_IEX_EMPTY_THRESHOLD = 1
_EMPTY_BAR_THRESHOLD = 3
_EMPTY_BAR_MAX_RETRIES = MAX_EMPTY_RETRIES
_FETCH_BARS_MAX_RETRIES = int(os.getenv("FETCH_BARS_MAX_RETRIES", "5"))
# Configurable backoff parameters for retry logic
_FETCH_BARS_BACKOFF_BASE = float(os.getenv("FETCH_BARS_BACKOFF_BASE", "2"))
_FETCH_BARS_BACKOFF_CAP = float(os.getenv("FETCH_BARS_BACKOFF_CAP", "5"))
_ENABLE_HTTP_FALLBACK = os.getenv("ENABLE_HTTP_FALLBACK", "0").strip().lower() not in {
    "0",
    "false",
}

# Track fallback usage to avoid repeated Alpaca requests for the same window
_FALLBACK_WINDOWS: set[tuple[str, str, int, int]] = set()

# Track consecutive empty Alpaca responses across all symbols to temporarily
# disable Alpaca fetching when upstream repeatedly returns empty payloads.
_ALPACA_DISABLE_THRESHOLD = 3
_alpaca_empty_streak = 0
_alpaca_disabled_until: _dt.datetime | None = None


def _fallback_key(symbol: str, timeframe: str, start: _dt.datetime, end: _dt.datetime) -> tuple[str, str, int, int]:
    return (symbol, timeframe, int(start.timestamp()), int(end.timestamp()))


def _mark_fallback(symbol: str, timeframe: str, start: _dt.datetime, end: _dt.datetime) -> None:
    provider = getattr(get_settings(), "backup_data_provider", "yahoo")
    log_backup_provider_used(
        provider,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
    )
    _FALLBACK_WINDOWS.add(_fallback_key(symbol, timeframe, start, end))


def _used_fallback(symbol: str, timeframe: str, start: _dt.datetime, end: _dt.datetime) -> bool:
    return _fallback_key(symbol, timeframe, start, end) in _FALLBACK_WINDOWS

def _symbol_exists(symbol: str) -> bool:
    """Return True if the symbol exists according to Alpaca or the local list."""
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/meta"
    try:
        resp = _HTTP_SESSION.get(url, timeout=clamp_request_timeout(2.0))
        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                data = {}
            if str(data.get("symbol", "")).upper() == symbol.upper():
                return True
        elif resp.status_code == 404:
            return False
    except Exception:
        pass
    path = os.getenv("AI_TRADING_TICKERS_CSV") or os.getenv("TICKERS_FILE_PATH")
    if not path:
        try:
            from importlib.resources import files as pkg_files
            p = pkg_files("ai_trading.data").joinpath("tickers.csv")
            path = str(p) if p.is_file() else os.path.join(os.getcwd(), "tickers.csv")
        except Exception:
            path = os.path.join(os.getcwd(), "tickers.csv")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return any(line.strip().upper() == symbol.upper() for line in fh)
    except OSError:
        return False

_VALID_FEEDS = {"iex", "sip"}
_VALID_ADJUSTMENTS = {"raw", "split", "dividend", "all"}
_VALID_TIMEFRAMES = {"1Min", "5Min", "15Min", "1Hour", "1Day"}


def _outside_market_hours(start: _dt.datetime, end: _dt.datetime) -> bool:
    """Return True if both ``start`` and ``end`` fall outside market hours."""
    try:
        from ai_trading.utils.base import is_market_open as _is_open

        return not (_is_open(start) or _is_open(end))
    except Exception:  # pragma: no cover - fallback to retrying
        return False


def _validate_alpaca_params(
    start: _dt.datetime, end: _dt.datetime, timeframe: str, feed: str, adjustment: str
) -> None:
    """Raise ``ValueError`` if request parameters are invalid."""
    if start >= end:
        raise ValueError("invalid_time_window")
    if feed not in _VALID_FEEDS:
        raise ValueError("invalid_feed")
    if adjustment not in _VALID_ADJUSTMENTS:
        raise ValueError("invalid_adjustment")
    if timeframe not in _VALID_TIMEFRAMES:
        raise ValueError("invalid_timeframe")


def _window_has_trading_session(start: _dt.datetime, end: _dt.datetime) -> bool:
    """Return True if any trading session overlaps the ``start``/``end`` window."""
    day = start.date()
    end_day = end.date()
    while day <= end_day:
        if is_trading_day(day):
            open_dt, close_dt = rth_session_utc(day)
            if end > open_dt and start < close_dt:
                return True
        day += _dt.timedelta(days=1)
    return False


def _has_alpaca_keys() -> bool:
    """Return True if Alpaca API credentials appear configured."""
    return bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))


def get_cached_minute_timestamp(symbol: str) -> int | None:
    """Return cached last bar timestamp for symbol."""
    rec = _MINUTE_CACHE.get(symbol)
    return rec[0] if rec else None


def set_cached_minute_timestamp(symbol: str, ts_epoch_s: int) -> None:
    """Store last bar timestamp with current insertion time."""
    now_s = int(_dt.datetime.now(tz=UTC).timestamp())
    _MINUTE_CACHE[symbol] = (int(ts_epoch_s), now_s)


def clear_cached_minute_timestamp(symbol: str) -> None:
    """Remove cached entry for symbol."""
    _MINUTE_CACHE.pop(symbol, None)


def age_cached_minute_timestamps(max_age_seconds: int) -> int:
    """Drop cache entries older than max_age_seconds (based on inserted time)."""
    now_s = int(_dt.datetime.now(tz=UTC).timestamp())
    to_del = [sym for sym, (_, ins) in _MINUTE_CACHE.items() if now_s - ins > max_age_seconds]
    for sym in to_del:
        _MINUTE_CACHE.pop(sym, None)
    return len(to_del)


def last_minute_bar_age_seconds(symbol: str) -> int | None:
    """Age in seconds of last cached minute bar for symbol, or None if absent."""
    ts = get_cached_minute_timestamp(symbol)
    if ts is None:
        return None
    now_s = int(_dt.datetime.now(tz=UTC).timestamp())
    return max(0, now_s - int(ts))


_DEFAULT_FEED = "iex"
_SIP_UNAUTHORIZED = os.getenv("ALPACA_SIP_UNAUTHORIZED", "").strip().lower() in {
    "1",
    "true",
    "yes",
}
_ALLOW_SIP = os.getenv("ALPACA_ALLOW_SIP", "").strip().lower() in {
    "1",
    "true",
    "yes",
}
_SIP_DISALLOWED_WARNED = False
_SIP_PRECHECK_DONE = False


def _sip_fallback_allowed(session: HTTPSession, headers: dict[str, str], timeframe: str) -> bool:
    """Return True if SIP fallback should be attempted."""
    global _SIP_UNAUTHORIZED, _SIP_DISALLOWED_WARNED, _SIP_PRECHECK_DONE
    if not _ALLOW_SIP:
        if not _SIP_DISALLOWED_WARNED:
            logger.warning(
                "SIP_DISABLED",
                extra=_norm_extra({"provider": "alpaca", "feed": "sip", "timeframe": timeframe}),
            )
            _SIP_DISALLOWED_WARNED = True
        return False
    if _SIP_UNAUTHORIZED:
        return False
    if _SIP_PRECHECK_DONE:
        return True
    _SIP_PRECHECK_DONE = True
    url = "https://data.alpaca.markets/v2/stocks/bars"
    params = {"symbols": "AAPL", "timeframe": timeframe, "limit": 1, "feed": "sip"}
    use_session_get = hasattr(session, "__dict__") and ("get" in getattr(session, "__dict__", {}))
    try:
        if use_session_get:
            resp = session.get(url, params=params, headers=headers, timeout=clamp_request_timeout(5))
        else:
            resp = requests.get(url, params=params, headers=headers, timeout=clamp_request_timeout(5))
    except Exception as e:  # pragma: no cover - best effort
        logger.debug(
            "SIP_PRECHECK_FAILED",
            extra=_norm_extra({"provider": "alpaca", "feed": "sip", "timeframe": timeframe, "error": str(e)}),
        )
        return True
    if getattr(resp, "status_code", None) in (401, 403):
        _incr("data.fetch.unauthorized", value=1.0, tags={"provider": "alpaca", "feed": "sip", "timeframe": timeframe})
        metrics.unauthorized += 1
        _SIP_UNAUTHORIZED = True
        os.environ["ALPACA_SIP_UNAUTHORIZED"] = "1"
        logger.warning(
            "UNAUTHORIZED_SIP",
            extra=_norm_extra({"provider": "alpaca", "status": "precheck", "feed": "sip", "timeframe": timeframe}),
        )
        return False
    return True


class _FinnhubFetcherStub:
    """Minimal stub with a fetch() method; tests monkeypatch this."""

    is_stub = True

    def fetch(self, *args, **kwargs):
        raise NotImplementedError


fh_fetcher = _FinnhubFetcherStub()


def get_last_available_bar(symbol: str) -> pd.DataFrame:
    """Placeholder; tests monkeypatch this to return a last available daily bar."""
    raise NotImplementedError("Tests should monkeypatch get_last_available_bar")


def _default_window_for(timeframe: Any) -> tuple[_dt.datetime, _dt.datetime]:
    """Derive [start, end] when callers omit them."""
    now = _dt.datetime.now(tz=UTC)
    end = now - _dt.timedelta(minutes=1)
    tf = str(timeframe).lower()
    if "day" in tf:
        days = int(os.getenv("DATA_LOOKBACK_DAYS_DAILY", "200"))
    else:
        days = int(os.getenv("DATA_LOOKBACK_DAYS_MINUTE", "5"))
    start = end - _dt.timedelta(days=days)
    return (start, end)


def _flatten_and_normalize_ohlcv(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Make YF/other OHLCV DataFrames uniform.

    - flatten MultiIndex columns
    - lower/snake columns
    - ensure 'close' exists (fallback to 'adj_close')
    - de-duplicate & sort index, convert index to UTC and tz-naive
    """
    pd = _ensure_pandas()
    if pd is None:
        return []  # type: ignore[return-value]
    if isinstance(df.columns, pd.MultiIndex):
        try:
            lvl0 = set(map(str, df.columns.get_level_values(0)))
            if {"Open", "High", "Low", "Close", "Adj Close", "Volume"} & lvl0:
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = ["_".join([str(x) for x in tup if x is not None]) for tup in df.columns]
        except (AttributeError, IndexError, TypeError):
            df.columns = ["_".join([str(x) for x in tup if x is not None]) for tup in df.columns]
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            tz = df.index.tz
            if tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
        except (AttributeError, TypeError, ValueError):
            pass
        df = df[~df.index.duplicated(keep="last")].sort_index()
    if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            df[col] = pd.Series(dtype="float64")
    return df


def _yahoo_get_bars(symbol: str, start: Any, end: Any, interval: str) -> pd.DataFrame:
    """Return a DataFrame with a tz-aware 'timestamp' column between start and end."""
    pd = _ensure_pandas()
    yf = _ensure_yfinance()
    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)
    if pd is None:
        return []  # type: ignore[return-value]
    if getattr(yf, "download", None) is None:
        idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        cols = ["open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=cols, index=idx).reset_index()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*auto_adjust.*", module="yfinance")
        df = yf.download(
            symbol,
            start=start_dt,
            end=end_dt,
            interval=interval,
            auto_adjust=True,
            threads=False,
            progress=False,
            group_by="column",
        )
    if df is None or df.empty:
        metrics.empty_payload += 1
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


def _backup_get_bars(symbol: str, start: Any, end: Any, interval: str) -> pd.DataFrame:
    """Route to configured backup provider or return empty DataFrame."""
    provider = getattr(get_settings(), "backup_data_provider", "yahoo")
    if provider == "yahoo":
        logger.info("USING_BACKUP_PROVIDER", extra={"provider": provider, "symbol": symbol})
        return _yahoo_get_bars(symbol, start, end, interval)
    pd_local = _ensure_pandas()
    if provider in ("", "none"):
        logger.info("BACKUP_PROVIDER_DISABLED", extra={"symbol": symbol})
    else:
        logger.warning("UNKNOWN_BACKUP_PROVIDER", extra={"provider": provider, "symbol": symbol})
    if pd_local is None:
        return []  # type: ignore[return-value]
    idx = pd_local.DatetimeIndex([], tz="UTC", name="timestamp")
    cols = ["open", "high", "low", "close", "volume"]
    return pd_local.DataFrame(columns=cols, index=idx).reset_index()


def _post_process(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV DataFrame or return empty."""
    pd = _ensure_pandas()
    if pd is None:
        return []  # type: ignore[return-value]
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()
    return _flatten_and_normalize_ohlcv(df)


def _ensure_http_client():
    try:
        from importlib import import_module

        client = import_module("ai_trading.utils.http")
        logger.debug("HTTP_INIT_PRIMARY", extra={"transport": "ai_trading.utils.http"})
        return client
    except ImportError:  # pragma: no cover - optional dependency
        logger.debug("HTTP_INIT_FALLBACK", extra={"transport": "requests"})
        return None


def _ensure_pandas():
    global pd
    if pd is None:
        try:
            pd = load_pandas()
        except Exception:  # pragma: no cover - optional dependency
            pd = None
    return pd


def _ensure_yfinance():
    global yf
    if getattr(yf, "download", None) is None:
        try:
            import yfinance as _yf  # type: ignore

            try:
                if hasattr(_yf, "set_tz_cache_location"):
                    os.makedirs("/tmp/py-yfinance", exist_ok=True)
                    _yf.set_tz_cache_location("/tmp/py-yfinance")
            except OSError:
                pass
            yf = _yf
        except ImportError:  # pragma: no cover - optional dependency
            yf = _YFinancePlaceholder()
            logger.info("YFINANCE_MISSING", extra={"hint": "pip install yfinance"})
    return yf


def _ensure_requests():
    global requests, ConnectionError, HTTPError, RequestException, Timeout
    if getattr(requests, "get", None) is None:
        try:
            import requests as _requests  # type: ignore
            from requests.exceptions import (
                ConnectionError as _ConnectionError,
                HTTPError as _HTTPError,
                RequestException as _RequestException,
                Timeout as _Timeout,
            )

            requests = _requests
            ConnectionError = _ConnectionError
            HTTPError = _HTTPError
            RequestException = _RequestException
            Timeout = _Timeout
        except Exception:  # pragma: no cover - optional dependency
            requests = _RequestsModulePlaceholder()
    return requests


def _parse_bars(symbol: str, content_type: str, body: bytes) -> pd.DataFrame:
    """Parse raw bar data into a normalized DataFrame.

    Supports a minimal subset of JSON or CSV payloads.  Raises
    ``DataFetchError`` when parsing fails or when ``pandas`` is unavailable.
    """
    pd = _ensure_pandas()
    if pd is None:
        raise DataFetchError("pandas not available")
    try:
        if "json" in (content_type or "").lower():
            import json

            data = json.loads(body.decode() or "{}")
            if isinstance(data, dict) and "bars" in data:
                data = data["bars"]
            df = pd.DataFrame(data)
        else:
            import io

            df = pd.read_csv(io.BytesIO(body))
    except Exception as exc:  # pragma: no cover - narrow parsing
        raise DataFetchError(f"parse error: {exc}") from exc
    return _flatten_and_normalize_ohlcv(df, symbol)


def _alpaca_get_bars(
    client: Any,
    symbol: str,
    start: Any,
    end: Any,
    timeframe: str = "1Day",
) -> pd.DataFrame:
    """Fetch bars from an Alpaca-style client."""
    pd = _ensure_pandas()
    if pd is None:
        raise DataFetchError("pandas not available")
    if client is None or not hasattr(client, "get_bars"):
        raise DataFetchError("invalid client")
    try:
        bars = client.get_bars(symbol, start=start, end=end, timeframe=timeframe)
    except Exception as exc:  # pragma: no cover - client variability
        raise DataFetchError(str(exc)) from exc
    if isinstance(bars, pd.DataFrame):
        return _flatten_and_normalize_ohlcv(bars, symbol)
    try:
        return _flatten_and_normalize_ohlcv(pd.DataFrame(bars), symbol)
    except Exception as exc:  # pragma: no cover - conversion failure
        raise DataFetchError(f"invalid bars: {exc}") from exc


def get_daily(symbol: str, start: Any, end: Any) -> pd.DataFrame:
    """Fetch daily bars for ``symbol`` using a Yahoo-style endpoint."""
    pd = _ensure_pandas()
    _ensure_requests()
    if pd is None:
        raise DataFetchError("pandas not available")
    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)
    url = _build_daily_url(symbol, start_dt, end_dt)
    try:
        resp = _HTTP_SESSION.get(url, timeout=clamp_request_timeout(10))
    except Exception as exc:  # pragma: no cover - network variance
        raise DataFetchError(str(exc)) from exc
    if getattr(resp, "status_code", 0) != 200:
        raise DataFetchError(f"http {getattr(resp, 'status_code', 'unknown')}")
    ctype = resp.headers.get("Content-Type", "") if getattr(resp, "headers", None) else ""
    return _parse_bars(symbol, ctype, resp.content)


def fetch_daily_data_async(
    symbols: list[str],
    start: Any,
    end: Any,
    *,
    timeout: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch daily bars for multiple symbols concurrently."""
    pd = _ensure_pandas()
    if pd is None:
        raise DataFetchError("pandas not available")
    http = _ensure_http_client()
    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)
    urls = [_build_daily_url(sym, start_dt, end_dt) for sym in symbols]
    timeout = clamp_request_timeout(timeout)
    results = http.map_get(urls, timeout=timeout)
    out: dict[str, pd.DataFrame] = {}
    for sym, (res, err) in zip(symbols, results):
        if err or res is None:
            raise DataFetchError(str(err))
        _, status, body = res
        if status != 200:
            raise DataFetchError(f"http {status}")
        out[sym] = _parse_bars(sym, "application/json", body)
    return out


# Singleton holder for DataFetcher instances
_FETCHER_SINGLETON: Any | None = None


def build_fetcher(config: Any):
    """Return a market data fetcher with safe fallbacks."""
    global _FETCHER_SINGLETON
    if _FETCHER_SINGLETON is not None:
        return _FETCHER_SINGLETON
    try:
        from ai_trading.alpaca_api import ALPACA_AVAILABLE
    except Exception:  # pragma: no cover - optional dependency
        ALPACA_AVAILABLE = False

    bot_mod = importlib.import_module("ai_trading.core.bot_engine")
    DataFetcher = bot_mod.DataFetcher
    _ensure_http_client()
    yf_mod = _ensure_yfinance()
    req_mod = _ensure_requests()

    alpaca_ok = bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))
    has_keys = alpaca_ok
    if ALPACA_AVAILABLE and has_keys:
        logger.info("DATA_FETCHER_BUILD", extra={"source": "alpaca"})
        fetcher = DataFetcher()
        setattr(fetcher, "source", "alpaca")
        _FETCHER_SINGLETON = fetcher
        return fetcher
    if getattr(yf_mod, "download", None) is not None and getattr(req_mod, "get", None) is not None:
        logger.info("DATA_FETCHER_BUILD", extra={"source": "yfinance"})
        fetcher = DataFetcher()
        setattr(fetcher, "source", "yfinance")
        _FETCHER_SINGLETON = fetcher
        return fetcher
    if getattr(req_mod, "get", None) is not None:
        logger.warning("DATA_FETCHER_BUILD_FALLBACK", extra={"source": "yahoo-requests"})
        fetcher = DataFetcher()
        setattr(fetcher, "source", "fallback")
        _FETCHER_SINGLETON = fetcher
        return fetcher
    logger.error("DATA_FETCHER_UNAVAILABLE", extra={"reason": "no deps"})
    raise DataFetchError("No market data source available")


def _fetch_bars(
    symbol: str, start: Any, end: Any, timeframe: str, *, feed: str = _DEFAULT_FEED, adjustment: str = "raw"
) -> pd.DataFrame:
    """Fetch bars from Alpaca v2 with alt-feed fallback."""
    pd = _ensure_pandas()
    _ensure_requests()
    if pd is None:
        raise RuntimeError("pandas not available")
    _start = ensure_datetime(start)
    _end = ensure_datetime(end)
    # Normalize timestamps to the minute to avoid querying empty slices
    _start = _start.replace(second=0, microsecond=0)
    _end = _end.replace(second=0, microsecond=0)
    _interval = _canon_tf(timeframe)
    _feed = _canon_feed(feed or _DEFAULT_FEED)
    _validate_alpaca_params(_start, _end, _interval, _feed, adjustment)
    try:
        if not _window_has_trading_session(_start, _end):
            raise ValueError("window_no_trading_session")
    except ValueError as e:
        if "window_no_trading_session" in str(e):
            tf_key = (symbol, _interval)
            _SKIPPED_SYMBOLS.add(tf_key)
            _IEX_EMPTY_COUNTS.pop(tf_key, None)
            logger.info(
                "DATA_WINDOW_NO_SESSION",
                extra=_norm_extra(
                    {
                        "provider": "alpaca",
                        "feed": _feed,
                        "timeframe": _interval,
                        "symbol": symbol,
                        "start": _start.isoformat(),
                        "end": _end.isoformat(),
                    }
                ),
            )
            return pd.DataFrame()
        raise
    global _alpaca_disabled_until
    if _alpaca_disabled_until and datetime.now(UTC) < _alpaca_disabled_until:
        interval_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "60m", "1Day": "1d"}
        fb_int = interval_map.get(_interval)
        if fb_int:
            _mark_fallback(symbol, _interval, _start, _end)
            return _backup_get_bars(symbol, _start, _end, interval=fb_int)
    if _used_fallback(symbol, _interval, _start, _end):
        interval_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "60m", "1Day": "1d"}
        fb_int = interval_map.get(_interval)
        if fb_int:
            return _backup_get_bars(symbol, _start, _end, interval=fb_int)
    global _SIP_DISALLOWED_WARNED
    if _feed == "sip" and not _ALLOW_SIP:
        if not _SIP_DISALLOWED_WARNED:
            logger.warning(
                "SIP_DISABLED",
                extra=_norm_extra({"provider": "alpaca", "feed": _feed, "timeframe": _interval}),
            )
            _SIP_DISALLOWED_WARNED = True
        # Do not silently rewrite the requested feed when explicitly set to SIP.
        # Allow the request to proceed and handle unauthorized gracefully.

    def _tags() -> dict[str, str]:
        return {"provider": "alpaca", "symbol": symbol, "feed": _feed, "timeframe": _interval}

    if _feed == "sip" and _SIP_UNAUTHORIZED:
        _incr("data.fetch.unauthorized", value=1.0, tags=_tags())
        metrics.unauthorized += 1
        logger.warning(
            "UNAUTHORIZED_SIP",
            extra=_norm_extra({"provider": "alpaca", "status": "unauthorized", "feed": _feed, "timeframe": _interval}),
        )
        return pd.DataFrame()

    headers = {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
    }
    timeout_v = clamp_request_timeout(10)
    session = _HTTP_SESSION

    # Mutable state for retry tracking
    start_time = time.monotonic()
    _state = {"corr_id": None, "retries": 0, "providers": []}
    max_retries = _FETCH_BARS_MAX_RETRIES

    def _req(
        session: HTTPSession,
        fallback: tuple[str, str, _dt.datetime, _dt.datetime] | None,
        *,
        headers: dict[str, str],
        timeout: float | tuple[float, float],
    ) -> pd.DataFrame:
        nonlocal _interval, _feed, _start, _end
        global _SIP_UNAUTHORIZED, _alpaca_empty_streak, _alpaca_disabled_until
        _state["providers"].append(_feed)

        def _attempt_fallback(fb: tuple[str, str, _dt.datetime, _dt.datetime]) -> pd.DataFrame | None:
            nonlocal _interval, _feed, _start, _end
            fb_interval, fb_feed, fb_start, fb_end = fb
            if fb_feed == "sip" and not _sip_fallback_allowed(session, headers, fb_interval):
                return None
            from_feed = _feed
            _interval, _feed, _start, _end = fb
            provider_fallback.labels(
                from_provider=f"alpaca_{from_feed}",
                to_provider=f"alpaca_{fb_feed}",
            ).inc()
            _incr("data.fetch.fallback_attempt", value=1.0, tags=_tags())
            payload = _format_fallback_payload_df(_interval, _feed, _start, _end)
            logger.info("DATA_SOURCE_FALLBACK_ATTEMPT", extra={"provider": "alpaca", "fallback": payload})
            return _req(session, None, headers=headers, timeout=timeout)
        params = {
            "symbols": symbol,
            "timeframe": _interval,
            "start": _start.isoformat(),
            "end": _end.isoformat(),
            "limit": 10000,
            "feed": _feed,
            "adjustment": adjustment,
        }
        url = "https://data.alpaca.markets/v2/stocks/bars"
        # Prefer an instance-level patched session.get when present (tests),
        # otherwise route through the module-level `requests.get` so tests that
        # monkeypatch `df.requests.get` can intercept deterministically.
        use_session_get = hasattr(session, "__dict__") and ("get" in getattr(session, "__dict__", {}))
        prev_corr = _state.get("corr_id")
        try:
            if use_session_get:
                resp = session.get(url, params=params, headers=headers, timeout=timeout)
            else:
                resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp is None or not hasattr(resp, "status_code"):
                raise ValueError("invalid_response")
            status = resp.status_code
            text = (resp.text or "").strip()
            ctype = (resp.headers.get("Content-Type") or "").lower()
            corr_id = (
                resp.headers.get("x-request-id")
                or resp.headers.get("apca-request-id")
                or resp.headers.get("x-correlation-id")
            )
            _state["corr_id"] = corr_id
        except Timeout as e:
            log_extra = {
                "url": url,
                "symbol": symbol,
                "feed": _feed,
                "timeframe": _interval,
                "params": params,
            }
            if prev_corr:
                log_extra["previous_correlation_id"] = prev_corr
            attempt = _state["retries"] + 1
            remaining = max_retries - attempt
            log_extra["remaining_retries"] = remaining
            log_fetch_attempt("alpaca", error=str(e), **log_extra)
            logger.warning(
                "DATA_SOURCE_HTTP_ERROR",
                extra=_norm_extra({"provider": "alpaca", "feed": _feed, "timeframe": _interval, "error": str(e)}),
            )
            _incr("data.fetch.timeout", value=1.0, tags=_tags())
            metrics.timeout += 1
            if fallback:
                result = _attempt_fallback(fallback)
                if result is not None:
                    return result
            if attempt >= max_retries:
                logger.error(
                    "FETCH_RETRIES_EXHAUSTED",
                    extra=_norm_extra(
                        {
                            "provider": "alpaca",
                            "feed": _feed,
                            "timeframe": _interval,
                            "symbol": symbol,
                            "error": str(e),
                            "correlation_id": _state["corr_id"],
                        }
                    ),
                )
                raise
            _state["retries"] = attempt
            backoff = min(_FETCH_BARS_BACKOFF_BASE ** (_state["retries"] - 1), _FETCH_BARS_BACKOFF_CAP)
            logger.debug(
                "RETRY_FETCH_ERROR",
                extra=_norm_extra(
                    {
                        "provider": "alpaca",
                        "feed": _feed,
                        "timeframe": _interval,
                        "symbol": symbol,
                        "start": _start.isoformat(),
                        "end": _end.isoformat(),
                        "correlation_id": _state["corr_id"],
                        "retry_delay": backoff,
                        "attempt": _state["retries"],
                        "remaining_retries": max_retries - _state["retries"],
                        "previous_correlation_id": prev_corr,
                    }
                ),
            )
            time.sleep(backoff)
            return _req(session, None, headers=headers, timeout=timeout)
        except ConnectionError as e:
            log_extra = {
                "url": url,
                "symbol": symbol,
                "feed": _feed,
                "timeframe": _interval,
                "params": params,
            }
            if prev_corr:
                log_extra["previous_correlation_id"] = prev_corr
            attempt = _state["retries"] + 1
            remaining = max_retries - attempt
            log_extra["remaining_retries"] = remaining
            log_fetch_attempt("alpaca", error=str(e), **log_extra)
            logger.warning(
                "DATA_SOURCE_HTTP_ERROR",
                extra=_norm_extra({"provider": "alpaca", "feed": _feed, "timeframe": _interval, "error": str(e)}),
            )
            _incr("data.fetch.connection_error", value=1.0, tags=_tags())
            if fallback:
                result = _attempt_fallback(fallback)
                if result is not None:
                    return result
            if attempt >= max_retries:
                logger.error(
                    "FETCH_RETRIES_EXHAUSTED",
                    extra=_norm_extra(
                        {
                            "provider": "alpaca",
                            "feed": _feed,
                            "timeframe": _interval,
                            "symbol": symbol,
                            "error": str(e),
                            "correlation_id": _state["corr_id"],
                        }
                    ),
                )
                raise
            _state["retries"] = attempt
            backoff = min(_FETCH_BARS_BACKOFF_BASE ** (_state["retries"] - 1), _FETCH_BARS_BACKOFF_CAP)
            logger.debug(
                "RETRY_FETCH_ERROR",
                extra=_norm_extra(
                    {
                        "provider": "alpaca",
                        "feed": _feed,
                        "timeframe": _interval,
                        "symbol": symbol,
                        "start": _start.isoformat(),
                        "end": _end.isoformat(),
                        "correlation_id": _state["corr_id"],
                        "retry_delay": backoff,
                        "attempt": _state["retries"],
                        "remaining_retries": max_retries - _state["retries"],
                        "previous_correlation_id": prev_corr,
                    }
                ),
            )
            time.sleep(backoff)
            return _req(session, None, headers=headers, timeout=timeout)
        except (HTTPError, RequestException, ValueError, KeyError) as e:
            log_extra = {
                "url": url,
                "symbol": symbol,
                "feed": _feed,
                "timeframe": _interval,
                "params": params,
            }
            if prev_corr:
                log_extra["previous_correlation_id"] = prev_corr
            attempt = _state["retries"] + 1
            remaining = max_retries - attempt
            log_extra["remaining_retries"] = remaining
            log_fetch_attempt("alpaca", error=str(e), **log_extra)
            logger.warning(
                "DATA_SOURCE_HTTP_ERROR",
                extra=_norm_extra({"provider": "alpaca", "feed": _feed, "timeframe": _interval, "error": str(e)}),
            )
            _incr("data.fetch.error", value=1.0, tags=_tags())
            if fallback:
                result = _attempt_fallback(fallback)
                if result is not None:
                    return result
            if attempt >= max_retries:
                logger.error(
                    "FETCH_RETRIES_EXHAUSTED",
                    extra=_norm_extra(
                        {
                            "provider": "alpaca",
                            "feed": _feed,
                            "timeframe": _interval,
                            "symbol": symbol,
                            "error": str(e),
                            "correlation_id": _state["corr_id"],
                        }
                    ),
                )
                raise
            _state["retries"] = attempt
            backoff = min(_FETCH_BARS_BACKOFF_BASE ** (_state["retries"] - 1), _FETCH_BARS_BACKOFF_CAP)
            logger.debug(
                "RETRY_FETCH_ERROR",
                extra=_norm_extra(
                    {
                        "provider": "alpaca",
                        "feed": _feed,
                        "timeframe": _interval,
                        "symbol": symbol,
                        "start": _start.isoformat(),
                        "end": _end.isoformat(),
                        "correlation_id": _state["corr_id"],
                        "retry_delay": backoff,
                        "attempt": _state["retries"],
                        "remaining_retries": max_retries - _state["retries"],
                        "previous_correlation_id": prev_corr,
                    }
                ),
            )
            time.sleep(backoff)
            return _req(session, None, headers=headers, timeout=timeout)
        payload: dict[str, Any] | list[Any] = {}
        if status != 400 and text:
            if "json" in ctype:
                try:
                    payload = resp.json()
                except ValueError:
                    payload = {}
        data = []
        if isinstance(payload, dict):
            if "bars" in payload and isinstance(payload["bars"], list):
                data = payload["bars"]
            elif symbol in payload and isinstance(payload[symbol], dict) and ("bars" in payload[symbol]):
                data = payload[symbol]["bars"]
        elif isinstance(payload, list):
            data = payload
        log_extra = {
            "url": url,
            "symbol": symbol,
            "feed": _feed,
            "timeframe": _interval,
            "params": params,
            "correlation_id": _state["corr_id"],
        }
        if prev_corr:
            log_extra["previous_correlation_id"] = prev_corr
        if status == 400:
            log_extra_with_remaining = {"remaining_retries": max_retries - _state["retries"], **log_extra}
            log_fetch_attempt("alpaca", status=status, error="bad_request", **log_extra_with_remaining)
            raise ValueError("Invalid feed or bad request")
        if status in (401, 403):
            _incr("data.fetch.unauthorized", value=1.0, tags=_tags())
            metrics.unauthorized += 1
            log_extra_with_remaining = {"remaining_retries": max_retries - _state["retries"], **log_extra}
            log_fetch_attempt("alpaca", status=status, error="unauthorized", **log_extra_with_remaining)
            logger.warning(
                "UNAUTHORIZED_SIP" if _feed == "sip" else "DATA_SOURCE_UNAUTHORIZED",
                extra=_norm_extra(
                    {"provider": "alpaca", "status": "unauthorized", "feed": _feed, "timeframe": _interval}
                ),
            )
            if _feed == "sip":
                _SIP_UNAUTHORIZED = True
                os.environ["ALPACA_SIP_UNAUTHORIZED"] = "1"
                return pd.DataFrame()
            if fallback:
                result = _attempt_fallback(fallback)
                if result is not None:
                    return result
            raise ValueError("unauthorized")
        if status == 429:
            _incr("data.fetch.rate_limited", value=1.0, tags=_tags())
            metrics.rate_limit += 1
            log_extra_with_remaining = {"remaining_retries": max_retries - _state["retries"], **log_extra}
            log_fetch_attempt("alpaca", status=status, error="rate_limited", **log_extra_with_remaining)
            logger.warning(
                "DATA_SOURCE_RATE_LIMITED",
                extra=_norm_extra(
                    {"provider": "alpaca", "status": "rate_limited", "feed": _feed, "timeframe": _interval}
                ),
            )
            if fallback:
                result = _attempt_fallback(fallback)
                if result is not None:
                    return result
            raise ValueError("rate_limited")
        df = pd.DataFrame(data)
        if df.empty:
            attempt = _state["retries"] + 1
            remaining_retries = max(0, max_retries - attempt)
            log_extra_with_remaining = {"remaining_retries": remaining_retries, **log_extra}
            if attempt <= max_retries:
                log_fetch_attempt("alpaca", status=status, error="empty", **log_extra_with_remaining)
            metrics.empty_payload += 1
            is_empty_error = isinstance(payload, dict) and payload.get("error") == "empty"
            if fallback:
                _incr("data.fetch.empty", value=1.0, tags=_tags())
            if _feed == "iex" and is_empty_error:
                key = (symbol, _interval)
                cnt = _IEX_EMPTY_COUNTS.get(key, 0) + 1
                _IEX_EMPTY_COUNTS[key] = cnt
                prev = _state.get("corr_id")
                if _ALLOW_SIP and not _SIP_UNAUTHORIZED:
                    result = _attempt_fallback((_interval, "sip", _start, _end))
                    sip_corr = _state.get("corr_id")
                    if result is not None and not getattr(result, "empty", True):
                        _IEX_EMPTY_COUNTS.pop(key, None)
                        return result
                    msg = "IEX_EMPTY_SIP_UNAUTHORIZED" if _SIP_UNAUTHORIZED else "IEX_EMPTY_SIP_EMPTY"
                    logger.error(
                        msg,
                        extra=_norm_extra(
                            {
                                "provider": "alpaca",
                                "symbol": symbol,
                                "timeframe": _interval,
                                "feed": "iex",
                                "occurrences": cnt,
                                "correlation_id": prev,
                                "sip_correlation_id": sip_corr,
                            }
                        ),
                    )
                    return result if result is not None else pd.DataFrame()
                logger.error(
                    "IEX_EMPTY_NO_SIP",
                    extra=_norm_extra(
                        {
                            "provider": "alpaca",
                            "symbol": symbol,
                            "timeframe": _interval,
                            "feed": "iex",
                            "occurrences": _IEX_EMPTY_COUNTS[key],
                            "correlation_id": prev,
                        }
                    ),
                )
                return pd.DataFrame()
            if _interval.lower() in {"1day", "day", "1d"}:
                try:
                    mdf = _fetch_bars(symbol, _start, _end, "1Min", feed=_feed, adjustment=adjustment)
                except (ValueError, RuntimeError):
                    mdf = pd.DataFrame()
                if not mdf.empty:
                    try:
                        if "timestamp" in mdf.columns:
                            mdf["timestamp"] = pd.to_datetime(mdf["timestamp"], utc=True)
                            mdf.set_index("timestamp", inplace=True)
                        from ai_trading.data.bars import _resample_minutes_to_daily as _resample_to_daily

                        rdf = _resample_to_daily(mdf)
                    except (ImportError, ValueError, TypeError, KeyError):
                        mdf = pd.DataFrame()
                    else:
                        if rdf is not None and (not rdf.empty):
                            return rdf
            _now = datetime.now(UTC)
            _key = (symbol, "AVAILABLE", _now.date().isoformat(), _feed, _interval)
            try:
                _open = is_market_open()
            except Exception:  # pragma: no cover - defensive
                _open = False
            if _open:
                if _empty_should_emit(_key, _now):
                    lvl = _empty_classify(is_market_open=True)
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
                                "symbol": symbol,
                                "start": _start.isoformat(),
                                "end": _end.isoformat(),
                                "correlation_id": corr_id,
                            }
                        ),
                    )
            if fallback:
                result = _attempt_fallback(fallback)
                if result is not None:
                    return result
            key = (symbol, _interval)
            if (
                _feed == "iex"
                and _IEX_EMPTY_COUNTS.get(key, 0) >= _IEX_EMPTY_THRESHOLD
                and _ALLOW_SIP
                and not _SIP_UNAUTHORIZED
                and _sip_fallback_allowed(session, headers, _interval)
            ):
                logger.warning(
                    "ALPACA_IEX_EMPTY_SWITCH_SIP",
                    extra=_norm_extra(
                        {
                            "provider": "alpaca",
                            "symbol": symbol,
                            "timeframe": _interval,
                            "correlation_id": _state["corr_id"],
                        }
                    ),
                )
                _incr("data.fetch.feed_switch", value=1.0, tags=_tags())
                try:
                    metrics.feed_switch += 1
                except Exception:
                    pass
                result = _attempt_fallback((_interval, "sip", _start, _end))
                if result is not None:
                    if not getattr(result, "empty", True):
                        _IEX_EMPTY_COUNTS.pop(key, None)
                    return result
            reason = (
                "market_closed"
                if (
                    _outside_market_hours(_start, _end)
                    or not _window_has_trading_session(_start, _end)
                )
                else "symbol_delisted_or_wrong_feed"
            )
            if reason == "symbol_delisted_or_wrong_feed":
                if _symbol_exists(symbol):
                    reason = "feed_error"
                else:
                    _state["retries"] = max_retries
            if (
                _state["retries"] == 0
                and _feed == "iex"
                and reason in {"symbol_delisted_or_wrong_feed", "feed_error"}
                and _ALLOW_SIP
                and not _SIP_UNAUTHORIZED
            ):
                result = _attempt_fallback((_interval, "sip", _start, _end))
                if result is not None:
                    return result
            if _state["retries"] >= 1:
                hint = (
                    "Market likely closed for requested window"
                    if reason == "market_closed"
                    else (
                        "Feed returned empty or wrong feed"
                        if reason == "feed_error"
                        else "Symbol may be delisted or feed may be incorrect"
                    )
                )
                logger.error(
                    "ALPACA_EMPTY_RESPONSE",
                    extra=_norm_extra(
                        {
                            "provider": "alpaca",
                            "feed": _feed,
                            "timeframe": _interval,
                            "symbol": symbol,
                            "start": _start.isoformat(),
                            "end": _end.isoformat(),
                            "correlation_id": _state["corr_id"],
                            "reason": reason,
                            "hint": hint,
                        }
                    ),
                )
                _alpaca_empty_streak += 1
                if _alpaca_empty_streak > _ALPACA_DISABLE_THRESHOLD:
                    _alpaca_disabled_until = datetime.now(UTC) + _dt.timedelta(minutes=5)
                remaining_retries = max_retries - _state["retries"]
                logger.warning(
                    "ALPACA_FETCH_ABORTED"
                    if remaining_retries > 0
                    else "ALPACA_FETCH_RETRY_LIMIT",
                    extra=_norm_extra(
                        {
                            "provider": "alpaca",
                            "status": "empty",
                            "feed": _feed,
                            "timeframe": _interval,
                            "symbol": symbol,
                            "start": _start.isoformat(),
                            "end": _end.isoformat(),
                            "correlation_id": _state["corr_id"],
                            "retries": _state["retries"],
                            "remaining_retries": remaining_retries,
                            "reason": reason,
                        }
                    ),
                )
                return None
            if str(_interval).lower() not in {"1day", "day", "1d"}:
                if _state["retries"] < max_retries:
                    if _outside_market_hours(_start, _end):
                        logger.info(
                            "ALPACA_FETCH_MARKET_CLOSED",
                            extra=_norm_extra(
                                {
                                    "provider": "alpaca",
                                    "status": "market_closed",
                                    "feed": _feed,
                                    "timeframe": _interval,
                                    "symbol": symbol,
                                    "start": _start.isoformat(),
                                    "end": _end.isoformat(),
                                    "correlation_id": _state["corr_id"],
                                }
                            ),
                        )
                        return None
                    _state["retries"] += 1
                    backoff = min(
                        _FETCH_BARS_BACKOFF_BASE ** (_state["retries"] - 1),
                        _FETCH_BARS_BACKOFF_CAP,
                    )
                    elapsed = time.monotonic() - start_time
                    logger.debug(
                        "RETRY_EMPTY_BARS",
                        extra=_norm_extra(
                            {
                                "provider": "alpaca",
                                "feed": _feed,
                                "timeframe": _interval,
                                "symbol": symbol,
                                "start": _start.isoformat(),
                                "end": _end.isoformat(),
                                "correlation_id": _state["corr_id"],
                                "retry_delay": backoff,
                                "previous_correlation_id": prev_corr,
                                "attempt": _state["retries"],
                                "remaining_retries": max_retries - _state["retries"],
                                "total_elapsed": elapsed,
                            }
                        ),
                    )
                    time.sleep(backoff)
                    return _req(session, None, headers=headers, timeout=timeout)
                logger.warning(
                    "ALPACA_FETCH_RETRY_LIMIT",
                    extra=_norm_extra(
                        {
                            "provider": "alpaca",
                            "status": "empty",
                            "feed": _feed,
                            "timeframe": _interval,
                            "symbol": symbol,
                            "start": _start.isoformat(),
                            "end": _end.isoformat(),
                            "correlation_id": _state["corr_id"],
                            "retries": _state["retries"],
                            "reason": reason,
                        }
                    ),
                )
                return None
            if (not _open) and str(_interval).lower() in {"1day", "day", "1d"}:
                from ai_trading.utils.lazy_imports import load_pandas as _lp
                pd_mod = _lp()
                try:
                    return pd_mod.DataFrame()
                except Exception:
                    return pd.DataFrame()
            remaining_retries = max_retries - _state["retries"]
            logger.warning(
                "ALPACA_FETCH_ABORTED"
                if remaining_retries > 0
                else "ALPACA_FETCH_RETRY_LIMIT",
                extra=_norm_extra(
                    {
                        "provider": "alpaca",
                        "status": "empty",
                        "feed": _feed,
                        "timeframe": _interval,
                        "symbol": symbol,
                        "start": _start.isoformat(),
                        "end": _end.isoformat(),
                        "correlation_id": _state["corr_id"],
                        "retries": _state["retries"],
                        "remaining_retries": remaining_retries,
                        "reason": reason,
                    }
                ),
            )
            return None
        _alpaca_empty_streak = 0
        ts_col = None
        for c in df.columns:
            if c.lower() in ("t", "timestamp", "time"):
                ts_col = c
                break
        if ts_col:
            df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)
        elif "timestamp" not in df.columns:
            df["timestamp"] = pd.to_datetime([], utc=True)
        rename = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])
        df.set_index("timestamp", inplace=True, drop=False)
        _IEX_EMPTY_COUNTS.pop((symbol, _interval), None)
        log_extra_success = {"remaining_retries": max_retries - _state["retries"], **log_extra}
        log_fetch_attempt("alpaca", status=status, **log_extra_success)
        _incr("data.fetch.success", value=1.0, tags=_tags())
        return df

    priority = list(provider_priority())
    max_fb = max_data_fallbacks()
    alt_feed = None
    fallback = None
    if max_fb >= 1:
        try:
            idx = priority.index(f"alpaca_{_feed}")
        except ValueError:
            idx = -1
        for prov in priority[idx + 1:]:
            if prov in {"alpaca_iex", "alpaca_sip"}:
                candidate = prov.split("_")[1]
                if not (candidate == "sip" and _SIP_UNAUTHORIZED):
                    alt_feed = candidate
                    break
        if alt_feed is not None:
            fallback = (_interval, alt_feed, _start, _end)
    df = _req(session, fallback, headers=headers, timeout=timeout_v)
    if (df is None or getattr(df, "empty", True)) and _ENABLE_HTTP_FALLBACK:
        interval_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "60m", "1Day": "1d"}
        y_int = interval_map.get(_interval)
        providers_tried = set(_state["providers"])
        can_use_sip = _ALLOW_SIP and not _SIP_UNAUTHORIZED
        yahoo_allowed = (
            (can_use_sip and {"iex", "sip"}.issubset(providers_tried) and max_fb >= 2)
            or (not can_use_sip and "iex" in providers_tried and max_fb >= 1)
        )
        if y_int and yahoo_allowed and "yahoo" in priority:
            try:
                alt_df = _yahoo_get_bars(symbol, _start, _end, interval=y_int)
            except Exception:  # pragma: no cover - network variance
                alt_df = pd.DataFrame()
            if alt_df is not None and (not alt_df.empty):
                provider_fallback.labels(
                    from_provider=f"alpaca_{_feed}", to_provider="yahoo"
                ).inc()
                logger.info(
                    "DATA_SOURCE_FALLBACK_ATTEMPT",
                    extra=_norm_extra({"provider": "yahoo", "fallback": {"interval": y_int}}),
                )
                _mark_fallback(symbol, _interval, _start, _end)
                return alt_df
    return df


def get_minute_df(symbol: str, start: Any, end: Any, feed: str | None = None) -> pd.DataFrame:
    """Minute bars fetch with provider fallback and downgraded errors.
    Also updates in-memory minute cache for freshness checks."""
    pd = _ensure_pandas()
    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)
    tf_key = (symbol, "1Min")
    if tf_key in _SKIPPED_SYMBOLS:
        logger.debug("SKIP_SYMBOL_EMPTY_BARS", extra={"symbol": symbol})
        return pd.DataFrame() if pd is not None else []  # type: ignore[return-value]
    record_attempt(symbol, "1Min")
    used_backup = False
    enable_finnhub = os.getenv("ENABLE_FINNHUB", "1").lower() not in ("0", "false")
    has_finnhub = (
        os.getenv("FINNHUB_API_KEY")
        and fh_fetcher is not None
        and not getattr(fh_fetcher, "is_stub", False)
    )
    use_finnhub = enable_finnhub and bool(has_finnhub)
    df = None
    if _has_alpaca_keys():
        try:
            feed_to_use = feed or _DEFAULT_FEED
            if (
                feed_to_use == "iex"
                and _IEX_EMPTY_COUNTS.get(tf_key, 0) > 0
                and _ALLOW_SIP
                and not _SIP_UNAUTHORIZED
            ):
                feed_to_use = "sip"
                payload = _format_fallback_payload_df("1Min", "sip", start_dt, end_dt)
                logger.info("DATA_SOURCE_FALLBACK_ATTEMPT", extra={"provider": "alpaca", "fallback": payload})
            df = _fetch_bars(symbol, start_dt, end_dt, "1Min", feed=feed_to_use)
        except (EmptyBarsError, ValueError, RuntimeError) as e:
            if isinstance(e, EmptyBarsError):
                now = datetime.now(UTC)
                if end_dt > now or start_dt > now:
                    logger.info(
                        "ALPACA_EMPTY_BAR_FUTURE",
                        extra={"symbol": symbol, "timeframe": "1Min"},
                    )
                    _EMPTY_BAR_COUNTS.pop(tf_key, None)
                    _IEX_EMPTY_COUNTS.pop(tf_key, None)
                    mark_success(symbol, "1Min")
                    return pd.DataFrame() if pd is not None else []  # type: ignore[return-value]
                try:
                    market_open = is_market_open()
                except Exception:  # pragma: no cover - defensive
                    market_open = True
                if not market_open:
                    logger.info(
                        "ALPACA_EMPTY_BAR_MARKET_CLOSED",
                        extra={"symbol": symbol, "timeframe": "1Min"},
                    )
                    _EMPTY_BAR_COUNTS.pop(tf_key, None)
                    _IEX_EMPTY_COUNTS.pop(tf_key, None)
                    mark_success(symbol, "1Min")
                    return pd.DataFrame() if pd is not None else []  # type: ignore[return-value]
                cnt = _EMPTY_BAR_COUNTS.get(tf_key, 0) + 1
                _EMPTY_BAR_COUNTS[tf_key] = cnt
                if cnt > _EMPTY_BAR_MAX_RETRIES:
                    logger.error(
                        "ALPACA_EMPTY_BAR_MAX_RETRIES",
                        extra={"symbol": symbol, "timeframe": "1Min", "occurrences": cnt},
                    )
                    log_empty_retries_exhausted(
                        "alpaca",
                        symbol=symbol,
                        timeframe="1Min",
                        feed=feed or _DEFAULT_FEED,
                        retries=cnt,
                    )
                    _SKIPPED_SYMBOLS.add(tf_key)
                    raise EmptyBarsError(
                        f"empty_bars: symbol={symbol}, timeframe=1Min, max_retries={cnt}"
                    ) from e
                if cnt >= _EMPTY_BAR_THRESHOLD:
                    backoff = min(2 ** (cnt - _EMPTY_BAR_THRESHOLD), 60)
                    ctx = {
                        "symbol": symbol,
                        "timeframe": "1Min",
                        "occurrences": cnt,
                        "backoff": backoff,
                        "finnhub_enabled": use_finnhub,
                        "feed": feed or _DEFAULT_FEED,
                    }
                    logger.warning("ALPACA_EMPTY_BAR_BACKOFF", extra=ctx)
                    time.sleep(backoff)
                    alt_feed = None
                    if max_data_fallbacks() >= 1:
                        prio = provider_priority()
                        cur = feed or _DEFAULT_FEED
                        try:
                            idx = prio.index(f"alpaca_{cur}")
                        except ValueError:
                            idx = -1
                        for prov in prio[idx + 1:]:
                            if prov in {"alpaca_iex", "alpaca_sip"}:
                                alt_feed = prov.split("_", 1)[1]
                                break
                    if alt_feed and alt_feed != (feed or _DEFAULT_FEED):
                        try:
                            df_alt = _fetch_bars(symbol, start_dt, end_dt, "1Min", feed=alt_feed)
                        except (EmptyBarsError, ValueError, RuntimeError) as alt_err:
                            logger.debug(
                                "ALPACA_ALT_FEED_FAILED",
                                extra={"symbol": symbol, "from_feed": feed or _DEFAULT_FEED, "to_feed": alt_feed, "err": str(alt_err)},
                            )
                            df_alt = None
                        else:
                            if df_alt is not None and not getattr(df_alt, "empty", True):
                                logger.info(
                                    "ALPACA_ALT_FEED_SUCCESS",
                                    extra={"symbol": symbol, "from_feed": feed or _DEFAULT_FEED, "to_feed": alt_feed, "timeframe": "1Min"},
                                )
                                _EMPTY_BAR_COUNTS.pop(tf_key, None)
                                _IEX_EMPTY_COUNTS.pop(tf_key, None)
                                mark_success(symbol, "1Min")
                                return df_alt
                    if end_dt - start_dt > _dt.timedelta(days=1):
                        short_start = end_dt - _dt.timedelta(days=1)
                        logger.debug(
                            "ALPACA_SHORT_WINDOW_RETRY",
                            extra={
                                "symbol": symbol,
                                "timeframe": "1Min",
                                "start": short_start.isoformat(),
                                "end": end_dt.isoformat(),
                                "feed": feed or _DEFAULT_FEED,
                            },
                        )
                        try:
                            df_short = _fetch_bars(symbol, short_start, end_dt, "1Min", feed=feed or _DEFAULT_FEED)
                        except (EmptyBarsError, ValueError, RuntimeError):
                            df_short = None
                        else:
                            if df_short is not None and not getattr(df_short, "empty", True):
                                logger.info(
                                    "ALPACA_SHORT_WINDOW_SUCCESS",
                                    extra={
                                        "symbol": symbol,
                                        "timeframe": "1Min",
                                        "feed": feed or _DEFAULT_FEED,
                                        "start": short_start.isoformat(),
                                        "end": end_dt.isoformat(),
                                    },
                                )
                                _EMPTY_BAR_COUNTS.pop(tf_key, None)
                                _IEX_EMPTY_COUNTS.pop(tf_key, None)
                                mark_success(symbol, "1Min")
                                return df_short
                        try:
                            df = _backup_get_bars(symbol, start_dt, end_dt, interval="1m")
                            used_backup = True
                        except Exception as alt_err:  # pragma: no cover - network failure
                            logger.warning(
                                "ALT_PROVIDER_FAILED",
                                extra={"symbol": symbol, "err": str(alt_err)},
                            )
                        df = None
                    if df is None or getattr(df, "empty", True):
                        _SKIPPED_SYMBOLS.add(tf_key)
                        logger.warning(
                            "ALPACA_EMPTY_BAR_SKIP",
                            extra={
                                "symbol": symbol,
                                "timeframe": "1Min",
                                "occurrences": cnt,
                            },
                        )
                        return pd.DataFrame() if pd is not None else []  # type: ignore[return-value]
                else:
                    logger.debug(
                        "ALPACA_EMPTY_BARS",
                        extra={"symbol": symbol, "timeframe": "1Min", "feed": feed or _DEFAULT_FEED, "occurrences": cnt},
                    )
                    df = None
            else:
                logger.warning(
                    "ALPACA_FETCH_FAILED", extra={"symbol": symbol, "err": str(e)}
                )
                df = None
    else:
        logger.warning("ALPACA_API_KEY_MISSING", extra={"symbol": symbol, "timeframe": "1Min"})
        df = None
    if df is None or getattr(df, "empty", True):
        if use_finnhub:
            try:
                df = fh_fetcher.fetch(symbol, start_dt, end_dt, resolution="1")
            except (FinnhubAPIException, ValueError, NotImplementedError) as e:
                logger.debug("FINNHUB_FETCH_FAILED", extra={"symbol": symbol, "err": str(e)})
                df = None
        elif not enable_finnhub:
            warn_finnhub_disabled_no_data(symbol)
        else:
            log_finnhub_disabled(symbol)
    if df is None or getattr(df, "empty", True):
        max_span = _dt.timedelta(days=8)
        total_span = end_dt - start_dt
        if total_span > max_span:
            logger.warning(
                "YF_1M_RANGE_SPLIT",
                extra={
                    "symbol": symbol,
                    "start": start_dt.isoformat(),
                    "end": end_dt.isoformat(),
                    "max_days": 8,
                },
            )
            dfs: list[pd.DataFrame] = []  # type: ignore[var-annotated]
            cur_start = start_dt
            while cur_start < end_dt:
                cur_end = min(cur_start + max_span, end_dt)
                dfs.append(_backup_get_bars(symbol, cur_start, cur_end, interval="1m"))
                used_backup = True
                cur_start = cur_end
            if pd is not None and dfs:
                df = pd.concat(dfs, ignore_index=True)
            elif dfs:
                df = dfs[0]
            else:
                df = pd.DataFrame() if pd is not None else []  # type: ignore[assignment]
        else:
            df = _backup_get_bars(symbol, start_dt, end_dt, interval="1m")
            used_backup = True
    try:
        if pd is not None and isinstance(df, pd.DataFrame) and (not df.empty):
            if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
                last_ts = int(pd.Timestamp(df.index[-1]).tz_convert("UTC").timestamp())
            elif "timestamp" in df.columns:
                last_ts = int(pd.Timestamp(df["timestamp"].iloc[-1]).tz_convert("UTC").timestamp())
            else:
                last_ts = None
            if last_ts is not None:
                set_cached_minute_timestamp(symbol, last_ts)
            _EMPTY_BAR_COUNTS.pop(tf_key, None)
            _IEX_EMPTY_COUNTS.pop(tf_key, None)
            mark_success(symbol, "1Min")
            if used_backup:
                _mark_fallback(symbol, "1Min", start_dt, end_dt)
    except (ValueError, TypeError, KeyError, AttributeError):
        pass
    return _post_process(df)


def get_daily_df(
    symbol: str,
    start: Any | None = None,
    end: Any | None = None,
    *,
    feed: str | None = None,
    adjustment: str | None = None,
) -> pd.DataFrame:
    """Thin wrapper around :func:`bars.get_bars_df` for daily bars."""
    try:
        from ai_trading.alpaca_api import get_bars_df as _get_bars_df
    except Exception as exc:  # pragma: no cover - optional dependency
        raise DataFetchError("Alpaca API unavailable") from exc

    return _get_bars_df(
        symbol,
        timeframe="1Day",
        start=start,
        end=end,
        feed=feed,
        adjustment=adjustment,
    )


def get_bars(
    symbol: str, timeframe: str, start: Any, end: Any, *, feed: str | None = None, adjustment: str | None = None
) -> pd.DataFrame | None:
    """Compatibility wrapper delegating to _fetch_bars."""
    S = get_settings()
    if S is None:
        from ai_trading.config import management as _cfg

        _cfg.reload_env()
        S = get_settings()
        if S is None:
            raise RuntimeError("Configuration is unavailable")
    # If a client-like object is passed for `feed`, route via client helper for tests
    if feed is not None and not isinstance(feed, str):
        return _alpaca_get_bars(feed, symbol, start, end, timeframe=_canon_tf(timeframe))
    if feed is None:
        prio = provider_priority(S)
        for prov in prio:
            if prov.startswith("alpaca_"):
                feed = prov.split("_", 1)[1]
                break
        feed = feed or S.alpaca_data_feed
    adjustment = adjustment or S.alpaca_adjustment
    return _fetch_bars(symbol, start, end, timeframe, feed=feed, adjustment=adjustment)


def get_bars_batch(
    symbols: list[str], timeframe: str, start: Any, end: Any, *, feed: str | None = None, adjustment: str | None = None
) -> dict[str, pd.DataFrame]:
    """Fetch bars for multiple symbols via get_bars."""
    return {sym: get_bars(sym, timeframe, start, end, feed=feed, adjustment=adjustment) for sym in symbols}


def fetch_minute_yfinance(symbol: str, start_dt: _dt.datetime, end_dt: _dt.datetime) -> pd.DataFrame:
    """Explicit helper for tests and optional direct Yahoo minute fetch."""
    df = _yahoo_get_bars(symbol, start_dt, end_dt, interval="1m")
    return _post_process(df)


def is_market_open() -> bool:
    """Return True if the market is currently open.

    Falls back to ``True`` when the detailed calendar check is unavailable.
    """
    try:
        from ai_trading.utils.base import is_market_open as _is_open

        return bool(_is_open())
    except Exception:
        return True


def _build_daily_url(symbol: str, start: datetime, end: datetime) -> str:
    start_s = int(start.timestamp())
    end_s = int(end.timestamp())
    return (
        "https://query1.finance.yahoo.com/v8/finance/chart/" f"{symbol}?period1={start_s}&period2={end_s}&interval=1d",
    )


__all__ = [
    "_DEFAULT_FEED",
    "_VALID_FEEDS",
    "_ALLOW_SIP",
    "_SIP_UNAUTHORIZED",
    "ensure_datetime",
    "bars_time_window_day",
    "_parse_bars",
    "_alpaca_get_bars",
    "get_daily",
    "fetch_daily_data_async",
    "_yahoo_get_bars",
    "_backup_get_bars",
    "_fetch_bars",
    "get_bars",
    "get_bars_batch",
    "fetch_minute_yfinance",
    "is_market_open",
    "get_last_available_bar",
    "fh_fetcher",
    "get_minute_df",
    "get_daily_df",
    "metrics",
    "build_fetcher",
    "DataFetchError",
    "DataFetchException",
    "FinnhubAPIException",
    "get_cached_minute_timestamp",
    "set_cached_minute_timestamp",
    "clear_cached_minute_timestamp",
    "age_cached_minute_timestamps",
    "last_minute_bar_age_seconds",
    "_build_daily_url",
]
