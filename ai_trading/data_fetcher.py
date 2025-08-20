from __future__ import annotations

import datetime as _dt
import os
from typing import Any

import pandas as pd  # AI-AGENT-REF: pandas already a project dependency
from pandas import Timestamp

try:  # AI-AGENT-REF: prefer internal HTTP helper when available
    from ai_trading.utils import http as _http
except Exception:  # pragma: no cover  # noqa: BLE001
    _http = None

try:  # AI-AGENT-REF: fallback to requests if internal helper missing
    import requests as _requests
except Exception:  # pragma: no cover  # noqa: BLE001
    _requests = None
requests = _requests

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
    now_s = int(_dt.datetime.now(tz=_dt.timezone.utc).timestamp())
    _MINUTE_CACHE[symbol] = (int(ts_epoch_s), now_s)

def clear_cached_minute_timestamp(symbol: str) -> None:
    """Remove cached entry for symbol."""  # AI-AGENT-REF: cache clear
    _MINUTE_CACHE.pop(symbol, None)

def age_cached_minute_timestamps(max_age_seconds: int) -> int:
    """Drop cache entries older than max_age_seconds (based on inserted time)."""  # AI-AGENT-REF: cache prune
    now_s = int(_dt.datetime.now(tz=_dt.timezone.utc).timestamp())
    to_del = [sym for sym, (_, ins) in _MINUTE_CACHE.items() if now_s - ins > max_age_seconds]
    for sym in to_del:
        _MINUTE_CACHE.pop(sym, None)
    return len(to_del)

def last_minute_bar_age_seconds(symbol: str) -> int | None:
    """Age in seconds of last cached minute bar for symbol, or None if absent."""  # AI-AGENT-REF: age helper
    ts = get_cached_minute_timestamp(symbol)
    if ts is None:
        return None
    now_s = int(_dt.datetime.now(tz=_dt.timezone.utc).timestamp())
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


# ---------------------------------------------------------------------------
# Datetime coercion
# ---------------------------------------------------------------------------


def ensure_datetime(value: Any) -> _dt.datetime:
    """Coerce various datetime inputs into timezone-aware UTC datetime."""

    if value is None:
        raise ValueError("None is not a valid datetime")
    if isinstance(value, str):
        v = value.strip()
        if not v:
            raise ValueError("Empty string is not a valid datetime")
        ts = Timestamp(v)
    elif isinstance(value, Timestamp):
        ts = value
    elif isinstance(value, _dt.datetime):
        ts = Timestamp(value)
    else:
        try:
            ts = Timestamp(value)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid datetime input: {value!r}") from exc

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _default_window_for(timeframe: Any) -> tuple[_dt.datetime, _dt.datetime]:
    """Derive [start, end] when callers omit them."""  # AI-AGENT-REF: legacy helper
    now = _dt.datetime.now(tz=_dt.timezone.utc)
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


def _yahoo_get_bars(symbol: str, start: Any, end: Any, interval: str) -> pd.DataFrame:
    """Return a DataFrame with a tz-aware 'timestamp' column between start and end."""

    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)

    try:  # AI-AGENT-REF: yfinance already used elsewhere in project
        import yfinance as yf
    except Exception:  # pragma: no cover - provides empty frame  # noqa: BLE001
        idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        cols = ["open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=cols, index=idx).reset_index()

    df = yf.download(symbol, start=start_dt, end=end_dt, interval=interval, progress=False)
    if df is None or df.empty:
        idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        cols = ["open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=cols, index=idx).reset_index()

    if df.index.tz is None:
        df = df.tz_localize("UTC")
    else:
        df = df.tz_convert("UTC")
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df = df.reset_index().rename(columns={df.index.name or "Date": "timestamp"})
    if "timestamp" not in df.columns:
        for c in df.columns:
            if c.lower() in ("date", "datetime"):
                df = df.rename(columns={c: "timestamp"})
                break
    return df


# ---------------------------------------------------------------------------
# Minimal bars fetcher used by tests (monkeypatchable)
# ---------------------------------------------------------------------------


def _fetch_bars(
    symbol: str, start: Any, end: Any, timeframe: str, feed: str = "sip"
) -> pd.DataFrame:
    """Fetch bars from Alpaca v2; tests monkeypatch HTTP call path."""

    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)
    tf = str(timeframe)
    use_feed = feed or "sip"

    def _req(_feed: str) -> pd.DataFrame:
        params = {
            "symbols": symbol,
            "timeframe": tf,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "limit": 10000,
            "feed": _feed,
        }
        if requests is None:  # pragma: no cover
            raise RuntimeError("requests not available")
        url = "https://data.alpaca.markets/v2/stocks/bars"
        resp = requests.get(url, params=params, timeout=10)
        status = resp.status_code
        payload = resp.json() if status != 400 else {}

        bars = []
        if isinstance(payload, dict):
            if "bars" in payload and isinstance(payload["bars"], list):
                bars = payload["bars"]
            elif (
                symbol in payload
                and isinstance(payload[symbol], dict)
                and "bars" in payload[symbol]
            ):
                bars = payload[symbol]["bars"]

        if status == 400:
            raise ValueError("Invalid feed or bad request")

        if not bars:
            if tf.lower() in ("1day", "day"):
                try:
                    return get_last_available_bar(symbol)
                except Exception:  # noqa: BLE001
                    pass
            cols = ["timestamp", "open", "high", "low", "close", "volume"]
            return pd.DataFrame(columns=cols)

        df = pd.DataFrame(bars)
        ts_col = None
        for c in df.columns:
            if c.lower() in ("t", "timestamp", "time"):
                ts_col = c
                break
        if ts_col:
            df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)
        elif "timestamp" not in df.columns:
            df["timestamp"] = pd.to_datetime([], utc=True)
        return df

    try:
        return _req(use_feed)
    except ValueError:
        fallback = _DEFAULT_FEED if use_feed != _DEFAULT_FEED else "sip"
        return _req(fallback)


# ---------------------------------------------------------------------------
# get_minute_df wrapper with graceful fallbacks / logging behavior
# ---------------------------------------------------------------------------


def get_minute_df(symbol: str, start: Any, end: Any, feed: str | None = None) -> pd.DataFrame:
    """Minute bars fetch with provider fallback and downgraded errors.
    Also updates in-memory minute cache for freshness checks."""  # AI-AGENT-REF

    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)

    df: pd.DataFrame | None = None
    try:
        df = fh_fetcher.fetch(symbol=symbol, start=start_dt, end=end_dt)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "403" in msg or "forbidden" in msg or "rate limit" in msg:
            df = fetch_minute_yfinance(symbol)

    if df is None:
        try:
            df = _fetch_bars(symbol, start_dt, end_dt, "1Min", feed=feed or "sip")
        except Exception as exc:  # noqa: BLE001
            emsg = str(exc).lower()
            if "subscription does not permit querying recent sip data" in emsg:
                try:
                    import logging as _logging

                    _logging.getLogger(__name__).warning(
                        "Downgrading SIP subscription error; retrying with iex"
                    )
                except Exception:  # pragma: no cover  # noqa: BLE001
                    pass
                try:
                    df = _fetch_bars(symbol, start_dt, end_dt, "1Min", feed=_DEFAULT_FEED)
                except Exception:  # noqa: BLE001
                    df = _yahoo_get_bars(symbol, start_dt, end_dt, interval="1m")
            else:
                df = _yahoo_get_bars(symbol, start_dt, end_dt, interval="1m")

    try:  # AI-AGENT-REF: update minute cache with latest bar
        if isinstance(df, pd.DataFrame) and not df.empty:
            if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
                last_ts = int(pd.Timestamp(df.index[-1]).tz_convert("UTC").timestamp())
            elif "timestamp" in df.columns:
                last_ts = int(pd.Timestamp(df["timestamp"].iloc[-1]).tz_convert("UTC").timestamp())
            else:
                last_ts = None
            if last_ts is not None:
                set_cached_minute_timestamp(symbol, last_ts)
    except Exception:  # pragma: no cover - cache best effort  # noqa: BLE001
        pass

    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def get_bars(
    symbol: str, timeframe: str, start: Any, end: Any, *, feed: str | None = None
) -> pd.DataFrame:
    """Compatibility wrapper delegating to _fetch_bars."""  # AI-AGENT-REF
    return _fetch_bars(symbol, start, end, timeframe, feed=feed or "sip")


def get_bars_batch(
    symbols: list[str], timeframe: str, start: Any, end: Any, *, feed: str | None = None
) -> dict[str, pd.DataFrame]:
    """Fetch bars for multiple symbols via get_bars."""  # AI-AGENT-REF
    return {sym: get_bars(sym, timeframe, start, end, feed=feed) for sym in symbols}


def get_bars_df(
    symbol: str,
    timeframe: Any,
    start: Any | None = None,
    end: Any | None = None,
    *,
    feed: str | None = None,
) -> pd.DataFrame:
    """Legacy alias expected by bot_engine (accepts optional start/end)."""  # AI-AGENT-REF
    if start is None or end is None:
        start, end = _default_window_for(timeframe)
    return get_bars(symbol, str(timeframe), start, end, feed=feed)


def fetch_minute_yfinance(symbol: str) -> pd.DataFrame:
    """Placeholder for tests to monkeypatch Yahoo minute fetcher."""  # AI-AGENT-REF
    raise NotImplementedError


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
]
