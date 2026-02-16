"""Yahoo Finance batch download helpers."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:  # optional dependency; tests stub behaviour when absent
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency missing
    yf = None  # type: ignore[assignment]

from ai_trading.config.management import get_env
from ai_trading.data.normalize import ensure_ohlcv
from ai_trading.logging import get_logger

logger = get_logger(__name__)


def _env_float(name: str, default: float) -> float:
    value = get_env(name, str(default), cast=float)
    return float(default if value is None else value)


def _env_int(name: str, default: int) -> int:
    value = get_env(name, str(default), cast=int)
    return int(default if value is None else value)


YF_TIMEOUT = _env_float("YF_TIMEOUT", 8.0)
YF_RETRIES = _env_int("YF_RETRIES", 3)
YF_BACKOFF = _env_float("YF_BACKOFF", 0.7)
YF_CHUNK_SIZE = _env_int("YF_CHUNK_SIZE", 40)
YF_CACHE_DIR = Path(get_env("YF_CACHE_DIR", "/var/cache/ai-trading-bot/yf"))
YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_VALID_INTERVALS = frozenset(
    {
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "60m",
        "90m",
        "1h",
        "4h",
        "1d",
        "5d",
        "1wk",
        "1mo",
        "3mo",
    }
)
_INTERVAL_ALIASES = {
    "1min": "1m",
    "1minute": "1m",
    "1hour": "1h",
}


def _sleep_backoff(attempt: int) -> None:
    delay = max(0.1, (attempt + 1) * YF_BACKOFF)
    time.sleep(delay)


def _download_batch(
    tickers: List[str], *, start=None, end=None, period="1y", interval="1d"
) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance_unavailable")
    return yf.download(
        tickers=" ".join(tickers),
        period=period if start is None else None,
        start=start,
        end=end,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        timeout=YF_TIMEOUT,
        threads=False,
        repair=True,
    )


def _cache_key(tickers: List[str], start, end, period: str, interval: str) -> str:
    payload = json.dumps([tickers, str(start), str(end), period, interval])
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    day = dt.date.today().isoformat()
    return f"{day}-{interval}-{digest}.parquet"


def normalize_yf_interval(interval: str | None) -> str | None:
    if interval is None:
        return None
    try:
        lowered = str(interval).strip().lower()
    except Exception:
        logger.debug("YF_INTERVAL_NORMALIZE_FAILED", extra={"interval": interval}, exc_info=True)
        return None
    if not lowered:
        return None
    normalized = _INTERVAL_ALIASES.get(lowered, lowered)
    return normalized if normalized in _VALID_INTERVALS else None


def _cache_read_or_none(key: str) -> Optional[pd.DataFrame]:
    path = YF_CACHE_DIR / key
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        logger.debug("YF_CACHE_READ_FAILED", extra={"path": str(path)}, exc_info=True)
        return None


def _cache_write(key: str, df: pd.DataFrame) -> None:
    path = YF_CACHE_DIR / key
    try:
        df.to_parquet(path)
    except Exception:
        logger.debug("YF_CACHE_WRITE_FAILED", extra={"path": str(path)}, exc_info=True)


def fetch_yf_batched(
    tickers: Iterable[str], *, start=None, end=None, period="1y", interval="1d"
) -> Dict[str, Optional[pd.DataFrame]]:
    tickers_list = [t.strip().upper() for t in tickers if str(t).strip()]
    tickers_unique = list(dict.fromkeys(tickers_list))
    out: Dict[str, Optional[pd.DataFrame]] = {ticker: None for ticker in tickers_unique}
    normalized_interval = normalize_yf_interval(interval) or "1d"
    pytest_mode = "pytest" in sys.modules or str(os.getenv("PYTEST_RUNNING", "")).strip().lower() in {"1", "true", "yes", "on"}
    if pytest_mode and not os.getenv("PYTEST_YF_ALLOW_NETWORK"):
        return out

    for i in range(0, len(tickers_unique), YF_CHUNK_SIZE):
        chunk = tickers_unique[i : i + YF_CHUNK_SIZE]
        cache_key = _cache_key(chunk, start, end, period, normalized_interval)
        df: pd.DataFrame | None = _cache_read_or_none(cache_key)
        if df is None:
            for attempt in range(max(1, YF_RETRIES)):
                try:
                    df = _download_batch(
                        chunk,
                        start=start,
                        end=end,
                        period=period,
                        interval=normalized_interval,
                    )
                    break
                except Exception as exc:  # pragma: no cover - network error surface
                    _sleep_backoff(attempt)
                    df = None
            if df is None:
                continue
            _cache_write(cache_key, df)
        if isinstance(df.columns, pd.MultiIndex):
            level0 = df.columns.get_level_values(0)
            for symbol in chunk:
                if symbol in level0:
                    slice_df = df[symbol].copy()
                else:
                    slice_df = pd.DataFrame()
                out[symbol] = ensure_ohlcv(slice_df)
        else:
            symbol = chunk[0] if chunk else None
            if symbol:
                out[symbol] = ensure_ohlcv(df.copy())
    return out


__all__ = ["fetch_yf_batched", "normalize_yf_interval"]
