from __future__ import annotations

import datetime as _dt
import importlib
import sys
from typing import Any

from ai_trading.utils.lazy_imports import load_pandas
from ai_trading.data.empty_bar_backoff import (
    _SKIPPED_SYMBOLS,
    _EMPTY_BAR_COUNTS,
    MAX_EMPTY_RETRIES,
    record_attempt,
    mark_success,
)
from ai_trading.data.metrics import provider_fallback, fetch_retry_total
from ai_trading.config.settings import provider_priority, max_data_fallbacks
from ai_trading.logging import log_backup_provider_used, get_logger
from ai_trading.logging.normalize import normalize_extra as _norm_extra
from ai_trading.data.provider_monitor import provider_monitor

from . import (
    EmptyBarsError,
    _backup_get_bars,
    _canon_tf,
    _fetch_bars,
    _mark_fallback,
)
_fetch_module = sys.modules.get(__package__)
if _fetch_module is None:  # pragma: no cover - defensive import path
    _fetch_module = importlib.import_module(__package__)

pd = load_pandas()
logger = get_logger(__name__)

_EMPTY_BAR_MAX_RETRIES = MAX_EMPTY_RETRIES


def _empty_df() -> Any:
    if pd is None:  # pragma: no cover - defensive
        return []
    return pd.DataFrame()


def _next_feed(cur_feed: str) -> str | None:
    """Return the next alpaca feed to try based on provider priority."""
    prio = list(provider_priority())
    try:
        idx = prio.index(f"alpaca_{cur_feed}")
    except ValueError:
        return None
    limit = max_data_fallbacks()
    for prov in prio[idx + 1 : idx + 1 + limit]:
        if prov.startswith("alpaca_"):
            return prov.split("_", 1)[1]
    return None


def _http_fallback(
    symbol: str,
    start: _dt.datetime,
    end: _dt.datetime,
    timeframe: str,
    *,
    from_feed: str,
):
    """Attempt HTTP-based backup retrieval when enabled."""

    if not getattr(_fetch_module, "_ENABLE_HTTP_FALLBACK", False):
        return None
    tf_norm = _canon_tf(timeframe)
    interval_map = {
        "1Min": "1m",
        "5Min": "5m",
        "15Min": "15m",
        "1Hour": "60m",
        "1Day": "1d",
    }
    interval = interval_map.get(tf_norm)
    if interval is None:
        return None
    df = _backup_get_bars(symbol, start, end, interval=interval)
    if df is None or getattr(df, "empty", True):
        return df
    provider_fallback.labels(from_provider=f"alpaca_{from_feed}", to_provider="yahoo").inc()
    provider_monitor.record_switchover(f"alpaca_{from_feed}", "yahoo")
    logger.info(
        "DATA_SOURCE_FALLBACK_ATTEMPT",
        extra=_norm_extra({"provider": "yahoo", "fallback": {"interval": interval}}),
    )
    _mark_fallback(symbol, tf_norm, start, end)
    return df


def _fetch_feed(
    symbol: str,
    start: _dt.datetime,
    end: _dt.datetime,
    timeframe: str,
    *,
    feed: str,
) -> Any:
    """Fetch bars while enforcing empty-bar backoff and feed fallback."""
    tf_key = (symbol, timeframe)
    if tf_key in _SKIPPED_SYMBOLS:
        logger.debug(
            "SKIP_SYMBOL_EMPTY_BARS", extra={"symbol": symbol, "timeframe": timeframe}
        )
        return _empty_df()

    record_attempt(symbol, timeframe)

    def _maybe_http_fallback(
        source_feed: str, fb_start: _dt.datetime, fb_end: _dt.datetime
    ) -> Any:
        fb_df = _http_fallback(
            symbol,
            fb_start,
            fb_end,
            timeframe,
            from_feed=source_feed,
        )
        if fb_df is not None and not getattr(fb_df, "empty", True):
            _EMPTY_BAR_COUNTS.pop(tf_key, None)
            mark_success(symbol, timeframe)
        return fb_df

    try:
        df = _fetch_bars(symbol, start, end, timeframe, feed=feed)
    except EmptyBarsError:
        cnt = _EMPTY_BAR_COUNTS.get(tf_key, 0)
        if cnt >= _EMPTY_BAR_MAX_RETRIES:
            fallback_df = _maybe_http_fallback(feed, start, end)
            if fallback_df is not None and not getattr(fallback_df, "empty", True):
                return fallback_df
            _SKIPPED_SYMBOLS.add(tf_key)
            logger.error(
                "ALPACA_EMPTY_BAR_MAX_RETRIES",
                extra={"symbol": symbol, "timeframe": timeframe, "occurrences": cnt},
            )
            raise
        alt_feed = _next_feed(feed)
        if alt_feed:
            fetch_retry_total.labels(provider=f"alpaca_{feed}").inc()
            provider_fallback.labels(
                from_provider=f"alpaca_{feed}", to_provider=f"alpaca_{alt_feed}"
            ).inc()
            provider_monitor.record_switchover(
                f"alpaca_{feed}", f"alpaca_{alt_feed}"
            )
            log_backup_provider_used(
                f"alpaca_{alt_feed}",
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
            )
            if end - start > _dt.timedelta(days=1):
                start = end - _dt.timedelta(days=1)
            try:
                df = _fetch_bars(symbol, start, end, timeframe, feed=alt_feed)
            except EmptyBarsError:
                fallback_df = _maybe_http_fallback(alt_feed, start, end)
                if fallback_df is not None and not getattr(fallback_df, "empty", True):
                    return fallback_df
                _SKIPPED_SYMBOLS.add(tf_key)
                logger.warning(
                    "ALPACA_EMPTY_BAR_SKIP",
                    extra={"symbol": symbol, "timeframe": timeframe, "feed": alt_feed},
                )
                return fallback_df if fallback_df is not None else _empty_df()
            else:
                if df is None or getattr(df, "empty", True):
                    fallback_df = _maybe_http_fallback(alt_feed, start, end)
                    if fallback_df is not None and not getattr(fallback_df, "empty", True):
                        return fallback_df
                    return fallback_df if fallback_df is not None else df
                _EMPTY_BAR_COUNTS.pop(tf_key, None)
                mark_success(symbol, timeframe)
                return df
        fallback_df = _maybe_http_fallback(feed, start, end)
        if fallback_df is not None and not getattr(fallback_df, "empty", True):
            return fallback_df
        _SKIPPED_SYMBOLS.add(tf_key)
        logger.warning(
            "ALPACA_EMPTY_BARS", extra={"symbol": symbol, "timeframe": timeframe, "feed": feed}
        )
        return fallback_df if fallback_df is not None else _empty_df()
    else:
        if df is None or getattr(df, "empty", True):
            fallback_df = _maybe_http_fallback(feed, start, end)
            if fallback_df is not None and not getattr(fallback_df, "empty", True):
                return fallback_df
            return fallback_df if fallback_df is not None else df
        _EMPTY_BAR_COUNTS.pop(tf_key, None)
        mark_success(symbol, timeframe)
        return df


__all__ = [
    "_fetch_feed",
    "_SKIPPED_SYMBOLS",
    "_EMPTY_BAR_COUNTS",
    "_EMPTY_BAR_MAX_RETRIES",
]
