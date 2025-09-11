from __future__ import annotations

import datetime as _dt
from typing import Any

from ai_trading.utils.lazy_imports import load_pandas
from ai_trading.data.empty_bar_backoff import _SKIPPED_SYMBOLS, record_attempt, mark_success
from ai_trading.data.metrics import provider_fallback, fetch_retry_total
from ai_trading.config.management import MAX_EMPTY_RETRIES
from ai_trading.config.settings import provider_priority, max_data_fallbacks
from ai_trading.logging import log_backup_provider_used, get_logger
from ai_trading.data.provider_monitor import provider_monitor

from . import EmptyBarsError, _fetch_bars

pd = load_pandas()
logger = get_logger(__name__)

_EMPTY_BAR_COUNTS: dict[tuple[str, str], int] = {}
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

    try:
        df = _fetch_bars(symbol, start, end, timeframe, feed=feed)
    except EmptyBarsError:
        cnt = _EMPTY_BAR_COUNTS.get(tf_key, 0) + 1
        _EMPTY_BAR_COUNTS[tf_key] = cnt
        if cnt >= _EMPTY_BAR_MAX_RETRIES:
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
                _SKIPPED_SYMBOLS.add(tf_key)
                logger.warning(
                    "ALPACA_EMPTY_BAR_SKIP",
                    extra={"symbol": symbol, "timeframe": timeframe, "feed": alt_feed},
                )
                return _empty_df()
            else:
                _EMPTY_BAR_COUNTS.pop(tf_key, None)
                mark_success(symbol, timeframe)
                return df
        _SKIPPED_SYMBOLS.add(tf_key)
        logger.warning(
            "ALPACA_EMPTY_BARS", extra={"symbol": symbol, "timeframe": timeframe, "feed": feed}
        )
        return _empty_df()
    else:
        _EMPTY_BAR_COUNTS.pop(tf_key, None)
        mark_success(symbol, timeframe)
        return df


__all__ = [
    "_fetch_feed",
    "_SKIPPED_SYMBOLS",
    "_EMPTY_BAR_COUNTS",
    "_EMPTY_BAR_MAX_RETRIES",
]
