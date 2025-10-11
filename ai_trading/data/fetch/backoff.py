from __future__ import annotations

import datetime as _dt
import datetime as _dt
import importlib
import os
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
from ai_trading.config.management import get_env
from ai_trading.logging import log_backup_provider_used, get_logger
from ai_trading.logging.normalize import normalize_extra as _norm_extra
from ai_trading.data.provider_monitor import provider_monitor
from ai_trading.utils.time import monotonic_time

from . import (
    EmptyBarsError,
    _backup_get_bars,
    _canon_tf,
    _fetch_bars,
    _mark_fallback,
    _record_feed_switch,
    _resolve_backup_provider,
    _yahoo_get_bars,
)
_fetch_module = sys.modules.get(__package__)
if _fetch_module is None:  # pragma: no cover - defensive import path
    _fetch_module = importlib.import_module(__package__)

pd = load_pandas()
logger = get_logger(__name__)

_EMPTY_BAR_MAX_RETRIES = MAX_EMPTY_RETRIES
_PROVIDER_COOLDOWNS: dict[tuple[str, str], float] = {}
_PROVIDER_DECISION_CACHE: tuple[float, float] = (120.0, 0.0)


def _provider_decision_window() -> float:
    """Return provider decision cooldown window in seconds."""

    global _PROVIDER_DECISION_CACHE
    cached_value, cached_at = _PROVIDER_DECISION_CACHE
    now = monotonic_time()
    if now - cached_at < 60.0:
        return cached_value

    window: float | None = None
    try:
        window = get_env("AI_TRADING_PROVIDER_DECISION_SECS", None, cast=float)
    except Exception:
        window = None
    if window is None:
        raw = os.getenv("AI_TRADING_PROVIDER_DECISION_SECS", "").strip()
        if raw:
            try:
                window = float(raw)
            except Exception:
                window = None
    if window is None:
        window = 120.0

    value = max(float(window), 0.0)
    _PROVIDER_DECISION_CACHE = (value, now)
    return value


def _provider_switch_cooldown_seconds() -> float:
    getter = getattr(_fetch_module, "_provider_switch_cooldown_seconds", None)
    if not callable(getter):
        return 0.0
    try:
        value = float(getter())
    except Exception:
        return 0.0
    return max(value, 0.0)


def _disable_primary_provider() -> None:
    duration = _provider_switch_cooldown_seconds()
    if duration <= 0:
        return
    try:
        provider_monitor.disable("alpaca", duration=duration)
    except Exception:
        pass


def _primary_on_cooldown(key: tuple[str, str]) -> tuple[bool, float]:
    expiry = _PROVIDER_COOLDOWNS.get(key)
    if expiry is None:
        return False, 0.0
    now = monotonic_time()
    if now >= expiry:
        _PROVIDER_COOLDOWNS.pop(key, None)
        return False, 0.0
    remaining = max(0.0, expiry - now)
    return True, remaining


def _apply_provider_cooldown(
    key: tuple[str, str],
    *,
    symbol: str,
    timeframe: str,
    provider: str,
    fallback_provider: str | None = None,
) -> None:
    window = _provider_decision_window()
    if window <= 0.0:
        return
    expiry = monotonic_time() + window
    previous = _PROVIDER_COOLDOWNS.get(key)
    if previous is not None and previous >= expiry:
        return
    _PROVIDER_COOLDOWNS[key] = expiry
    extra = {
        "symbol": symbol,
        "timeframe": timeframe,
        "provider": provider,
        "cooldown_seconds": round(window, 3),
    }
    if fallback_provider:
        extra["fallback_provider"] = fallback_provider
    logger.info("PRIMARY_PROVIDER_COOLDOWN_SET", extra=extra)


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
        try:
            limit = max_data_fallbacks()
        except Exception:
            limit = 1
        if limit <= 0:
            return None
        fallback_map = {"iex": "sip", "sip": "iex"}
        return fallback_map.get(cur_feed)
    limit = max_data_fallbacks()
    for prov in prio[idx + 1 : idx + 1 + limit]:
        if prov.startswith("alpaca_"):
            return prov.split("_", 1)[1]
    fallback_map = {"iex": "sip", "sip": "iex"}
    return fallback_map.get(cur_feed)


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
    provider_str, normalized_provider = _resolve_backup_provider()
    resolved_provider = normalized_provider or provider_str
    provider_fallback.labels(
        from_provider=f"alpaca_{from_feed}", to_provider=resolved_provider
    ).inc()
    provider_monitor.record_switchover(f"alpaca_{from_feed}", resolved_provider)
    logger.info(
        "DATA_SOURCE_FALLBACK_ATTEMPT",
        extra=_norm_extra({"provider": "yahoo", "fallback": {"interval": interval}}),
    )
    _mark_fallback(
        symbol,
        tf_norm,
        start,
        end,
        from_provider=f"alpaca_{from_feed}",
        fallback_df=df,
        resolved_provider=resolved_provider,
        resolved_feed=normalized_provider or None,
    )
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
    tf_norm = _canon_tf(timeframe)

    cooldown_active, cooldown_remaining = _primary_on_cooldown(tf_key)
    attempted_primary = False
    try:
        if cooldown_active:
            logger.info(
                "PRIMARY_PROVIDER_IN_COOLDOWN",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "provider": f"alpaca_{feed}",
                    "cooldown_remaining": round(cooldown_remaining, 3),
                },
            )
            df = None
            empty_error = True
        else:
            attempted_primary = True
            df = _fetch_bars(symbol, start, end, timeframe, feed=feed)
            empty_error = df is None or getattr(df, "empty", True)
            if not empty_error:
                _EMPTY_BAR_COUNTS.pop(tf_key, None)
                _PROVIDER_COOLDOWNS.pop(tf_key, None)
                mark_success(symbol, timeframe)
                _disable_primary_provider()
                return df
    except EmptyBarsError:
        df = None
        empty_error = True

    logger.info("USING_BACKUP_PROVIDER")
    attempts = _EMPTY_BAR_COUNTS.get(tf_key, 0)
    enable_http = bool(getattr(_fetch_module, "_ENABLE_HTTP_FALLBACK", False))
    try:
        fallback_budget = int(max_data_fallbacks())
    except Exception:
        fallback_budget = 0
    if empty_error and attempts >= _EMPTY_BAR_MAX_RETRIES:
        fetch_skips = getattr(_fetch_module, "_SKIPPED_SYMBOLS", None)
        if isinstance(fetch_skips, set):
            fetch_skips.add(tf_key)
        if enable_http and fallback_budget > 0:
            http_df = _http_fallback(
                symbol,
                start,
                end,
                timeframe,
                from_feed=feed,
            )
            if http_df is not None and not getattr(http_df, "empty", True):
                _EMPTY_BAR_COUNTS.pop(tf_key, None)
                mark_success(symbol, timeframe)
                _apply_provider_cooldown(
                    tf_key,
                    symbol=symbol,
                    timeframe=timeframe,
                    provider=f"alpaca_{feed}",
                    fallback_provider="yahoo",
                )
                _disable_primary_provider()
                return http_df
        raise EmptyBarsError(
            f"empty_bars: symbol={symbol}, timeframe={timeframe}, max_retries={attempts}"
        )
    if feed in ("iex", "sip") and attempted_primary:
        alt_feed = "sip" if feed == "iex" else "iex"
        logger.info("ALPACA_FEED_SWITCH", extra={"from": feed, "to": alt_feed})
        shrink = _dt.timedelta(days=1) if tf_norm.endswith("Min") else _dt.timedelta(0)
        alt_start = start
        if shrink > _dt.timedelta(0):
            truncated = end - shrink
            if truncated > start:
                alt_start = truncated
        try:
            df_alt = _fetch_bars(symbol, alt_start, end, timeframe, feed=alt_feed)
        except EmptyBarsError:
            df_alt = None
        if df_alt is not None and not getattr(df_alt, "empty", True):
            logger.warning("BACKUP_PROVIDER_USED")
            _record_feed_switch(symbol, timeframe, feed, alt_feed)
            _EMPTY_BAR_COUNTS.pop(tf_key, None)
            _PROVIDER_COOLDOWNS.pop(tf_key, None)
            mark_success(symbol, timeframe)
            _disable_primary_provider()
            return df_alt

    if enable_http and fallback_budget > 0:
        http_df = _http_fallback(
            symbol,
            start,
            end,
            timeframe,
            from_feed=feed,
        )
        if http_df is not None and not getattr(http_df, "empty", True):
            _EMPTY_BAR_COUNTS.pop(tf_key, None)
            mark_success(symbol, timeframe)
            _apply_provider_cooldown(
                tf_key,
                symbol=symbol,
                timeframe=timeframe,
                provider=f"alpaca_{feed}",
                fallback_provider="yahoo",
            )
            _disable_primary_provider()
            return http_df

    yf_map = getattr(_fetch_module, "_YF_INTERVAL_MAP", {})
    yf_interval = yf_map.get(tf_norm, tf_norm.lower())
    yahoo_df = _yahoo_get_bars(symbol, start, end, yf_interval)
    _mark_fallback(
        symbol,
        tf_norm,
        start,
        end,
        from_provider=f"alpaca_{feed}",
        fallback_df=yahoo_df,
        resolved_provider="yahoo",
        resolved_feed=None,
    )
    _apply_provider_cooldown(
        tf_key,
        symbol=symbol,
        timeframe=timeframe,
        provider=f"alpaca_{feed}",
        fallback_provider="yahoo",
    )
    if yahoo_df is not None and not getattr(yahoo_df, "empty", True):
        _EMPTY_BAR_COUNTS.pop(tf_key, None)
        mark_success(symbol, timeframe)
        _disable_primary_provider()
    return yahoo_df


__all__ = [
    "_fetch_feed",
    "_SKIPPED_SYMBOLS",
    "_EMPTY_BAR_COUNTS",
    "_EMPTY_BAR_MAX_RETRIES",
]
