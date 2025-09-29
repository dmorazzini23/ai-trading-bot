from __future__ import annotations
import asyncio
import datetime as _dt
import gc
import importlib
import logging
import os
import sys
import time
import warnings
import weakref
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Dict, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo
from ai_trading.utils.lazy_imports import load_pandas
from ai_trading.utils.time import monotonic_time


from ai_trading.data.timeutils import ensure_utc_datetime
from ai_trading.data.market_calendar import is_trading_day, rth_session_utc
from ai_trading.logging.empty_policy import classify as _empty_classify
from ai_trading.logging.empty_policy import record as _empty_record
from ai_trading.logging.empty_policy import should_emit as _empty_should_emit
from ai_trading.logging.normalize import canon_timeframe as _canon_tf
from ai_trading.logging.normalize import normalize_extra as _norm_extra
from ai_trading.logging import (
    log_throttled_event,
    log_backup_provider_used,
    log_empty_retries_exhausted,
    log_fetch_attempt,
    log_finnhub_disabled,
    provider_log_deduper,
    record_provider_log_suppressed,
    warn_finnhub_disabled_no_data,
    get_logger,
)
from ai_trading.logging.emit_once import emit_once
from ai_trading.config.management import MAX_EMPTY_RETRIES, get_env
from ai_trading.config.settings import (
    provider_priority,
    max_data_fallbacks,
    alpaca_feed_failover,
    alpaca_empty_to_backup,
)
from ai_trading.data.empty_bar_backoff import (
    _EMPTY_BAR_COUNTS,
    _SKIPPED_SYMBOLS,
    mark_success,
    record_attempt,
)
from ai_trading.data.metrics import (
    metrics,
    provider_fallback,
    provider_disabled,
    provider_disable_total,
)
from ai_trading.data.provider_monitor import provider_monitor
from ai_trading.core.daily_fetch_memo import get_daily_df_memoized
from .normalize import normalize_ohlcv_df
from ai_trading.monitoring.alerts import AlertSeverity, AlertType
from ai_trading.net.http import HTTPSession, get_http_session, reload_host_limit_if_env_changed
from ai_trading.utils.http import clamp_request_timeout
from ai_trading.utils import safe_to_datetime
from ai_trading.utils.env import (
    alpaca_credential_status,
    resolve_alpaca_feed,
    is_data_feed_downgraded,
    get_data_feed_override,
    get_data_feed_downgrade_reason,
)
from ai_trading.data.finnhub import fh_fetcher, FinnhubAPIException
from . import fallback_order
from .validators import validate_adjustment, validate_feed
from .._alpaca_guard import should_import_alpaca_sdk

logger = get_logger(__name__)


def _sip_allowed() -> bool:
    """Return ``True`` when SIP access is permitted for the current process."""

    override = globals().get("_ALLOW_SIP")
    if override is not None:
        return bool(override)
    try:
        explicit = get_env("ALPACA_ALLOW_SIP", None, cast=bool)
    except Exception:
        explicit = None
    if explicit is not None:
        return bool(explicit)
    raw_allow = os.getenv("ALPACA_ALLOW_SIP")
    if raw_allow is not None:
        return raw_allow.strip().lower() in {"1", "true", "yes", "on"}
    raw_entitlement = os.getenv("ALPACA_HAS_SIP")
    if raw_entitlement is not None:
        return raw_entitlement.strip().lower() in {"1", "true", "yes", "on"}
    return True


def _ordered_fallbacks(primary_feed: str) -> List[str]:
    """Return fallback feeds ordered by preference for *primary_feed*."""

    normalized = (primary_feed or "").strip().lower()
    if normalized == "iex":
        return ["sip", "yahoo"] if _sip_allowed() else ["yahoo"]
    if normalized == "sip":
        return ["yahoo"]
    return ["yahoo"]


_cycle_feed_override: Dict[str, str] = {}
_override_set_ts: Dict[str, float] = {}
_ALLOW_SIP: Optional[bool] | None = None
_OVERRIDE_TTL_S = 600.0
_ENV_STAMP: tuple[str | None, str | None] | None = None


def _env_signature() -> tuple[str | None, str | None]:
    return (
        os.getenv("ALPACA_DATA_FEED"),
        os.getenv("ALPACA_SIP_UNAUTHORIZED"),
    )


def _clear_cycle_overrides() -> None:
    _cycle_feed_override.clear()
    _override_set_ts.clear()


def reload_env_settings() -> None:
    """Reset cached override state when environment configuration changes."""

    global _ENV_STAMP
    _clear_cycle_overrides()
    _ENV_STAMP = _env_signature()


def _ensure_override_state_current() -> None:
    global _ENV_STAMP
    sig = _env_signature()
    if _ENV_STAMP is None:
        _ENV_STAMP = sig
        return
    if sig != _ENV_STAMP:
        _ENV_STAMP = sig
        _clear_cycle_overrides()


def _record_override(symbol: str, feed: str) -> None:
    normalized = str(feed).strip().lower()
    _cycle_feed_override[symbol] = normalized
    _override_set_ts[symbol] = time.time()
    _remember_fallback_for_cycle(_get_cycle_id(), symbol, "1Min", normalized)


def _clear_override(symbol: str) -> None:
    _cycle_feed_override.pop(symbol, None)
    _override_set_ts.pop(symbol, None)


def _get_cached_or_primary(symbol: str, primary_feed: str) -> str:
    cached = _cycle_feed_override.get(symbol)
    if cached:
        ts = _override_set_ts.get(symbol, 0.0)
        if ts and (time.time() - ts) <= _OVERRIDE_TTL_S:
            return cached
        _clear_override(symbol)
    normalized_primary = str(primary_feed or "iex").strip().lower() or "iex"
    return normalized_primary


def _cache_fallback(symbol: str, feed: str) -> None:
    if not feed:
        return
    _record_override(symbol, feed)


async def run_with_concurrency(limit: int, coros):
    """Execute *coros* concurrently while keeping at most *limit* in flight."""

    max_concurrency = max(1, int(limit or 1))
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run(index: int, coro):
        async with semaphore:
            try:
                result = await coro
            except BaseException as exc:  # noqa: BLE001 - surface alongside results
                return index, exc
            return index, result

    pending: set[asyncio.Task[tuple[int, Any]]] = set()
    results: dict[int, Any] = {}
    total = 0
    iterator = iter(coros)

    while True:
        while len(pending) < max_concurrency:
            try:
                coro = next(iterator)
            except StopIteration:
                break
            pending.add(asyncio.create_task(_run(total, coro)))
            total += 1

        if not pending:
            break

        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            index, value = task.result()
            results[index] = value

    ordered_results = [results[i] for i in range(total)] if total else []
    succeeded = sum(1 for item in ordered_results if not isinstance(item, Exception))
    failed = total - succeeded
    return ordered_results, succeeded, failed


_daily_memo: Dict[Tuple[str, str], Tuple[float, Any]] = {}
_DAILY_TTL_S = 60.0


def daily_fetch_memo(key: Tuple[str, str], value_factory):
    """Memoize intraday daily fetch results for a short TTL."""

    now = time.time()
    cached = _daily_memo.get(key)
    if cached is not None:
        ts, value = cached
        if (now - ts) < _DAILY_TTL_S:
            return value
    try:
        value = value_factory()
    except StopIteration:
        _daily_memo.pop(key, None)
        return None
    _daily_memo[key] = (now, value)
    return value


def _emit_capture_record(
    message: str,
    *,
    level: int = logging.WARNING,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Emit ``message`` directly to pytest's ``caplog`` handler when active."""

    try:
        handler = _find_pytest_capture_handler()
        if handler is None:
            return
        base_logger = getattr(logger, "logger", logging.getLogger(__name__))
        try:
            record = base_logger.makeRecord(
                base_logger.name,
                level,
                __file__,
                0,
                message,
                None,
                None,
                extra=dict(extra or {}),
            )
        except Exception:
            record = base_logger.makeRecord(base_logger.name, level, __file__, 0, message, (), None)
            if extra:
                for key, value in extra.items():
                    setattr(record, key, value)
        try:
            record.message = record.getMessage()
        except Exception:
            record.message = message
        handler.emit(record)
    except Exception:
        # Avoid interfering with production logging when pytest internals change.
        return


_YF_INTERVAL_MAP = {
    "1Min": "1m",
    "5Min": "5m",
    "15Min": "15m",
    "1Hour": "60m",
    "1Day": "1d",
}


# Lightweight indirection to support tests monkeypatching `data_fetcher.get_settings`
def get_settings():  # pragma: no cover - simple alias for tests
    from ai_trading.config.settings import get_settings as _get

    settings = _get()
    _ensure_override_state_current()
    return settings


# Module-level session reused across requests
_HTTP_SESSION: HTTPSession = get_http_session()


_JSON_PRIMITIVE_TYPES = (str, int, float, bool, type(None))


def _coerce_json_primitives(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a JSON-serialisable copy of *data* for structured logging."""

    if not data:
        return {}
    safe: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, bool):
            safe[key] = value
        elif isinstance(value, (int,)):
            safe[key] = int(value)
        elif isinstance(value, float):
            safe[key] = float(value)
        elif isinstance(value, _JSON_PRIMITIVE_TYPES):
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


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


def _normalize_feed_value(feed: object) -> str:
    """Return canonical feed string while enforcing validation."""

    try:
        candidate = str(feed).strip().lower()
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"invalid feed: {feed!r}") from exc

    if not candidate:
        raise ValueError("invalid feed: ''")

    if candidate.startswith("alpaca_"):
        candidate = candidate.split("_", 1)[1]

    validate_feed(candidate)
    return candidate


def _to_feed_str(feed: object) -> str:
    """Return canonical feed string with strict validation."""

    return _normalize_feed_value(feed)


class DataFetchError(Exception):
    """Error raised when market data retrieval fails."""  # AI-AGENT-REF: stable public symbol


class MissingOHLCVColumnsError(DataFetchError):
    """Raised when a provider omits required OHLCV columns."""


# Dedicated error for SIP authorization failures so callers can branch.
class UnauthorizedSIPError(DataFetchError):
    """Raised when Alpaca SIP requests fail due to authorization issues."""


# Backwards compat alias
DataFetchException = DataFetchError


class EmptyBarsError(DataFetchError, ValueError):
    """Raised when a data provider returns no bars for a request."""


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

# Track consecutive error="empty" responses specifically from the IEX feed to
# allow proactive SIP fallback on subsequent requests.
_IEX_EMPTY_COUNTS: dict[tuple[str, str], int] = {}
_IEX_EMPTY_THRESHOLD = 1
# Track consecutive Alpaca `error="empty"` responses per (symbol, timeframe)
# to short-circuit further Alpaca requests and fall back to the secondary
# provider when the upstream repeatedly returns empty payloads.
_ALPACA_EMPTY_ERROR_COUNTS: dict[tuple[str, str], int] = {}
_ALPACA_EMPTY_ERROR_THRESHOLD = int(os.getenv("ALPACA_EMPTY_ERROR_THRESHOLD", "2"))
_EMPTY_BAR_THRESHOLD = 3
_EMPTY_BAR_MAX_RETRIES = MAX_EMPTY_RETRIES
_FETCH_BARS_MAX_RETRIES = int(os.getenv("FETCH_BARS_MAX_RETRIES", "5"))
# Configurable backoff parameters for retry logic
_FETCH_BARS_BACKOFF_BASE = float(os.getenv("FETCH_BARS_BACKOFF_BASE", "2"))
_FETCH_BARS_BACKOFF_CAP = float(os.getenv("FETCH_BARS_BACKOFF_CAP", "5"))
_http_fallback_env = os.getenv("ENABLE_HTTP_FALLBACK")
if _http_fallback_env is None:
    _ENABLE_HTTP_FALLBACK = True
else:
    _ENABLE_HTTP_FALLBACK = _http_fallback_env.strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

# Track fallback usage to avoid repeated Alpaca requests for the same window
_FALLBACK_WINDOWS: set[tuple[str, str, int, int]] = set()
# Soft memory of fallback usage per (symbol, timeframe) to suppress repeated
# primary-provider attempts for slightly shifted windows in the same cycle.
_FALLBACK_UNTIL: dict[tuple[str, str], int] = {}
_FALLBACK_METADATA: dict[tuple[str, str, int, int], dict[str, str]] = {}
_FALLBACK_TTL_SECONDS = int(os.getenv("FALLBACK_TTL_SECONDS", "180"))
# Track backup provider log emissions to avoid duplicate INFO spam for the same
# symbol/timeframe per cycle. The mapping is pruned as cycles advance to avoid
# unbounded growth.
_BACKUP_USAGE_LOGGED: dict[str, set[tuple[str, str]]] = {}
_BACKUP_USAGE_MAX_CYCLES = 6

# Throttled Yahoo Finance warnings per event -> cycle -> symbols
_YF_WARNING_CACHE: dict[str, dict[str, set[tuple[str, str]]]] = {}
_YF_WARNING_MAX_CYCLES = 6


def _cycle_bucket(store: dict[str, set[tuple[str, str]]], max_cycles: int) -> tuple[str, set[tuple[str, str]]]:
    """Return the per-cycle set for ``store`` while pruning old entries."""

    cycle_id = _get_cycle_id()
    bucket = store.setdefault(cycle_id, set())
    if len(store) > max_cycles:
        for key in list(store.keys()):
            if key == cycle_id:
                continue
            store.pop(key, None)
            if len(store) <= max_cycles:
                break
    return cycle_id, bucket


def _missing_alpaca_warning_context() -> tuple[bool, dict[str, object]]:
    extra: dict[str, object] = {}

    if os.getenv("ALPACA_SIP_UNAUTHORIZED") == "1" or _is_sip_unauthorized():
        extra["sip_locked"] = True
        return False, extra

    try:
        settings = get_settings()
    except Exception:
        settings = None

    provider = str(getattr(settings, "data_provider", "alpaca") or "").strip()
    if provider:
        extra["provider"] = provider
    provider_normalized = provider.lower()
    if provider_normalized and "alpaca" not in provider_normalized:
        return False, extra

    override = None
    try:
        override = get_data_feed_override()
    except Exception:
        override = None
    if override:
        extra["override"] = override
        override_norm = override.strip().lower()
        if override_norm and override_norm not in {"iex", "sip"} and not override_norm.startswith("alpaca"):
            # When feed is forced to a non-Alpaca provider (e.g. yahoo), skip warning.
            return False, extra

    try:
        resolved_feed = resolve_alpaca_feed(None)
    except Exception:
        resolved_feed = "iex"

    if resolved_feed is None:
        extra["feed"] = "disabled"
        return False, extra

    extra["feed"] = resolved_feed
    return True, extra


def _warn_missing_alpaca(symbol: str, timeframe: str) -> None:
    """Emit an Alpaca credential warning at most once per UTC day per symbol."""

    should_warn, extra = _missing_alpaca_warning_context()
    if not should_warn:
        return

    extra.update({"symbol": symbol, "timeframe": timeframe})
    emit_once(
        logger,
        f"ALPACA_API_KEY_MISSING:{symbol}:{timeframe}",
        "warning",
        "ALPACA_API_KEY_MISSING",
        **extra,
    )


def _log_yf_warning(event: str, symbol: str, timeframe: str, extra: dict[str, object]) -> None:
    """Log Yahoo Finance warnings once per cycle/event/symbol."""

    store = _YF_WARNING_CACHE.setdefault(event, {})
    _, bucket = _cycle_bucket(store, _YF_WARNING_MAX_CYCLES)
    key = (str(symbol).upper(), str(timeframe).upper())
    if key in bucket:
        return
    logger.warning(event, extra=extra)
    bucket.add(key)

# Track feed failovers so we avoid redundant retries for the same symbol/timeframe.
_FEED_OVERRIDE_BY_TF: dict[tuple[str, str], str] = {}
_FEED_SWITCH_LOGGED: set[tuple[str, str, str]] = set()
_FEED_SWITCH_HISTORY: list[tuple[str, str, str]] = []
_FEED_FAILOVER_ATTEMPTS: dict[tuple[str, str], set[str]] = {}


def _preferred_feed_failover() -> tuple[str, ...]:
    try:
        feeds = alpaca_feed_failover()
    except Exception:
        return ()
    return tuple(feeds or ())


def _should_use_backup_on_empty() -> bool:
    try:
        return bool(alpaca_empty_to_backup())
    except Exception:
        return True


def _iter_preferred_feeds(symbol: str, timeframe: str, current_feed: str) -> tuple[str, ...]:
    key = (symbol, timeframe)
    attempted = _FEED_FAILOVER_ATTEMPTS.setdefault(key, set())
    feeds = list(_preferred_feed_failover())
    if not feeds:
        try:
            providers = provider_priority()
        except Exception:
            providers = ()
        for provider in providers:
            if not provider:
                continue
            if isinstance(provider, str) and provider.startswith("alpaca_"):
                feeds.append(provider.split("_", 1)[1])
            elif isinstance(provider, str):
                feeds.append(provider)
    out: list[str] = []
    sip_locked = _is_sip_unauthorized()
    for raw in feeds:
        try:
            candidate = _to_feed_str(raw)
        except ValueError:
            continue
        if candidate == current_feed:
            continue
        if candidate in attempted:
            continue
        if candidate == "sip" and sip_locked:
            attempted.add(candidate)
            continue
        attempted.add(candidate)
        out.append(candidate)
    return tuple(out)


def _record_feed_switch(symbol: str, timeframe: str, from_feed: str, to_feed: str) -> None:
    key = (symbol, timeframe)
    _FEED_OVERRIDE_BY_TF[key] = to_feed
    _record_override(symbol, to_feed)
    attempted = _FEED_FAILOVER_ATTEMPTS.setdefault(key, set())
    attempted.add(to_feed)
    _FEED_SWITCH_HISTORY.append((symbol, timeframe, to_feed))
    if from_feed == "iex":
        _IEX_EMPTY_COUNTS.pop(key, None)
    log_key = (symbol, timeframe, to_feed)
    if log_key not in _FEED_SWITCH_LOGGED:
        logger.info(
            "ALPACA_FEED_SWITCH",
            extra=_norm_extra(
                {
                    "provider": "alpaca",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "from_feed": from_feed,
                    "to_feed": to_feed,
                }
            ),
        )
        _FEED_SWITCH_LOGGED.add(log_key)


# Track consecutive empty Alpaca responses across all symbols to temporarily
# disable Alpaca fetching when upstream repeatedly returns empty payloads.
_ALPACA_DISABLE_THRESHOLD = 3
_alpaca_empty_streak = 0
_alpaca_disabled_until: _dt.datetime | None = None
_ALPACA_DISABLED_ALERTED = False
# Track consecutive disable events for logging purposes
_alpaca_disable_count = 0


def is_primary_provider_enabled() -> bool:
    """Return ``True`` when Alpaca minute data is not in a cooldown window."""

    if _alpaca_disabled_until is None:
        return True
    return _dt.datetime.now(UTC) >= _alpaca_disabled_until


# Emit a one-time explanatory log when Alpaca keys are missing to make
# backup-provider usage obvious in production logs without spamming.
_ALPACA_KEYS_MISSING_LOGGED = False


def _disable_alpaca(duration: _dt.timedelta) -> None:
    """Disable Alpaca as a data source for ``duration``."""

    global _alpaca_disabled_until, _ALPACA_DISABLED_ALERTED, _alpaca_disable_count

    # Exponential backoff to avoid rapid flip/flop between providers. Each
    # consecutive disable doubles the cooldown window up to the monitor's
    # ``max_cooldown`` ceiling.
    count = _alpaca_disable_count + 1
    backoff = duration * (2 ** (count - 1))
    max_cooldown = getattr(provider_monitor, "max_cooldown", None)
    if max_cooldown is not None:
        cap = _dt.timedelta(seconds=max_cooldown)
        if backoff > cap:
            backoff = cap
    _alpaca_disabled_until = datetime.now(UTC) + backoff
    _ALPACA_DISABLED_ALERTED = False
    _alpaca_disable_count = count
    try:
        logger.warning(
            "ALPACA_TEMP_DISABLED",
            extra=_norm_extra(
                {
                    "provider": "alpaca",
                    "disabled_until": _alpaca_disabled_until.isoformat(),
                    "backoff_seconds": backoff.total_seconds(),
                    "disable_count": _alpaca_disable_count,
                }
            ),
        )
    except Exception:
        pass


# Register disable callback with provider monitor so repeated failures trigger
# temporary provider deactivation and downstream fallbacks.
provider_monitor.register_disable_callback("alpaca", _disable_alpaca)


def _fallback_key(symbol: str, timeframe: str, start: _dt.datetime, end: _dt.datetime) -> tuple[str, str, int, int]:
    return (symbol, timeframe, int(start.timestamp()), int(end.timestamp()))


def _mark_fallback(
    symbol: str,
    timeframe: str,
    start: _dt.datetime,
    end: _dt.datetime,
    *,
    from_provider: str | None = None,
    fallback_feed: str | None = None,
    fallback_df: Any | None = None,
    resolved_provider: str | None = None,
    resolved_feed: str | None = None,
) -> None:
    """Record usage of the configured backup provider.

    Side effects
    -----------
    * Logs a "backup provider used" message once per unique window.
    * Notifies the :mod:`provider_monitor` of the switchover.
    * Appends ``(provider, symbol)`` to the shared registries in
      :mod:`ai_trading.data.fetch.fallback_order` for test introspection.
    """

    configured_provider, configured_normalized = _resolve_backup_provider()

    def _normalize(value: Any | None) -> str | None:
        if value is None:
            return None
        try:
            normalized_value = str(value).strip()
        except Exception:
            return None
        return normalized_value or None

    def _normalize_lower(value: Any | None) -> str | None:
        normalized_value = _normalize(value)
        if normalized_value is None:
            return None
        return normalized_value.lower()

    provider_from_df: str | None = None
    feed_from_df: str | None = None
    if fallback_df is not None:
        try:
            attrs = getattr(fallback_df, "attrs", None)
        except Exception:  # pragma: no cover - attrs access best-effort
            attrs = None
        if isinstance(attrs, Mapping):
            provider_from_df = _normalize(
                attrs.get("data_provider") or attrs.get("fallback_provider")
            )
            feed_from_df = _normalize(attrs.get("data_feed") or attrs.get("fallback_feed"))

    provider_hint = _normalize(resolved_provider) or provider_from_df
    if provider_hint is None:
        provider_hint = _normalize(configured_normalized) or _normalize(configured_provider)

    feed_hint = (
        feed_from_df
        or _normalize(resolved_feed)
        or _normalize(fallback_feed)
    )
    if feed_hint is None and provider_hint and provider_hint.startswith("alpaca_"):
        feed_hint = provider_hint.split("_", 1)[1]

    provider_hint_lower = _normalize_lower(provider_hint)
    feed_hint_lower = _normalize_lower(feed_hint)
    resolved_provider_lower = _normalize_lower(resolved_provider)
    resolved_feed_lower = _normalize_lower(resolved_feed)
    fallback_feed_lower = _normalize_lower(fallback_feed)
    provider_from_df_lower = _normalize_lower(provider_from_df)
    feed_from_df_lower = _normalize_lower(feed_from_df)

    yahoo_detected = any(
        hint == "yahoo"
        for hint in (
            resolved_provider_lower,
            resolved_feed_lower,
            fallback_feed_lower,
            provider_hint_lower,
            provider_from_df_lower,
            feed_hint_lower,
            feed_from_df_lower,
        )
    )

    if yahoo_detected:
        provider_hint = "yahoo"
        provider_hint_lower = "yahoo"

    provider_for_register = provider_hint or _normalize(configured_provider) or "yahoo"
    provider_for_register_lower = _normalize_lower(provider_for_register)

    if from_provider:
        current = str(from_provider).strip().lower()
        if current and provider_for_register_lower and current == provider_for_register_lower:
            return

    key = _fallback_key(symbol, timeframe, start, end)
    fallback_order.register_fallback(provider_for_register, symbol)
    metadata: dict[str, str] = {"fallback_provider": provider_for_register}
    log_extra: dict[str, str] = dict(metadata)
    if configured_provider and _normalize(configured_provider) != provider_for_register:
        metadata["configured_fallback_provider"] = _normalize(configured_provider) or configured_provider
    if from_provider:
        metadata["from_provider"] = from_provider
        log_extra["from_provider"] = from_provider
    if feed_hint:
        metadata["fallback_feed"] = feed_hint
        log_extra["fallback_feed"] = feed_hint
    _FALLBACK_METADATA[key] = metadata
    # Emit once per unique (symbol, timeframe, window)
    if key not in _FALLBACK_WINDOWS:
        payload = log_backup_provider_used(
            provider_for_register,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            extra=log_extra,
        )
        logger.warning("BACKUP_PROVIDER_USED", extra=payload)
        provider_monitor.record_switchover(
            from_provider or "alpaca",
            provider_for_register,
        )
    _FALLBACK_WINDOWS.add(key)
    fallback_name: str | None = None
    if feed_hint:
        try:
            fallback_name = _normalize_feed_value(feed_hint)
        except Exception:
            fallback_name = str(feed_hint)
    elif provider_for_register and provider_for_register.startswith("alpaca_"):
        fallback_name = provider_for_register.split("_", 1)[1]
    elif provider_for_register:
        fallback_name = str(provider_for_register)
    if fallback_name:
        fallback_clean = str(fallback_name).strip().lower()
        if fallback_clean:
            _remember_fallback_for_cycle(_get_cycle_id(), symbol, timeframe, fallback_clean)
    # Also remember at a coarser granularity for a short TTL to avoid
    # repeated primary-provider retries for small window shifts in the same run.
    try:
        now_s = int(_dt.datetime.now(tz=UTC).timestamp())
    except Exception:
        now_s = int(time.time())
    _FALLBACK_UNTIL[(symbol, timeframe)] = now_s + max(30, _FALLBACK_TTL_SECONDS)


def _used_fallback(symbol: str, timeframe: str, start: _dt.datetime, end: _dt.datetime) -> bool:
    return _fallback_key(symbol, timeframe, start, end) in _FALLBACK_WINDOWS


def _annotate_df_source(
    df: pd.DataFrame | None,
    *,
    provider: str,
    feed: str | None = None,
) -> pd.DataFrame | None:
    """Attach provider/feed metadata to ``df`` when supported."""

    pd_local = _ensure_pandas()
    if pd_local is None or df is None or not isinstance(df, pd_local.DataFrame):
        return df
    try:
        attrs = df.attrs  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - extremely defensive
        return df

    try:
        attrs["data_provider"] = provider
        attrs.setdefault("fallback_provider", provider)
        if feed:
            attrs["data_feed"] = feed
            attrs.setdefault("fallback_feed", feed)
    except Exception:  # pragma: no cover - metadata best-effort only
        pass
    return df


def _normalize_with_attrs(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV frame while preserving existing attributes."""

    attrs: dict[str, Any] = {}
    try:
        attrs = dict(getattr(df, "attrs", {}) or {})
    except (AttributeError, TypeError):
        attrs = {}
    normalized = normalize_ohlcv_df(df)
    if attrs:
        try:
            normalized.attrs.update(attrs)
        except (AttributeError, TypeError, ValueError):
            pass
    return normalized


def _restore_timestamp_column(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Ensure ``df`` exposes a ``timestamp`` column aligned with its index."""

    if df is None:
        return None

    pd_local = _ensure_pandas()
    if pd_local is None or not isinstance(df, pd_local.DataFrame):
        return df

    if "timestamp" in df.columns:
        return df

    try:
        index_values = df.index
    except AttributeError:
        return df

    try:
        # Copy to avoid mutating shared references downstream.
        frame = df.copy()
    except Exception:  # pragma: no cover - defensive fallback
        frame = df

    try:
        frame.insert(0, "timestamp", index_values)
    except Exception:
        try:
            frame["timestamp"] = index_values
        except Exception:  # pragma: no cover - last-resort guard
            return frame
    return frame


# --- BEGIN: universal OHLCV normalization helper ---
try:
    import pandas as _pd_norm_helper  # ok if already imported elsewhere; safe to repeat
except ImportError:  # pragma: no cover - optional dependency missing
    _pd_norm_helper = None


def ensure_ohlcv_schema(
    df: Any,
    *,
    source: str,
    frequency: str,
    _pd: Any | None = None,
) -> "pd.DataFrame":
    """Return a DataFrame with canonical OHLCV columns and UTC timestamp index."""

    pd_local = _pd if _pd is not None else _ensure_pandas()
    if pd_local is None:
        raise DataFetchError("pandas_unavailable")

    if df is None:
        raise DataFetchError("DATA_FETCH_EMPTY")

    if not isinstance(df, pd_local.DataFrame):
        try:
            df = pd_local.DataFrame(df)
        except Exception as exc:  # pragma: no cover - defensive conversion
            raise DataFetchError("DATA_FETCH_EMPTY") from exc

    if df.empty:
        raise DataFetchError("DATA_FETCH_EMPTY")

    work_df = df.copy()

    rename_map: dict[Any, str] = {}
    ts_candidates: dict[str, Any] = {}
    for col in list(work_df.columns):
        key = str(col).strip().lower()
        if key in {"timestamp", "time", "datetime", "date", "t"}:
            ts_candidates.setdefault("timestamp", col)
            rename_map[col] = "timestamp"
        elif key in {"open", "o"}:
            rename_map[col] = "open"
        elif key in {"high", "h"}:
            rename_map[col] = "high"
        elif key in {"low", "l"}:
            rename_map[col] = "low"
        elif key in {"close", "c"}:
            rename_map[col] = "close"
        elif key in {"adj close", "adj_close", "adjclose", "adjusted_close"}:
            rename_map[col] = "adj_close"
        elif key in {"volume", "v"}:
            rename_map[col] = "volume"

    if rename_map:
        work_df = work_df.rename(columns=rename_map)

    timestamp_col = None
    if "timestamp" in work_df.columns:
        timestamp_col = "timestamp"
    elif ts_candidates:
        # Prefer whichever candidate was recorded first
        timestamp_col = next(iter(ts_candidates.values()))
        work_df = work_df.rename(columns={timestamp_col: "timestamp"})
        timestamp_col = "timestamp"

    if timestamp_col is None:
        if isinstance(work_df.index, pd_local.DatetimeIndex):
            timestamp_col = "timestamp"
            work_df = work_df.reset_index()
            first_column = work_df.columns[0]
            work_df = work_df.rename(columns={first_column: "timestamp"})
        else:
            raise MissingOHLCVColumnsError(
                f"missing timestamp column | source={source} frequency={frequency}"
            )

    warn_prefix = f"{str(source or 'unknown')}:{str(frequency or 'unknown')}"
    timestamps = safe_to_datetime(
        work_df[timestamp_col],
        utc=True,
        context=warn_prefix,
        _warn_key=f"SAFE_TO_DATETIME:{warn_prefix.upper()}",
    )
    work_df["timestamp"] = timestamps
    work_df = work_df.dropna(subset=["timestamp"])

    if work_df.empty:
        raise DataFetchError("DATA_FETCH_EMPTY")

    if "close" not in work_df.columns and "adj_close" in work_df.columns:
        work_df["close"] = work_df["adj_close"]

    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in work_df.columns]
    if missing:
        raise MissingOHLCVColumnsError(
            f"OHLCV_COLUMNS_MISSING | missing={missing} source={source} frequency={frequency}"
        )

    if "volume" not in work_df.columns:
        work_df["volume"] = 0

    work_df = work_df.set_index("timestamp", drop=False)
    work_df = work_df.sort_index()
    if getattr(work_df.index, "has_duplicates", False):
        work_df = work_df[~work_df.index.duplicated(keep="last")]

    work_df = work_df.dropna(subset=["open", "high", "low", "close", "volume"], how="any")

    if work_df.empty:
        raise DataFetchError("DATA_FETCH_EMPTY")

    desired_order = ["timestamp", "open", "high", "low", "close", "volume"]
    other_cols = [col for col in work_df.columns if col not in desired_order]
    ordered_cols = desired_order + other_cols
    work_df = work_df.reindex(columns=ordered_cols)

    if hasattr(work_df.index, "tz"):
        if work_df.index.tz is None:
            work_df.index = work_df.index.tz_localize("UTC")
        else:
            work_df.index = work_df.index.tz_convert("UTC")
    work_df.index.name = "timestamp"
    return work_df


def _normalize_ohlcv_df(df, _pd: Any | None = None):
    """Backwards-compatible shim that delegates to :func:`ensure_ohlcv_schema`."""

    if df is None:
        return None
    try:
        normalized = ensure_ohlcv_schema(
            df,
            source="unknown",
            frequency="unknown",
            _pd=_pd if _pd is not None else _pd_norm_helper,
        )
    except (DataFetchError, MissingOHLCVColumnsError):
        return None
    except Exception:
        return None
    try:
        normalized = normalize_ohlcv_df(normalized)
    except Exception:
        return normalized
    restored = _restore_timestamp_column(normalized)
    return restored if restored is not None else normalized


# --- END: universal OHLCV normalization helper ---


def _empty_ohlcv_frame(pd_local: Any | None = None) -> pd.DataFrame | None:
    """Return an empty, normalized OHLCV DataFrame."""

    if pd_local is None:
        pd_local = _ensure_pandas()
    if pd_local is None:
        return None
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    base = pd_local.DataFrame({col: [] for col in cols})
    return normalize_ohlcv_df(base)


def _resolve_backup_provider() -> tuple[str, str]:
    provider_val = getattr(get_settings(), "backup_data_provider", "yahoo")
    provider_str = str(provider_val).strip()
    normalized = provider_str.lower()
    return provider_str, normalized


_CAPTURE_HANDLER_REF: weakref.ReferenceType[logging.Handler] | None = None
_CAPTURE_LOCK = Lock()
_FINNHUB_CAPTURE_KEYS: set[str] = set()


def _pytest_logging_active() -> bool:
    """Return ``True`` when pytest's logging capture is active."""

    if os.getenv("PYTEST_RUNNING") in {"1", "true", "True"}:
        return True
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True
    try:
        root_handlers = getattr(logging.getLogger(), "handlers", [])
        return any(h.__class__.__name__ == "LogCaptureHandler" for h in root_handlers)
    except Exception:  # pragma: no cover - defensive
        return False


def _find_pytest_capture_handler() -> logging.Handler | None:
    """Return the active ``LogCaptureHandler`` when pytest's caplog is in use."""

    if not _pytest_logging_active():
        return None

    global _CAPTURE_HANDLER_REF
    ref = _CAPTURE_HANDLER_REF
    handler = ref() if ref is not None else None
    if handler is not None and not getattr(handler, "closed", False) and getattr(handler, "records", None) is not None:
        return handler

    with _CAPTURE_LOCK:
        ref = _CAPTURE_HANDLER_REF
        handler = ref() if ref is not None else None
        if handler is not None and not getattr(handler, "closed", False) and getattr(handler, "records", None) is not None:
            return handler

        root = logging.getLogger()
        for existing in getattr(root, "handlers", []):
            try:
                if existing.__class__.__name__ == "LogCaptureHandler" and getattr(existing, "records", None) is not None and not getattr(existing, "closed", False):
                    _CAPTURE_HANDLER_REF = weakref.ref(existing)
                    return existing
            except Exception:
                continue

        for obj in gc.get_objects():
            try:
                if isinstance(obj, logging.Handler) and obj.__class__.__name__ == "LogCaptureHandler":
                    if getattr(obj, "records", None) is None or getattr(obj, "closed", False):
                        continue
                    _CAPTURE_HANDLER_REF = weakref.ref(obj)
                    return obj
            except Exception:
                continue

        _CAPTURE_HANDLER_REF = None
        return None


def _log_with_capture(level: int, message: str, extra: Mapping[str, Any] | None = None) -> None:
    """Log ``message`` and mirror it to pytest's caplog handler when active."""

    payload = dict(extra) if extra else None
    logger.log(level, message, extra=payload)

    handler = _find_pytest_capture_handler()
    if handler is None:
        return

    record_extra = dict(extra) if extra else None
    try:
        record = logger.makeRecord(
            logger.name,
            level,
            __file__,
            0,
            message,
            (),
            None,
            extra=record_extra,
        )
    except Exception:
        record = logger.makeRecord(logger.name, level, __file__, 0, message, (), None)
        if record_extra:
            for key, value in record_extra.items():
                setattr(record, key, value)

    handler_level = getattr(handler, "level", logging.NOTSET)
    if isinstance(handler_level, int) and handler_level > logging.NOTSET and record.levelno < handler_level:
        return

    try:
        handler.emit(record)
    except Exception:
        pass


def _log_fetch_minute_empty(
    provider_feed: str,
    reason: str,
    detail: str | None = None,
    *,
    symbol: str | None = None,
) -> None:
    extra = {
        "provider": provider_feed,
        "timeframe": "1Min",
        "reason": reason,
    }
    if detail:
        extra["detail"] = detail
    if symbol:
        extra["symbol"] = symbol
    _log_with_capture(logging.WARNING, "FETCH_MINUTE_EMPTY", extra=extra)


def get_fallback_metadata(
    symbol: str,
    timeframe: str,
    start: _dt.datetime,
    end: _dt.datetime,
) -> dict[str, str] | None:
    """Return recorded fallback metadata for ``symbol`` and window if available."""

    key = _fallback_key(symbol, timeframe, start, end)
    if key in _FALLBACK_METADATA:
        return dict(_FALLBACK_METADATA[key])
    return None


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


def _validate_alpaca_params(start: _dt.datetime, end: _dt.datetime, timeframe: str, feed: str, adjustment: str) -> None:
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
            try:
                open_dt, close_dt = rth_session_utc(day)
            except (RuntimeError, ValueError):
                day += _dt.timedelta(days=1)
                continue
            if end > open_dt and start < close_dt:
                return True
        day += _dt.timedelta(days=1)
    return False


_ALPACA_CREDS_CACHE: tuple[bool, float] | None = None
_ALPACA_CREDS_TTL_SECONDS = 120.0


def _has_alpaca_keys() -> bool:
    """Return True if Alpaca API credentials appear configured.

    The detection prefers cached TradingConfig-derived credentials to avoid
    false negatives when environment aliases differ between trading and data
    subsystems. Results are memoized briefly to avoid repeated config reloads.
    """

    global _ALPACA_CREDS_CACHE
    now = monotonic_time()
    if is_data_feed_downgraded():
        _ALPACA_CREDS_CACHE = (False, now)
        return False

    if _ALPACA_CREDS_CACHE is not None:
        cached_value, cached_ts = _ALPACA_CREDS_CACHE
        if now - cached_ts < _ALPACA_CREDS_TTL_SECONDS:
            return cached_value

    # Prefer credential truth from the live execution engine when available.
    try:
        from ai_trading.execution import live_trading as _live_exec

        truth_fn = getattr(_live_exec, "get_cached_credential_truth", None)
    except Exception:
        truth_fn = None
    if callable(truth_fn):
        try:
            has_key_live, has_secret_live, ts = truth_fn()
        except Exception:
            has_key_live = has_secret_live = False
            ts = 0.0
        else:
            if ts:
                if has_key_live and has_secret_live:
                    _ALPACA_CREDS_CACHE = (True, now)
                    return True
                # When live trading explicitly reports missing credentials,
                # cache the negative result but continue checking env/env-config
                # to allow startup environments to recover later in the cycle.
                _ALPACA_CREDS_CACHE = (False, now)

    has_key, has_secret = alpaca_credential_status()
    if has_key and has_secret:
        _ALPACA_CREDS_CACHE = (True, now)
        return True

    resolved = False
    try:
        from ai_trading.config.management import TradingConfig

        cfg = TradingConfig.from_env(allow_missing_drawdown=True)
    except Exception:
        cfg = None
    if cfg is not None:
        key = getattr(cfg, "alpaca_api_key", None)
        secret = getattr(cfg, "alpaca_secret_key", None)
        resolved = bool(key) and bool(secret)

    if resolved:
        _ALPACA_CREDS_CACHE = (True, now)
        return True

    # Restore cache entry when no sources yield credentials.
    _ALPACA_CREDS_CACHE = (False, now)
    return False


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


try:
    _cfg_default = get_settings()
    _DEFAULT_FEED = (
        getattr(_cfg_default, "data_feed", None) or getattr(_cfg_default, "alpaca_data_feed", "iex") or "iex"
    )
except Exception:  # pragma: no cover - defensive default
    _DEFAULT_FEED = "iex"

_DATA_FEED_OVERRIDE = get_data_feed_override()
if _DATA_FEED_OVERRIDE:
    logger.info(
        "DATA_PROVIDER_DOWNGRADED",
        extra={
            "from": f"alpaca_{_DEFAULT_FEED or 'iex'}",
            "to": _DATA_FEED_OVERRIDE,
            "reason": get_data_feed_downgrade_reason() or "missing_credentials",
        },
    )


def _env_flag(key: str, default: bool = False) -> bool:
    """Return truthy flag for ``key`` honouring ``default`` when unset."""

    raw = os.getenv(key)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _prefers_sip() -> bool:
    feed = os.getenv("ALPACA_DATA_FEED", "iex").strip().lower()
    if "sip" in feed:
        return True
    failover = os.getenv("ALPACA_FEED_FAILOVER", "")
    if any(part.strip().lower() == "sip" for part in failover.split(",")):
        return True
    try:
        settings = get_settings()
    except Exception:
        return False
    failover_tuple = getattr(settings, "alpaca_feed_failover", ()) or ()
    return any(str(part).strip().lower() == "sip" for part in failover_tuple)


_SIP_UNAUTHORIZED = _env_flag("ALPACA_SIP_UNAUTHORIZED")
_HAS_SIP = _env_flag("ALPACA_HAS_SIP")
_SIP_DISALLOWED_WARNED = False
_SIP_PRECHECK_DONE = False
_SIP_UNAVAILABLE_LOGGED: set[tuple[str, str]] = set()
_SIP_UNAUTHORIZED_UNTIL: float | None = None
_CYCLE_FALLBACK_FEED: dict[tuple[str, str, str], str] = {}


def _now_monotonic() -> float:
    return monotonic_time()


def _pytest_active() -> bool:
    return bool(os.getenv("PYTEST_RUNNING") or os.getenv("PYTEST_CURRENT_TEST"))


def _is_sip_unauthorized() -> bool:
    global _SIP_UNAUTHORIZED, _SIP_UNAUTHORIZED_UNTIL
    if not _SIP_UNAUTHORIZED:
        return False
    if _SIP_UNAUTHORIZED_UNTIL is None:
        return True
    if _now_monotonic() >= _SIP_UNAUTHORIZED_UNTIL:
        _SIP_UNAUTHORIZED = False
        _SIP_UNAUTHORIZED_UNTIL = None
        os.environ.pop("ALPACA_SIP_UNAUTHORIZED", None)
        return False
    return True


def _mark_sip_unauthorized(cooldown_s: float = 1800.0) -> None:
    global _SIP_UNAUTHORIZED, _SIP_UNAUTHORIZED_UNTIL
    _SIP_UNAUTHORIZED = True
    try:
        cooldown = max(60.0, float(cooldown_s))
    except Exception:
        cooldown = 1800.0
    _SIP_UNAUTHORIZED_UNTIL = _now_monotonic() + cooldown
    if _pytest_active():
        os.environ.pop("ALPACA_SIP_UNAUTHORIZED", None)
    else:
        os.environ["ALPACA_SIP_UNAUTHORIZED"] = "1"


def _clear_sip_lockout_for_tests() -> None:
    """Reset SIP authorization lockout when running under pytest."""

    global _SIP_UNAUTHORIZED, _SIP_UNAUTHORIZED_UNTIL
    _SIP_UNAUTHORIZED = False
    _SIP_UNAUTHORIZED_UNTIL = None
    os.environ.pop("ALPACA_SIP_UNAUTHORIZED", None)


def _get_cycle_id() -> str:
    cycle_env = os.environ.get("AI_TRADING_CYCLE_ID")
    if cycle_env:
        return str(cycle_env)
    bot_engine = sys.modules.get("ai_trading.core.bot_engine")
    if bot_engine is not None:
        try:
            cycle_val = getattr(bot_engine, "_GLOBAL_CYCLE_ID", None)
        except Exception:
            cycle_val = None
        if cycle_val is not None:
            return str(cycle_val)
    return "default-cycle"


def _fallback_cache_for_cycle(cycle_id: str, symbol: str, timeframe: str) -> str | None:
    return _CYCLE_FALLBACK_FEED.get((cycle_id, symbol, timeframe))


def _remember_fallback_for_cycle(cycle_id: str, symbol: str, timeframe: str, feed: str) -> None:
    if not feed:
        return
    _CYCLE_FALLBACK_FEED[(cycle_id, symbol, timeframe)] = feed


def _reset_provider_auth_state_for_tests() -> None:
    global _ALLOW_SIP
    _clear_sip_lockout_for_tests()
    _SIP_UNAVAILABLE_LOGGED.clear()
    _CYCLE_FALLBACK_FEED.clear()
    _ALLOW_SIP = None


def _sip_configured() -> bool:
    if os.getenv("PYTEST_RUNNING"):
        return _sip_allowed()
    if not _sip_allowed():
        return False
    if _sip_allowed() and _HAS_SIP:
        return True
    try:
        feeds = alpaca_feed_failover()
    except Exception:
        feeds = ("sip",)
    if not feeds:
        feeds = ("sip",)
    return any(str(feed).strip().lower() == "sip" for feed in feeds)


def _log_sip_unavailable(symbol: str, timeframe: str, reason: str = "UNAUTHORIZED_SIP") -> None:
    key = (symbol, timeframe)
    if key in _SIP_UNAVAILABLE_LOGGED:
        return
    extra = {"provider": "alpaca", "feed": "sip", "symbol": symbol, "timeframe": timeframe}
    if reason == "UNAUTHORIZED_SIP":
        extra["status"] = "unauthorized"
    logger.warning(reason, extra=_norm_extra(extra))
    _SIP_UNAVAILABLE_LOGGED.add(key)


def _sip_fallback_allowed(session: HTTPSession | None, headers: dict[str, str], timeframe: str) -> bool:
    """Return True if SIP fallback should be attempted."""
    if session is None or not hasattr(session, "get"):
        raise ValueError("session_required")
    global _SIP_DISALLOWED_WARNED, _SIP_PRECHECK_DONE
    # In tests, allow SIP fallback without performing precheck to avoid
    # consuming mocked responses intended for the actual fallback request.
    if os.getenv("PYTEST_RUNNING") or os.getenv("PYTEST_CURRENT_TEST"):
        if _is_sip_unauthorized():
            return False
        if _SIP_PRECHECK_DONE:
            return True
        _SIP_PRECHECK_DONE = True
        return True
    if not _sip_allowed():
        return False
    if not _sip_configured():
        if not _sip_allowed() and not _SIP_DISALLOWED_WARNED:
            logger.warning(
                "SIP_FEED_DISABLED",
                extra=_norm_extra({"provider": "alpaca", "feed": "sip", "timeframe": timeframe}),
            )
            _SIP_DISALLOWED_WARNED = True
        return False
    if _is_sip_unauthorized():
        return False
    if _SIP_PRECHECK_DONE:
        return True
    _SIP_PRECHECK_DONE = True
    url = "https://data.alpaca.markets/v2/stocks/bars"
    params = {"symbols": "AAPL", "timeframe": timeframe, "limit": 1, "feed": "sip"}
    try:
        resp = session.get(url, params=params, headers=headers, timeout=clamp_request_timeout(5))
    except Exception as e:  # pragma: no cover - best effort
        logger.debug(
            "SIP_PRECHECK_FAILED",
            extra=_norm_extra({"provider": "alpaca", "feed": "sip", "timeframe": timeframe, "error": str(e)}),
        )
        return True
    if getattr(resp, "status_code", None) in (401, 403):
        _incr(
            "data.fetch.unauthorized",
            value=1.0,
            tags={"provider": "alpaca", "feed": "sip", "timeframe": timeframe},
        )
        metrics.unauthorized += 1
        provider_monitor.record_failure("alpaca", "unauthorized")
        logger.warning(
            "UNAUTHORIZED_SIP",
            extra=_norm_extra({"provider": "alpaca", "status": "precheck", "feed": "sip", "timeframe": timeframe}),
        )
        _mark_sip_unauthorized()
        return False
    return True


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


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename provider-specific OHLCV aliases to canonical column names."""
    pd_local = _ensure_pandas()
    if pd_local is None or df is None or not hasattr(df, "columns"):
        return df

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    alias_groups = {
        "timestamp": {"timestamp", "time", "t", "ts"},
        "open": {"open", "o"},
        "high": {"high", "h"},
        "low": {"low", "l"},
        "close": {"close", "c", "price"},
        "volume": {"volume", "v"},
    }
    for canonical, aliases in alias_groups.items():
        for alias in list(aliases):
            if alias not in df.columns or alias == canonical:
                continue
            if canonical in df.columns:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            category=FutureWarning,
                            message="The behavior of array concatenation with empty entries is deprecated.",
                        )
                        df[canonical] = df[canonical].combine_first(df[alias])
                except AttributeError:  # pragma: no cover - non-Series columns
                    pass
                df.drop(columns=[alias], inplace=True)
            else:
                df.rename(columns={alias: canonical}, inplace=True)
    return df


def _flatten_and_normalize_ohlcv(
    df: pd.DataFrame,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> pd.DataFrame:
    """Make YF/other OHLCV DataFrames uniform.

    - flatten MultiIndex columns
    - lower/snake columns
    - ensure 'close' exists (fallback to 'adj_close')
    - de-duplicate & sort index, convert timestamp fields to UTC while keeping them tz-aware
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
    normalize_ohlcv_columns(df)

    timeframe_norm = str(timeframe or "").lower()
    is_daily = "day" in timeframe_norm or timeframe_norm.endswith("d")
    if is_daily:
        if "adj_close" in df.columns:
            df["close"] = df["adj_close"]
    elif "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        extra = {
            "symbol": symbol,
            "timeframe": timeframe,
            "missing_columns": missing,
            "columns": [str(col) for col in getattr(df, "columns", [])],
            "rows": int(getattr(df, "shape", (0, 0))[0]),
        }
        extra = {k: v for k, v in extra.items() if v is not None}
        logger.error("OHLCV_COLUMNS_MISSING", extra=extra)
        reason = "close_column_missing" if "close" in missing else "ohlcv_columns_missing"
        err = MissingOHLCVColumnsError(reason)
        setattr(err, "fetch_reason", reason)
        setattr(err, "missing_columns", tuple(missing))
        setattr(err, "symbol", symbol)
        setattr(err, "timeframe", timeframe)
        raise err

    # Ensure the primary OHLCV columns are numeric before downstream checks.
    for col in required:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:  # pragma: no cover - defensive fallback
                try:
                    df[col] = pd.Series(df[col]).apply(lambda value: pd.to_numeric(value, errors="coerce"))
                except Exception:
                    # Leave the column as-is if coercion repeatedly fails; the
                    # downstream NaN guard will handle missing values.
                    pass

    close_series = df.get("close")
    if close_series is not None:
        try:
            all_nan = bool(pd.isna(close_series).all())
        except Exception:  # pragma: no cover - defensive fallback
            try:
                all_nan = bool(close_series.isna().all())  # type: ignore[attr-defined]
            except Exception:
                all_nan = False
        if all_nan:
            extra = {
                "symbol": symbol,
                "timeframe": timeframe,
                "rows": int(getattr(df, "shape", (0, 0))[0]),
            }
            extra = {k: v for k, v in extra.items() if v is not None}
            logger.error("OHLCV_CLOSE_ALL_NAN", extra=extra)
            err = DataFetchError("close_column_all_nan")
            setattr(err, "fetch_reason", "close_column_all_nan")
            setattr(err, "symbol", symbol)
            setattr(err, "timeframe", timeframe)
            raise err

    df_out = df

    if isinstance(df_out.index, pd.DatetimeIndex):
        try:
            tz = df_out.index.tz
        except Exception:  # pragma: no cover - defensive
            tz = None
        try:
            if tz is not None:
                df_out.index = df_out.index.tz_convert("UTC")
            else:
                df_out.index = df_out.index.tz_localize("UTC")
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            duplicated = df_out.index.duplicated(keep="last")
        except Exception:  # pragma: no cover - defensive
            duplicated = None
        has_duplicates = False
        if duplicated is not None:
            try:
                has_duplicates = bool(duplicated.any())
            except Exception:  # pragma: no cover - defensive
                has_duplicates = False
        if has_duplicates:
            mask = ~duplicated
            try:
                needs_filter = not bool(mask.all())
            except Exception:  # pragma: no cover - defensive
                needs_filter = True
            if needs_filter:
                df_out = df_out.loc[mask]
        is_monotonic_attr = getattr(df_out.index, "is_monotonic_increasing", True)
        try:
            is_monotonic = bool(is_monotonic_attr()) if callable(is_monotonic_attr) else bool(is_monotonic_attr)
        except Exception:  # pragma: no cover - defensive
            is_monotonic = True
        if not is_monotonic:
            df_out.sort_index(inplace=True)
    if "timestamp" in getattr(df_out, "columns", []):
        try:
            ts_series = df_out["timestamp"]
        except Exception:  # pragma: no cover - defensive
            ts_series = None
        if ts_series is not None:
            tz = None
            try:
                tz = getattr(ts_series.dt, "tz", None)  # type: ignore[attr-defined]
            except Exception:
                tz = None
            try:
                if tz is not None:
                    df_out["timestamp"] = ts_series.dt.tz_convert("UTC")  # type: ignore[attr-defined]
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="errors='ignore' is deprecated and will raise in a future version.",
                            category=FutureWarning,
                        )
                        converted = pd.to_datetime(ts_series, utc=True, errors="ignore")
                    df_out["timestamp"] = converted
            except (AttributeError, TypeError, ValueError):
                pass
    if "timestamp" not in getattr(df_out, "columns", []) and isinstance(df_out.index, pd.DatetimeIndex):
        index_name = df_out.index.name or "index"
        df_reset = df_out.reset_index()
        if index_name != "timestamp":
            df_reset.rename(columns={index_name: "timestamp"}, inplace=True)
        df_out = df_reset

    # Avoid timestamp being simultaneously a column and an index label.
    try:
        index_names = list(getattr(df_out.index, "names", []) or [])
    except TypeError:  # pragma: no cover - non-list names container
        index_names = []
    if "timestamp" in df_out.columns and index_names and any(name == "timestamp" for name in index_names):
        new_names = [None if name == "timestamp" else name for name in index_names]
        try:
            df_out.index = df_out.index.set_names(new_names)
        except AttributeError:  # e.g. DatetimeIndex with single name attribute
            if getattr(df_out.index, "name", None) == "timestamp":
                df_out.index = df_out.index.rename(None)

    context_label = "ohlcv" if symbol is None else f"ohlcv {symbol}"
    tf_label = str(timeframe or "").strip() or "unknown"
    if "timestamp" in getattr(df_out, "columns", []):
        try:
            ts_index = safe_to_datetime(
                df_out["timestamp"],
                context=f"{context_label} {tf_label}",
            )
            df_out = df_out.assign(timestamp=pd.Series(ts_index, index=df_out.index))
        except Exception:
            pass
    if "timestamp" in getattr(df_out, "columns", []):
        try:
            df_out = df_out[df_out["timestamp"].notna()]
        except Exception:
            pass
    required_cols = ["open", "high", "low", "close", "volume"]
    try:
        df_out = df_out.dropna(subset=required_cols, how="any")
    except Exception:
        for col in required_cols:
            if col in getattr(df_out, "columns", []):
                try:
                    df_out = df_out[df_out[col].notna()]
                except Exception:
                    continue
    if "timestamp" in getattr(df_out, "columns", []):
        try:
            df_out = df_out.sort_values("timestamp")
            df_out = df_out.drop_duplicates(subset=["timestamp"], keep="last")
        except Exception:
            pass

    return df_out


def _mutate_dataframe_in_place(target: Any, source: Any) -> Any:
    """Coerce ``target`` to match ``source`` while retaining object identity."""

    if source is None:
        return None

    pd_local = _ensure_pandas()
    if pd_local is None:
        return source

    dataframe_type = getattr(pd_local, "DataFrame", None)
    if dataframe_type is None or not isinstance(source, dataframe_type):
        return source
    if not isinstance(target, dataframe_type):
        return source
    if target is source:
        return target

    try:
        target.drop(target.index, inplace=True)
    except Exception:
        pass
    try:
        target.drop(columns=list(target.columns), inplace=True, errors="ignore")
    except Exception:
        pass

    for column in source.columns:
        try:
            target[column] = source[column].to_numpy()
        except Exception:
            target[column] = list(source[column])

    try:
        target.index = source.index.copy()
    except Exception:
        try:
            target.index = source.index
        except Exception:
            pass

    try:
        target.attrs.clear()
        target.attrs.update(getattr(source, "attrs", {}) or {})
    except Exception:
        pass

    return target


def _set_price_reliability(
    df: pd.DataFrame,
    *,
    reliable: bool,
    reason: str | None = None,
) -> None:
    try:
        attrs = getattr(df, "attrs", None)
        if not isinstance(attrs, dict):
            return
        attrs["price_reliable"] = bool(reliable)
        if reason:
            attrs["price_reliable_reason"] = str(reason)
        else:
            attrs.pop("price_reliable_reason", None)
    except Exception:
        pass


def _yahoo_get_bars(symbol: str, start: Any, end: Any, interval: str) -> pd.DataFrame:
    """Return a DataFrame with a tz-aware 'timestamp' column between start and end."""
    pd = _ensure_pandas()
    yf = _ensure_yfinance()
    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)
    if str(interval).lower() in {"1m", "1min", "1minute"}:
        safe_end = _last_complete_minute(pd)
        if end_dt > safe_end:
            end_dt = max(start_dt, safe_end)
    window_seconds = max((end_dt - start_dt).total_seconds(), 0.0)
    if pd is None:
        return []  # type: ignore[return-value]
    if getattr(yf, "download", None) is None:
        empty_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=empty_cols)
    error_cls = None
    try:
        error_cls = getattr(getattr(yf, "shared", None), "YFPricesMissingError", None)
    except Exception:
        error_cls = None

    def _download(_start: _dt.datetime) -> pd.DataFrame | None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*auto_adjust.*", module="yfinance")
            return yf.download(
                symbol,
                start=_start,
                end=end_dt,
                interval=interval,
                auto_adjust=True,
                threads=False,
                progress=False,
                group_by="column",
            )

    try:
        df = _download(start_dt)
    except Exception as exc:  # pragma: no cover - yfinance error formatting varies
        message = str(exc).lower()
        if (error_cls and isinstance(exc, error_cls)) or "possibly delisted" in message:
            _log_yf_warning(
                "YF_PRICES_MISSING",
                symbol,
                interval,
                {
                    "symbol": symbol,
                    "interval": interval,
                    "detail": str(exc),
                },
            )
            metrics.empty_payload += 1
            idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
            cols = ["open", "high", "low", "close", "volume"]
            return pd.DataFrame(columns=cols, index=idx).reset_index()
        raise

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*auto_adjust.*", module="yfinance")
    if df is None or getattr(df, "empty", True):
        metrics.empty_payload += 1
        if window_seconds < 180:
            widened_start = start_dt - _dt.timedelta(minutes=3)
            try:
                widened = _download(widened_start)
            except Exception as exc:
                message = str(exc).lower()
                if (error_cls and isinstance(exc, error_cls)) or "possibly delisted" in message:
                    _log_yf_warning(
                        "YF_PRICES_MISSING",
                        symbol,
                        interval,
                        {
                            "symbol": symbol,
                            "interval": interval,
                            "detail": str(exc),
                            "phase": "widened",
                        },
                    )
                    widened = None
                else:
                    widened = None
            if widened is not None and not getattr(widened, "empty", True):
                df = widened
        if df is None or getattr(df, "empty", True):
            _log_yf_warning(
                "YAHOO_EDGE_WINDOW_EMPTY",
                symbol,
                interval,
                {
                    "symbol": symbol,
                    "interval": interval,
                    "start": start_dt.isoformat(),
                    "end": end_dt.isoformat(),
                    "window_seconds": window_seconds,
                },
            )
            idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
            cols = ["open", "high", "low", "close", "volume"]
            return pd.DataFrame(columns=cols, index=idx).reset_index()
    df = df.reset_index().rename(columns={df.index.name or "Date": "timestamp"})
    if "timestamp" not in df.columns:
        for c in df.columns:
            if c.lower() in ("date", "datetime"):
                df = df.rename(columns={c: "timestamp"})
                break
    df = _flatten_and_normalize_ohlcv(df, symbol, interval)
    return df


def _finnhub_resolution(interval: str) -> str | None:
    mapping = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "60m": "60",
        "1h": "60",
        "1d": "D",
        "day": "D",
    }
    return mapping.get(interval.lower())


def _finnhub_get_bars(symbol: str, start: Any, end: Any, interval: str) -> pd.DataFrame:
    pd_local = _ensure_pandas()
    if getattr(fh_fetcher, "fetch", None) is None or getattr(fh_fetcher, "is_stub", False):
        if pd_local is None:
            return []  # type: ignore[return-value]
        log_finnhub_disabled(symbol)
        idx = pd_local.DatetimeIndex([], tz="UTC", name="timestamp")
        cols = ["open", "high", "low", "close", "volume"]
        return pd_local.DataFrame(columns=cols, index=idx).reset_index()
    resolution = _finnhub_resolution(interval)
    if resolution is None:
        if pd_local is None:
            return []  # type: ignore[return-value]
        return pd_local.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)
    try:
        df = fh_fetcher.fetch(symbol, start_dt, end_dt, resolution=resolution)
        if isinstance(df, pd_local.DataFrame):
            if "close" in df.columns:
                for col in ("open", "high", "low"):
                    if col not in df.columns:
                        df[col] = df["close"]
            if "volume" not in df.columns:
                df["volume"] = 0.0
    except FinnhubAPIException:
        df = None
    except Exception:
        df = None
    if df is None:
        if pd_local is None:
            return []  # type: ignore[return-value]
        return pd_local.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    try:
        return _flatten_and_normalize_ohlcv(df, symbol, interval)
    except Exception:
        if pd_local is None:
            return []  # type: ignore[return-value]
        return pd_local.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def _backup_get_bars(symbol: str, start: Any, end: Any, interval: str) -> pd.DataFrame:
    """Route to configured backup provider or return empty DataFrame."""
    settings = get_settings()
    provider = getattr(settings, "backup_data_provider", "yahoo")
    provider_str = str(provider).strip()
    normalized = provider_str.lower()
    if normalized in {"finnhub", "finnhub_low_latency"}:
        df = _finnhub_get_bars(symbol, start, end, interval)
        if isinstance(df, list):  # pragma: no cover - defensive for stub returns
            return df
        if getattr(df, "empty", True):
            logger.warning(
                "BACKUP_PROVIDER_EMPTY",
                extra={"provider": provider, "symbol": symbol, "interval": interval},
            )
        if isinstance(df, pd.DataFrame):
            df = _normalize_with_attrs(df)
        return _annotate_df_source(df, provider=normalized, feed=normalized)
    if normalized == "yahoo":
        pd_local = _ensure_pandas()
        start_dt = ensure_datetime(start)
        end_dt = ensure_datetime(end)
        interval_norm = str(interval).lower()
        chunk_span = _dt.timedelta(days=7)
        needs_chunk = (
            interval_norm in {"1m", "1min", "1minute"}
            and end_dt - start_dt > chunk_span
        )
        frames: list[pd.DataFrame] = []  # type: ignore[var-annotated]
        if needs_chunk:
            _log_yf_warning(
                "YF_1M_RANGE_SPLIT",
                symbol,
                interval_norm,
                {
                    "symbol": symbol,
                    "interval": interval,
                    "start": start_dt.isoformat(),
                    "end": end_dt.isoformat(),
                    "max_days": 7,
                },
            )
            cur_start = start_dt
            while cur_start < end_dt:
                cur_end = min(cur_start + chunk_span, end_dt)
                frames.append(_yahoo_get_bars(symbol, cur_start, cur_end, interval))
                cur_start = cur_end
        _, bucket = _cycle_bucket(_BACKUP_USAGE_LOGGED, _BACKUP_USAGE_MAX_CYCLES)
        key = (str(symbol).upper(), str(interval))
        if key not in bucket:
            dedupe_key = f"USING_BACKUP_PROVIDER:{normalized}:{str(symbol).upper()}"
            ttl = getattr(settings, "logging_dedupe_ttl_s", 0)
            if provider_log_deduper.should_log(dedupe_key, int(ttl)):
                logger.info("USING_BACKUP_PROVIDER", extra={"provider": provider, "symbol": symbol})
            else:
                record_provider_log_suppressed("USING_BACKUP_PROVIDER")
            bucket.add(key)
        if frames:
            if pd_local is not None:
                valid_frames = [frame for frame in frames if isinstance(frame, pd_local.DataFrame)]
                combined: Any
                if valid_frames:
                    try:
                        combined = pd_local.concat(valid_frames, ignore_index=True)
                    except Exception:
                        combined = valid_frames[0]
                else:
                    combined = frames[0]
                if isinstance(combined, pd_local.DataFrame) and "timestamp" in combined.columns:
                    try:
                        combined["timestamp"] = pd_local.to_datetime(
                            combined["timestamp"], utc=True, errors="coerce"
                        )
                    except Exception:
                        pass
                    else:
                        combined = combined.sort_values("timestamp")
                        combined = combined.drop_duplicates(subset="timestamp", keep="last")
                        combined = combined.reset_index(drop=True)
                if isinstance(combined, pd_local.DataFrame):
                    combined = _normalize_with_attrs(combined)
                    combined = _annotate_df_source(combined, provider=normalized, feed=normalized)
                return combined
            first = frames[0]
            if isinstance(first, list):  # pragma: no cover - pandas unavailable path
                return first  # type: ignore[return-value]
            if isinstance(first, pd_local.DataFrame):
                first = _normalize_with_attrs(first)
            return _annotate_df_source(first, provider=normalized, feed=normalized)
        df = _yahoo_get_bars(symbol, start_dt, end_dt, interval)
        if isinstance(df, pd_local.DataFrame):
            df = _normalize_with_attrs(df)
        return _annotate_df_source(df, provider=normalized, feed=normalized)
    pd_local = _ensure_pandas()
    if normalized in ("", "none"):
        logger.info("BACKUP_PROVIDER_DISABLED", extra={"symbol": symbol})
        return None
    else:
        logger.warning("UNKNOWN_BACKUP_PROVIDER", extra={"provider": provider, "symbol": symbol})
    if pd_local is None:
        return []  # type: ignore[return-value]
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    empty_df = pd_local.DataFrame({col: [] for col in cols})
    normalized_df = normalize_ohlcv_df(empty_df)
    return _annotate_df_source(normalized_df, provider=normalized or "none", feed=normalized or None)


def _is_normalized_ohlcv_frame(
    df: Any,
    pd_module: Any | None = None,
) -> bool:
    """Return ``True`` when *df* already satisfies normalization requirements."""

    if df is None:
        return False

    pd_local = pd_module if pd_module is not None else _ensure_pandas()
    if pd_local is None:
        return False

    dataframe_type = getattr(pd_local, "DataFrame", None)
    if dataframe_type is None or not isinstance(df, dataframe_type):
        return False

    expected_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    try:
        if list(df.columns) != expected_columns:
            return False
    except Exception:
        return False

    try:
        ts_series = df["timestamp"]
    except Exception:
        return False

    try:
        if ts_series.isna().any():
            return False
    except Exception:
        return False

    try:
        tz = getattr(getattr(ts_series, "dt", None), "tz", None)
    except Exception:
        tz = None
    if tz is None or str(tz) != "UTC":
        return False

    try:
        monotonic_attr = getattr(ts_series, "is_monotonic_increasing", True)
        is_monotonic = (
            bool(monotonic_attr()) if callable(monotonic_attr) else bool(monotonic_attr)
        )
    except Exception:
        is_monotonic = True
    if not is_monotonic:
        return False

    return True


def _post_process(
    df: pd.DataFrame | None,
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> pd.DataFrame | None:
    """Normalize OHLCV DataFrame while preserving explicit empties."""
    pd = _ensure_pandas()
    if pd is None:
        return df
    if df is None or getattr(df, "empty", True):
        return None
    candidate = df if _is_normalized_ohlcv_frame(df, pd) else _flatten_and_normalize_ohlcv(df, symbol, timeframe)
    try:
        normalized = normalize_ohlcv_df(candidate)
    except Exception:
        normalized = candidate

    if normalized is None:
        return None

    if normalized is df:
        return df

    if not isinstance(df, pd.DataFrame):
        return normalized

    result = _mutate_dataframe_in_place(df, normalized)
    return result if isinstance(result, pd.DataFrame) else normalized


def _verify_minute_continuity(df: pd.DataFrame | None, symbol: str, backfill: str | None = None) -> pd.DataFrame | None:
    """Verify 1-minute bar continuity and optionally backfill gaps."""

    pd_local = _ensure_pandas()
    if pd_local is None or df is None or getattr(df, "empty", True) or "timestamp" not in df.columns:
        return df

    timestamp_series = df["timestamp"]
    monotonic_attr = getattr(timestamp_series, "is_monotonic_increasing", None)
    try:
        needs_sort = bool(monotonic_attr is False)
    except Exception:  # pragma: no cover - defensive
        needs_sort = False
    if monotonic_attr is None:
        try:
            needs_sort = bool(timestamp_series.is_monotonic_increasing is False)
        except Exception:  # pragma: no cover - fallback
            needs_sort = True
    if needs_sort:
        df.sort_values("timestamp", inplace=True)
    ts = pd_local.DatetimeIndex(df["timestamp"])
    diffs = ts.to_series().diff().dt.total_seconds().iloc[1:]
    missing = diffs[diffs > 60]
    if missing.empty:
        return df

    gap_count = int(len(missing))
    log_throttled_event(
        logger,
        "MINUTE_GAPS_DETECTED",
        level=logging.WARNING,
        extra=_norm_extra({"symbol": symbol, "gap_count": gap_count}),
    )
    if not backfill:
        return df

    full_index = pd_local.date_range(ts.min(), ts.max(), freq="1min", tz=ts.tz)
    df = df.set_index("timestamp").reindex(full_index)
    df.index.name = "timestamp"

    if backfill == "ffill":
        df["close"] = df["close"].ffill()
        df["open"] = df["open"].fillna(df["close"])
        df["high"] = df["high"].fillna(df["close"])
        df["low"] = df["low"].fillna(df["close"])
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0)
    elif backfill == "interpolate":
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df[cols] = df[cols].interpolate(method="time")  # type: ignore[assignment]
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0)
        df[cols] = df[cols].ffill().bfill()

    return df.reset_index()


def _normalize_window_bounds(
    start: _dt.datetime,
    end: _dt.datetime,
    tz: ZoneInfo,
) -> tuple[pd.DatetimeIndex, pd.Timestamp, pd.Timestamp]:
    pd_local = _ensure_pandas()
    if pd_local is None:
        raise RuntimeError("pandas required for window normalization")
    start_ts = pd_local.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    end_ts = pd_local.Timestamp(end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    start_local = start_ts.tz_convert(tz)
    end_local = end_ts.tz_convert(tz)
    expected = pd_local.date_range(
        start_local,
        end_local,
        freq="min",
        tz=tz,
        inclusive="left",
    )
    return expected, start_ts.tz_convert("UTC"), end_ts.tz_convert("UTC")


def _repair_rth_minute_gaps(
    df: pd.DataFrame | None,
    *,
    symbol: str,
    start: _dt.datetime,
    end: _dt.datetime,
    tz: ZoneInfo,
) -> tuple[pd.DataFrame | None, dict[str, object], bool]:
    """Attempt to fill missing RTH minutes using the configured backup provider."""

    pd_local = _ensure_pandas()
    if pd_local is None or df is None or getattr(df, "empty", True):
        return df, {"expected": 0, "missing_after": 0, "gap_ratio": 0.0}, False

    expected_local, start_utc, end_utc = _normalize_window_bounds(start, end, tz)
    expected_utc = expected_local.tz_convert("UTC")
    expected_count = int(expected_utc.size)
    if expected_count == 0:
        return df, {"expected": 0, "missing_after": 0, "gap_ratio": 0.0}, False

    work_df = df
    mutated = False
    filled_backup = False
    try:
        timestamps = pd_local.to_datetime(df["timestamp"], utc=True)
    except Exception:
        return df, {"expected": expected_count, "missing_after": expected_count, "gap_ratio": 1.0}, False
    existing_index = pd_local.DatetimeIndex(timestamps)
    missing = expected_utc.difference(existing_index)
    try:
        provider_attr = None
        attrs = getattr(df, "attrs", None)
        if isinstance(attrs, dict):
            provider_attr = str(
                attrs.get("data_provider")
                or attrs.get("fallback_provider")
                or ""
            ).strip().lower()
    except Exception:
        provider_attr = None
    skip_backup_fill = provider_attr == "yahoo"
    used_backup = False
    if len(missing) > 0 and not skip_backup_fill:
        try:
            missing_start = missing.min()
            missing_end = missing.max() + _dt.timedelta(minutes=1)
        except ValueError:
            missing_start = None
            missing_end = None
        if missing_start is not None and missing_end is not None:
            try:
                fallback_df = _backup_get_bars(
                    symbol,
                    missing_start,
                    missing_end,
                    interval="1m",
                )
            except Exception:
                fallback_df = None
            else:
                fallback_df = _post_process(
                    fallback_df,
                    symbol=symbol,
                    timeframe="1Min",
                )
            if fallback_df is not None and not getattr(fallback_df, "empty", True):
                try:
                    fb_idx = pd_local.to_datetime(fallback_df["timestamp"], utc=True)
                except Exception:
                    fb_idx = pd_local.DatetimeIndex([])
                fallback_df = fallback_df.set_index(fb_idx)
                needed = fallback_df.loc[fallback_df.index.intersection(missing)]
                if not needed.empty:
                    used_backup = True
                    filled_backup = True
                    base_df = df.set_index(existing_index)
                    combined = pd_local.concat([base_df, needed])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined.sort_index(inplace=True)
                    work_df = combined.reset_index()
                    if "index" in work_df.columns and "timestamp" not in work_df.columns:
                        work_df.rename(columns={"index": "timestamp"}, inplace=True)
                    mutated = True
    if filled_backup:
        logger.info(
            "MINUTE_GAPS_BACKFILLED",
            extra={"symbol": symbol, "window_start": start.isoformat(), "window_end": end.isoformat()},
        )
    if mutated:
        try:
            combined_idx = pd_local.to_datetime(work_df["timestamp"], utc=True)
        except Exception:
            combined_idx = pd_local.DatetimeIndex([])
    else:
        combined_idx = existing_index
    missing_after = int(expected_utc.difference(combined_idx).size)
    gap_ratio = (missing_after / expected_count) if expected_count else 0.0
    tolerated = False
    if missing_after and gap_ratio <= 0.015:
        tolerated = True
        missing_after = 0
        gap_ratio = 0.0
    metadata: dict[str, object] = {
        "expected": expected_count,
        "missing_after": missing_after,
        "gap_ratio": gap_ratio,
        "window_start": start_utc,
        "window_end": end_utc,
        "used_backup": used_backup,
    }
    if tolerated:
        logger.info(
            "MINUTE_GAPS_TOLERATED",
            extra={"symbol": symbol, "gap_ratio": 0.0, "window_start": start.isoformat(), "window_end": end.isoformat()},
        )
    target_df = work_df if mutated else df
    try:
        attrs = target_df.attrs  # type: ignore[attr-defined]
        attrs.setdefault("symbol", symbol)
        attrs["_coverage_meta"] = metadata
    except Exception:
        pass
    return (work_df if mutated else df), metadata, used_backup


_SKIP_LOGGED: set[tuple[str, _dt.date]] = set()


def should_skip_symbol(
    df: pd.DataFrame,
    *,
    window: tuple[_dt.datetime, _dt.datetime],
    tz: str | ZoneInfo,
    max_gap_ratio: float,
) -> bool:
    """Return ``True`` when the DataFrame should be skipped due to coverage gaps."""

    pd_local = _ensure_pandas()
    if pd_local is None or df is None or getattr(df, "empty", True):
        return False
    tzinfo = ZoneInfo(tz) if isinstance(tz, str) else tz
    start, end = window
    expected_local, _, _ = _normalize_window_bounds(start, end, tzinfo)
    expected_count = int(expected_local.size)
    if expected_count == 0:
        return False
    try:
        timestamps = pd_local.to_datetime(df["timestamp"], utc=True).tz_convert(tzinfo)
    except Exception:
        timestamps = pd_local.DatetimeIndex([])
    missing_after = int(expected_local.difference(timestamps).size)
    gap_ratio = missing_after / expected_count if expected_count else 0.0
    metadata = getattr(df, "attrs", {}).get("_coverage_meta")
    if isinstance(metadata, dict):
        metadata.update(
            {
                "expected": expected_count,
                "missing_after": missing_after,
                "gap_ratio": gap_ratio,
            }
        )
    symbol = getattr(getattr(df, "attrs", {}), "get", lambda *_: None)("symbol")
    if callable(symbol):
        symbol = df.attrs.get("symbol")  # type: ignore[assignment]
    symbol_str = str(symbol) if symbol else "UNKNOWN"
    catastrophic_gap = gap_ratio >= 0.999
    skip = catastrophic_gap
    if skip:
        try:
            coverage_meta = df.attrs.setdefault("_coverage_meta", {})  # type: ignore[attr-defined]
        except Exception:
            coverage_meta = {}
        if isinstance(coverage_meta, dict):
            coverage_meta["skip_flagged"] = True
            coverage_meta["gap_ratio"] = gap_ratio
            coverage_meta["missing_after"] = missing_after
            coverage_meta["expected"] = expected_count
        key = (symbol_str, start.date())
        if key not in _SKIP_LOGGED:
            logger.warning(
                "SKIP_SYMBOL_INSUFFICIENT_INTRADAY_COVERAGE | symbol=%s missing=%d expected=%d gap_ratio=%s",
                symbol_str,
                missing_after,
                expected_count,
                f"{gap_ratio:.4%}",
            )
            _SKIP_LOGGED.add(key)
    elif gap_ratio > max_gap_ratio:
        try:
            coverage_meta = df.attrs.setdefault("_coverage_meta", {})  # type: ignore[attr-defined]
        except Exception:
            coverage_meta = {}
        if isinstance(coverage_meta, dict):
            coverage_meta["gap_ratio"] = gap_ratio
            coverage_meta["missing_after"] = missing_after
            coverage_meta["expected"] = expected_count
    return skip


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


def _last_complete_minute(pd_local: Any | None = None) -> _dt.datetime:
    """Return the most recent fully closed UTC minute."""

    pd_mod = pd_local if pd_local is not None else _ensure_pandas()
    if pd_mod is not None:
        try:
            ts = pd_mod.Timestamp.utcnow().floor("min") - pd_mod.Timedelta(minutes=1)
            tzinfo = getattr(ts, "tzinfo", None)
            if tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts.to_pydatetime()
        except Exception:  # pragma: no cover - defensive
            pass
    return datetime.now(UTC).replace(second=0, microsecond=0) - _dt.timedelta(minutes=1)


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
            for placeholder, real in (
                (RequestException, _RequestException),
                (Timeout, _Timeout),
                (ConnectionError, _ConnectionError),
                (HTTPError, _HTTPError),
            ):
                try:
                    placeholder.__bases__ = (real,)
                except TypeError:  # pragma: no cover - fallback for exotic class wrappers

                    class _Shim(real):  # type: ignore[misc]
                        pass

                    placeholder.__bases__ = (_Shim,)
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
        return _flatten_and_normalize_ohlcv(bars, symbol, timeframe)
    try:
        return _flatten_and_normalize_ohlcv(pd.DataFrame(bars), symbol, timeframe)
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
    if start is None:
        raise ValueError("start_required")
    if end is None:
        raise ValueError("end_required")
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


def build_fetcher(
    prefer: str | None = None,
    force_feed: str | None = None,
    *,
    cache_minutes: int = 15,
):
    """
    Returns a DataFetcher. If prefer/force_feed are provided, build a fresh, directed
    fetcher (no singleton). Import the class first to avoid local variable shadowing.
    """

    from ai_trading.core.bot_engine import DataFetcher  # import FIRST

    if prefer:
        prefer = _normalize_feed_value(prefer)
    if force_feed:
        force_feed = _normalize_feed_value(force_feed)

    if prefer or force_feed:
        f = DataFetcher(prefer=prefer, force_feed=force_feed, cache_minutes=cache_minutes)
        setattr(f, "source", "directed")
        return f

    global _FETCHER_SINGLETON
    if _FETCHER_SINGLETON is not None:
        return _FETCHER_SINGLETON

    _FETCHER_SINGLETON = DataFetcher(cache_minutes=cache_minutes)
    setattr(_FETCHER_SINGLETON, "source", "singleton")
    return _FETCHER_SINGLETON


def retry_empty_fetch_once(
    *,
    delay: float,
    attempt: int,
    max_retries: int,
    previous_correlation_id: str | None,
    total_elapsed: float,
) -> dict[str, Any]:
    """Return structured metadata for a single empty-bar retry.

    Parameters
    ----------
    delay:
        Backoff delay in seconds before the next request.
    attempt:
        The 1-indexed retry attempt number.
    max_retries:
        Maximum number of retries allowed.
    previous_correlation_id:
        Correlation ID from the prior request, if any.
    total_elapsed:
        Total time elapsed in seconds since the initial request.

    Returns
    -------
    dict[str, Any]
        Mapping used for structured logging of the retry.
    """

    return {
        "retry_delay": delay,
        "delay": delay,
        "previous_correlation_id": previous_correlation_id,
        "attempt": attempt,
        "remaining_retries": max_retries - attempt,
        "total_elapsed": total_elapsed,
    }


def _fetch_bars(
    symbol: str, start: Any, end: Any, timeframe: str, *, feed: str = _DEFAULT_FEED, adjustment: str = "raw"
) -> pd.DataFrame | None:
    """Fetch bars from Alpaca v2 with alt-feed fallback."""
    pd = _ensure_pandas()
    _ensure_requests()
    if pd is None:
        raise RuntimeError("pandas not available")
    if start is None:
        raise ValueError("start_required")
    if end is None:
        raise ValueError("end_required")
    _start = ensure_datetime(start)
    _end = ensure_datetime(end)
    # Normalize timestamps to the minute to avoid querying empty slices
    _start = _start.replace(second=0, microsecond=0)
    _end = _end.replace(second=0, microsecond=0)
    if _canon_tf(timeframe) == "1Min":
        last_complete_minute = _last_complete_minute(pd)
        if _end > last_complete_minute:
            _end = max(_start, last_complete_minute)
    session = _HTTP_SESSION
    if session is None or not hasattr(session, "get"):
        raise ValueError("session_required")
    _interval = _canon_tf(timeframe)
    explicit_feed_request = isinstance(feed, str)
    _feed = _to_feed_str(feed or _DEFAULT_FEED)

    # Mutable state for retry tracking shared by nested helpers.
    _state: dict[str, Any] = {
        "corr_id": None,
        "retries": 0,
        "providers": [],
        "empty_metric_emitted": False,
    }

    def _tags(*, provider: str | None = None, feed: str | None = None) -> dict[str, str]:
        tag_provider = provider if provider is not None else "alpaca"
        tag_feed = _feed if feed is None else feed
        return {"provider": tag_provider, "symbol": symbol, "feed": tag_feed, "timeframe": _interval}

    def _run_backup_fetch(interval_code: str, *, from_provider: str | None = None) -> pd.DataFrame:
        provider_str, normalized_provider = _resolve_backup_provider()
        resolved_provider = normalized_provider or provider_str
        feed_tag = normalized_provider or provider_str
        tags = _tags(provider=resolved_provider, feed=feed_tag)
        _incr("data.fetch.fallback_attempt", value=1.0, tags=tags)
        _state["last_fallback_feed"] = feed_tag
        fallback_df = _backup_get_bars(symbol, _start, _end, interval=interval_code)
        annotated_df = _annotate_df_source(
            fallback_df,
            provider=resolved_provider,
            feed=normalized_provider or None,
        )
        _mark_fallback(
            symbol,
            _interval,
            _start,
            _end,
            from_provider=from_provider or f"alpaca_{_feed}",
            fallback_df=annotated_df,
            resolved_provider=resolved_provider,
            resolved_feed=normalized_provider or None,
        )
        if annotated_df is not None and not getattr(annotated_df, "empty", True):
            _incr("data.fetch.success", value=1.0, tags=tags)
        return annotated_df
    resolved_feed = resolve_alpaca_feed(_feed)
    if resolved_feed is None:
        provider_fallback.labels(from_provider=f"alpaca_{_feed}", to_provider="yahoo").inc()
        logger.warning(
            "ALPACA_FEED_SWITCHOVER",
            extra=_norm_extra(
                {
                    "provider": "alpaca",
                    "requested_feed": _feed,
                    "timeframe": _interval,
                    "symbol": symbol,
                    "fallback": "yahoo",
                }
            ),
        )
        yf_interval = _YF_INTERVAL_MAP.get(_interval, _interval.lower())
        return _run_backup_fetch(yf_interval)
    _feed = resolved_feed
    adjustment_norm = adjustment.lower() if isinstance(adjustment, str) else adjustment
    validate_adjustment(adjustment_norm)
    _validate_alpaca_params(_start, _end, _interval, _feed, adjustment_norm)
    try:
        if not _window_has_trading_session(_start, _end):
            raise ValueError("window_no_trading_session")
    except ValueError as e:
        if "window_no_trading_session" in str(e):
            tf_key = (symbol, _interval)
            _SKIPPED_SYMBOLS.discard(tf_key)
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
            fallback_df = None
            if _ENABLE_HTTP_FALLBACK:
                interval_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "60m", "1Day": "1d"}
                fb_interval = interval_map.get(_interval)
                if fb_interval:
                    provider_str, normalized_provider = _resolve_backup_provider()
                    resolved_provider = normalized_provider or provider_str
                    from_provider_label = f"alpaca_{_feed}"
                    if from_provider_label == "alpaca_iex":
                        from_provider_label = "alpaca_sip"
                    fallback_df = _run_backup_fetch(
                        fb_interval,
                        from_provider=from_provider_label,
                    )
            if fallback_df is not None and not getattr(fallback_df, "empty", True):
                return fallback_df
            empty_df = _empty_ohlcv_frame(pd)
            return empty_df if empty_df is not None else pd.DataFrame()
        raise
    if not _has_alpaca_keys():
        global _ALPACA_KEYS_MISSING_LOGGED
        if not _ALPACA_KEYS_MISSING_LOGGED:
            try:
                logger.warning(
                    "ALPACA_KEYS_MISSING_USING_BACKUP",
                    extra={
                        "provider": getattr(get_settings(), "backup_data_provider", "yahoo"),
                        "hint": "Set ALPACA_API_KEY, ALPACA_SECRET_KEY, and ALPACA_BASE_URL to use Alpaca data",
                    },
                )
                provider_monitor.alert_manager.create_alert(
                    AlertType.SYSTEM,
                    AlertSeverity.CRITICAL,
                    "Alpaca credentials missing; using backup provider",
                    metadata={"provider": getattr(get_settings(), "backup_data_provider", "yahoo")},
                )
            except Exception:
                pass
            _ALPACA_KEYS_MISSING_LOGGED = True
        interval_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "60m", "1Day": "1d"}
        fb_int = interval_map.get(_interval)
        if fb_int:
            return _run_backup_fetch(fb_int)
        return pd.DataFrame()
    global _alpaca_disabled_until, _ALPACA_DISABLED_ALERTED, _alpaca_empty_streak, _alpaca_disable_count
    if _alpaca_disabled_until:
        now = datetime.now(UTC)
        if now < _alpaca_disabled_until:
            if not _ALPACA_DISABLED_ALERTED:
                try:
                    provider_monitor.alert_manager.create_alert(
                        AlertType.SYSTEM,
                        AlertSeverity.CRITICAL,
                        "Primary data provider alpaca disabled",
                        metadata={
                            "provider": "alpaca",
                            "disabled_until": _alpaca_disabled_until.isoformat(),
                        },
                    )
                except Exception:  # pragma: no cover - alerting best effort
                    logger.exception("ALERT_FAILURE", extra={"provider": "alpaca"})
                _ALPACA_DISABLED_ALERTED = True
            try:
                logger.warning(
                    "PRIMARY_PROVIDER_TEMP_DISABLED",
                    extra=_norm_extra(
                        {
                            "provider": "alpaca",
                            "feed": _feed,
                            "timeframe": _interval,
                            "disabled_until": _alpaca_disabled_until.isoformat(),
                        }
                    ),
                )
            except Exception:
                pass
            if explicit_feed_request and _SIP_UNAUTHORIZED and _feed == "iex":
                raise ValueError("rate_limited")
            interval_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "60m", "1Day": "1d"}
            fb_int = interval_map.get(_interval)
            if fb_int:
                return _run_backup_fetch(fb_int)
        else:
            _alpaca_disabled_until = None
            _ALPACA_DISABLED_ALERTED = False
            _alpaca_empty_streak = 0
            prev_disable_count = _alpaca_disable_count
            provider_disabled.labels(provider="alpaca").set(0)
            provider_monitor.record_success("alpaca")
            _alpaca_disable_count = 0
            try:
                logger.info(
                    "PRIMARY_PROVIDER_REENABLED",
                    extra=_norm_extra(
                        {
                            "provider": "alpaca",
                            "feed": _feed,
                            "timeframe": _interval,
                            "disable_count": prev_disable_count,
                        }
                    ),
                )
            except Exception:
                pass
    if _used_fallback(symbol, _interval, _start, _end):
        interval_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "60m", "1Day": "1d"}
        fb_int = interval_map.get(_interval)
        if fb_int:
            return _run_backup_fetch(fb_int)
    # Respect recent fallback TTL at coarse granularity
    try:
        now_s = int(_dt.datetime.now(tz=UTC).timestamp())
    except Exception:
        now_s = int(time.time())
    until = _FALLBACK_UNTIL.get((symbol, _interval))
    if until and now_s < until:
        interval_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "60m", "1Day": "1d"}
        fb_int = interval_map.get(_interval)
        if fb_int:
            return _run_backup_fetch(fb_int)
    global _SIP_DISALLOWED_WARNED
    if _feed == "sip" and not _sip_configured():
        if not _sip_allowed() and not _SIP_DISALLOWED_WARNED:
            logger.warning(
                "SIP_FEED_DISABLED",
                extra=_norm_extra({"provider": "alpaca", "feed": _feed, "timeframe": _interval}),
            )
            _SIP_DISALLOWED_WARNED = True
        if explicit_feed_request:
            _log_sip_unavailable(symbol, _interval, "SIP_UNAVAILABLE")
        else:
            _log_sip_unavailable(symbol, _interval, "SIP_UNAVAILABLE")
            interval_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "60m", "1Day": "1d"}
            fb_int = interval_map.get(_interval)
            if fb_int:
                return _run_backup_fetch(fb_int)
            empty_df = _empty_ohlcv_frame(pd)
            return empty_df if empty_df is not None else pd.DataFrame()

    if _feed == "sip" and _is_sip_unauthorized():
        _log_sip_unavailable(symbol, _interval)
        _incr("data.fetch.unauthorized", value=1.0, tags=_tags())
        metrics.unauthorized += 1
        provider_monitor.record_failure("alpaca", "unauthorized")
        interval_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "60m", "1Day": "1d"}
        fb_int = interval_map.get(_interval)
        if fb_int:
            return _run_backup_fetch(fb_int)
        return pd.DataFrame()

    headers = {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
    }
    timeout_v = clamp_request_timeout(10)

    # Track request start time for retry/backoff telemetry
    start_time = monotonic_time()
    max_retries = _FETCH_BARS_MAX_RETRIES

    def _push_to_caplog(
        message: str,
        *,
        level: int = logging.WARNING,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        handler = _find_pytest_capture_handler()
        if handler is None:
            return
        root_handlers = getattr(logging.getLogger(), "handlers", None)
        if isinstance(root_handlers, list) and handler in root_handlers:
            return
        _emit_capture_record(message, level=level, extra=extra)

    def _log_retry_limit_warning(payload: dict[str, Any]) -> None:
        """Emit a retry-limit warning and surface it to pytest caplog when active."""

        logger.warning("ALPACA_FETCH_RETRY_LIMIT", extra=_norm_extra(payload))
        _push_to_caplog("ALPACA_FETCH_RETRY_LIMIT")

    def _req(
        session: HTTPSession,
        fallback: tuple[str, str, _dt.datetime, _dt.datetime] | None,
        *,
        headers: dict[str, str],
        timeout: float | tuple[float, float],
    ) -> pd.DataFrame:
        nonlocal _interval, _feed, _start, _end
        global _SIP_UNAUTHORIZED, _alpaca_empty_streak, _alpaca_disabled_until, _alpaca_disable_count, _ALPACA_DISABLED_ALERTED
        _state["providers"].append(_feed)

        if session is None or not hasattr(session, "get"):
            raise ValueError("session_required")

        reload_host_limit_if_env_changed(session)

        def _attempt_fallback(
            fb: tuple[str, str, _dt.datetime, _dt.datetime], *, skip_check: bool = False
        ) -> pd.DataFrame | None:
            """Execute a provider fallback attempt.

            Side effects
            -----------
            * Increments the ``provider_fallback`` metric.
            * Records the switchover with the provider monitor.
            * Appends the target provider and symbol to the shared registries.
            """

            nonlocal _interval, _feed, _start, _end
            fb_interval, fb_feed, fb_start, fb_end = fb
            if fb_feed == "sip" and (not skip_check):
                if not _sip_fallback_allowed(session, headers, fb_interval):
                    _log_sip_unavailable(symbol, fb_interval, "UNAUTHORIZED_SIP")
                    return None
            from_feed = _feed
            _interval, _feed, _start, _end = fb
            from_provider_name = f"alpaca_{from_feed}"
            to_provider_name = f"alpaca_{fb_feed}"
            provider_fallback.labels(
                from_provider=f"alpaca_{from_feed}",
                to_provider=f"alpaca_{fb_feed}",
            ).inc()
            _incr("data.fetch.fallback_attempt", value=1.0, tags=_tags())
            _state["last_fallback_feed"] = fb_feed
            payload = _format_fallback_payload_df(_interval, _feed, _start, _end)
            logger.info(
                "DATA_SOURCE_FALLBACK_ATTEMPT",
                extra={"provider": "alpaca", "fallback": payload},
            )
            result = _req(session, None, headers=headers, timeout=timeout)
            if result is not None and not getattr(result, "empty", True):
                _record_feed_switch(symbol, fb_interval, from_feed, fb_feed)
            if result is not None and not _used_fallback(symbol, fb_interval, fb_start, fb_end):
                _mark_fallback(
                    symbol,
                    fb_interval,
                    fb_start,
                    fb_end,
                    from_provider=from_provider_name,
                    fallback_feed=fb_feed,
                    fallback_df=result,
                    resolved_provider=to_provider_name,
                    resolved_feed=fb_feed,
                )
            return result

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
        # Prefer an instance-level patched ``session.get`` when present (tests);
        # otherwise route through the module-level ``requests.get`` so tests
        # that monkeypatch ``df.requests.get`` can intercept deterministically.
        use_session_get = hasattr(session, "get")
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
            if status < 400:
                _ALPACA_DISABLED_ALERTED = False
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
            provider_monitor.record_failure("alpaca", "timeout", str(e))
            fallback_target = fallback
            if (
                fallback_target is None
                and _feed == "iex"
                and _sip_allowed()
                and not _is_sip_unauthorized()
            ):
                fallback_target = (_interval, "sip", _start, _end)
            if fallback_target:
                result = _attempt_fallback(fallback_target, skip_check=True)
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
            provider_monitor.record_failure("alpaca", "connection_error", str(e))
            if fallback:
                result = _attempt_fallback(fallback, skip_check=True)
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
            if isinstance(e, ValueError) and str(e) == "rate_limited" and _feed == "iex" and _is_sip_unauthorized():
                raise
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
                result = _attempt_fallback(fallback, skip_check=True)
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
        data: list[Any] = []
        if isinstance(payload, dict):
            bars_payload = payload.get("bars")
            if isinstance(bars_payload, list):
                data = bars_payload
            elif isinstance(bars_payload, dict):
                # Alpaca v2 multi-symbol payload nests bars under the symbol key.
                for sym_key, sym_bars in bars_payload.items():
                    if isinstance(sym_key, str) and sym_key.upper() == symbol.upper():
                        if isinstance(sym_bars, dict) and isinstance(sym_bars.get("bars"), list):
                            data = sym_bars.get("bars", [])
                        else:
                            data = sym_bars
                        break
            elif symbol in payload and isinstance(payload[symbol], dict) and ("bars" in payload[symbol]):
                data = payload[symbol]["bars"]
        elif isinstance(payload, list):
            data = payload
        if data is None:
            data = []
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
        delay = _state.get("delay")
        if delay is not None:
            log_extra["delay"] = delay
        retry_delay = _state.get("retry_delay")
        if retry_delay is not None and "retry_delay" not in log_extra:
            log_extra["retry_delay"] = retry_delay
        if status == 400:
            log_extra_with_remaining = {"remaining_retries": max_retries - _state["retries"], **log_extra}
            log_fetch_attempt("alpaca", status=status, error="bad_request", **log_extra_with_remaining)
            if "invalid feed" in text.lower():
                provider_fallback.labels(from_provider=f"alpaca_{_feed}", to_provider="yahoo").inc()
                provider_monitor.record_switchover(f"alpaca_{_feed}", "yahoo")
                logger.warning(
                    "ALPACA_FEED_SWITCHOVER",
                    extra=_norm_extra(
                        {
                            "provider": "alpaca",
                            "requested_feed": _feed,
                            "timeframe": _interval,
                            "symbol": symbol,
                            "fallback": "yahoo",
                            "reason": "invalid_feed",
                        }
                    ),
                )
                yf_interval = _YF_INTERVAL_MAP.get(_interval, _interval.lower())
                return _run_backup_fetch(yf_interval)
            raise ValueError("Invalid feed or bad request")
        if status in (401, 403):
            _incr("data.fetch.unauthorized", value=1.0, tags=_tags())
            metrics.unauthorized += 1
            provider_id = "alpaca"
            if _feed in {"sip", "iex"}:
                provider_id = f"alpaca_{_feed}"
            provider_monitor.record_failure(provider_id, "unauthorized")
            log_extra_with_remaining = {"remaining_retries": max_retries - _state["retries"], **log_extra}
            log_fetch_attempt("alpaca", status=status, error="unauthorized", **log_extra_with_remaining)
            if _feed == "sip":
                _mark_sip_unauthorized()
                if _sip_allowed():
                    _log_sip_unavailable(symbol, _interval, "UNAUTHORIZED_SIP")
                is_fallback_request = any(p != "sip" for p in _state.get("providers", [])[:-1])
                if is_fallback_request:
                    raise ValueError("rate_limited")
                interval_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "60m", "1Day": "1d"}
                fb_int = interval_map.get(_interval)
                if fb_int:
                    try:
                        fallback_df = _backup_get_bars(symbol, _start, _end, interval=fb_int)
                    except Exception:
                        fallback_df = pd.DataFrame()
                    provider_str, normalized_provider = _resolve_backup_provider()
                    resolved_provider = normalized_provider or provider_str
                    return _annotate_df_source(
                        fallback_df,
                        provider=resolved_provider,
                        feed=normalized_provider or None,
                    )
                return pd.DataFrame()
            if _feed != "sip":
                logger.warning(
                    "DATA_SOURCE_UNAUTHORIZED",
                    extra=_norm_extra(
                        {"provider": "alpaca", "status": "unauthorized", "feed": _feed, "timeframe": _interval}
                    ),
                )
            if fallback:
                result = _attempt_fallback(fallback)
                if result is not None:
                    return result
            raise ValueError("unauthorized")
        if status == 429:
            requested_feed = _feed
            _incr("data.fetch.rate_limited", value=1.0, tags=_tags())
            sip_locked = _is_sip_unauthorized()
            if requested_feed == "iex" and (sip_locked or _SIP_UNAUTHORIZED):
                raise ValueError("rate_limited")
            metrics.rate_limit += 1
            provider_id = "alpaca"
            if _feed in {"sip", "iex"}:
                provider_id = f"alpaca_{_feed}"
            provider_monitor.record_failure(provider_id, "rate_limit")
            log_extra_with_remaining = {"remaining_retries": max_retries - _state["retries"], **log_extra}
            log_fetch_attempt("alpaca", status=status, error="rate_limited", **log_extra_with_remaining)
            logger.warning(
                "DATA_SOURCE_RATE_LIMITED",
                extra=_norm_extra(
                    {"provider": "alpaca", "status": "rate_limited", "feed": _feed, "timeframe": _interval}
                ),
            )
            retry_after = 0
            try:
                retry_after = int(float(resp.headers.get("Retry-After", "0")))
            except Exception:
                retry_after = 0

            if retry_after > 0:
                already_disabled = provider_monitor.is_disabled("alpaca")
                if not already_disabled and _alpaca_disabled_until:
                    try:
                        already_disabled = datetime.now(UTC) < _alpaca_disabled_until
                    except Exception:
                        already_disabled = False
                if not already_disabled:
                    _incr("data.fetch.provider_disabled", value=1.0, tags=_tags())
                provider_monitor.disable("alpaca", duration=retry_after)
                interval_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "60m", "1Day": "1d"}
                fb_int = interval_map.get(_interval)
                if fb_int:
                    provider_str, normalized_provider = _resolve_backup_provider()
                    resolved_provider = normalized_provider or provider_str
                    provider_fallback.labels(
                        from_provider=f"alpaca_{_feed}", to_provider=resolved_provider
                    ).inc()
                    provider_monitor.record_switchover(
                        f"alpaca_{_feed}", resolved_provider
                    )
                    return _run_backup_fetch(fb_int)
                return pd.DataFrame()
            fallback_target = fallback
            if (
                fallback_target is None
                and requested_feed == "iex"
                and _sip_allowed()
                and not sip_locked
            ):
                try:
                    remaining_fallbacks = max_data_fallbacks()
                except Exception:
                    remaining_fallbacks = 1
                if remaining_fallbacks > 0:
                    fallback_target = (_interval, "sip", _start, _end)
            if fallback_target:
                result = _attempt_fallback(fallback_target)
                if result is not None:
                    return result
                raise ValueError("rate_limited")
            attempt = _state["retries"] + 1
            if requested_feed == "iex" and sip_locked:
                fallback_viable = False
                if fallback_target:
                    _, fb_feed, _, _ = fallback_target
                    if fb_feed != "sip":
                        fallback_viable = True
                    elif _sip_configured() and not sip_locked:
                        fallback_viable = True
                if not fallback_viable:
                    raise ValueError("rate_limited")
            if attempt >= max_retries:
                if fallback:
                    result = _attempt_fallback(fallback)
                    if result is not None:
                        return result
                raise ValueError("rate_limited")
            _state["retries"] = attempt
            backoff = min(_FETCH_BARS_BACKOFF_BASE ** (_state["retries"] - 1), _FETCH_BARS_BACKOFF_CAP)
            logger.debug(
                "RETRY_AFTER_RATE_LIMIT",
                extra=_norm_extra(
                    {
                        "provider": "alpaca",
                        "feed": _feed,
                        "timeframe": _interval,
                        "symbol": symbol,
                        "retry_delay": backoff,
                        "attempt": _state["retries"],
                        "remaining_retries": max_retries - _state["retries"],
                    }
                ),
            )
            time.sleep(backoff)
            return _req(session, fallback, headers=headers, timeout=timeout)
        df = pd.DataFrame(data)
        if data:
            _state.pop("empty_attempts", None)
        if df.empty:
            empty_attempts = _state.get("empty_attempts", 0) + 1
            _state["empty_attempts"] = empty_attempts
            logger.info(
                "EMPTY_BARS_DETECTED",
                extra={
                    "symbol": symbol,
                    "timeframe": _interval,
                    "feed": _feed,
                    "attempt": empty_attempts,
                },
            )
            if not _state.get("empty_metric_emitted"):
                _state["empty_metric_emitted"] = True
                _incr("data.fetch.empty", value=1.0, tags=_tags())
            attempt = _state["retries"] + 1
            remaining_retries = max(0, max_retries - attempt)
            can_retry_timeframe = str(_interval).lower() not in {"1day", "day", "1d"}
            planned_retry_meta: dict[str, Any] = {}
            planned_backoff: float | None = None
            outside_market_hours = _outside_market_hours(_start, _end) if can_retry_timeframe else False
            if attempt <= max_retries and can_retry_timeframe and _state["retries"] < max_retries:
                if not outside_market_hours:
                    planned_backoff = min(
                        _FETCH_BARS_BACKOFF_BASE ** (_state["retries"]),
                        _FETCH_BARS_BACKOFF_CAP,
                    )
                    if _state["retries"] >= 1:
                        planned_retry_meta = _coerce_json_primitives(
                            retry_empty_fetch_once(
                                delay=planned_backoff,
                                attempt=attempt,
                                max_retries=max_retries,
                                previous_correlation_id=prev_corr,
                                total_elapsed=monotonic_time() - start_time,
                            )
                        )
            should_backoff_first_empty = True
            if _ENABLE_HTTP_FALLBACK and not outside_market_hours:
                try:
                    remaining_fallbacks = max_data_fallbacks()
                except Exception:
                    should_backoff_first_empty = True
                else:
                    should_backoff_first_empty = remaining_fallbacks <= 0
            fallback_enabled = bool(fallback) or bool(_ENABLE_HTTP_FALLBACK)
            if not fallback_enabled:
                fallback_enabled = _should_use_backup_on_empty()
            retries_enabled = remaining_retries > 0 and not outside_market_hours
            if outside_market_hours and not (retries_enabled or fallback_enabled):
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
                raise EmptyBarsError(
                    "alpaca_empty: symbol="
                    f"{symbol}, timeframe={_interval}, feed={_feed}, reason=market_closed"
                )
            logged_attempt = False
            if empty_attempts == 1 and should_backoff_first_empty and remaining_retries > 0:
                retry_delay = _state.get("delay", 0.25)
                _state["delay"] = retry_delay
                log_extra["delay"] = retry_delay
                _state["retries"] = attempt
                try:
                    market_open = is_market_open()
                except Exception:
                    market_open = False
                if market_open:
                    _now = datetime.now(UTC)
                    _key = (symbol, "AVAILABLE", _now.date().isoformat(), _feed, _interval)
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
                                    "correlation_id": _state["corr_id"],
                                }
                            ),
                        )
                        _push_to_caplog("EMPTY_DATA", level=lvl)
                logger.warning(
                    "RETRY_SCHEDULED",
                    extra={
                        "symbol": symbol,
                        "timeframe": _interval,
                        "delay": retry_delay,
                        "reason": "empty_bars",
                    },
                )
                log_payload = {
                    **planned_retry_meta,
                    "remaining_retries": remaining_retries,
                    **log_extra,
                }
                log_fetch_attempt("alpaca", status=status, error="empty", **log_payload)
                logged_attempt = True
                time.sleep(retry_delay)
                return _req(session, fallback, headers=headers, timeout=timeout)
            else:
                _state.pop("delay", None)
                log_extra.pop("delay", None)
            if not logged_attempt and attempt <= max_retries:
                log_payload = {
                    **planned_retry_meta,
                    "remaining_retries": remaining_retries,
                    **log_extra,
                }
                log_fetch_attempt("alpaca", status=status, error="empty", **log_payload)
            persistent_empty = empty_attempts >= 2
            if persistent_empty:
                logger.warning(
                    "PERSISTENT_EMPTY_ABORT",
                    extra={
                        "symbol": symbol,
                        "timeframe": _interval,
                        "attempts": empty_attempts,
                    },
                )
                _state.pop("empty_attempts", None)
                if remaining_retries > 0:
                    payload = {
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
                    }
                    logger.warning(
                        "ALPACA_FETCH_ABORTED",
                        extra=_norm_extra(payload),
                    )
                    _push_to_caplog("ALPACA_FETCH_ABORTED", level=logging.WARNING, extra=payload)
                    _state["abort_logged"] = True
            metrics.empty_payload += 1
            is_empty_error = isinstance(payload, dict) and payload.get("error") == "empty"
            base_interval = _interval
            base_feed = _feed
            base_start = _start
            base_end = _end
            tf_key = (symbol, base_interval)
            if status == 200:
                for alt_feed in _iter_preferred_feeds(symbol, base_interval, base_feed):
                    result = _attempt_fallback((base_interval, alt_feed, base_start, base_end))
                    if result is None or getattr(result, "empty", True):
                        _interval, _feed, _start, _end = base_interval, base_feed, base_start, base_end
                        continue
                    return result
                _interval, _feed, _start, _end = base_interval, base_feed, base_start, base_end
            if is_empty_error:
                cnt = _ALPACA_EMPTY_ERROR_COUNTS.get(tf_key, 0) + 1
                _ALPACA_EMPTY_ERROR_COUNTS[tf_key] = cnt
                provider_monitor.record_failure("alpaca", "empty")
                if cnt >= _ALPACA_EMPTY_ERROR_THRESHOLD:
                    provider_str, normalized_provider = _resolve_backup_provider()
                    resolved_provider = normalized_provider or provider_str
                    provider_fallback.labels(
                        from_provider=f"alpaca_{_feed}", to_provider=resolved_provider
                    ).inc()
                    provider_monitor.record_switchover(
                        f"alpaca_{_feed}", resolved_provider
                    )
                    metrics.empty_fallback += 1
                    _ALPACA_EMPTY_ERROR_COUNTS.pop(tf_key, None)
                    _state["stop"] = True
                    interval_map = {
                        "1Min": "1m",
                        "5Min": "5m",
                        "15Min": "15m",
                        "1Hour": "60m",
                        "1Day": "1d",
                    }
                    fb_int = interval_map.get(_interval)
                    if fb_int:
                        return _run_backup_fetch(fb_int)
                    return pd.DataFrame()
            else:
                _ALPACA_EMPTY_ERROR_COUNTS.pop(tf_key, None)
            if fallback:
                fb_interval, fb_feed, fb_start, fb_end = fallback
                if fb_feed in _FEED_FAILOVER_ATTEMPTS.get(tf_key, set()):
                    fallback = None
            if fallback:
                fb_interval, fb_feed, fb_start, fb_end = fallback
                _FEED_FAILOVER_ATTEMPTS.setdefault(tf_key, set()).add(fb_feed)
                result = _attempt_fallback(fallback, skip_check=True)
                if result is not None and not getattr(result, "empty", True):
                    return result
                _interval, _feed, _start, _end = base_interval, base_feed, base_start, base_end
            if _feed == "iex" and is_empty_error:
                cnt = _IEX_EMPTY_COUNTS.get(tf_key, 0) + 1
                _IEX_EMPTY_COUNTS[tf_key] = cnt
                prev = _state.get("corr_id")
                sip_locked = _is_sip_unauthorized()
                allow_sip_fallback = _sip_configured() and not sip_locked
                sip_fallbacks_remaining: int | None = None
                if allow_sip_fallback:
                    try:
                        sip_fallbacks_remaining = max_data_fallbacks()
                    except Exception:
                        sip_fallbacks_remaining = None
                    if sip_fallbacks_remaining is not None and sip_fallbacks_remaining <= 0:
                        attempt = _state.get("retries", 0) + 1
                        if attempt >= max_retries:
                            allow_sip_fallback = False
                        else:
                            _state["retries"] = attempt
                            backoff = min(
                                _FETCH_BARS_BACKOFF_BASE ** (_state["retries"] - 1),
                                _FETCH_BARS_BACKOFF_CAP,
                            )
                            logger.debug(
                                "RETRY_AFTER_SIP_FALLBACK_DISABLED",
                                extra=_norm_extra(
                                    {
                                        "provider": "alpaca",
                                        "feed": _feed,
                                        "timeframe": _interval,
                                        "symbol": symbol,
                                        "retry_delay": backoff,
                                        "attempt": _state["retries"],
                                        "remaining_retries": max_retries - _state["retries"],
                                        "reason": "fallbacks_exhausted",
                                    }
                                ),
                            )
                            time.sleep(backoff)
                            return _req(session, fallback, headers=headers, timeout=timeout)
                if (
                    allow_sip_fallback
                    and "sip" not in _FEED_FAILOVER_ATTEMPTS.get(tf_key, set())
                ):
                    _FEED_FAILOVER_ATTEMPTS.setdefault(tf_key, set()).add("sip")
                    result = _attempt_fallback((base_interval, "sip", base_start, base_end))
                    sip_corr = _state.get("corr_id")
                    if result is not None and not getattr(result, "empty", True):
                        _IEX_EMPTY_COUNTS.pop(tf_key, None)
                        return result
                    _interval, _feed, _start, _end = base_interval, base_feed, base_start, base_end
                if allow_sip_fallback:
                    result = _attempt_fallback((_interval, "sip", _start, _end))
                    sip_corr = _state.get("corr_id")
                    if result is not None and not getattr(result, "empty", True):
                        _IEX_EMPTY_COUNTS.pop(tf_key, None)
                        _record_feed_switch(symbol, base_interval, base_feed, "sip")
                        return result
                    msg = "IEX_EMPTY_SIP_UNAUTHORIZED" if sip_locked else "IEX_EMPTY_SIP_EMPTY"
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
                reason = "UNAUTHORIZED_SIP" if sip_locked else "SIP_UNAVAILABLE"
                if reason != "UNAUTHORIZED_SIP" or _sip_allowed():
                    _log_sip_unavailable(symbol, _interval, reason)
                interval_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "60m", "1Day": "1d"}
                fb_int = interval_map.get(_interval)
                if fb_int:
                    return _run_backup_fetch(fb_int)
                logger.error(
                    "IEX_EMPTY_NO_SIP",
                    extra=_norm_extra(
                        {
                            "provider": "alpaca",
                            "symbol": symbol,
                            "timeframe": _interval,
                            "feed": "iex",
                            "occurrences": _IEX_EMPTY_COUNTS[tf_key],
                            "correlation_id": prev,
                        }
                    ),
                )
                return pd.DataFrame()
            if status == 200 and _should_use_backup_on_empty():
                interval_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "60m", "1Day": "1d"}
                fb_int = interval_map.get(base_interval)
                if fb_int:
                    return _run_backup_fetch(fb_int)
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
                    _push_to_caplog("EMPTY_DATA", level=lvl)
            if fallback:
                result = _attempt_fallback(fallback)
                if result is not None:
                    return result
            key = (symbol, _interval)
            sip_locked = _is_sip_unauthorized()
            if (
                _feed == "iex"
                and _IEX_EMPTY_COUNTS.get(key, 0) >= _IEX_EMPTY_THRESHOLD
                and _sip_configured()
                and not sip_locked
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
                if (_outside_market_hours(_start, _end) or not _window_has_trading_session(_start, _end))
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
                and _sip_configured()
                and not sip_locked
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
                    already_disabled = provider_monitor.is_disabled("alpaca")
                    if not already_disabled and _alpaca_disabled_until:
                        try:
                            already_disabled = datetime.now(UTC) < _alpaca_disabled_until
                        except Exception:
                            already_disabled = False
                    if not already_disabled:
                        _incr("data.fetch.provider_disabled", value=1.0, tags=_tags())
                    provider_monitor.disable("alpaca", duration=300)
                    _ALPACA_DISABLED_ALERTED = True
                    try:
                        provider_monitor.alert_manager.create_alert(
                            AlertType.SYSTEM,
                            AlertSeverity.CRITICAL,
                            "Primary data provider alpaca disabled",
                            metadata={
                                "provider": "alpaca",
                                "disabled_until": _alpaca_disabled_until.isoformat() if _alpaca_disabled_until else "",
                                "reason": "empty",
                            },
                        )
                    except Exception:  # pragma: no cover - alerting best effort
                        logger.exception("ALERT_FAILURE", extra={"provider": "alpaca"})
                remaining_retries = max_retries - _state["retries"]
                payload = {
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
                if remaining_retries > 0:
                    logger.warning("ALPACA_FETCH_ABORTED", extra=_norm_extra(payload))
                    _push_to_caplog("ALPACA_FETCH_ABORTED", level=logging.WARNING)
                    _state["abort_logged"] = True
                    return None
                _log_retry_limit_warning(payload)
                raise EmptyBarsError(
                    "alpaca_empty: symbol="
                    f"{symbol}, timeframe={_interval}, feed={_feed}, reason={reason},"
                    f" retries={_state['retries']}"
                )
            if can_retry_timeframe:
                if _state["retries"] < max_retries:
                    if outside_market_hours:
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
                        raise EmptyBarsError(
                            "alpaca_empty: symbol="
                            f"{symbol}, timeframe={_interval}, feed={_feed}, reason=market_closed"
                        )
                    if planned_backoff is not None:
                        _state["retries"] = attempt
                        _state["delay"] = planned_backoff
                        _state["retry_delay"] = planned_backoff
                        retry_meta = planned_retry_meta or _coerce_json_primitives(
                            retry_empty_fetch_once(
                                delay=planned_backoff,
                                attempt=attempt,
                                max_retries=max_retries,
                                previous_correlation_id=prev_corr,
                                total_elapsed=monotonic_time() - start_time,
                            )
                        )
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
                                    **retry_meta,
                                }
                            ),
                        )
                        time.sleep(planned_backoff)
                        return _req(session, None, headers=headers, timeout=timeout)
                payload = {
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
                _log_retry_limit_warning(payload)
                raise EmptyBarsError(
                    "alpaca_empty: symbol="
                    f"{symbol}, timeframe={_interval}, feed={_feed}, reason={reason},"
                    f" retries={_state['retries']}"
                )
            if (not _open) and str(_interval).lower() in {"1day", "day", "1d"}:
                from ai_trading.utils.lazy_imports import load_pandas as _lp

                pd_mod = _lp()
                try:
                    return pd_mod.DataFrame()
                except Exception:
                    return pd.DataFrame()
            remaining_retries = max_retries - _state["retries"]
            log_event = "ALPACA_FETCH_ABORTED" if remaining_retries > 0 else "ALPACA_FETCH_RETRY_LIMIT"
            if not (log_event == "ALPACA_FETCH_ABORTED" and _state.get("abort_logged")):
                extra_payload = _norm_extra(
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
                )
                logger.warning(log_event, extra=extra_payload)
                _push_to_caplog(log_event, level=logging.WARNING)
            payload = {
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
            if log_event == "ALPACA_FETCH_RETRY_LIMIT":
                _log_retry_limit_warning(payload)
                raise EmptyBarsError(
                    "alpaca_empty: symbol="
                    f"{symbol}, timeframe={_interval}, feed={_feed}, reason={reason},"
                    f" retries={_state['retries']}"
                )
            logger.warning("ALPACA_FETCH_ABORTED", extra=_norm_extra(payload))
            _push_to_caplog("ALPACA_FETCH_ABORTED", level=logging.WARNING)
            _state.pop("abort_logged", None)
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
        state_delay = _state.get("delay")
        if state_delay is not None:
            log_extra_success["delay"] = state_delay
        state_retry_delay = _state.get("retry_delay")
        if state_retry_delay is not None and "retry_delay" not in log_extra_success:
            log_extra_success["retry_delay"] = state_retry_delay
        log_fetch_attempt("alpaca", status=status, **log_extra_success)
        provider_monitor.record_success("alpaca")
        _ALPACA_DISABLED_ALERTED = False
        _alpaca_disable_count = 0
        success_tags = _tags()
        last_fallback_feed = _state.pop("last_fallback_feed", None)
        if last_fallback_feed:
            success_tags = _tags(feed=last_fallback_feed)
        _incr("data.fetch.success", value=1.0, tags=success_tags)
        return df

    priority = list(provider_priority())
    allow_sip = _sip_allowed()
    if not allow_sip:
        priority = [p for p in priority if p != "alpaca_sip"]
    max_fb = max_data_fallbacks()
    alt_feed = None
    fallback = None
    sip_locked_initial = _is_sip_unauthorized()
    if max_fb >= 1:
        if priority:
            try:
                idx = priority.index(f"alpaca_{_feed}")
            except ValueError:
                idx = -1
            # Consider both subsequent and preceding providers to find an Alpaca alt feed
            scan = list(priority[idx + 1 :]) + list(reversed(priority[: max(0, idx)]))
            for prov in scan:
                if prov == "alpaca_sip":
                    if allow_sip and _sip_configured() and not sip_locked_initial and _feed != "sip":
                        alt_feed = "sip"
                        break
                    continue
                if prov == "alpaca_iex" and _feed != "iex":
                    alt_feed = "iex"
                    break
        if alt_feed is not None:
            fallback = (_interval, alt_feed, _start, _end)
        elif _feed == "iex" and allow_sip and _sip_configured() and not sip_locked_initial:
            # Ensure a SIP fallback candidate exists for tests even when
            # provider priority is customized or empty.
            fallback = (_interval, "sip", _start, _end)
    # Attempt request with bounded retries when empty or transient issues occur
    normalized_feed = _normalize_feed_value(feed) if feed is not None else None
    df = None
    last_empty_error: EmptyBarsError | None = None

    empty_attempts = 0
    for _ in range(max(1, max_retries)):
        df = _req(session, fallback, headers=headers, timeout=timeout_v)
        if _state.get("stop"):
            break
        # Stop immediately when SIP is unauthorized; further retries won't help.
        if _feed == "sip" and _is_sip_unauthorized():
            break
        if df is not None and not getattr(df, "empty", True):
            break
        empty_attempts += 1
        if empty_attempts >= 2:
            metrics.empty_fallback += 1
            logger.info(
                "ALPACA_EMPTY_RESPONSE_THRESHOLD",
                extra=_norm_extra(
                    {
                        "provider": "alpaca",
                        "feed": _feed,
                        "timeframe": _interval,
                        "symbol": symbol,
                        "start": _start.isoformat(),
                        "end": _end.isoformat(),
                        "attempts": empty_attempts,
                    }
                ),
            )
            _push_to_caplog("ALPACA_EMPTY_RESPONSE_THRESHOLD", level=logging.INFO)
            break
        # Otherwise, loop to give the provider another chance
    if df is not None and not getattr(df, "empty", True):
        _ALPACA_EMPTY_ERROR_COUNTS.pop((symbol, _interval), None)
        return df
    if _ENABLE_HTTP_FALLBACK and (df is None or getattr(df, "empty", True)):
        interval_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "60m", "1Day": "1d"}
        y_int = interval_map.get(_interval)
        providers_tried = set(_state["providers"])
        can_use_sip = _sip_configured() and not _is_sip_unauthorized()
        yahoo_allowed = (can_use_sip and {"iex", "sip"}.issubset(providers_tried) and max_fb >= 2) or (
            not can_use_sip and "iex" in providers_tried and max_fb >= 1
        )
        if y_int and yahoo_allowed and "yahoo" in priority:
            try:
                alt_df = _yahoo_get_bars(symbol, _start, _end, interval=y_int)
            except Exception:  # pragma: no cover - network variance
                alt_df = pd.DataFrame()
            if alt_df is not None and (not alt_df.empty):
                provider_fallback.labels(from_provider=f"alpaca_{_feed}", to_provider="yahoo").inc()
                logger.info(
                    "DATA_SOURCE_FALLBACK_ATTEMPT",
                    extra=_norm_extra({"provider": "yahoo", "fallback": {"interval": y_int}}),
                )
                annotated_df = _annotate_df_source(
                    alt_df,
                    provider="yahoo",
                    feed="yahoo",
                )
                _mark_fallback(
                    symbol,
                    _interval,
                    _start,
                    _end,
                    from_provider=f"alpaca_{_feed}",
                    fallback_df=annotated_df,
                    resolved_provider="yahoo",
                    resolved_feed="yahoo",
                )
                return annotated_df
    if df is None or getattr(df, "empty", True):
        return None
    return df


def _fetch_minute_from_provider(
    symbol: str,
    feed: str,
    provider: str,
    start: Any,
    end: Any,
    **kwargs: Any,
):
    """Fetch minute bars from a specific provider/feed combination."""

    fetch_kwargs = dict(kwargs)
    normalized_feed = str(feed or "").strip().lower() or None
    provider_lower = provider.lower()
    if provider_lower == "yahoo":
        fetch_kwargs["feed"] = "yahoo"
    elif normalized_feed:
        fetch_kwargs["feed"] = normalized_feed
    return get_minute_df(symbol, start, end, **fetch_kwargs)


def get_minute_df(
    symbol: str,
    start: Any,
    end: Any,
    feed: str | None = None,
    *,
    backfill: str | None = None,
) -> pd.DataFrame | None:
    """Minute bars fetch with provider fallback and gap handling.

    Also updates in-memory minute cache for freshness checks."""
    pd = _ensure_pandas()
    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)
    last_complete_minute = _last_complete_minute(pd)
    if end_dt > last_complete_minute:
        end_dt = max(start_dt, last_complete_minute)
    window_has_session = _window_has_trading_session(start_dt, end_dt)
    tf_key = (symbol, "1Min")
    _ensure_override_state_current()
    normalized_feed = _normalize_feed_value(feed) if feed is not None else None
    backup_provider_str, backup_provider_normalized = _resolve_backup_provider()
    resolved_backup_provider = backup_provider_normalized or backup_provider_str
    resolved_backup_feed = backup_provider_normalized or None

    def _record_minute_fallback(
        *,
        frame: Any | None = None,
        timeframe: str = "1Min",
        window_start: _dt.datetime | None = None,
        window_end: _dt.datetime | None = None,
        from_feed: str | None = None,
    ) -> None:
        start_window = window_start or start_dt
        end_window = window_end or end_dt
        source_feed = from_feed or normalized_feed or _DEFAULT_FEED
        provider_tag = resolved_backup_provider or "yahoo"
        feed_tag = resolved_backup_feed or provider_tag or "yahoo"
        if frame is not None:
            try:
                attrs = getattr(frame, "attrs", {})
            except Exception:
                attrs = {}
            if isinstance(attrs, dict):
                provider_attr = attrs.get("data_provider") or attrs.get("fallback_provider")
                feed_attr = attrs.get("data_feed") or attrs.get("fallback_feed")
                if provider_attr:
                    provider_tag = str(provider_attr).strip() or provider_tag
                if feed_attr:
                    feed_tag = str(feed_attr).strip() or feed_tag
        tags = {
            "provider": str(provider_tag or "unknown"),
            "symbol": symbol,
            "feed": str(feed_tag or provider_tag or "unknown"),
            "timeframe": timeframe,
        }

        def _frame_has_rows(candidate: Any | None) -> bool:
            if candidate is None:
                return False
            empty_attr = getattr(candidate, "empty", None)
            if isinstance(empty_attr, bool):
                return not empty_attr
            if empty_attr is not None:
                try:
                    return not bool(empty_attr)
                except Exception:
                    return False
            try:
                return len(candidate) > 0  # type: ignore[arg-type]
            except Exception:
                return False

        frame_has_rows = _frame_has_rows(frame)
        _incr("data.fetch.fallback_attempt", value=1.0, tags=tags)
        if frame_has_rows:
            _incr("data.fetch.success", value=1.0, tags=tags)
        _mark_fallback(
            symbol,
            timeframe,
            start_window,
            end_window,
            from_provider=f"alpaca_{source_feed}",
            fallback_df=frame,
            resolved_provider=resolved_backup_provider,
            resolved_feed=resolved_backup_feed,
        )
    if normalized_feed is None:
        cached_cycle_feed = _fallback_cache_for_cycle(_get_cycle_id(), symbol, "1Min")
        if cached_cycle_feed:
            try:
                normalized_feed = _normalize_feed_value(cached_cycle_feed)
            except Exception:
                normalized_feed = str(cached_cycle_feed).strip().lower()
    if tf_key in _SKIPPED_SYMBOLS and window_has_session:
        logger.debug("SKIP_SYMBOL_EMPTY_BARS", extra={"symbol": symbol})
        raise EmptyBarsError(f"empty_bars: symbol={symbol}, timeframe=1Min, skipped=1")
    elif tf_key in _SKIPPED_SYMBOLS:
        _SKIPPED_SYMBOLS.discard(tf_key)
        _EMPTY_BAR_COUNTS.pop(tf_key, None)
    try:
        attempt = record_attempt(symbol, "1Min")
    except EmptyBarsError:
        cnt = _EMPTY_BAR_COUNTS.get(tf_key, MAX_EMPTY_RETRIES + 1)
        _log_with_capture(
            logging.ERROR,
            "ALPACA_EMPTY_BAR_MAX_RETRIES",
            extra={"symbol": symbol, "timeframe": "1Min", "occurrences": cnt},
        )
        log_empty_retries_exhausted(
            "alpaca",
            symbol=symbol,
            timeframe="1Min",
            feed=normalized_feed or _DEFAULT_FEED,
            retries=cnt,
        )
        raise
    attempt_count_snapshot = attempt
    used_backup = False
    success_marked = False
    fallback_logged = False
    enable_finnhub = os.getenv("ENABLE_FINNHUB", "1").lower() not in ("0", "false")
    has_finnhub = os.getenv("FINNHUB_API_KEY") and fh_fetcher is not None and not getattr(fh_fetcher, "is_stub", False)
    use_finnhub = enable_finnhub and bool(has_finnhub)
    finnhub_disabled_requested = False
    df = None
    last_empty_error: EmptyBarsError | None = None
    provider_str, backup_normalized = _resolve_backup_provider()
    backup_label = (backup_normalized or provider_str.lower() or "").strip()
    primary_label = f"alpaca_{normalized_feed or _DEFAULT_FEED}"
    force_primary_fetch = True
    if backup_label:
        active_provider = provider_monitor.active_provider(primary_label, backup_label)
        if active_provider == backup_label:
            try:
                refreshed_last_minute = _last_complete_minute(pd)
            except Exception:
                refreshed_last_minute = last_complete_minute
            backup_end_dt = end_dt
            if refreshed_last_minute is not None:
                candidate_end = min(backup_end_dt, refreshed_last_minute)
                backup_end_dt = max(start_dt, candidate_end)
                if backup_end_dt != end_dt:
                    end_dt = backup_end_dt
            try:
                df = _backup_get_bars(symbol, start_dt, backup_end_dt, interval="1m")
            except Exception:
                df = None
            else:
                df = _post_process(df, symbol=symbol, timeframe="1Min")
                if df is not None and not getattr(df, "empty", True):
                    used_backup = True
                    force_primary_fetch = False
                else:
                    df = None
    requested_feed = normalized_feed or _DEFAULT_FEED
    feed_to_use = requested_feed
    initial_feed = requested_feed
    override_feed: str | None = None
    proactive_switch = False
    switch_recorded = False

    if _has_alpaca_keys() and force_primary_fetch:
        try:
            override_raw = _FEED_OVERRIDE_BY_TF.get(tf_key) if feed is None else None
            if override_raw is not None:
                override_feed = _normalize_feed_value(override_raw)
            feed_to_use = override_feed or requested_feed
            initial_feed = requested_feed
            sip_locked_primary = _is_sip_unauthorized()
            if (
                feed_to_use == "iex"
                and _IEX_EMPTY_COUNTS.get(tf_key, 0) > 0
                and _sip_configured()
                and not sip_locked_primary
            ):
                feed_to_use = "sip"
                payload = _format_fallback_payload_df("1Min", "sip", start_dt, end_dt)
                logger.info("DATA_SOURCE_FALLBACK_ATTEMPT", extra={"provider": "alpaca", "fallback": payload})
                proactive_switch = True
            elif feed_to_use == "iex" and _IEX_EMPTY_COUNTS.get(tf_key, 0) > 0 and not _sip_configured():
                _log_sip_unavailable(symbol, "1Min", "SIP_UNAVAILABLE")
            df = _fetch_bars(symbol, start_dt, end_dt, "1Min", feed=feed_to_use)
            if (
                proactive_switch
                and feed_to_use != initial_feed
                and df is not None
                and not getattr(df, "empty", True)
            ):
                _record_feed_switch(symbol, "1Min", initial_feed, feed_to_use)
                switch_recorded = True
        except (EmptyBarsError, ValueError, RuntimeError, AttributeError) as e:
            provider_feed_label = f"alpaca_{feed_to_use}"
            if isinstance(e, EmptyBarsError):
                last_empty_error = e
                _log_fetch_minute_empty(provider_feed_label, "empty_bars", str(e), symbol=symbol)
                now = datetime.now(UTC)
                if end_dt > now or start_dt > now:
                    logger.info(
                        "ALPACA_EMPTY_BAR_FUTURE",
                        extra={"symbol": symbol, "timeframe": "1Min"},
                    )
                    _EMPTY_BAR_COUNTS.pop(tf_key, None)
                    _IEX_EMPTY_COUNTS.pop(tf_key, None)
                    _SKIPPED_SYMBOLS.discard(tf_key)
                    return pd.DataFrame() if pd is not None else []  # type: ignore[return-value]
                try:
                    market_open = is_market_open()
                except Exception:  # pragma: no cover - defensive
                    market_open = True
                if not market_open:
                    _log_with_capture(
                        logging.INFO,
                        "ALPACA_EMPTY_BAR_MARKET_CLOSED",
                        extra={"symbol": symbol, "timeframe": "1Min"},
                    )
                    _EMPTY_BAR_COUNTS.pop(tf_key, None)
                    _IEX_EMPTY_COUNTS.pop(tf_key, None)
                    _SKIPPED_SYMBOLS.discard(tf_key)
                    return pd.DataFrame() if pd is not None else []  # type: ignore[return-value]
                cnt = _EMPTY_BAR_COUNTS.get(tf_key, attempt)
                if cnt > _EMPTY_BAR_MAX_RETRIES:
                    _log_with_capture(
                        logging.ERROR,
                        "ALPACA_EMPTY_BAR_MAX_RETRIES",
                        extra={"symbol": symbol, "timeframe": "1Min", "occurrences": cnt},
                    )
                    log_empty_retries_exhausted(
                        "alpaca",
                        symbol=symbol,
                        timeframe="1Min",
                        feed=normalized_feed or _DEFAULT_FEED,
                        retries=cnt,
                    )
                    _SKIPPED_SYMBOLS.add(tf_key)
                    raise EmptyBarsError(f"empty_bars: symbol={symbol}, timeframe=1Min, max_retries={cnt}") from e
                if cnt >= _EMPTY_BAR_THRESHOLD:
                    backoff = min(2 ** (cnt - _EMPTY_BAR_THRESHOLD), 60)
                    ctx = {
                        "symbol": symbol,
                        "timeframe": "1Min",
                        "occurrences": cnt,
                        "backoff": backoff,
                        "finnhub_enabled": use_finnhub,
                        "feed": normalized_feed or _DEFAULT_FEED,
                    }
                    _log_with_capture(logging.WARNING, "ALPACA_EMPTY_BAR_BACKOFF", extra=ctx)
                    time.sleep(backoff)
                    alt_feed = None
                    max_fb = max_data_fallbacks()
                    attempted_feeds = _FEED_FAILOVER_ATTEMPTS.setdefault(tf_key, set())
                    current_feed = normalized_feed or _DEFAULT_FEED
                    sip_locked_backoff = _is_sip_unauthorized()
                    if max_fb >= 1 and len(attempted_feeds) < max_fb:
                        priority_order = list(provider_priority())
                        if not _sip_allowed():
                            priority_order = [
                                prov for prov in priority_order if prov != "alpaca_sip"
                            ]
                        try:
                            cur_idx = priority_order.index(f"alpaca_{current_feed}")
                        except ValueError:
                            cur_idx = -1
                        ordered_scan: list[str] = []
                        if cur_idx >= 0:
                            ordered_scan.extend(priority_order[cur_idx + 1 :])
                            ordered_scan.extend(reversed(priority_order[:cur_idx]))
                        else:
                            ordered_scan = priority_order
                        for prov in ordered_scan:
                            if not prov.startswith("alpaca_"):
                                continue
                            candidate = prov.split("_", 1)[1]
                            if candidate == current_feed:
                                continue
                            if candidate not in {"iex", "sip"}:
                                continue
                            if candidate in attempted_feeds:
                                continue
                            if candidate == "sip" and (not _sip_configured() or sip_locked_backoff):
                                continue
                            alt_feed = candidate
                            break
                    if (
                        alt_feed is None
                        and max_fb >= 1
                        and len(attempted_feeds) < max_fb
                        and current_feed == "iex"
                        and _sip_configured()
                        and not sip_locked_backoff
                        and "sip" not in attempted_feeds
                    ):
                        alt_feed = "sip"
                    if alt_feed and alt_feed != current_feed:
                        attempted_feeds.add(alt_feed)
                        try:
                            df_alt = _fetch_bars(symbol, start_dt, end_dt, "1Min", feed=alt_feed)
                        except (EmptyBarsError, ValueError, RuntimeError) as alt_err:
                            logger.debug(
                                "ALPACA_ALT_FEED_FAILED",
                                extra={
                                    "symbol": symbol,
                                    "from_feed": normalized_feed or _DEFAULT_FEED,
                                    "to_feed": alt_feed,
                                    "err": str(alt_err),
                                },
                            )
                            df_alt = None
                        else:
                            if df_alt is not None and not getattr(df_alt, "empty", True):
                                logger.info(
                                    "ALPACA_ALT_FEED_SUCCESS",
                                    extra={
                                        "symbol": symbol,
                                        "from_feed": normalized_feed or _DEFAULT_FEED,
                                        "to_feed": alt_feed,
                                        "timeframe": "1Min",
                                    },
                                )
                                _record_feed_switch(symbol, "1Min", current_feed, alt_feed)
                                _IEX_EMPTY_COUNTS.pop(tf_key, None)
                                df_alt = _post_process(df_alt, symbol=symbol, timeframe="1Min")
                                df_alt = _verify_minute_continuity(df_alt, symbol, backfill=backfill)
                                if df_alt is not None and not getattr(df_alt, "empty", True):
                                    mark_success(symbol, "1Min")
                                    _EMPTY_BAR_COUNTS.pop(tf_key, None)
                                    _SKIPPED_SYMBOLS.discard(tf_key)
                                    return df_alt
                                df = df_alt
                    if end_dt - start_dt > _dt.timedelta(days=1):
                        short_start = end_dt - _dt.timedelta(days=1)
                        logger.debug(
                            "ALPACA_SHORT_WINDOW_RETRY",
                            extra={
                                "symbol": symbol,
                                "timeframe": "1Min",
                                "start": short_start.isoformat(),
                                "end": end_dt.isoformat(),
                                "feed": normalized_feed or _DEFAULT_FEED,
                            },
                        )
                        try:
                            df_short = _fetch_bars(
                                symbol,
                                short_start,
                                end_dt,
                                "1Min",
                                feed=normalized_feed or _DEFAULT_FEED,
                            )
                        except (EmptyBarsError, ValueError, RuntimeError):
                            df_short = None
                        else:
                            if df_short is not None and not getattr(df_short, "empty", True):
                                logger.info(
                                    "ALPACA_SHORT_WINDOW_SUCCESS",
                                    extra={
                                        "symbol": symbol,
                                        "timeframe": "1Min",
                                        "feed": normalized_feed or _DEFAULT_FEED,
                                        "start": short_start.isoformat(),
                                        "end": end_dt.isoformat(),
                                    },
                                )
                                _IEX_EMPTY_COUNTS.pop(tf_key, None)
                                df_short = _post_process(df_short, symbol=symbol, timeframe="1Min")
                                df_short = _verify_minute_continuity(df_short, symbol, backfill=backfill)
                                if df_short is not None and not getattr(df_short, "empty", True):
                                    mark_success(symbol, "1Min")
                                    return df_short
                                df = df_short
                    df = None
                else:
                    logger.debug(
                        "ALPACA_EMPTY_BARS",
                        extra={
                            "symbol": symbol,
                            "timeframe": "1Min",
                            "feed": normalized_feed or _DEFAULT_FEED,
                            "occurrences": cnt,
                        },
                    )
                    df = None
            else:
                if isinstance(e, ValueError) and "invalid_time_window" in str(e):
                    _log_fetch_minute_empty(provider_feed_label, "invalid_time_window", str(e), symbol=symbol)
                    last_empty_error = EmptyBarsError(
                        f"empty_bars: symbol={symbol}, timeframe=1Min, reason=invalid_time_window"
                    )
                    last_empty_error.__cause__ = e  # type: ignore[attr-defined]
                else:
                    logger.warning("ALPACA_FETCH_FAILED", extra={"symbol": symbol, "err": str(e)})
                df = None
    else:
        _warn_missing_alpaca(symbol, "1Min")
        df = None

    if (
        df is not None
        and not getattr(df, "empty", True)
        and feed is None
        and feed_to_use != initial_feed
        and not switch_recorded
        and not used_backup
    ):
        existing_override = _FEED_OVERRIDE_BY_TF.get(tf_key)
        normalized_override: str | None = None
        if existing_override is not None:
            try:
                normalized_override = _normalize_feed_value(existing_override)
            except ValueError:
                normalized_override = str(existing_override).strip().lower() or None
        if normalized_override != feed_to_use:
            _record_feed_switch(symbol, "1Min", initial_feed, feed_to_use)
            switch_recorded = True
    if df is None or getattr(df, "empty", True):
        if use_finnhub:
            finnhub_df = None
            try:
                finnhub_df = _finnhub_get_bars(symbol, start_dt, end_dt, "1m")
            except (FinnhubAPIException, ValueError, NotImplementedError) as e:
                logger.debug("FINNHUB_FETCH_FAILED", extra={"symbol": symbol, "err": str(e)})
            else:
                logger.debug(
                    "FINNHUB_FETCH_SUCCESS",
                    extra={
                        "symbol": symbol,
                        "rows": getattr(finnhub_df, "shape", (0,))[0] if finnhub_df is not None else 0,
                    },
                )
                finnhub_df = _annotate_df_source(
                    finnhub_df,
                    provider="finnhub",
                    feed="finnhub",
                )
            df = finnhub_df
            used_backup = True
        elif not enable_finnhub:
            finnhub_disabled_requested = True
            warn_finnhub_disabled_no_data(
                symbol,
                timeframe="1Min",
                start=start_dt,
                end=end_dt,
            )
        else:
            log_finnhub_disabled(symbol)
    if df is None or getattr(df, "empty", True):
        max_span = _dt.timedelta(days=7)
        total_span = end_dt - start_dt
        if total_span > max_span:
            logger.warning(
                "YF_1M_RANGE_SPLIT",
                extra={
                    "symbol": symbol,
                    "interval": "1m",
                    "start": start_dt.isoformat(),
                    "end": end_dt.isoformat(),
                    "max_days": 7,
                },
            )
            _emit_capture_record(
                "YF_1M_RANGE_SPLIT",
                extra={
                    "symbol": symbol,
                    "interval": "1m",
                    "start": start_dt.isoformat(),
                    "end": end_dt.isoformat(),
                    "max_days": 7,
                },
            )
            logger.warning("YF_1", extra={"symbol": symbol, "interval": "1m"})
            _emit_capture_record("YF_1", extra={"symbol": symbol, "interval": "1m"})
            dfs: list[pd.DataFrame] = []  # type: ignore[var-annotated]
            cur_start = start_dt
            while cur_start < end_dt:
                cur_end = min(cur_start + max_span, end_dt)
                dfs.append(_backup_get_bars(symbol, cur_start, cur_end, interval="1m"))
                used_backup = True
                cur_start = cur_end
            if pd is not None and dfs:
                df = pd.concat(dfs, ignore_index=True)
                first_attrs = getattr(dfs[0], "attrs", {}) if dfs else {}
                provider_attr = first_attrs.get("data_provider") or first_attrs.get("fallback_provider")
                feed_attr = first_attrs.get("data_feed") or first_attrs.get("fallback_feed")
                if "timestamp" in df.columns:
                    try:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                    except Exception:
                        pass
                    else:
                        df = df.sort_values("timestamp")
                        df = df.drop_duplicates(subset="timestamp", keep="last")
                        df = df.reset_index(drop=True)
                if provider_attr:
                    df = _annotate_df_source(
                        df,
                        provider=str(provider_attr),
                        feed=str(feed_attr) if feed_attr else None,
                    )
            elif dfs:
                df = dfs[0]
            else:
                df = pd.DataFrame() if pd is not None else []  # type: ignore[assignment]
        else:
            df = _backup_get_bars(symbol, start_dt, end_dt, interval="1m")
            used_backup = True

    if used_backup and df is not None and not getattr(df, "empty", True):
        processed_df = _post_process(df, symbol=symbol, timeframe="1Min")
        if processed_df is not None and not getattr(processed_df, "empty", True):
            df = processed_df
            if not fallback_logged:
                _record_minute_fallback(frame=df)
                fallback_logged = True
            try:
                attrs = getattr(df, "attrs", None)
            except Exception:
                attrs = None
            if isinstance(attrs, dict):
                coverage_meta = attrs.get("_coverage_meta")
                if isinstance(coverage_meta, dict):
                    attrs["_coverage_meta"] = dict(coverage_meta)
            _set_price_reliability(df, reliable=True)
            mark_success(symbol, "1Min")
            success_marked = True
            _EMPTY_BAR_COUNTS.pop(tf_key, None)
            _SKIPPED_SYMBOLS.discard(tf_key)
    attempt_count_snapshot = max(attempt_count_snapshot, _EMPTY_BAR_COUNTS.get(tf_key, attempt_count_snapshot))
    allow_empty_return = not window_has_session
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
    except (ValueError, TypeError, KeyError, AttributeError):
        pass
    if finnhub_disabled_requested and (df is None or getattr(df, "empty", True)):
        warn_finnhub_disabled_no_data(
            symbol,
            timeframe="1Min",
            start=start_dt,
            end=end_dt,
        )
        dedupe_key = ":".join(
            [
                symbol,
                "1Min",
                f"{start_dt.isoformat()}->{end_dt.isoformat()}",
            ]
        )
        if dedupe_key not in _FINNHUB_CAPTURE_KEYS:
            _FINNHUB_CAPTURE_KEYS.add(dedupe_key)
            handler = _find_pytest_capture_handler()
            root_handlers = getattr(logging.getLogger(), "handlers", None)
            if not (isinstance(root_handlers, list) and handler in root_handlers):
                _emit_capture_record(
                    "FINNHUB_DISABLED_NO_DATA",
                    level=logging.INFO,
                    extra={
                        "symbol": symbol,
                        "timeframe": "1Min",
                        "start": start_dt.isoformat(),
                        "end": end_dt.isoformat(),
                    },
                )
    original_df = df
    if original_df is None:
        if allow_empty_return:
            if used_backup and not fallback_logged:
                _record_minute_fallback()
                fallback_logged = True
            _IEX_EMPTY_COUNTS.pop(tf_key, None)
            _SKIPPED_SYMBOLS.discard(tf_key)
            _EMPTY_BAR_COUNTS.pop(tf_key, None)
            return pd.DataFrame() if pd is not None else []  # type: ignore[return-value]
        if last_empty_error is not None:
            raise last_empty_error
        raise EmptyBarsError(f"empty_bars: symbol={symbol}, timeframe=1Min")
    if getattr(original_df, "empty", False):
        if allow_empty_return:
            if used_backup and not fallback_logged:
                _record_minute_fallback(frame=original_df)
                fallback_logged = True
            _IEX_EMPTY_COUNTS.pop(tf_key, None)
            _SKIPPED_SYMBOLS.discard(tf_key)
            return original_df
        if last_empty_error is not None:
            raise last_empty_error
        raise EmptyBarsError(f"empty_bars: symbol={symbol}, timeframe=1Min")
    try:
        df = _post_process(original_df, symbol=symbol, timeframe="1Min")
    except DataFetchError as exc:
        if used_backup:
            logger.warning(
                "YAHOO_PRICES_MISSING",
                extra={
                    "symbol": symbol,
                    "timeframe": "1Min",
                    "detail": str(exc),
                },
            )
            err = EmptyBarsError(
                f"empty_bars: symbol={symbol}, timeframe=1Min, provider=yahoo"
            )
            setattr(err, "fetch_reason", getattr(exc, "fetch_reason", "ohlcv_columns_missing"))
            raise err from exc
        raise
    if df is None:
        if allow_empty_return:
            if used_backup and not fallback_logged:
                _record_minute_fallback(frame=original_df)
                fallback_logged = True
            _IEX_EMPTY_COUNTS.pop(tf_key, None)
            _SKIPPED_SYMBOLS.discard(tf_key)
            return original_df if original_df is not None else df
        if last_empty_error is not None:
            raise last_empty_error
        raise EmptyBarsError(f"empty_bars: symbol={symbol}, timeframe=1Min")
    df = _verify_minute_continuity(df, symbol, backfill=backfill)
    coverage_meta: dict[str, object] = {"expected": 0, "missing_after": 0, "gap_ratio": 0.0}
    repair_used_backup = False
    tz_info = ZoneInfo("America/New_York")
    df, coverage_meta, repair_used_backup = _repair_rth_minute_gaps(
        df,
        symbol=symbol,
        start=start_dt,
        end=end_dt,
        tz=tz_info,
    )
    if repair_used_backup:
        used_backup = True
    try:
        attrs = getattr(df, "attrs", None)
        if isinstance(attrs, dict):
            attrs.setdefault("symbol", symbol)
            attrs["_coverage_meta"] = coverage_meta
    except Exception:
        pass

    def _gap_ratio_setting() -> float:
        try:
            env_ratio = get_env("AI_TRADING_GAP_RATIO_LIMIT", None, cast=float)
        except Exception:
            env_ratio = None
        if env_ratio is not None:
            try:
                return max(float(env_ratio) * 10000.0, 0.0)
            except (TypeError, ValueError):
                pass
        for key in ("DATA_MAX_GAP_RATIO_BPS", "MAX_GAP_RATIO_BPS"):
            try:
                value = get_env(key, None, cast=float)
            except Exception:
                continue
            if value is not None:
                try:
                    return max(float(value), 0.0)
                except (TypeError, ValueError):
                    continue
        return 50.0

    max_gap_bps = _gap_ratio_setting()
    max_gap_ratio = max(0.0, max_gap_bps / 10000.0)
    gap_ratio = float(coverage_meta.get("gap_ratio", 0.0))
    healthy_gap = gap_ratio <= max_gap_ratio
    severity = "good" if healthy_gap else "degraded"
    gap_reason = f"gap_ratio={gap_ratio * 100:.2f}%"
    if backup_label:
        provider_monitor.update_data_health(
            primary_label,
            backup_label,
            healthy=healthy_gap,
            reason=gap_reason,
            severity=severity,
        )
    try:
        settings_obj = get_settings()
    except Exception:
        settings_obj = None
    gap_limit = None
    if settings_obj is not None:
        try:
            data_settings = getattr(settings_obj, "data", None)
            if data_settings is not None:
                candidate = getattr(data_settings, "max_gap_ratio_intraday", None)
                if candidate is not None:
                    gap_limit = max(float(candidate), 0.0)
        except Exception:
            gap_limit = None
    if not healthy_gap:
        try:
            coverage_meta["status"] = "degraded"
            coverage_meta["gap_reason"] = gap_reason
        except Exception:
            pass
    if gap_limit is None:
        gap_limit = max_gap_ratio
    price_reliable = True
    unreliable_reason: str | None = None
    try:
        ratio_value = float(gap_ratio)
    except (TypeError, ValueError):
        ratio_value = 0.0
    if ratio_value > gap_limit:
        price_reliable = False
        unreliable_reason = f"gap_ratio={ratio_value * 100:.2f}%>limit={gap_limit * 100:.2f}%"
    _set_price_reliability(df, reliable=price_reliable, reason=unreliable_reason)
    if df is None or getattr(df, "empty", False):
        if allow_empty_return:
            if used_backup and not fallback_logged:
                _record_minute_fallback(frame=df)
                fallback_logged = True
            _IEX_EMPTY_COUNTS.pop(tf_key, None)
            _SKIPPED_SYMBOLS.discard(tf_key)
            if df is None:
                return original_df if original_df is not None else df
            return df
        if last_empty_error is not None:
            raise last_empty_error
        raise EmptyBarsError(f"empty_bars: symbol={symbol}, timeframe=1Min")
    if used_backup and (df is None or getattr(df, "empty", False)):
        logger.warning(
            "YAHOO_PRICES_MISSING",
            extra={"symbol": symbol, "timeframe": "1Min", "detail": "empty_frame"},
        )
    if not success_marked:
        mark_success(symbol, "1Min")
        success_marked = True
        if used_backup and not fallback_logged:
            _record_minute_fallback(frame=df)
            fallback_logged = True
        _IEX_EMPTY_COUNTS.pop(tf_key, None)
    source_label = (
        resolved_backup_provider
        if used_backup
        else primary_label
    )
    try:
        ensured_df = ensure_ohlcv_schema(
            df,
            source=source_label or "alpaca",
            frequency="1Min",
        )
        df = _mutate_dataframe_in_place(df, ensured_df)

        normalized_df = normalize_ohlcv_df(df)
        df = _mutate_dataframe_in_place(df, normalized_df)

        restored_df = _restore_timestamp_column(df)
        df = _mutate_dataframe_in_place(df, restored_df)
    except MissingOHLCVColumnsError as exc:
        logger.error(
            "OHLCV_COLUMNS_MISSING",
            extra={"source": source_label, "frequency": "1Min", "detail": str(exc)},
        )
        return None
    except DataFetchError as exc:
        logger.error(
            "DATA_FETCH_EMPTY",
            extra={"source": source_label, "frequency": "1Min", "detail": str(exc)},
        )
        return None
    return df


def get_daily_df(
    symbol: str,
    start: Any | None = None,
    end: Any | None = None,
    *,
    feed: str | None = None,
    adjustment: str | None = None,
) -> pd.DataFrame:
    """Fetch daily bars and ensure canonical OHLCV columns."""

    rename_map = {
        "t": "timestamp",
        "time": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }

    use_alpaca = should_import_alpaca_sdk()

    normalized_feed = _normalize_feed_value(feed) if feed is not None else None

    if feed is None:
        tf_key = (symbol, _canon_tf("1Day"))
        override = _FEED_OVERRIDE_BY_TF.get(tf_key)
        if override:
            normalized_feed = _normalize_feed_value(override)

    df: Any = None
    if not use_alpaca:
        start_dt = ensure_datetime(start if start is not None else datetime.now(UTC) - _dt.timedelta(days=10))
        end_dt = ensure_datetime(end if end is not None else datetime.now(UTC))
        df = _backup_get_bars(symbol, start_dt, end_dt, interval=_YF_INTERVAL_MAP.get("1Day", "1d"))
        if df is None or getattr(df, "empty", False):
            try:
                from ai_trading import alpaca_api as _bars_mod

                fallback_df = _bars_mod.get_bars_df(
                    symbol,
                    timeframe="1Day",
                    start=start,
                    end=end,
                    feed=normalized_feed,
                    adjustment=adjustment,
                )
            except Exception:  # pragma: no cover - optional dependency
                fallback_df = None
            if fallback_df is not None:
                df = fallback_df
    else:
        try:
            from ai_trading.alpaca_api import get_bars_df as _get_bars_df
        except Exception as exc:  # pragma: no cover - optional dependency
            raise DataFetchError("Alpaca API unavailable") from exc

    adjustment = adjustment or "raw"
    if isinstance(adjustment, str):
        adjustment = adjustment.lower()
    validate_adjustment(adjustment)

    if use_alpaca:
        df = _get_bars_df(
            symbol,
            timeframe="1Day",
            start=start,
            end=end,
            feed=normalized_feed,
            adjustment=adjustment,
        )

    pd_mod = _ensure_pandas()
    if pd_mod is None:
        return df

    if isinstance(df, pd_mod.DataFrame):
        if "timestamp" not in df.columns and getattr(df.index, "name", None) in {
            "t",
            "time",
            "timestamp",
        }:
            df = df.reset_index().rename(columns={df.index.name: "timestamp"})
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    try:
        df = ensure_ohlcv_schema(
            df,
            source=normalized_feed or "alpaca",
            frequency="1Day",
        )
        df = normalize_ohlcv_df(df)
        df = _restore_timestamp_column(df)
    except MissingOHLCVColumnsError as exc:
        logger.error(
            "OHLCV_COLUMNS_MISSING",
            extra={"source": normalized_feed or "alpaca", "frequency": "1Day", "detail": str(exc)},
        )
        return None
    except DataFetchError as exc:
        logger.error(
            "DATA_FETCH_EMPTY",
            extra={"source": normalized_feed or "alpaca", "frequency": "1Day", "detail": str(exc)},
        )
        return None
    return df


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
    normalized_feed: str | None = None
    if isinstance(feed, str):
        normalized_feed = _normalize_feed_value(feed)

    # Resolve feed preference from settings when not explicitly provided
    tf_norm = _canon_tf(timeframe)
    if normalized_feed is None:
        override = _FEED_OVERRIDE_BY_TF.get((symbol, tf_norm))
        if override:
            normalized_feed = _normalize_feed_value(override)
        else:
            prio = provider_priority(S)
            feed_candidate: str | None = None
            for prov in prio:
                if prov.startswith("alpaca_"):
                    feed_candidate = prov.split("_", 1)[1]
                    break
            if feed_candidate is None:
                feed_candidate = getattr(S, "data_feed", getattr(S, "alpaca_data_feed", "iex"))
            normalized_feed = _normalize_feed_value(feed_candidate)

    adjustment = adjustment or S.alpaca_adjustment
    if isinstance(adjustment, str):
        adjustment = adjustment.lower()
    validate_adjustment(adjustment)
    # If Alpaca credentials are missing, skip direct Alpaca HTTP calls and
    # fall back to the Yahoo helper to avoid noisy empty/unauthorized logs.
    if not _has_alpaca_keys():
        global _ALPACA_KEYS_MISSING_LOGGED
        if not _ALPACA_KEYS_MISSING_LOGGED:
            backup_provider = getattr(S, "backup_data_provider", "yahoo")
            try:
                logger.warning(
                    "ALPACA_KEYS_MISSING_USING_BACKUP",
                    extra={
                        "provider": backup_provider,
                        "hint": "Set ALPACA_API_KEY, ALPACA_SECRET_KEY, and ALPACA_BASE_URL to use Alpaca data",
                    },
                )
                provider_monitor.alert_manager.create_alert(
                    AlertType.SYSTEM,
                    AlertSeverity.CRITICAL,
                    "Alpaca credentials missing; using backup provider",
                    metadata={"provider": backup_provider},
                )
            except Exception:
                # Never allow diagnostics to break data path
                pass
            _ALPACA_KEYS_MISSING_LOGGED = True
    interval_map = {
        "1Min": "1m",
        "5Min": "5m",
        "15Min": "15m",
        "1Hour": "60m",
        "1Day": "1d",
        "1D": "1d",
        "1H": "60m",
    }
    y_int = interval_map.get(tf_norm, None)
    if y_int is None:
        # Best effort: daily vs intraday
        y_int = "1d" if tf_norm.lower() in {"1day", "day", "1d"} else "1m"
        try:
            return _backup_get_bars(symbol, ensure_datetime(start), ensure_datetime(end), interval=y_int)
        except Exception:
            # Defer to Alpaca path (will return None) to preserve behavior
            return _fetch_bars(symbol, start, end, timeframe, feed=feed, adjustment=adjustment)
    return _fetch_bars(symbol, start, end, timeframe, feed=normalized_feed, adjustment=adjustment)


def get_bars_batch(
    symbols: list[str], timeframe: str, start: Any, end: Any, *, feed: str | None = None, adjustment: str | None = None
) -> dict[str, pd.DataFrame | None]:
    """Fetch bars for multiple symbols via get_bars."""
    return {sym: get_bars(sym, timeframe, start, end, feed=feed, adjustment=adjustment) for sym in symbols}


def fetch_minute_yfinance(symbol: str, start_dt: _dt.datetime, end_dt: _dt.datetime) -> pd.DataFrame:
    """Explicit helper for tests and optional direct Yahoo minute fetch."""
    df = _yahoo_get_bars(symbol, start_dt, end_dt, interval="1m")
    return _post_process(df, symbol=symbol, timeframe="1m")


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
    "_HAS_SIP",
    "_SIP_UNAUTHORIZED",
    "_sip_allowed",
    "_ordered_fallbacks",
    "_is_sip_unauthorized",
    "_mark_sip_unauthorized",
    "_clear_sip_lockout_for_tests",
    "_reset_provider_auth_state_for_tests",
    "_get_cycle_id",
    "_fallback_cache_for_cycle",
    "reload_env_settings",
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
    "_get_cached_or_primary",
    "_cache_fallback",
    "_fetch_minute_from_provider",
    "get_minute_df",
    "get_daily_df",
    "run_with_concurrency",
    "daily_fetch_memo",
    "metrics",
    "build_fetcher",
    "DataFetchError",
    "MissingOHLCVColumnsError",
    "UnauthorizedSIPError",
    "DataFetchException",
    "FinnhubAPIException",
    "get_cached_minute_timestamp",
    "set_cached_minute_timestamp",
    "clear_cached_minute_timestamp",
    "get_fallback_metadata",
    "age_cached_minute_timestamps",
    "last_minute_bar_age_seconds",
    "_build_daily_url",
    "retry_empty_fetch_once",
    "is_primary_provider_enabled",
    "should_skip_symbol",
]
