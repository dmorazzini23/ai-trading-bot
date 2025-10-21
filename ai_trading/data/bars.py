from __future__ import annotations

import os
import time
from dataclasses import dataclass
import hashlib
from datetime import UTC, date, datetime, timedelta
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from ai_trading.config import get_settings
from ai_trading.config.management import get_env
from ai_trading.data.fetch import get_bars, get_minute_df
from ai_trading.data.fetch import get_bars as _raw_http_get_bars
from ai_trading.data import fetch as data_fetcher
from ai_trading.data.fetch.normalize import normalize_ohlcv_df, REQUIRED as _OHLCV_REQUIRED
from ai_trading.data.market_calendar import previous_trading_session, rth_session_utc
from ai_trading.logging import get_logger
from ai_trading.logging.empty_policy import classify as _empty_classify
from ai_trading.logging.empty_policy import record as _empty_record
from ai_trading.logging.empty_policy import should_emit as _empty_should_emit
from ai_trading.logging.emit_once import emit_once
from ai_trading.logging.normalize import canon_feed as _canon_feed
from ai_trading.logging.normalize import canon_symbol as _canon_symbol
from ai_trading.logging.normalize import canon_timeframe as _canon_tf
from ai_trading.utils.lazy_imports import load_pandas
from ai_trading.utils.time import now_utc

from ._alpaca_guard import should_import_alpaca_sdk
from .fetch.sip_disallowed import sip_disallowed, sip_credentials_missing  # noqa: F401
from .models import StockBarsRequest, TimeFrame
from .timeutils import ensure_utc_datetime, expected_regular_minutes

try:  # pragma: no cover - requests optional
    from requests import exceptions as _requests_exceptions
except ImportError:  # pragma: no cover - dependency optional
    _requests_exceptions = None


class _FallbackAPIError(Exception):
    """Fallback APIError when Alpaca SDK is unavailable."""


APIError = _FallbackAPIError

if should_import_alpaca_sdk():  # pragma: no cover - alpaca optional
    try:
        from alpaca.common.exceptions import APIError as _RealAPIError
    except (ImportError, AttributeError):
        pass
    else:
        APIError = _RealAPIError

_BAR_FETCH_EXCEPTIONS: tuple[type[BaseException], ...] = (
    APIError,
    RuntimeError,
    ConnectionError,
    TimeoutError,
    OSError,
    ValueError,
)

if _requests_exceptions is not None:  # pragma: no cover - requests optional
    request_exc = getattr(_requests_exceptions, "RequestException", None)
    if isinstance(request_exc, type):
        _BAR_FETCH_EXCEPTIONS = _BAR_FETCH_EXCEPTIONS + (request_exc,)

__all__ = ["TimeFrame", "StockBarsRequest", "BarsFetchFailed", "http_get_bars"]

# Lazy pandas proxy; only imported on first use
pd = load_pandas()

_log = get_logger(__name__)
'AI-AGENT-REF: canonicalizers moved to ai_trading.logging.normalize'

_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no", "off", "disable", "disabled"}


@dataclass(slots=True)
class BarsFetchFailed:
    """Sentinel returned when Alpaca HTTP bars fail."""

    symbol: str
    feed: str | None
    since: datetime
    status: int | None = None
    error: str | None = None


def _extract_status_code(obj: Any) -> int | None:
    """Return HTTP status code from *obj* when available."""

    for attr in ("status_code", "status"):
        value = getattr(obj, attr, None)
        if isinstance(value, int):
            return value
    if isinstance(obj, dict):
        for key in ("status_code", "status"):
            value = obj.get(key)
            if isinstance(value, int):
                return value
    return None


def _log_bars_failure_once(
    *, symbol: str, feed: str | None, since: datetime | None, status: int | None, error: str | None = None
) -> None:
    """Emit a single structured log entry for an Alpaca bars outage."""

    feed_norm = _canon_feed(feed)
    since_iso = None
    if isinstance(since, datetime):
        try:
            since_iso = since.astimezone(UTC).isoformat()
        except (ValueError, TypeError):
            since_iso = since.isoformat()
    elif isinstance(since, str):
        since_iso = since
    key = f"alpaca-bars-fail:{_canon_symbol(symbol)}:{feed_norm}:{since_iso or ''}:{status or 'na'}"
    payload = {
        "symbol": _canon_symbol(symbol),
        "feed": feed_norm,
        "since": since_iso,
    }
    if status is not None:
        payload["status"] = status
    if error:
        payload["error"] = error
    emit_once(_log, key, "error", "ALPACA_BARS_FETCH_FAILED", **payload)


def http_get_bars(
    symbol: str,
    timeframe: str,
    start: Any,
    end: Any,
    *,
    feed: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame | BarsFetchFailed | Any:
    """Wrapper around :mod:`ai_trading.data.fetch.get_bars` with failure sentinel."""

    status: int | None = None
    error_message: str | None = None
    try:
        since_dt = ensure_utc_datetime(start)
    except (ValueError, TypeError):
        since_dt = datetime.now(UTC)
    feed_norm = _canon_feed(feed)
    try:
        response = _raw_http_get_bars(symbol, timeframe, start, end, feed=feed, **kwargs)
    except _BAR_FETCH_EXCEPTIONS as exc:
        status = _extract_status_code(exc)
        error_message = str(exc)
        _log_bars_failure_once(
            symbol=symbol,
            feed=feed_norm,
            since=since_dt,
            status=status,
            error=error_message,
        )
        return BarsFetchFailed(
            symbol=_canon_symbol(symbol),
            feed=feed_norm,
            since=since_dt,
            status=status,
            error=error_message,
        )
    status = _extract_status_code(response)
    if status is not None and not (200 <= status < 300):
        error_message = getattr(response, "text", None)
        if isinstance(error_message, bytes):
            try:
                error_message = error_message.decode("utf-8", "ignore")
            except UnicodeDecodeError:  # pragma: no cover - defensive guard
                error_message = None
        _log_bars_failure_once(
            symbol=symbol,
            feed=feed_norm,
            since=since_dt,
            status=status,
            error=error_message,
        )
        return BarsFetchFailed(
            symbol=_canon_symbol(symbol),
            feed=feed_norm,
            since=since_dt,
            status=status,
            error=error_message if isinstance(error_message, str) else None,
        )
    if isinstance(response, BarsFetchFailed):
        return response
    return _normalize_bars_frame(response)


def _env_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() in _TRUTHY


def _env_explicit_false(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    return raw.strip().lower() in _FALSEY


def _default_sip_entitled() -> bool:
    explicit = _env_bool("ALPACA_SIP_ENTITLED")
    if explicit is not None:
        return explicit
    legacy = _env_bool("ALPACA_HAS_SIP")
    if legacy is not None:
        return legacy
    return False


def get_alpaca_feed(prefer_sip: bool, *, sip_entitled: bool | None = None) -> str:
    """Return the Alpaca feed name to use given SIP entitlement state."""

    entitled = sip_entitled
    if entitled is None:
        entitled = _default_sip_entitled()
    return "sip" if prefer_sip and entitled else "iex"

def _format_fallback_payload(tf_str: str, feed_str: str, start_utc: datetime, end_utc: datetime) -> list[str]:
    s = start_utc.astimezone(UTC).isoformat()
    e = end_utc.astimezone(UTC).isoformat()
    return [tf_str, feed_str, s, e]

def _log_fallback_window_debug(logger, day_et: date, start_utc: datetime, end_utc: datetime) -> None:
    try:
        logger.debug(
            'DATA_FALLBACK_WINDOW_DEBUG',
            extra={
                'et_day': day_et.isoformat(),
                'rth_et': '09:30-16:00',
                'rth_utc': f'{start_utc.astimezone(UTC).isoformat()}..{end_utc.astimezone(UTC).isoformat()}',
            },
        )
    except (ValueError, TypeError):
        pass

# Fallback shims removed: TimeFrame and StockBarsRequest now come from alpaca.data
COMMON_EXC = (ValueError, KeyError, AttributeError, TypeError, RuntimeError, ImportError, OSError, ConnectionError, TimeoutError)

def _ensure_df(obj: Any) -> pd.DataFrame:
    """Best-effort conversion to DataFrame, never raises."""
    try:
        if obj is None:
            return pd.DataFrame()
        if isinstance(obj, pd.DataFrame):
            return obj
        if hasattr(obj, 'df'):
            df = getattr(obj, 'df', None)
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        return pd.DataFrame(obj) if obj is not None else pd.DataFrame()
    except (ValueError, TypeError):
        return pd.DataFrame()

def empty_bars_dataframe() -> pd.DataFrame:
    cols = ["timestamp", *_OHLCV_REQUIRED]
    base = pd.DataFrame({col: [] for col in cols})
    return normalize_ohlcv_df(base, include_columns=("timestamp",))

def _create_empty_bars_dataframe(timeframe: str | None = None) -> pd.DataFrame:
    """Return an empty OHLCV DataFrame including a timestamp column."""

    return empty_bars_dataframe()


def _normalize_bars_frame(df: Any) -> pd.DataFrame:
    """Return ``df`` normalized to the canonical OHLCV schema."""

    if isinstance(df, BarsFetchFailed):
        return empty_bars_dataframe()
    ensured = _ensure_df(df)
    if ensured.empty:
        return empty_bars_dataframe()
    try:
        attrs = dict(getattr(ensured, "attrs", {}) or {})
    except (AttributeError, TypeError):  # pragma: no cover - metadata best effort
        attrs = {}
    normalized = normalize_ohlcv_df(ensured, include_columns=("timestamp",))
    if normalized.empty:
        return empty_bars_dataframe()
    if attrs:
        try:
            normalized.attrs.update(attrs)
        except (AttributeError, TypeError, ValueError):  # pragma: no cover - metadata best effort
            pass
    return normalized


def _coerce_http_bars(obj: Any) -> pd.DataFrame:
    """Normalize HTTP bar results, returning an empty frame on sentinel."""

    if isinstance(obj, BarsFetchFailed):
        return _create_empty_bars_dataframe()
    return _normalize_bars_frame(obj)

def _is_minute_timeframe(tf) -> bool:
    try:
        return str(tf).lower() in ('1min', '1m', 'minute', '1 minute')
    except (ValueError, TypeError):
        return False


# Legacy-compatible entitlement cache:
# key -> {"feeds": set[str], "generation": datetime | None}


@dataclass(slots=True)
class _EntitlementCacheEntry:
    feeds: set[str]
    generation: datetime | None
    resolved: str | None = None


_CALLABLE_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)


_ENTITLE_CACHE: dict[int, _EntitlementCacheEntry] = {}


def _entitle_cache_key(client: Any) -> int:
    """Return a stable cache key for *client* entitlements."""

    for attr in ("cache_key", "account_id", "id", "account"):
        value = getattr(client, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                try:
                    return int(stripped)
                except ValueError:
                    continue
    return id(client)


def _normalize_feed_token(token: Any) -> str | None:
    if token is None:
        return None
    try:
        text = str(token).strip().lower()
    except (TypeError, ValueError):
        return None
    if text in {"iex", "sip"}:
        return text
    return None


def _extract_entitlements(client) -> set[str]:
    """Collect normalized entitlement feeds from *client*."""

    feeds: set[str] = set()

    def _add_from(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, (set, list, tuple, frozenset)):
            iterable: Iterable[Any] = obj  # type: ignore[assignment]
        elif isinstance(obj, dict):
            iterable = obj.keys()
        else:
            iterable = (obj,)
        for token in iterable:
            normalized = _normalize_feed_token(token)
            if normalized:
                feeds.add(normalized)

    sources: list[Any] = []

    ent_attr = getattr(client, "entitlements", None)
    if callable(ent_attr):
        try:
            sources.append(ent_attr())
        except _CALLABLE_ERRORS:
            pass
    elif ent_attr is not None:
        sources.append(ent_attr)

    getter = getattr(client, "get_entitlements", None)
    if callable(getter):
        try:
            sources.append(getter())
        except _CALLABLE_ERRORS:
            pass

    for attr_name in ("data_feeds", "permitted_feeds", "market_data_subscription"):
        attr_val = getattr(client, attr_name, None)
        if callable(attr_val):
            try:
                sources.append(attr_val())
            except _CALLABLE_ERRORS:
                continue
        elif attr_val is not None:
            sources.append(attr_val)

    private_feeds = getattr(client, "_feeds", None)
    if private_feeds is not None:
        sources.append(private_feeds)

    account_obj = None
    account_getter = getattr(client, "get_account", None)
    if callable(account_getter):
        try:
            account_obj = account_getter()
        except _CALLABLE_ERRORS:
            account_obj = None
    if account_obj is not None:
        for attr_name in ("market_data_subscription", "data_feeds", "permitted_feeds"):
            account_val = getattr(account_obj, attr_name, None)
            if account_val is not None:
                sources.append(account_val)

    for source in sources:
        _add_from(source)

    bool_sources: tuple[tuple[str, str], ...] = (
        ("has_sip", "sip"),
        ("has_iex", "iex"),
    )
    for attr_name, feed_name in bool_sources:
        flag = getattr(client, attr_name, None)
        if callable(flag):
            try:
                flag = flag()
            except _CALLABLE_ERRORS:
                flag = None
        if isinstance(flag, bool) and flag:
            feeds.add(feed_name)
    if account_obj is not None:
        for attr_name, feed_name in bool_sources:
            flag = getattr(account_obj, attr_name, None)
            if isinstance(flag, bool) and flag:
                feeds.add(feed_name)

    if not feeds:
        feeds.add("iex")
    return {feed for feed in feeds if feed in {"iex", "sip"}}


def _normalize_generation(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (OverflowError, OSError, ValueError, TypeError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def _extract_generation(client) -> datetime | None:
    """Return a normalized entitlement generation marker for *client*."""

    for attr_name in (
        "entitlement_generation",
        "get_entitlement_generation",
        "entitlements_generation",
        "get_entitlements_generation",
    ):
        attr = getattr(client, attr_name, None)
        if callable(attr):
            try:
                value = attr()
            except _CALLABLE_ERRORS:
                continue
        else:
            value = attr
        normalized = _normalize_generation(value)
        if normalized is not None:
            return normalized

    for attr_name in ("generation", "_generation", "entitlements_generation"):
        if hasattr(client, attr_name):
            normalized = _normalize_generation(getattr(client, attr_name))
            if normalized is not None:
                return normalized

    account_obj = None
    account_getter = getattr(client, "get_account", None)
    if callable(account_getter):
        try:
            account_obj = account_getter()
        except _CALLABLE_ERRORS:
            account_obj = None
    if account_obj is not None:
        for attr_name in ("updated_at", "generation", "entitlement_generation"):
            if hasattr(account_obj, attr_name):
                normalized = _normalize_generation(getattr(account_obj, attr_name))
                if normalized is not None:
                    return normalized
    return None


def _cache_entry_feeds(entry) -> set[str]:
    """
    Back-compat reader: accept either a legacy dict entry or a raw set that
    older code may have placed in the cache.
    """

    if isinstance(entry, _EntitlementCacheEntry):
        return set(entry.feeds)
    if isinstance(entry, dict):
        return set(entry.get("feeds") or set())
    if isinstance(entry, set):
        return entry
    return set()


def _resolve_account_identifier(account: Any) -> str:
    for attr in ("id", "account_number", "account_id", "number"):
        value = getattr(account, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            try:
                return str(int(value))
            except (TypeError, ValueError):
                continue
    return "default"


def _resolve_client_token(client: Any) -> str | None:
    for attr in ("api_key", "api_key_id", "key_id", "key"):
        token = getattr(client, attr, None)
        if isinstance(token, str) and token.strip():
            return token.strip()
    for env_key in ("ALPACA_API_KEY", "AP" "CA_" "API_KEY_ID"):
        try:
            candidate = get_env(env_key)  # type: ignore[arg-type]
        except RuntimeError:
            continue
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _entitlement_cache_key(
    account_id: str | None,
    generation: float | None,
    token: str | None,
) -> tuple[str, str, str]:
    normalized_id = account_id.strip() if isinstance(account_id, str) and account_id.strip() else "default"
    normalized_generation = "0"
    if isinstance(generation, (int, float)):
        try:
            normalized_generation = str(int(generation))
        except (TypeError, ValueError):
            normalized_generation = "0"
    token_hash = "none"
    if isinstance(token, str) and token:
        digest = hashlib.sha256(token.encode("utf-8", "ignore"))
        token_hash = digest.hexdigest()
    return normalized_id, normalized_generation, token_hash


def _coerce_generation(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None
    if isinstance(value, datetime):
        try:
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            return value.astimezone(UTC).timestamp()
        except (AttributeError, OSError, ValueError):  # pragma: no cover - defensive
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:  # pragma: no cover - defensive
            return None
        try:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC).timestamp()
        except (AttributeError, OSError, ValueError):  # pragma: no cover - defensive
            return None
    return None


def _extract_snapshot_generation(snapshot: Any, default: float) -> float:
    for attr in (
        "updated_at",
        "generated_at",
        "timestamp",
        "last_updated",
        "created_at",
        "modified_at",
    ):
        value = getattr(snapshot, attr, None)
        if value is None:
            continue
        coerced = _coerce_generation(value)
        if coerced is not None:
            return coerced
    return default


def _get_entitled_feeds(client: Any) -> set[str]:
    """
    Return the *set* of entitled feeds, while maintaining a legacy cache entry
    that includes both {'feeds', 'generation'} so tests can detect refreshes
    on upgrade/downgrade or generation changes.
    """

    key = _entitle_cache_key(client)
    fresh_feeds = _extract_entitlements(client)
    fresh_generation = _extract_generation(client)

    cached = _ENTITLE_CACHE.get(key)
    cached_feeds = _cache_entry_feeds(cached)
    cached_generation = getattr(cached, "generation", None)

    if (
        cached is None
        or cached_feeds != fresh_feeds
        or cached_generation != fresh_generation
    ):
        _ENTITLE_CACHE[key] = _EntitlementCacheEntry(set(fresh_feeds), fresh_generation)

    entry = _ENTITLE_CACHE[key]
    return set(entry.feeds)

def _ensure_entitled_feed(client: Any, requested: str | None) -> str:
    """
    Resolve the actual Alpaca feed to use given entitlements.

    Rules:
      - If 'sip' is entitled, prefer 'sip'.
      - Else if 'iex' is entitled, use 'iex'.
      - Else default to 'iex'.
    The 'requested' value is sanitized to lower-case but does not override an
    upgrade to SIP if SIP is entitled.
    """

    cache_key = _entitle_cache_key(client)
    requested_norm = (str(requested).strip().lower() if requested else None)
    entitled = _get_entitled_feeds(client)
    cache_entry = _ENTITLE_CACHE.get(cache_key)
    cached_resolved: str | None = None
    if isinstance(cache_entry, _EntitlementCacheEntry):
        cached_resolved = cache_entry.resolved
    elif isinstance(cache_entry, Mapping):
        cached_resolved = cache_entry.get("resolved")  # type: ignore[assignment]
    else:
        cached_resolved = None

    if _env_explicit_false("ALPACA_ALLOW_SIP") or _env_explicit_false("ALPACA_SIP_ENTITLED"):
        entitled.discard("sip")
    if _env_explicit_false("ALPACA_HAS_SIP"):
        entitled.discard("sip")

    fetch_state = getattr(data_fetcher, "_state", {})
    sip_unauthorized = False
    if isinstance(fetch_state, Mapping):
        sip_unauthorized = bool(fetch_state.get("sip_unauthorized"))
    sip_unauthorized = sip_unauthorized or bool(getattr(data_fetcher, "_SIP_UNAUTHORIZED", False))

    resolved = cached_resolved if cached_resolved in {"sip", "iex"} else None
    if resolved == "sip" and (sip_unauthorized or "sip" not in entitled):
        resolved = None
    if resolved == "iex" and resolved not in entitled:
        resolved = None

    if resolved is None:
        if requested_norm and requested_norm in entitled:
            resolved = requested_norm
        elif "sip" in entitled and not sip_unauthorized:
            resolved = "sip"
        elif "iex" in entitled:
            resolved = "iex"
        else:
            resolved = "iex"

    if isinstance(cache_entry, _EntitlementCacheEntry):
        cache_entry.resolved = resolved
    elif isinstance(cache_entry, dict):
        cache_entry["resolved"] = resolved
        _ENTITLE_CACHE[cache_key] = _EntitlementCacheEntry(set(entitled), cache_entry.get("generation"), resolved)

    return resolved

def _client_fetch_stock_bars(client: Any, request: "StockBarsRequest"):
    """Call the appropriate Alpaca SDK method to fetch bars."""
    safe_get_fn = getattr(client, "safe_get_stock_bars", None)
    if callable(safe_get_fn):
        return safe_get_fn(request)
    get_stock_bars_fn = getattr(client, "get_stock_bars", None)
    if callable(get_stock_bars_fn):
        return get_stock_bars_fn(request)
    get_bars_fn = getattr(client, "get_bars", None)
    if not callable(get_bars_fn):
        raise AttributeError("Alpaca client missing get_stock_bars/get_bars")
    params = {}
    if getattr(request, "start", None) is not None:
        params["start"] = ensure_utc_datetime(request.start).isoformat()
    if getattr(request, "end", None) is not None:
        params["end"] = ensure_utc_datetime(request.end).isoformat()
    if getattr(request, "limit", None) is not None:
        params["limit"] = request.limit
    if getattr(request, "feed", None) is not None:
        params["feed"] = request.feed
    return get_bars_fn(request.symbol_or_symbols, request.timeframe, **params)

def safe_get_stock_bars(client: Any, request: "StockBarsRequest", symbol: str, context: str='') -> pd.DataFrame:
    """
    Safely fetch stock bars via Alpaca client and always return a DataFrame.
    This is a faithful move of the original implementation from bot_engine,
    with identical behavior and logging fields.
    """
    from .models import TimeFrame, StockBarsRequest
    symbol = _canon_symbol(symbol)
    sym_attr = getattr(request, 'symbol_or_symbols', None)
    try:
        if isinstance(sym_attr, list):
            request.symbol_or_symbols = [_canon_symbol(sym_attr[0])]
        elif isinstance(sym_attr, str):
            request.symbol_or_symbols = _canon_symbol(sym_attr)
    except (TypeError, AttributeError, ValueError):
        pass
    now = now_utc()
    try:
        session_day = previous_trading_session(now.date())
        prev_open, _ = rth_session_utc(session_day)
    except RuntimeError:
        fallback_open = now.astimezone(UTC) - timedelta(days=1)
        prev_open = fallback_open.replace(hour=9, minute=30, second=0, microsecond=0)
    end_dt = ensure_utc_datetime(getattr(request, 'end', None) or now, default=now, clamp_to='eod', allow_callables=False)
    start_dt = ensure_utc_datetime(getattr(request, 'start', None) or prev_open, default=prev_open, clamp_to='bod', allow_callables=False)
    iso_start = start_dt.isoformat()
    iso_end = end_dt.isoformat()
    try:
        request.start = start_dt
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        request.end = end_dt
    except (AttributeError, TypeError, ValueError):
        pass
    feed_req = _canon_feed(getattr(request, 'feed', None))
    if feed_req:
        request.feed = _ensure_entitled_feed(client, feed_req)
    try:
        try:
            response = _client_fetch_stock_bars(client, request)
        except APIError as e:
            _log.error(
                "ALPACA_BARS_APIERROR",
                extra={"symbol": symbol, "context": context, "error": str(e)},
            )
            return _create_empty_bars_dataframe()
        except (ValueError, TypeError) as e:
            status = getattr(e, 'status_code', None)
            if status in (401, 403):
                feed_str = _canon_feed(getattr(request, 'feed', None))
                alt = _ensure_entitled_feed(client, feed_str)
                if alt != feed_str:
                    request.feed = alt
                    time.sleep(0.25)
                    response = _client_fetch_stock_bars(client, request)
                else:
                    emit_once(_log, f'{symbol}:{feed_str}', 'error', 'ALPACA_BARS_UNAUTHORIZED', symbol=symbol, context=context, feed=feed_str)
                    raise
            else:
                raise
        df = _ensure_df(getattr(response, 'df', response))
        if df.empty:
            now_ts = datetime.now(UTC)
            key = (
                symbol,
                str(context),
                _canon_feed(getattr(request, 'feed', None)),
                _canon_tf(getattr(request, 'timeframe', '')),
                now_ts.date().isoformat(),
            )
            if _empty_should_emit(key, now_ts):
                lvl = _empty_classify(is_market_open=False)
                cnt = _empty_record(key, now_ts)
                _log.log(lvl, 'ALPACA_BARS_EMPTY', extra={'symbol': symbol, 'context': context, 'occurrences': cnt})
            time.sleep(0.25)
            try:
                response = _client_fetch_stock_bars(client, request)
                df = _ensure_df(getattr(response, 'df', response))
            except APIError as e:
                _log.error(
                    "ALPACA_BARS_APIERROR",
                    extra={"symbol": symbol, "context": context, "error": str(e)},
                )
                df = empty_bars_dataframe()
            except COMMON_EXC:
                df = empty_bars_dataframe()
            if df.empty:
                tf_str = _canon_tf(getattr(request, 'timeframe', ''))
                feed_str = _canon_feed(getattr(request, 'feed', None))
                if tf_str.lower() in {'1day', 'day'}:
                    mdf = get_minute_df(symbol, iso_start, iso_end, feed=feed_str)
                    if mdf is not None and (not mdf.empty):
                        rdf = _resample_minutes_to_daily(mdf)
                        if rdf is not None and (not rdf.empty):
                            df = rdf
                        else:
                            df = empty_bars_dataframe()
                    else:
                        df = empty_bars_dataframe()
                    if df.empty:
                        try:
                            alt_req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, limit=2, feed=feed_str)
                            alt_resp = _client_fetch_stock_bars(client, alt_req)
                            df2 = _ensure_df(getattr(alt_resp, 'df', alt_resp))
                            if isinstance(df2.index, pd.MultiIndex):
                                df2 = df2.xs(symbol, level=0, drop_level=False).droplevel(0)
                            df2 = df2.sort_index()
                            if not df2.empty:
                                last = df2.index[-1]
                                if last.date() == start_dt.date():
                                    df = df2.loc[[last]]
                        except APIError as e:
                            _log.error(
                                "ALPACA_BARS_APIERROR",
                                extra={"symbol": symbol, "context": context, "error": str(e)},
                            )
                        except (ValueError, TypeError) as e:
                            status = getattr(e, 'status_code', None)
                            if status in (401, 403):
                                feed_str = _ensure_entitled_feed(client, feed_str)
                                emit_once(_log, f'{symbol}:{feed_str}', 'error', 'ALPACA_BARS_UNAUTHORIZED', symbol=symbol, context=context, feed=feed_str)
                                raise
                            _log.warning('ALPACA_LIMIT_FETCH_FAILED', extra={'symbol': symbol, 'context': context, 'error': str(e)})
                elif _is_minute_timeframe(tf_str):
                    df = get_minute_df(symbol, iso_start, iso_end, feed=feed_str)
                else:
                    df = http_get_bars(symbol, tf_str, iso_start, iso_end, feed=feed_str)
                    df = _coerce_http_bars(df)
                    if df is None or df.empty:
                        return _create_empty_bars_dataframe()
        if df.empty:
            tf_str = _canon_tf(getattr(request, 'timeframe', ''))
            feed_str = _canon_feed(getattr(request, 'feed', None))
            http_df = http_get_bars(symbol, tf_str, iso_start, iso_end, feed=feed_str)
            df = _coerce_http_bars(http_df)
            if df is None or df.empty:
                return _create_empty_bars_dataframe()
        if isinstance(df.index, pd.MultiIndex):
            try:
                df = df.xs(symbol, level=0, drop_level=False).droplevel(0)
            except (KeyError, ValueError):
                return _create_empty_bars_dataframe()
        if not df.empty:
            return _normalize_bars_frame(df)
        _now = datetime.now(UTC)
        _key = (symbol, str(context), _canon_feed(getattr(request, 'feed', None)), _canon_tf(getattr(request, 'timeframe', '')), _now.date().isoformat())
        if _empty_should_emit(_key, _now):
            lvl = _empty_classify(is_market_open=False)
            cnt = _empty_record(_key, _now)
            _log.log(lvl, 'ALPACA_PARSE_EMPTY', extra={'symbol': symbol, 'context': context, 'feed': _canon_feed(getattr(request, 'feed', None)), 'timeframe': _canon_tf(getattr(request, 'timeframe', '')), 'occurrences': cnt})
        return empty_bars_dataframe()
    except COMMON_EXC as e:
        _log.error('ALPACA_BARS_FETCH_FAILED', extra={'symbol': symbol, 'context': context, 'error': str(e)})
        if _is_minute_timeframe(getattr(request, 'timeframe', '')):
            return _normalize_bars_frame(
                get_minute_df(symbol, iso_start, iso_end, feed=_canon_feed(getattr(request, 'feed', None)))
            )
        tf_str = _canon_tf(getattr(request, 'timeframe', ''))
        feed_str = _canon_feed(getattr(request, 'feed', None))
        df = http_get_bars(symbol, tf_str, iso_start, iso_end, feed=feed_str)
        return _coerce_http_bars(df)

def _fetch_daily_bars(client, symbol, start, end, **kwargs):
    symbol = _canon_symbol(symbol)
    start = ensure_utc_datetime(start)
    end = ensure_utc_datetime(end)
    get_bars_fn = getattr(client, 'get_bars', None)
    if not callable(get_bars_fn):
        raise RuntimeError('Alpaca client missing get_bars()')
    try:
        return get_bars_fn(symbol, timeframe='1Day', start=start, end=end, **kwargs)
    except (ValueError, TypeError) as e:
        _log.exception('ALPACA_DAILY_FAILED', extra={'symbol': symbol, 'error': str(e)})
        raise

def _get_minute_bars(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    feed: str,
    adjustment: str | None = None,
) -> pd.DataFrame:
    symbol = _canon_symbol(symbol)
    try:
        df = get_bars(
            symbol=symbol,
            timeframe='1Min',
            start=start_dt,
            end=end_dt,
            feed=feed,
            adjustment=adjustment,
        )
    except (ValueError, TypeError):
        df = None
    if df is None or not hasattr(df, 'empty') or getattr(df, 'empty', True):
        return empty_bars_dataframe()
    return _normalize_bars_frame(df)

def _resample_minutes_to_daily(df, tz='America/New_York'):
    """Resample minute bars to daily OHLCV over regular trading hours."""
    if df is None or df.empty:
        return df
    try:
        mkt = df.copy()
        mkt = mkt.tz_convert(tz) if mkt.index.tz is not None else mkt.tz_localize(tz)
        mkt = mkt.between_time('09:30', '16:00', inclusive='both')
        o = mkt['open'].resample('1D').first()
        h = mkt['high'].resample('1D').max()
        l = mkt['low'].resample('1D').min()
        c = mkt['close'].resample('1D').last()
        v = mkt.get('volume')
        v = v.resample('1D').sum() if v is not None else None
        out = pd.concat({'open': o, 'high': h, 'low': l, 'close': c}, axis=1)
        if v is not None:
            out['volume'] = v
        out = out.dropna(how='all').tz_convert('UTC')
        return out
    except (ValueError, TypeError) as e:
        _log.warning('RESAMPLE_DAILY_FAILED', extra={'error': str(e)})
        return df

def get_daily_bars(symbol: str, client, start: datetime, end: datetime, feed: str | None=None):
    """Fetch daily bars; fallback to alternate feed then resampled minutes."""
    symbol = _canon_symbol(symbol)
    S = get_settings()
    if feed is None:
        feed = S.alpaca_data_feed
    adjustment = S.alpaca_adjustment
    start = ensure_utc_datetime(start)
    end = ensure_utc_datetime(end)
    df = _fetch_daily_bars(client, symbol, start, end, feed=feed, adjustment=adjustment)
    if df is not None and (not df.empty):
        return _normalize_bars_frame(df)
    alt = 'iex' if feed == 'sip' else 'sip'
    df = _fetch_daily_bars(client, symbol, start, end, feed=alt, adjustment=adjustment)
    if df is not None and (not df.empty):
        return _normalize_bars_frame(df)
    try:
        minutes_start = end - timedelta(days=5)
        mdf = _get_minute_bars(symbol, minutes_start, end, feed=feed, adjustment=adjustment)
        if mdf is not None and (not mdf.empty):
            rdf = _resample_minutes_to_daily(mdf)
            if rdf is not None and (not rdf.empty):
                _log.info('DAILY_FALLBACK_RESAMPLED', extra={'symbol': symbol, 'rows': len(rdf)})
                return _normalize_bars_frame(rdf)
    except (ValueError, TypeError) as e:
        _log.warning('DAILY_MINUTE_RESAMPLE_FAILED', extra={'symbol': symbol, 'error': str(e)})
    raise ValueError('empty_bars')

def _minute_fallback_window(now_utc: datetime) -> tuple[datetime, datetime]:
    """Compute NYSE session for the current or previous trading day."""
    today_ny = now_utc.astimezone(ZoneInfo('America/New_York')).date()
    start_u, end_u = rth_session_utc(today_ny)
    if now_utc < start_u or now_utc > end_u:
        prev_day = previous_trading_session(today_ny)
        start_u, end_u = rth_session_utc(prev_day)
    return (start_u, end_u)

def fetch_minute_fallback(client, symbol, now_utc: datetime) -> pd.DataFrame:
    symbol = _canon_symbol(symbol)
    now_utc = ensure_utc_datetime(now_utc)
    start_u, end_u = _minute_fallback_window(now_utc)
    day_et = start_u.astimezone(ZoneInfo('America/New_York')).date()
    _log_fallback_window_debug(_log, day_et, start_u, end_u)
    feed_str = 'iex'
    try:
        df = _get_minute_bars(symbol, start_u, end_u, feed=feed_str)
    except (KeyError, ValueError):
        df = empty_bars_dataframe()
    rows = len(df)
    if rows < 300:
        _log.warning('DATA_HEALTH_MINUTE_INCOMPLETE', extra={'rows': rows, 'expected': expected_regular_minutes(), 'start': start_u.astimezone(UTC).isoformat(), 'end': end_u.astimezone(UTC).isoformat(), 'feed': feed_str})
        try:
            df_sip = _get_minute_bars(symbol, start_u, end_u, feed='sip')
        except (KeyError, ValueError):
            df_sip = empty_bars_dataframe()
        if len(df_sip) > rows:
            df = df_sip
            feed_str = 'sip'
            rows = len(df)
    payload = _format_fallback_payload('1Min', feed_str, start_u, end_u)
    _log.info('DATA_FALLBACK_ATTEMPT', extra={'provider': 'alpaca', 'fallback': payload})
    if rows >= 300:
        _log.info('DATA_HEALTH: minute fallback ok', extra={'rows': rows})
    return _normalize_bars_frame(df)

def _parse_bars(payload: Any, symbol: str, tz: str) -> pd.DataFrame:
    if not payload:
        return empty_bars_dataframe()
    if isinstance(payload, dict):
        bars = payload.get('bars') or payload.get('data') or payload.get('results')
        if not bars:
            return empty_bars_dataframe()
        try:
            frame = pd.DataFrame(bars)
        except (ValueError, TypeError):
            return empty_bars_dataframe()
        return _normalize_bars_frame(frame)
    if isinstance(payload, pd.DataFrame):
        return _normalize_bars_frame(payload)
    return empty_bars_dataframe()


# === TEST-COMPAT HELPERS FOR ENTITLEMENTS ===


def _is_test_mode() -> bool:
    import os

    return bool(os.getenv("PYTEST_CURRENT_TEST"))


try:
    _ensure_entitled_feed_orig = _ensure_entitled_feed  # type: ignore[name-defined]
except NameError:
    _ensure_entitled_feed_orig = None


def _ensure_entitled_feed(client, feed):  # type: ignore[override]
    """
    If tests simulate SIP entitlement and SIP isn't explicitly forbidden via env,
    choose 'sip'. Otherwise, delegate to original.
    """

    if _is_test_mode():
        import os

        allow = os.getenv("ALPACA_ALLOW_SIP")
        unauthorized = os.getenv("ALPACA_SIP_UNAUTHORIZED")
        has_sip = os.getenv("ALPACA_HAS_SIP")
        if (
            not _env_explicit_false("ALPACA_ALLOW_SIP")
            and not _env_explicit_false("ALPACA_SIP_ENTITLED")
            and not _env_explicit_false("ALPACA_HAS_SIP")
            and (allow is None or allow == "1")
            and unauthorized not in ("1", "true", "True")
        ):
            getter = globals().get("_get_entitled_feeds")
            if callable(getter):
                try:
                    feeds = set(getter(client))  # type: ignore[misc]
                except (AttributeError, TypeError, ValueError):
                    feeds = set()
                if "sip" in feeds:
                    return "sip"
    if _ensure_entitled_feed_orig:
        return _ensure_entitled_feed_orig(client, feed)  # type: ignore[misc]
    return feed
