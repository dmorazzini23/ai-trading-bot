from __future__ import annotations
import datetime as dt
from datetime import timezone
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from ai_trading.net.http import HTTPSession, get_http_session
from ai_trading.exc import RequestException
from ai_trading.utils.http import clamp_request_timeout
import importlib
from ai_trading.logging import get_logger
from ai_trading.config.management import is_shadow_mode
from ai_trading.logging.normalize import canon_symbol as _canon_symbol
from ai_trading.metrics import get_counter, get_histogram
from time import monotonic as _mono

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

_log = get_logger(__name__)
RETRY_HTTP_CODES = {429, 500, 502, 503, 504}
RETRYABLE_HTTP_STATUSES = tuple(RETRY_HTTP_CODES)
_UTC = timezone.utc  # AI-AGENT-REF: prefer stdlib UTC
_HTTP: HTTPSession = get_http_session()

# Lightweight Prometheus metrics (no-op when client unavailable)
_alpaca_calls_total = get_counter("alpaca_calls_total", "Total Alpaca calls")
_alpaca_errors_total = get_counter("alpaca_errors_total", "Total Alpaca call errors")
_alpaca_call_latency = get_histogram(
    "alpaca_call_latency_seconds",
    "Latency of Alpaca calls",
)


from zoneinfo import ZoneInfo


def eastern_tz() -> ZoneInfo:
    """Return America/New_York tzinfo using stdlib zoneinfo (Py3.12)."""
    return ZoneInfo("America/New_York")  # AI-AGENT-REF: rely solely on stdlib


EASTERN_TZ = eastern_tz()

def _is_intraday_unit(unit_tok: str) -> bool:
    """Return True for minute or hour-based timeframes."""
    return unit_tok in ('Min', 'Hour')

def _unit_from_norm(tf_norm: str) -> tuple[str, str]:
    """Return (unit_name, suffix) from a normalized timeframe string."""
    mapping = {
        "Min": "Minute",
        "Minute": "Minute",
        "Hour": "Hour",
        "Day": "Day",
        "Week": "Week",
        "Month": "Month",
    }
    for suffix, name in mapping.items():
        if tf_norm.endswith(suffix):
            return name, suffix
    return "Day", "Day"
def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


# Detect SDK availability without importing heavy modules at import time
ALPACA_AVAILABLE = (
    _module_exists("alpaca")
    and _module_exists("alpaca.trading.client")
    and (
        _module_exists("alpaca.data.historical.stock")
        or _module_exists("alpaca.data.historical")
        or _module_exists("alpaca.data")
    )
)
HAS_PANDAS: bool = _module_exists("pandas")  # AI-AGENT-REF: expose pandas availability


def initialize() -> None:
    """Ensure required Alpaca SDK modules are importable.

    Raises:
        RuntimeError: If the Alpaca SDK cannot be imported, providing a
            helpful message for installation.
    """
    try:
        importlib.import_module("alpaca.trading.client")
        try:
            importlib.import_module("alpaca.data.historical.stock")
        except ModuleNotFoundError:
            try:
                importlib.import_module("alpaca.data.historical")
            except ModuleNotFoundError:
                importlib.import_module("alpaca.data")
    except Exception as exc:  # pragma: no cover - exercised in tests
        raise RuntimeError("alpaca-py SDK is required") from exc

if not ALPACA_AVAILABLE:  # pragma: no cover - exercised in tests
    from dataclasses import dataclass
    from enum import Enum
    from typing import Any

    class TimeFrameUnit(Enum):
        Minute = "Min"
        Hour = "Hour"
        Day = "Day"

    @dataclass(frozen=True)
    class TimeFrame:
        amount: int = 1
        unit: TimeFrameUnit = TimeFrameUnit.Day

        def __str__(self) -> str:
            return f"{self.amount}{self.unit.value}"

    # Pre-defined shorthand attributes mirroring alpaca-py
    TimeFrame.Minute = TimeFrame(1, TimeFrameUnit.Minute)  # type: ignore[attr-defined]
    TimeFrame.Hour = TimeFrame(1, TimeFrameUnit.Hour)  # type: ignore[attr-defined]
    TimeFrame.Day = TimeFrame()  # type: ignore[attr-defined]

    @dataclass
    class StockBarsRequest:
        symbol_or_symbols: Any
        timeframe: Any
        start: Any | None = None
        end: Any | None = None
        limit: int | None = None
        adjustment: str | None = None
        feed: str | None = None
        sort: str | None = None
        asof: str | None = None
        currency: str | None = None

        def __init__(
            self,
            symbol_or_symbols: Any,
            timeframe: Any,
            *,
            start: Any | None = None,
            end: Any | None = None,
            limit: int | None = None,
            adjustment: str | None = None,
            feed: str | None = None,
            sort: str | None = None,
            asof: str | None = None,
            currency: str | None = None,
            **extra: Any,
        ) -> None:
            self.symbol_or_symbols = symbol_or_symbols
            self.timeframe = timeframe
            self.start = start
            self.end = end
            self.limit = limit
            self.adjustment = adjustment
            self.feed = feed
            self.sort = sort
            self.asof = asof
            self.currency = currency
            for k, v in extra.items():
                setattr(self, k, v)

def _make_client_order_id(prefix: str='ai') -> str:
    return f'{prefix}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}'
generate_client_order_id = _make_client_order_id


def get_trading_client_cls():
    """Return the Alpaca TradingClient class via lazy import."""
    from alpaca.trading.client import TradingClient  # type: ignore

    return TradingClient


def get_data_client_cls():
    """Return the Alpaca StockHistoricalDataClient class via lazy import."""
    from alpaca.data.historical.stock import StockHistoricalDataClient  # type: ignore

    return StockHistoricalDataClient


def get_api_error_cls():
    """Return the Alpaca APIError class via lazy import."""
    from alpaca.common.exceptions import APIError  # type: ignore

    return APIError


def list_orders_wrapper(api: Any, *args: Any, **kwargs: Any):
    """Adapter for ``get_orders`` methods lacking ``list_orders``.

    Parameters
    ----------
    api:
        Client exposing ``get_orders``.
    *args, **kwargs:
        Forwarded to ``get_orders`` with the ``status`` keyword preserved.
    """
    status = kwargs.pop("status", None)
    if status is None:
        return api.get_orders(*args, **kwargs)  # type: ignore[attr-defined]

    enum_val: Any = status
    try:  # optional enum mapping for alpaca-py
        enums_mod = __import__("alpaca.trading.enums", fromlist=[""])
        enum_cls = getattr(enums_mod, "QueryOrderStatus", None) or getattr(
            enums_mod, "OrderStatus", None
        )
        if enum_cls is not None:
            enum_val = getattr(enum_cls, str(status).upper(), status)
    except Exception:
        pass

    try:  # build filter object when request class available
        requests_mod = __import__(
            "alpaca.trading.requests", fromlist=["GetOrdersRequest"]
        )
        GetOrdersRequest = getattr(requests_mod, "GetOrdersRequest")
        filter_obj = GetOrdersRequest(statuses=[enum_val])
        return api.get_orders(*args, filter=filter_obj, **kwargs)  # type: ignore[attr-defined]
    except Exception:
        kwargs["status"] = enum_val
        return api.get_orders(*args, **kwargs)  # type: ignore[attr-defined]


def _data_classes():
    """Return Alpaca data request classes lazily."""
    from alpaca.data import StockBarsRequest, TimeFrame, TimeFrameUnit  # type: ignore

    return StockBarsRequest, TimeFrame, TimeFrameUnit


def get_stock_bars_request_cls():
    if ALPACA_AVAILABLE:
        cls, _, _ = _data_classes()
        return cls
    return StockBarsRequest


def get_timeframe_cls():
    if ALPACA_AVAILABLE:
        _, cls, unit_cls = _data_classes()

        class _TF(cls):  # type: ignore[misc]
            def __init__(self, amount: int = 1, unit=unit_cls.Day):  # type: ignore[assignment]
                super().__init__(amount, unit)

        return _TF
    return TimeFrame


def get_timeframe_unit_cls():
    if ALPACA_AVAILABLE:
        _, _, cls = _data_classes()
        return cls
    return TimeFrameUnit

def _normalize_timeframe_for_tradeapi(tf_raw):
    """Support string pass-through and alpaca TimeFrame objects."""
    try:
        TimeFrame = get_timeframe_cls()
    except Exception:
        TimeFrame = None
    if isinstance(tf_raw, str):
        s = tf_raw.strip()
        return s if s[:1].isdigit() else f'1{s.capitalize()}'
    if TimeFrame is not None and isinstance(tf_raw, TimeFrame):
        unit = getattr(tf_raw.unit, 'name', str(tf_raw.unit)).title()
        return f'{tf_raw.amount}{unit}'
    return str(tf_raw)

def _to_utc(dtobj: dt.datetime) -> dt.datetime:
    """Ensure a ``datetime`` is timezone-aware and in UTC."""
    if dtobj.tzinfo is None:
        return dtobj.replace(tzinfo=dt.timezone.utc)
    return dtobj.astimezone(dt.timezone.utc)

def _fmt_rfc3339_z(dtobj: dt.datetime) -> str:
    """Format a UTC datetime to RFC3339 ``YYYY-MM-DDTHH:MM:SSZ``."""
    d = _to_utc(dtobj).replace(microsecond=0)
    return d.strftime('%Y-%m-%dT%H:%M:%SZ')

def _format_start_end_for_tradeapi(timeframe: str, start, end):
    """Daily => YYYY-MM-DD; intraday => RFC3339Z in UTC."""
    from ai_trading.utils.datetime import compose_daily_params, compose_intraday_params, ensure_datetime
    sd = ensure_datetime(start) if start is not None else None
    ed = ensure_datetime(end) if end is not None else None
    is_daily = str(timeframe).lower() in {'1day', 'day', 'daily'}
    params = compose_daily_params(sd, ed) if is_daily else compose_intraday_params(sd, ed)
    return (params['start'], params['end'])

def _get_rest(*, bars: bool = False) -> Any:
    """Return a new `alpaca-py` client instance.

    Parameters
    ----------
    bars:
        When ``True`` return a :class:`StockHistoricalDataClient`; otherwise a
        :class:`TradingClient`.
    """

    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    oauth = os.getenv("ALPACA_OAUTH")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    if oauth and (key or secret):
        raise RuntimeError(
            "Provide either ALPACA_API_KEY/ALPACA_SECRET_KEY or ALPACA_OAUTH, not both"
        )

    if bars:
        StockHistoricalDataClient = get_data_client_cls()

        if oauth:
            return StockHistoricalDataClient(
                oauth_token=oauth,
            )
        return StockHistoricalDataClient(
            api_key=key,
            secret_key=secret,
        )

    TradingClient = get_trading_client_cls()

    is_paper = bool(base_url and "paper" in base_url.lower())
    if oauth:
        return TradingClient(
            oauth_token=oauth,
            paper=is_paper,
            url_override=base_url,
        )
    return TradingClient(
        api_key=key,
        secret_key=secret,
        paper=is_paper,
        url_override=base_url,
    )

def _bars_time_window(timeframe: Any) -> tuple[str, str]:
    now = dt.datetime.now(tz=_UTC)
    end = now - dt.timedelta(minutes=1)

    try:  # pragma: no cover - imported lazily
        TimeFrame = get_timeframe_cls()
    except Exception:  # pragma: no cover - optional dependency missing
        TimeFrame = None

    is_daily = False
    if TimeFrame is not None and isinstance(timeframe, TimeFrame):
        unit = getattr(getattr(timeframe, "unit", None), "name", "")
        is_daily = unit == "Day"
    else:
        tf_str = str(timeframe).strip().lower()
        is_daily = tf_str.endswith("day") or tf_str == "day"

    if is_daily:
        days = int(os.getenv("DATA_LOOKBACK_DAYS_DAILY", 10))
    else:
        days = int(os.getenv("DATA_LOOKBACK_DAYS_MINUTE", 5))
    start = end - dt.timedelta(days=days)
    return (_fmt_rfc3339_z(start), _fmt_rfc3339_z(end))


# AI-AGENT-REF: ensure clear error when pandas missing
def _require_pandas(consumer: str = "this function"):
    """Return imported pandas module or raise a helpful ImportError."""
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover - import time failure
        raise ImportError(
            f"Missing required dependency 'pandas' for {consumer}. Install with: pip install pandas"
        ) from e
    return pd


# Optional retry/backoff support using tenacity
try:  # pragma: no cover - optional dependency wrapper
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
    )

    def _with_retry(callable_):
        return retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.25, max=2.0),
            retry=retry_if_exception_type(Exception),
        )(callable_)

except Exception:  # pragma: no cover - absent tenacity
    retry = None  # type: ignore

    def _with_retry(callable_):  # type: ignore
        return callable_


def get_bars_df(
    symbol: str,
    timeframe: str | Any = "1Min",
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    adjustment: str | None = None,
    feed: str | None = None,
) -> "pd.DataFrame":
    """Fetch bars for ``symbol`` and return a normalized DataFrame."""
    symbol = _canon_symbol(symbol)
    APIError = get_api_error_cls()
    StockBarsRequest = get_stock_bars_request_cls()
    TimeFrame = get_timeframe_cls()
    TimeFrameUnit = get_timeframe_unit_cls()

    _pd = _require_pandas("get_bars_df")
    rest = _get_rest(bars=True)
    feed = feed or os.getenv('ALPACA_DATA_FEED', 'iex')
    adjustment = adjustment or os.getenv('ALPACA_ADJUSTMENT', 'all')
    tf_raw = timeframe
    tf_norm = _normalize_timeframe_for_tradeapi(tf_raw)
    unit_name, suffix = _unit_from_norm(tf_norm)
    try:
        amount = int(tf_norm[: len(tf_norm) - len(suffix)])
    except ValueError:
        amount = 1
    try:
        unit_enum = getattr(TimeFrameUnit, unit_name)
    except AttributeError:
        unit_enum = TimeFrameUnit.Day
    tf_obj = tf_raw if isinstance(tf_raw, TimeFrame) else TimeFrame(amount, unit_enum)
    if end is not None:
        from ai_trading.utils.datetime import ensure_datetime
        try:
            end_dt = ensure_datetime(end)
            if end_dt.date() > dt.date.today():
                _log.warning(
                    "END_DATE_AFTER_TODAY",
                    extra={
                        "requested_end": end_dt.date().isoformat(),
                        "today": dt.date.today().isoformat(),
                    },
                )
        except (ValueError, TypeError):
            pass
    if start is None or end is None:
        start, end = _bars_time_window(tf_obj)
    start_s, end_s = _format_start_end_for_tradeapi(tf_norm, start, end)
    try:
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=tf_obj,
            start=start_s,
            end=end_s,
            adjustment=adjustment,
            feed=feed,
        )
        _start_t = _mono()
        _err: Exception | None = None
        try:
            call = rest.get_stock_bars
            if retry is not None:
                call = _with_retry(call)
            df = call(req).df
        except Exception as e:
            _err = e
            raise
        finally:
            try:
                _alpaca_calls_total.inc()
                _alpaca_call_latency.observe(max(0.0, _mono() - _start_t))
                if _err is not None:
                    _alpaca_errors_total.inc()
            except Exception:
                pass
        if isinstance(df, _pd.DataFrame) and (not df.empty):
            return df.reset_index(drop=False)
        return _pd.DataFrame()
    except APIError as e:
        req = {
            "timeframe_raw": str(tf_raw),
            "timeframe_norm": tf_norm,
            "feed": feed,
            "start": start_s,
            "end": end_s,
            "adjustment": adjustment,
        }
        body = ""
        try:
            body = e.response.text
        except (ValueError, TypeError):
            pass
        _log.error(
            "ALPACA_FAIL",
            extra={
                "symbol": symbol,
                "timeframe": tf_norm,
                "feed": feed,
                "start": start_s,
                "end": end_s,
                "status_code": getattr(e, "status_code", None),
                "endpoint": "alpaca/bars",
                "query_params": req,
                "body": body,
            },
        )
        return _pd.DataFrame()


class AlpacaOrderError(RuntimeError):
    """Base exception for order submission issues."""


class AlpacaOrderHTTPError(AlpacaOrderError):
    def __init__(self, status_code: int, message: str, payload: Optional[dict[str, Any]] | None = None):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.payload = payload or {}


class AlpacaOrderNetworkError(AlpacaOrderError):
    pass


@dataclass(frozen=True)
class _AlpacaConfig:
    base_url: str
    key_id: str | None
    secret_key: str | None
    shadow: bool

    @staticmethod
    def from_env() -> "_AlpacaConfig":
        base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
        key = os.getenv("ALPACA_API_KEY_ID")
        sec = os.getenv("ALPACA_API_SECRET_KEY")
        shadow_env = os.getenv("ALPACA_SHADOW", "")
        shadow = is_shadow_mode() or str(shadow_env).strip().lower() in {"1", "true", "yes", "on"}
        return _AlpacaConfig(base, key, sec, shadow)


def _ts() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _as_int(n: Any) -> int:
    return int(round(float(n)))


def _to_public_dict(resp_json: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "id",
        "client_order_id",
        "symbol",
        "qty",
        "side",
        "type",
        "time_in_force",
        "limit_price",
        "stop_price",
        "status",
        "created_at",
        "submitted_at",
        "filled_qty",
    ]
    return {k: resp_json.get(k) for k in fields if k in resp_json}


def _sdk_submit(
    client: Any,
    *,
    symbol: str,
    qty: int,
    side: str,
    type: str,
    time_in_force: str,
    limit_price: float | None,
    stop_price: float | None,
    idempotency_key: str | None,
    timeout: float | int | None,
) -> dict[str, Any]:
    submit = getattr(client, "submit_order", None)
    if submit is None:
        raise AttributeError("client.submit_order is not available")

    kwargs: dict[str, Any] = dict(
        symbol=symbol,
        qty=str(qty),
        side=side,
        type=type,
        time_in_force=time_in_force,
    )
    if limit_price is not None:
        kwargs["limit_price"] = str(limit_price)
    if stop_price is not None:
        kwargs["stop_price"] = str(stop_price)
    if idempotency_key:
        kwargs["client_order_id"] = idempotency_key

    # Optional retry wrapper for SDK submit
    call = submit if retry is None else _with_retry(submit)
    _start_t = _mono()
    _err: Exception | None = None
    try:
        order = call(**kwargs)
    except Exception as e:
        _err = e
        raise
    finally:
        try:
            _alpaca_calls_total.inc()
            _alpaca_call_latency.observe(max(0.0, _mono() - _start_t))
            if _err is not None:
                _alpaca_errors_total.inc()
        except Exception:
            pass
    if hasattr(order, "_raw"):
        data = dict(order._raw)  # type: ignore[attr-defined]
    elif hasattr(order, "__dict__"):
        data = dict(order.__dict__)
    else:
        try:
            data = json.loads(json.dumps(order))
        except Exception:
            data = {
                "id": getattr(order, "id", None),
                "symbol": symbol,
                "qty": str(qty),
                "side": side,
                "status": getattr(order, "status", None),
            }
    return _to_public_dict(data)


def _http_submit(
    cfg: _AlpacaConfig,
    *,
    symbol: str,
    qty: int,
    side: str,
    type: str,
    time_in_force: str,
    limit_price: float | None,
    stop_price: float | None,
    idempotency_key: str | None,
    timeout: float | int | None,
) -> dict[str, Any]:
    url = f"{cfg.base_url}/v2/orders"
    headers = {
        "APCA-API-KEY-ID": cfg.key_id or "",
        "APCA-API-SECRET-KEY": cfg.secret_key or "",
        "Content-Type": "application/json",
    }
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key

    payload: dict[str, Any] = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": type,
        "time_in_force": time_in_force,
    }
    if limit_price is not None:
        payload["limit_price"] = str(limit_price)
    if stop_price is not None:
        payload["stop_price"] = str(stop_price)

    _start_t = _mono()
    _err: Exception | None = None
    try:
        timeout_v = clamp_request_timeout(timeout or 10)
        call = _HTTP.post
        if retry is not None:
            call = _with_retry(call)
        resp = call(url, headers=headers, json=payload, timeout=timeout_v)
    except RequestException as e:  # pragma: no cover - network error path
        _err = e
        raise AlpacaOrderNetworkError(f"Network error calling {url}: {e}") from e
    finally:
        try:
            _alpaca_calls_total.inc()
            _alpaca_call_latency.observe(max(0.0, _mono() - _start_t))
            if _err is not None or resp.status_code >= 400:  # type: ignore[name-defined]
                _alpaca_errors_total.inc()
        except Exception:
            pass

    try:
        content: dict[str, Any] | None = resp.json()
    except Exception:
        content = None

    if resp.status_code >= 400:
        msg = "rate limited" if resp.status_code == 429 else (content or {}).get("message") or resp.text
        raise AlpacaOrderHTTPError(resp.status_code, msg, payload=content or {})

    return _to_public_dict(content or {})


def submit_order(
    symbol: str,
    side: str,
    *,
    qty: int | float | str,
    type: str = "market",
    time_in_force: str = "day",
    limit_price: float | None = None,
    stop_price: float | None = None,
    shadow: bool | None = None,
    timeout: float | int | None = None,
    idempotency_key: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Submit an order via Alpaca SDK or HTTP.

    Parameters
    ----------
    symbol:
        Asset ticker to trade.
    side:
        ``"buy"`` or ``"sell"`` direction for the order.
    qty:
        Quantity of shares to submit (keyword-only).

    Notes
    -----
    - Supports shadow mode for offline testing.
    - Raises :class:`AttributeError` if a provided client lacks ``submit_order``.
    - Maps HTTP errors (incl. 429) to :class:`AlpacaOrderHTTPError`.
    """
    # AI-AGENT-REF: expose deterministic submit_order with shadow + HTTP fallback
    cfg = _AlpacaConfig.from_env()
    do_shadow = cfg.shadow if shadow is None else bool(shadow)
    q_int = _as_int(qty)
    timeout = clamp_request_timeout(timeout)
    idempotency_key = idempotency_key or generate_client_order_id()

    if do_shadow:
        oid = f"shadow-{uuid.uuid4().hex[:16]}"
        return {
            "id": oid,
            "client_order_id": idempotency_key,
            "symbol": symbol,
            "qty": str(q_int),
            "side": side,
            "type": type,
            "time_in_force": time_in_force,
            "status": "accepted",
            "submitted_at": _ts(),
            "filled_qty": "0",
        }

    if client is not None:
        return _sdk_submit(
            client,
            symbol=symbol,
            qty=q_int,
            side=side,
            type=type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
            idempotency_key=idempotency_key,
            timeout=timeout,
        )

    try:
        from alpaca.trading.client import TradingClient as _REST
        rest = _REST(
            api_key=cfg.key_id,
            secret_key=cfg.secret_key,
            url_override=cfg.base_url,
        )
        return _sdk_submit(
            rest,
            symbol=symbol,
            qty=q_int,
            side=side,
            type=type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
            idempotency_key=idempotency_key,
            timeout=timeout,
        )
    except ModuleNotFoundError:
        pass

    return _http_submit(
        cfg,
        symbol=symbol,
        qty=q_int,
        side=side,
        type=type,
        time_in_force=time_in_force,
        limit_price=limit_price,
        stop_price=stop_price,
        idempotency_key=idempotency_key,
        timeout=timeout,
    )

def alpaca_get(*_a, **_k):
    return None

def start_trade_updates_stream(*_a, **_k):
    return None
__all__ = [
    'ALPACA_AVAILABLE',
    'is_shadow_mode',
    'RETRY_HTTP_CODES',
    'RETRYABLE_HTTP_STATUSES',
    'submit_order',
    'AlpacaOrderError',
    'AlpacaOrderHTTPError',
    'AlpacaOrderNetworkError',
    'generate_client_order_id',
    'list_orders_wrapper',
    '_bars_time_window',
    'get_bars_df',
    'alpaca_get',
    'start_trade_updates_stream',
    'initialize',
]
