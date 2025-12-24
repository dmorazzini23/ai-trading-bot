from __future__ import annotations
import datetime as dt
from datetime import timezone
import json
import os
import time
import uuid
import inspect
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING
from threading import RLock

try:
    from alpaca.trading.client import TradingClient
    ALPACA_AVAILABLE = True
except Exception:
    TradingClient = None  # type: ignore[assignment]
    ALPACA_AVAILABLE = False

try:  # pragma: no cover - exercised via stub classes in tests
    from alpaca.trading.requests import (
        LimitOrderRequest as _LimitOrderRequest,
        MarketOrderRequest as _MarketOrderRequest,
        StopOrderRequest as _StopOrderRequest,
        StopLimitOrderRequest as _StopLimitOrderRequest,
    )
except Exception:
    _MarketOrderRequest = _LimitOrderRequest = _StopOrderRequest = _StopLimitOrderRequest = None  # type: ignore[assignment]


if _MarketOrderRequest is None:  # pragma: no cover - fallback when SDK unavailable
    @dataclass
    class _OrderRequest:
        symbol: Any
        qty: Any
        side: Any
        time_in_force: Any
        limit_price: Any | None = None
        stop_price: Any | None = None
        client_order_id: str | None = None


    class _MarketOrderRequest(_OrderRequest):
        pass


    class _LimitOrderRequest(_OrderRequest):
        pass


    class _StopOrderRequest(_OrderRequest):
        pass


    class _StopLimitOrderRequest(_OrderRequest):
        pass


MarketOrderRequest = _MarketOrderRequest
LimitOrderRequest = _LimitOrderRequest
StopOrderRequest = _StopOrderRequest
StopLimitOrderRequest = _StopLimitOrderRequest

# Only used for type hints; does NOT run at import time.
if TYPE_CHECKING:
    from ai_trading.net.http import HTTPSession  # pragma: no cover


def _lazy_http_session() -> "Optional[HTTPSession]":
    """Return the shared HTTP session, importing lazily to avoid cycles."""

    try:
        from ai_trading.net.http import get_http_session  # local import breaks the cycle

        return get_http_session()
    except Exception:
        return None


_HTTP_SESSION: "Optional[HTTPSession]" = None
_pending_orders_lock = RLock()
_pending_orders: dict[str, dict[str, Any]] = {}
partial_fill_tracker: dict[str, float] = {}
partial_fills: dict[str, dict[str, Any]] = {}


def _get_http_session() -> "HTTPSession":
    """Return the shared HTTP session, raising if it is unavailable."""

    global _HTTP_SESSION

    if _HTTP_SESSION is None:
        _HTTP_SESSION = _lazy_http_session()

    session = _HTTP_SESSION
    if session is None:
        raise RuntimeError("HTTP session is not available yet")
    return session


class _HTTPShim:
    """Attribute-forwarding proxy exposing the shared HTTP session."""

    def __getattr__(self, name):  # pragma: no cover - exercised in tests
        session = _get_http_session()
        try:
            return getattr(session, name)
        except AttributeError:
            if name == "post":
                # Provide a best-effort default when the underlying session
                # lacks ``post`` (e.g. minimal test shims without requests).
                def _post(url, *args, **kwargs):
                    return session.request("POST", url, *args, **kwargs)

                setattr(session, "post", _post)
                return getattr(session, name)
            raise

    def __setattr__(self, name, value):  # pragma: no cover - exercised in tests
        if name.startswith("_"):
            return super().__setattr__(name, value)
        setattr(_get_http_session(), name, value)


_HTTP = _HTTPShim()
from ai_trading.exc import RequestException
from ai_trading.utils.http import clamp_request_timeout
import importlib
import sys
from ai_trading.logging import get_logger
try:
    from ai_trading.config.management import is_shadow_mode, _resolve_alpaca_env
except ImportError:  # pragma: no cover - fallback for tests stubbing config
    def is_shadow_mode() -> bool:
        return False

    def _resolve_alpaca_env():
        return None, None, "https://paper-api.alpaca.markets"
from ai_trading.logging.normalize import canon_symbol as _canon_symbol
from ai_trading.metrics import get_counter, get_histogram
from ai_trading.utils.optional_dep import missing
from ai_trading.utils.time import monotonic_time

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

_log = get_logger(__name__)
RETRY_HTTP_CODES = {429, 500, 502, 503, 504}
RETRYABLE_HTTP_STATUSES = tuple(RETRY_HTTP_CODES)
_UTC = timezone.utc  # AI-AGENT-REF: prefer stdlib UTC

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

ALPACA_AVAILABLE = ALPACA_AVAILABLE and not missing("alpaca", "alpaca")
_TEST_FLAG_VALUES = {"1", "true", "yes", "on"}
if (
    str(os.getenv("PYTEST_RUNNING", "")).strip().lower() in _TEST_FLAG_VALUES
    or str(os.getenv("TESTING", "")).strip().lower() in _TEST_FLAG_VALUES
    or os.getenv("PYTEST_CURRENT_TEST")
):
    ALPACA_AVAILABLE = False
HAS_PANDAS: bool = not missing("pandas", "pandas")
_ALPACA_SERVICE_AVAILABLE: bool = True


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


def is_alpaca_service_available() -> bool:
    """Return ``True`` when Alpaca API requests are currently authenticated."""

    return _ALPACA_SERVICE_AVAILABLE


def _set_alpaca_service_available(value: bool) -> None:
    global _ALPACA_SERVICE_AVAILABLE
    _ALPACA_SERVICE_AVAILABLE = bool(value)


if not ALPACA_AVAILABLE:  # pragma: no cover - exercised in tests
    from dataclasses import dataclass
    from enum import Enum
    from typing import Any

    class TimeFrameUnit(Enum):
        Minute = "Min"
        Hour = "Hour"
        Day = "Day"
        Week = "Week"
        Month = "Month"

    @dataclass
    class TimeFrame:
        amount: int = 1
        unit: TimeFrameUnit = TimeFrameUnit.Day

        def __str__(self) -> str:
            return f"{self.amount}{self.unit.value}"

    # Pre-defined shorthand attributes mirroring alpaca-py
    TimeFrame.Minute = TimeFrame(1, TimeFrameUnit.Minute)  # type: ignore[attr-defined]
    TimeFrame.Hour = TimeFrame(1, TimeFrameUnit.Hour)  # type: ignore[attr-defined]
    TimeFrame.Day = TimeFrame()  # type: ignore[attr-defined]
    _week_unit = getattr(TimeFrameUnit, "Week", None)
    if _week_unit is not None:
        TimeFrame.Week = TimeFrame(1, _week_unit)  # type: ignore[attr-defined]
    _month_unit = getattr(TimeFrameUnit, "Month", None)
    if _month_unit is not None:
        TimeFrame.Month = TimeFrame(1, _month_unit)  # type: ignore[attr-defined]

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


def _make_client_order_id(prefix: str = "ai") -> str:
    minute_bucket = int(time.time() // 60)
    return f"{prefix}-{minute_bucket}-{uuid.uuid4().hex[:8]}"


generate_client_order_id = _make_client_order_id


def _ensure_trading_client_cls():
    global TradingClient, ALPACA_AVAILABLE
    try:
        from alpaca.trading.client import TradingClient as _TradingClient
    except Exception:
        TradingClient = None  # type: ignore[assignment]
        ALPACA_AVAILABLE = False
        return None
    TradingClient = _TradingClient  # type: ignore[assignment]
    ALPACA_AVAILABLE = True
    return TradingClient


def get_trading_client_cls():
    """Return the Alpaca TradingClient class if available."""

    client_cls = _ensure_trading_client_cls()
    if client_cls is None:
        class _UnavailableTradingClient:
            def __init__(self, *_a, **_k):
                raise RuntimeError("alpaca-py TradingClient not available")

        return _UnavailableTradingClient
    return client_cls


class TradingClientAdapter:
    """Adapter that exposes ``cancel_order`` regardless of SDK shape.

    The modern alpaca-py ``TradingClient`` exposes ``cancel_order_by_id`` and
    ``cancel_orders`` helpers, but older call sites – including our validation
    logic – expect a ``cancel_order`` method. This adapter provides that method
    while transparently proxying all other attribute access to the wrapped
    client. The adapter stores the underlying instance on
    ``_ai_trading_wrapped_client`` so validation can recognise the concrete SDK
    type and avoid emitting compatibility warnings.
    """

    __slots__ = (
        "_client",
        "_ai_trading_wrapped_client",
        "__ai_trading_adapter__",
        "list_orders",
        "list_positions",
        "__dict__",
    )

    def __init__(self, client: Any):
        self._client = client
        self._ai_trading_wrapped_client = client
        self.__ai_trading_adapter__ = "trading_client"

        if hasattr(client, "list_orders"):
            self.list_orders = client.list_orders  # type: ignore[assignment]
        else:
            orders_shim = self._build_list_orders_shim()
            if orders_shim is not None:
                self.list_orders = orders_shim  # type: ignore[assignment]

        if hasattr(client, "list_positions"):
            self.list_positions = client.list_positions  # type: ignore[assignment]
        else:
            positions_shim = self._build_list_positions_shim()
            if positions_shim is not None:
                self.list_positions = positions_shim  # type: ignore[assignment]

    def __getattr__(self, item: str) -> Any:
        return getattr(self._client, item)

    def __dir__(self) -> list[str]:  # pragma: no cover - convenience only
        merged = set(dir(self._client))
        merged.update(["cancel_order", "_ai_trading_wrapped_client"])
        return sorted(merged)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic aid
        return f"TradingClientAdapter({self._client!r})"

    def _build_list_orders_shim(self):
        get_orders = getattr(self._client, "get_orders", None)
        if not callable(get_orders):
            return None

        try:
            sig = inspect.signature(get_orders)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            sig = None
        accepts_status = bool(sig and "status" in sig.parameters)
        accepts_filter = bool(sig and "filter" in sig.parameters)

        def _list_orders(*args: Any, **kwargs: Any):
            status = kwargs.pop("status", None)
            if status is None:
                return get_orders(*args, **kwargs)

            enum_val: Any = status
            try:  # pragma: no cover - best effort import
                enums_mod = __import__("alpaca.trading.enums", fromlist=[""])
                enum_cls = getattr(enums_mod, "QueryOrderStatus", None) or getattr(
                    enums_mod, "OrderStatus", None
                )
                if enum_cls is not None:
                    enum_val = getattr(enum_cls, str(status).upper(), status)
            except Exception:
                pass

            if accepts_filter:
                try:
                    requests_mod = __import__(
                        "alpaca.trading.requests", fromlist=["GetOrdersRequest"]
                    )
                    GetOrdersRequest = getattr(requests_mod, "GetOrdersRequest")
                except Exception:
                    GetOrdersRequest = None
                if GetOrdersRequest is not None:
                    try:
                        filter_obj = GetOrdersRequest(statuses=[enum_val])
                        return get_orders(*args, filter=filter_obj, **kwargs)
                    except TypeError:
                        # Fall back to status kwargs below
                        pass

            if accepts_status:
                kwargs["status"] = enum_val
                return get_orders(*args, **kwargs)

            kwargs["status"] = enum_val
            return get_orders(*args, **kwargs)

        return _list_orders

    def _build_list_positions_shim(self):
        get_all_positions = getattr(self._client, "get_all_positions", None)
        if not callable(get_all_positions):
            return None

        def _list_positions(*_args: Any, **_kwargs: Any):
            return get_all_positions()

        return _list_positions

    def cancel_order(self, order_id: Any) -> Any:
        """Cancel an order by delegating to the wrapped SDK client."""

        cancel_by_id = getattr(self._client, "cancel_order_by_id", None)
        if callable(cancel_by_id):
            return cancel_by_id(order_id)

        cancel_orders = getattr(self._client, "cancel_orders", None)
        if not callable(cancel_orders):  # pragma: no cover - defensive guard
            raise AttributeError("cancel_order not supported by wrapped client")

        CancelOrdersRequest: Any | None
        try:  # pragma: no cover - exercised indirectly in integration tests
            from alpaca.trading.requests import CancelOrdersRequest  # type: ignore
        except Exception:
            CancelOrdersRequest = None

        if CancelOrdersRequest is None:

            class _FallbackCancelOrdersRequest:
                def __init__(self, **kwargs: Any):
                    if not kwargs:
                        raise TypeError("payload required")
                    self.payload = kwargs

            CancelOrdersRequest = _FallbackCancelOrdersRequest

        last_error: Exception | None = None
        init_variants = (
            {"order_id": order_id},
            {"order_ids": [order_id]},
            {"client_order_id": order_id},
        )

        for init_kwargs in init_variants:
            try:
                request_obj = CancelOrdersRequest(**init_kwargs)
            except TypeError as exc:
                last_error = exc
                continue

            for caller in (
                lambda ro=request_obj: cancel_orders(ro),
                lambda ro=request_obj: cancel_orders(request=ro),
                lambda ro=request_obj: cancel_orders(cancel_orders_request=ro),
            ):
                try:
                    return caller()
                except TypeError as exc:
                    last_error = exc
                    continue

        raise RuntimeError(
            "Alpaca client cancel_orders shim could not adapt provided API"
        ) from last_error


def get_data_client_cls():
    """Return the Alpaca StockHistoricalDataClient class via lazy import."""
    try:
        from alpaca.data.historical.stock import StockHistoricalDataClient  # type: ignore

        return StockHistoricalDataClient
    except Exception:
        class _UnavailableDataClient:
            def __init__(self, *_a, **_k):
                self._reason = "alpaca-py StockHistoricalDataClient not available"

            def get_stock_bars(self, *_a, **_k):
                raise RuntimeError(self._reason)

        return _UnavailableDataClient


def get_api_error_cls():
    """Return the Alpaca APIError class via lazy import."""
    try:
        from alpaca.common.exceptions import APIError  # type: ignore
    except Exception:
        class APIError(Exception):
            """Fallback APIError when alpaca-py is unavailable."""

            pass

    return APIError


def list_orders_wrapper(api: Any, *args: Any, **kwargs: Any):
    """Adapter for ``get_orders`` methods lacking ``list_orders``.

    ``status`` is forwarded as a top-level keyword to preserve compatibility
    with SDKs expecting a simple string or enum. Other parameters are passed
    through unchanged. For newer SDKs accepting a ``filter`` object, this
    constructs :class:`GetOrdersRequest` when available.
    """
    status = kwargs.pop("status", None)

    try:
        use_filter = "filter" in inspect.signature(api.get_orders).parameters
    except Exception:  # pragma: no cover - defensive
        use_filter = False

    if use_filter and status is not None:
        try:
            requests_mod = importlib.import_module("alpaca.trading.requests")
            enums_mod = importlib.import_module("alpaca.trading.enums")
            req_cls = getattr(requests_mod, "GetOrdersRequest")
            enum_cls = getattr(enums_mod, "QueryOrderStatus")
            enum_val = getattr(enum_cls, str(status).upper(), status)
            filt = req_cls(statuses=[enum_val])
        except Exception:
            pass
        else:
            return api.get_orders(*args, filter=filt, **kwargs)

    if status is not None:
        enum_val: Any = status
        try:  # optional enum mapping for alpaca-py
            enums_mod = importlib.import_module("alpaca.trading.enums")
            enum_cls = getattr(enums_mod, "QueryOrderStatus", None) or getattr(enums_mod, "OrderStatus", None)
            if enum_cls is not None:
                enum_val = getattr(enum_cls, str(status).upper(), status)
        except Exception:
            pass
        kwargs["status"] = enum_val
    return api.get_orders(*args, **kwargs)  # type: ignore[attr-defined]


def _data_classes():
    """Return Alpaca data request classes lazily."""
    try:
        from alpaca.data import StockBarsRequest as _StockBarsRequest, TimeFrame as _TimeFrame, TimeFrameUnit as _TimeFrameUnit  # type: ignore

        return _StockBarsRequest, _TimeFrame, _TimeFrameUnit
    except Exception:
        return StockBarsRequest, TimeFrame, TimeFrameUnit


def get_stock_bars_request_cls():
    if ALPACA_AVAILABLE:
        cls, _, _ = _data_classes()
        return cls
    return StockBarsRequest


def get_timeframe_cls():
    """Return the package-level :class:`TimeFrame` wrapper.

    The wrapper guarantees that ``TimeFrame()`` defaults to ``1 Day`` and that
    ``amount`` and ``unit`` attributes are always present.
    """
    from .timeframe import TimeFrame

    return TimeFrame


def get_timeframe_unit_cls():
    """Return the package-level ``TimeFrameUnit`` enum."""
    from .timeframe import TimeFrameUnit

    return TimeFrameUnit


def _normalize_timeframe_for_tradeapi(tf_raw):
    """Return (canonical_string, TimeFrame) for ``tf_raw`` input."""
    from ai_trading.timeframe import canonicalize_timeframe

    tf_obj = canonicalize_timeframe(tf_raw)
    unit = getattr(tf_obj.unit, "name", str(tf_obj.unit)).title()
    return f"{tf_obj.amount}{unit}", tf_obj


def _to_utc(dtobj: dt.datetime) -> dt.datetime:
    """Ensure a ``datetime`` is timezone-aware and in UTC."""
    if dtobj.tzinfo is None:
        return dtobj.replace(tzinfo=dt.timezone.utc)
    return dtobj.astimezone(dt.timezone.utc)


def _fmt_rfc3339_z(dtobj: dt.datetime) -> str:
    """Format a UTC datetime to RFC3339 ``YYYY-MM-DDTHH:MM:SSZ``."""
    d = _to_utc(dtobj).replace(microsecond=0)
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_start_end_for_tradeapi(timeframe: str, start, end):
    """Return request datetimes plus canonical string forms for logging."""
    from ai_trading.utils.datetime import (
        compose_daily_params,
        compose_intraday_params,
        ensure_datetime,
    )

    if start is None or end is None:
        raise ValueError("start and end must be provided")

    sd = ensure_datetime(start)
    ed = ensure_datetime(end)

    is_daily = str(timeframe).lower() in {"1day", "day", "daily"}
    if is_daily:
        req_start = dt.datetime.combine(sd.date(), dt.time())
        req_end = dt.datetime.combine(ed.date(), dt.time())
        params = compose_daily_params(sd, ed)
    else:
        req_start = sd.astimezone(_UTC)
        req_end = ed.astimezone(_UTC)
        params = compose_intraday_params(sd, ed)

    return req_start, req_end, params["start"], params["end"]


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
        raise RuntimeError("Provide either ALPACA_API_KEY/ALPACA_SECRET_KEY or ALPACA_OAUTH, not both")

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


def _bars_time_window(timeframe: Any) -> tuple[dt.datetime, dt.datetime]:
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
    if is_daily:
        start = dt.datetime.combine(start.date(), dt.time())
        end = dt.datetime.combine(end.date(), dt.time())
    else:
        start = start.astimezone(_UTC)
        end = end.astimezone(_UTC)
    return (start, end)


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
if missing("tenacity", "retry"):
    retry = object()  # type: ignore[assignment]

    def _with_retry(callable_):  # type: ignore
        def _wrapper(*args, **kwargs):
            attempts = 0
            while True:
                try:
                    return callable_(*args, **kwargs)
                except RequestException:
                    attempts += 1
                    if attempts >= 2:
                        raise

        return _wrapper

else:  # pragma: no cover - optional dependency wrapper
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

    _pd = _require_pandas("get_bars_df")
    rest_factories: list[Any] = []
    seen_factories: set[int] = set()

    def _add_factory(candidate: Any) -> None:
        if not callable(candidate):
            return
        ident = id(candidate)
        if ident in seen_factories:
            return
        seen_factories.add(ident)
        rest_factories.append(candidate)

    module_obj = sys.modules.get("ai_trading.alpaca_api")
    if module_obj is not None:
        _add_factory(getattr(module_obj, "_get_rest", None))
    _add_factory(globals().get("_get_rest"))
    if not rest_factories:  # pragma: no cover - defensive fallback when globals missing
        module_self = importlib.import_module("ai_trading.alpaca_api")
        _add_factory(getattr(module_self, "_get_rest", None))

    rest_factories.sort(
        key=lambda fn: 1 if getattr(fn, "__module__", None) == __name__ else 0
    )

    rest: Any | None = None
    last_error: Exception | None = None
    for factory in rest_factories:
        try:
            rest = factory(bars=True)
        except TypeError as exc:
            last_error = exc
            try:
                rest = factory(True)
            except TypeError:
                continue
        except ImportError as exc:
            last_error = exc
            continue
        if rest is not None:
            break
    if rest is None:
        if last_error is not None:
            raise RuntimeError("_get_rest unavailable") from last_error
        raise RuntimeError("_get_rest unavailable")
    feed = feed or os.getenv("ALPACA_DATA_FEED", "iex")
    adjustment = adjustment or os.getenv("ALPACA_ADJUSTMENT", "all")
    tf_raw = timeframe
    tf_norm, tf_obj = _normalize_timeframe_for_tradeapi(tf_raw)
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
    req_start, req_end, start_s, end_s = _format_start_end_for_tradeapi(tf_norm, start, end)
    try:
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=tf_obj,
            start=req_start,
            end=req_end,
            adjustment=adjustment,
            feed=feed,
        )
        # Some lightweight SDK stubs normalise timezone-aware datetimes to naive UTC.
        # Restore the explicit UTC tzinfo so downstream code can rely on aware values.
        if hasattr(req, "start") and isinstance(req_start, dt.datetime) and req_start.tzinfo is not None:
            req_start_attr = getattr(req, "start")
            if isinstance(req_start_attr, dt.datetime) and req_start_attr.tzinfo is None:
                setattr(req, "start", req_start)
        if hasattr(req, "end") and isinstance(req_end, dt.datetime) and req_end.tzinfo is not None:
            req_end_attr = getattr(req, "end")
            if isinstance(req_end_attr, dt.datetime) and req_end_attr.tzinfo is None:
                setattr(req, "end", req_end)
        df: "pd.DataFrame" | Any
        max_attempts = 3
        base_delay = 0.5
        attempt = 0
        last_error: Exception | None = None
        while attempt < max_attempts:
            attempt += 1
            _start_t = monotonic_time()
            error: Exception | None = None
            try:
                response = rest.get_stock_bars(req)
                df = response.df
                last_error = None
                break
            except APIError as api_exc:
                error = api_exc
                last_error = api_exc
                status_code = getattr(api_exc, "status_code", None)
                should_retry = bool(
                    status_code in RETRY_HTTP_CODES and attempt < max_attempts
                )
                if should_retry:
                    delay = min(base_delay * (2 ** (attempt - 1)), 4.0)
                    _log.warning(
                        "ALPACA_RATE_LIMIT_RETRY",
                        extra={
                            "symbol": symbol,
                            "timeframe": tf_norm,
                            "status_code": status_code,
                            "attempt": attempt,
                            "delay": round(delay, 3),
                        },
                    )
                    try:
                        time.sleep(delay)
                    except Exception:
                        pass
                    continue
                raise
            except RequestException as req_exc:
                error = req_exc
                last_error = req_exc
                if attempt < max_attempts:
                    delay = min(base_delay * (2 ** (attempt - 1)), 4.0)
                    _log.warning(
                        "ALPACA_NETWORK_RETRY",
                        extra={
                            "symbol": symbol,
                            "timeframe": tf_norm,
                            "attempt": attempt,
                            "delay": round(delay, 3),
                            "error": str(req_exc),
                        },
                    )
                    try:
                        time.sleep(delay)
                    except Exception:
                        pass
                    continue
                raise
            except Exception as exc:
                error = exc
                last_error = exc
                raise
            finally:
                try:
                    _alpaca_calls_total.inc()
                    _alpaca_call_latency.observe(
                        max(0.0, monotonic_time() - _start_t)
                    )
                    if error is not None:
                        _alpaca_errors_total.inc()
                except Exception:
                    pass
        else:
            if last_error is not None:
                raise last_error
            return _pd.DataFrame()
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


class AlpacaAuthenticationError(AlpacaOrderHTTPError):
    """Raised when Alpaca rejects a request due to invalid credentials."""

    def __init__(self, message: str, payload: Optional[dict[str, Any]] | None = None):
        super().__init__(401, message, payload=payload)


@dataclass(frozen=True)
class _AlpacaConfig:
    base_url: str
    key_id: str | None
    secret_key: str | None
    shadow: bool

    @staticmethod
    def from_env() -> "_AlpacaConfig":
        key, sec, base = _resolve_alpaca_env()
        base = (base or "https://paper-api.alpaca.markets").rstrip("/")
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


def _record_client_order_id(client: Any | None, idempotency_key: str | None) -> None:
    """Record generated client order IDs on the provided client object.

    The Alpaca client historically exposed ``client_order_ids`` while newer
    tests and helpers expect ``ids``. To keep both code paths working, we
    append the generated idempotency key to each attribute, initialising a
    list where necessary. Failures are intentionally swallowed to avoid
    leaking broker errors back to callers.
    """

    if client is None or not idempotency_key:
        return

    for attr in ("ids", "client_order_ids"):
        try:
            collection = getattr(client, attr)
        except AttributeError:
            try:
                setattr(client, attr, [])
                collection = getattr(client, attr)
            except Exception:
                continue

        if not hasattr(collection, "append"):
            continue

        try:
            collection.append(idempotency_key)
        except Exception:
            continue


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

    try:
        sig = inspect.signature(submit)
    except (TypeError, ValueError):  # pragma: no cover - builtin/descriptor
        sig = None

    params: dict[str, inspect.Parameter] = sig.parameters if sig is not None else {}
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    ) if params else True

    first_named: str | None = None
    for p in params.values():
        if p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if p.name == "self":
            continue
        first_named = p.name
        break

    use_request_object = False
    if "order_data" in params:
        use_request_object = True
    elif first_named == "order_data":
        use_request_object = True

    request_obj: Any | None = None
    request_kwargs: dict[str, Any]
    if use_request_object:
        order_type = str(type or "market").lower()
        request_cls_map: dict[str, type] = {}
        if MarketOrderRequest is not None:
            request_cls_map["market"] = MarketOrderRequest
        if LimitOrderRequest is not None:
            request_cls_map["limit"] = LimitOrderRequest
        if StopOrderRequest is not None:
            request_cls_map["stop"] = StopOrderRequest
        if StopLimitOrderRequest is not None:
            request_cls_map["stop_limit"] = StopLimitOrderRequest

        req_cls = request_cls_map.get(order_type)
        if req_cls is None:
            # Fallback to any available request class (preferring market)
            for candidate in (
                MarketOrderRequest,
                LimitOrderRequest,
                StopOrderRequest,
                StopLimitOrderRequest,
            ):
                if candidate is not None:
                    req_cls = candidate
                    break

        if req_cls is not None:
            request_kwargs = {
                "symbol": symbol,
                "qty": str(qty),
                "side": side,
                "time_in_force": time_in_force,
            }
            if limit_price is not None:
                request_kwargs["limit_price"] = str(limit_price)
            if stop_price is not None:
                request_kwargs["stop_price"] = str(stop_price)
            if idempotency_key:
                request_kwargs["client_order_id"] = idempotency_key
            try:
                request_obj = req_cls(**request_kwargs)  # type: ignore[misc]
            except Exception:
                request_obj = None

    extra_kwargs: dict[str, Any] = {}
    if request_obj is not None:
        if idempotency_key and (has_var_kw or "idempotency_key" in params):
            extra_kwargs["idempotency_key"] = idempotency_key
        if timeout is not None and (has_var_kw or "timeout" in params):
            extra_kwargs["timeout"] = timeout
    else:
        use_request_object = False

    legacy_kwargs: dict[str, Any] = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": type,
        "time_in_force": time_in_force,
    }
    if limit_price is not None:
        legacy_kwargs["limit_price"] = str(limit_price)
    if stop_price is not None:
        legacy_kwargs["stop_price"] = str(stop_price)
    if idempotency_key:
        legacy_kwargs["client_order_id"] = idempotency_key

    # Optional retry wrapper for SDK submit
    disable_retry = client is None
    try:
        if os.getenv("PYTEST_RUNNING"):
            disable_retry = True
    except Exception:
        pass
    selected_retry = None if disable_retry else retry
    call = submit if selected_retry is None else _with_retry(submit)
    _start_t = monotonic_time()
    _err: Exception | None = None
    try:
        if use_request_object and request_obj is not None:
            order = call(order_data=request_obj, **extra_kwargs)
        else:
            order = call(**legacy_kwargs)
    except Exception as e:
        _err = e
        raise
    finally:
        try:
            _alpaca_calls_total.inc()
            _alpaca_call_latency.observe(max(0.0, monotonic_time() - _start_t))
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
    if "status" not in data or data.get("status") in (None, ""):
        data["status"] = "accepted"
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

    _start_t = monotonic_time()
    _err: Exception | None = None
    try:
        timeout_v = clamp_request_timeout(timeout or 10)
        session = _get_http_session()
        call = session.post
        if retry is not None:
            call = _with_retry(call)
        resp = call(url, headers=headers, json=payload, timeout=timeout_v)
    except RequestException as e:  # pragma: no cover - network error path
        _err = e
        raise AlpacaOrderNetworkError(f"Network error calling {url}: {e}") from e
    finally:
        try:
            _alpaca_calls_total.inc()
            _alpaca_call_latency.observe(max(0.0, monotonic_time() - _start_t))
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
    pytest_mode = os.getenv("PYTEST_RUNNING")
    if pytest_mode:
        try:
            from ai_trading.config.management import reload_trading_config

            reload_trading_config()
        except Exception:
            pass

    cfg = _AlpacaConfig.from_env()
    explicit_shadow = shadow if shadow is not None else None
    do_shadow = cfg.shadow if shadow is None else bool(shadow)
    if (
        shadow is None
        and client is not None
        and pytest_mode
        and str(os.getenv("SHADOW_MODE", "")).strip().lower() in {"1", "true", "yes", "on"}
    ):
        do_shadow = False
    q_int = _as_int(qty)
    # AI-AGENT-REF: Add quantity validation before submission
    if q_int <= 0:
        raise ValueError(f"Invalid quantity: {qty}. Must be a positive integer.")
    timeout = clamp_request_timeout(timeout)
    symbol_part = str(symbol or "").strip().upper() or "UNKNOWN"
    side_part = str(side or "").strip().lower() or "buy"
    prefix = f"{symbol_part}-{side_part}"
    idempotency_key = idempotency_key or generate_client_order_id(prefix)

    def _ensure_client_order_id(payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        if idempotency_key and payload.get("client_order_id") in (None, ""):
            enriched = dict(payload)
            enriched["client_order_id"] = idempotency_key
            return enriched
        return payload

    _record_client_order_id(client, idempotency_key)

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
        order = _sdk_submit(
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
        _record_client_order_id(client, idempotency_key)
        return _ensure_client_order_id(order)

    try:
        from alpaca.trading.client import TradingClient as _REST

        rest = _REST(
            api_key=cfg.key_id,
            secret_key=cfg.secret_key,
            url_override=cfg.base_url,
        )
        return _ensure_client_order_id(
            _sdk_submit(
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
        )
    except ModuleNotFoundError:
        pass

    return _ensure_client_order_id(
        _http_submit(
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
    ))


def alpaca_get(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float | int | tuple[float | int, float | int] | None = None,
) -> dict[str, Any]:
    """Perform a GET request against the Alpaca REST API.

    Parameters
    ----------
    path:
        Endpoint path (``/v2/...``) or fully-qualified URL.
    params:
        Optional query string parameters to include with the request.
    timeout:
        Optional override for the request timeout in seconds. Values are
        clamped using :func:`ai_trading.utils.http.clamp_request_timeout` to
        match :func:`_http_submit`.
    """

    cfg = _AlpacaConfig.from_env()
    if path.startswith(("http://", "https://")):
        url = path
    else:
        url = f"{cfg.base_url}/{path.lstrip('/')}"

    headers = {
        "APCA-API-KEY-ID": cfg.key_id or "",
        "APCA-API-SECRET-KEY": cfg.secret_key or "",
        "Accept": "application/json",
    }

    _start_t = monotonic_time()
    _err: Exception | None = None
    resp: Any | None = None
    try:
        timeout_v = clamp_request_timeout(timeout or 10)
        session = _get_http_session()
        call = session.get
        if retry is not None:
            call = _with_retry(call)
        resp = call(url, headers=headers, params=params, timeout=timeout_v)
    except RequestException as exc:  # pragma: no cover - network error path
        _err = exc
        raise RequestException(f"Network error calling {url}: {exc}") from exc
    finally:
        try:
            _alpaca_calls_total.inc()
            _alpaca_call_latency.observe(max(0.0, monotonic_time() - _start_t))
            status = getattr(resp, "status_code", 0) if resp is not None else 0
            if _err is not None or status >= 400:
                _alpaca_errors_total.inc()
        except Exception:
            pass

    try:
        content = resp.json() if resp is not None else None
    except Exception:
        content = None

    status_code = getattr(resp, "status_code", 0)
    text = getattr(resp, "text", "") if resp is not None else ""

    if 200 <= status_code < 400:
        _set_alpaca_service_available(True)

    if status_code == 401:
        payload = content if isinstance(content, dict) else {}
        message = ""
        if isinstance(payload, dict):
            message = str(payload.get("message") or text)
        else:
            message = text
        _set_alpaca_service_available(False)
        _log.critical(
            "ALPACA_AUTH_FAILURE",
            extra={
                "endpoint": url,
                "status": status_code,
                "shadow_mode": cfg.shadow,
            },
        )
        raise AlpacaAuthenticationError(message or "Alpaca authentication failed", payload=payload)

    if status_code >= 400:
        payload = content if isinstance(content, dict) else {}
        message = ""
        if isinstance(payload, dict):
            message = str(payload.get("message") or text)
        else:
            message = text
        raise AlpacaOrderHTTPError(status_code, message or "Alpaca request failed", payload=payload)

    if not isinstance(content, dict):
        if content is None:
            return {}
        return {"data": content}

    for key in ("quote", "trade", "bar"):
        value = content.get(key)
        if isinstance(value, dict):
            return value

    return content


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def handle_trade_update(event: Any) -> None:
    """Handle trade update events and log fill transitions."""

    if event is None:
        return
    event_type = getattr(event, "event", None) or getattr(event, "event_type", None)
    event_label = str(event_type or "").strip().lower()
    if not event_label:
        return
    order = getattr(event, "order", None) or getattr(event, "order_data", None)
    if order is None:
        return
    order_id = getattr(order, "id", None) or getattr(order, "order_id", None)
    if not order_id:
        return
    order_key = str(order_id)
    symbol = getattr(order, "symbol", None)
    filled_qty = _coerce_float(getattr(order, "filled_qty", None))
    filled_avg_price = _coerce_float(getattr(order, "filled_avg_price", None))
    payload = {
        "order_id": order_key,
        "symbol": symbol,
        "filled_qty": filled_qty,
        "filled_avg_price": filled_avg_price,
        "event": event_label,
    }

    if event_label in {"partial_fill", "partial_filled"}:
        if order_key in partial_fill_tracker:
            return
        partial_fill_tracker[order_key] = monotonic_time()
        partial_fills[order_key] = dict(payload)
        _log.info("ORDER_PARTIAL_FILL", extra=payload)
        return
    if event_label in {"fill", "filled"}:
        partial_fill_tracker.pop(order_key, None)
        partial_fills.pop(order_key, None)
        _log.info("ORDER_FILLED", extra=payload)
        return


def start_trade_updates_stream(*_a, **_k):
    return None


__all__ = [
    "ALPACA_AVAILABLE",
    "is_shadow_mode",
    "is_alpaca_service_available",
    "RETRY_HTTP_CODES",
    "RETRYABLE_HTTP_STATUSES",
    "submit_order",
    "AlpacaOrderError",
    "AlpacaOrderHTTPError",
    "AlpacaAuthenticationError",
    "AlpacaOrderNetworkError",
    "generate_client_order_id",
    "list_orders_wrapper",
    "_bars_time_window",
    "get_bars_df",
    "alpaca_get",
    "start_trade_updates_stream",
    "partial_fill_tracker",
    "partial_fills",
    "handle_trade_update",
    "initialize",
    "_HTTP",
    "_pending_orders_lock",
    "_pending_orders",
]
