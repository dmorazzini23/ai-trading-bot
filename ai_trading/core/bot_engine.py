# fmt: off
"""Trading bot core engine.

This module uses ``pickle`` for regime model persistence; model paths are
validated before deserialization.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Any, Dict, Iterable, TYPE_CHECKING, cast
from zoneinfo import ZoneInfo  # AI-AGENT-REF: timezone conversions
from functools import cached_property, lru_cache

from json import JSONDecodeError
# Safe 'requests' import with stub + RequestException binding
try:  # pragma: no cover
    import requests  # type: ignore
    RequestException = requests.exceptions.RequestException  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover  # AI-AGENT-REF: narrow import handling
    class RequestException(Exception):
        pass

    # Minimal stub so runtime calls fail gracefully into COMMON_EXC
    class _RequestsStub:
        class exceptions:
            RequestException = RequestException

        def get(self, *a, **k):
            raise RequestException("requests not installed")

    requests = _RequestsStub()  # type: ignore

# Reusable HTTP session
try:  # pragma: no cover
    from ai_trading.net.http import HTTPSession, get_http_session

    _HTTP_SESSION: HTTPSession = get_http_session()
except Exception:  # pragma: no cover - fallback when requests missing
    class _HTTPStub:
        def get(self, *a, **k):  # type: ignore[no-untyped-def]
            raise RequestException("HTTPSession unavailable")

    _HTTP_SESSION = _HTTPStub()  # type: ignore

from threading import Lock
import warnings

import ai_trading.data.fetch as data_fetcher_module
from ai_trading.data.fetch import (
    DataFetchError,
    get_bars,
    get_bars_batch,
    get_minute_df,
    get_cached_minute_timestamp,
    last_minute_bar_age_seconds,
)
if TYPE_CHECKING:  # pragma: no cover - type hints only
    from ai_trading.risk.engine import RiskEngine, TradeSignal
    from ai_trading.data.bars import TimeFrame, StockBarsRequest
from ai_trading.utils.time import last_market_session
from ai_trading.capital_scaling import capital_scale, update_if_present
from ai_trading.utils.datetime import ensure_datetime
from ai_trading.data.timeutils import (
    ensure_utc_datetime as _ensure_utc_dt,  # AI-AGENT-REF: callable-aware UTC coercion
    nyse_session_utc as _nyse_session_utc,  # AI-AGENT-REF: derive NYSE RTH in UTC
    previous_business_day as _prev_bus_day,  # AI-AGENT-REF: default previous session
)
from ai_trading.data_validation import is_valid_ohlcv
from ai_trading.utils import health_check as _health_check
from ai_trading.alpaca_api import (
    ALPACA_AVAILABLE,
    get_bars_df,  # AI-AGENT-REF: canonical bar fetcher (auto start/end)
    get_trading_client_cls,
    get_data_client_cls,
    get_api_error_cls,
)
from ai_trading.utils.pickle_safe import safe_pickle_load

if TYPE_CHECKING:  # pragma: no cover - typing only
    from alpaca.common.exceptions import APIError  # type: ignore
    from alpaca.trading.client import TradingClient  # type: ignore
else:
    class APIError(Exception):
        """Fallback APIError when alpaca-py is unavailable."""

        pass

from ai_trading.config.management import (
    get_env,
    is_shadow_mode,
    TradingConfig,
)
from ai_trading.position_sizing import get_max_position_size
from ai_trading.settings import get_settings, get_alpaca_secret_key_plain


def _alpaca_available() -> bool:
    """Return ``True`` if the Alpaca SDK is importable."""

    return ALPACA_AVAILABLE
import ai_trading.data.bars as _bars
from ai_trading.data.bars import (
    _create_empty_bars_dataframe,
    _resample_minutes_to_daily,
)

# Deprecated legacy proxies; prefer ai_trading.data.bars
StockBarsRequest = _bars.StockBarsRequest
safe_get_stock_bars = _bars.safe_get_stock_bars

try:  # pragma: no cover
    from alpaca.data.historical import StockHistoricalDataClient  # type: ignore
except Exception:  # pragma: no cover
    class StockHistoricalDataClient:  # type: ignore[no-redef]
        """Fallback when alpaca-py is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401, ARG002
            raise ImportError("alpaca-py not installed")

def _parse_timeframe(tf: Any) -> _bars.TimeFrame:
    """Map configuration values to :class:`_bars.TimeFrame` enums."""

    if isinstance(tf, _bars.TimeFrame):
        return tf
    tf_map = {
        "1day": _bars.TimeFrame.Day,
        "1d": _bars.TimeFrame.Day,
        "day": _bars.TimeFrame.Day,
        "1min": _bars.TimeFrame.Minute,
        "1m": _bars.TimeFrame.Minute,
        "minute": _bars.TimeFrame.Minute,
    }
    key = str(tf).lower()
    if key in tf_map:
        return tf_map[key]
    raise ValueError(f"Unsupported timeframe: {tf}")

if os.getenv("BOT_SHOW_DEPRECATIONS", "").lower() in {"1", "true", "yes"}:
    warnings.filterwarnings("default", category=DeprecationWarning)
    warnings.warn(
        "bot_engine.py is deprecated",
        DeprecationWarning,
    )  # AI-AGENT-REF: deprecation notice


# One place to define the common exception family (module-scoped)
COMMON_EXC = (
    TypeError,
    ValueError,
    KeyError,
    JSONDecodeError,
    RequestException,
    TimeoutError,
    ImportError,
)

# Environment keys (defined early to support import-time fallbacks)
# Allow NEWS_API_KEY to serve as the default sentiment key if SENTIMENT_API_KEY is unset.  # AI-AGENT-REF: env alias
SENTIMENT_API_KEY: str | None = os.getenv("SENTIMENT_API_KEY") or os.getenv("NEWS_API_KEY")
NEWS_API_KEY: str | None = os.getenv("NEWS_API_KEY")
SENTIMENT_API_URL: str = os.getenv("SENTIMENT_API_URL", "")
TESTING = os.getenv("TESTING", "").lower() == "true"

SKLEARN_AVAILABLE = bool(importlib.util.find_spec("sklearn") or "sklearn" in sys.modules)
FINNHUB_AVAILABLE = bool(importlib.util.find_spec("finnhub") or "finnhub" in sys.modules)




def _rf_class():
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("sklearn not available")
    sklearn_ensemble = importlib.import_module("sklearn.ensemble")
    return sklearn_ensemble.RandomForestClassifier


def _bayesian_ridge():
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("sklearn not available")
    sklearn_linear = importlib.import_module("sklearn.linear_model")
    return sklearn_linear.BayesianRidge


def _ridge():
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("sklearn not available")
    sklearn_linear = importlib.import_module("sklearn.linear_model")
    return sklearn_linear.Ridge


trading_client = None
data_client = None


def _validate_trading_api(api: Any) -> bool:
    """Ensure trading ``api`` exposes ``list_orders``.

    Maps ``get_orders`` to ``list_orders`` when necessary and warns when the
    client is not an instance of :class:`TradingClient`.
    """  # AI-AGENT-REF: verify Alpaca client contract
    if api is None:
        logger_once.error("ALPACA_CLIENT_MISSING", key="alpaca_client_missing")
        return False
    if not hasattr(api, "list_orders"):
        if hasattr(api, "get_orders"):
            def _list_orders_wrapper(*args: Any, **kwargs: Any):  # type: ignore[override]
                """Proxy ``list_orders`` to ``get_orders`` with compatible kwargs."""

                status = kwargs.pop("status", None)
                if status is None:
                    return api.get_orders(*args, **kwargs)  # type: ignore[attr-defined]

                import inspect

                params = inspect.signature(api.get_orders).parameters

                enum_val: Any = status
                try:  # pragma: no cover - optional import paths
                    enums_mod = __import__("alpaca.trading.enums", fromlist=[""])
                    enum_cls = getattr(enums_mod, "QueryOrderStatus", None) or getattr(
                        enums_mod, "OrderStatus", None
                    )
                    if enum_cls is not None:
                        enum_val = getattr(enum_cls, str(status).upper(), status)
                except Exception:  # pragma: no cover - optional import paths
                    pass

                if "status" in params:
                    kwargs["status"] = enum_val
                    return api.get_orders(*args, **kwargs)  # type: ignore[attr-defined]

                req = None
                try:  # pragma: no cover - optional import paths
                    requests_mod = __import__("alpaca.trading.requests", fromlist=[""])
                    enums_mod = __import__("alpaca.trading.enums", fromlist=[""])
                    enum_cls = getattr(enums_mod, "QueryOrderStatus", None) or getattr(
                        enums_mod, "OrderStatus", None
                    )
                    if enum_cls is not None:
                        enum_val = getattr(enum_cls, str(status).upper(), status)
                    req_cls = getattr(requests_mod, "GetOrdersRequest")
                    req = req_cls(statuses=[enum_val])
                except Exception:
                    pass
                if req is not None:
                    try:
                        return api.get_orders(*args, filter=req, **kwargs)  # type: ignore[attr-defined]
                    except TypeError:
                        return api.get_orders(req, *args, **kwargs)  # type: ignore[attr-defined]

                kwargs["status"] = status
                return api.get_orders(*args, **kwargs)  # type: ignore[attr-defined]

            setattr(api, "list_orders", _list_orders_wrapper)  # type: ignore[attr-defined]
            logger_once.warning(
                "API_GET_ORDERS_MAPPED", key="alpaca_get_orders_mapped"
            )
        else:
            logger_once.error(
                "ALPACA_LIST_ORDERS_MISSING", key="alpaca_list_orders_missing"
            )
            if not is_shadow_mode():
                raise RuntimeError("Alpaca client missing list_orders method")
            return False
    TradingClient = get_trading_client_cls()
    if not isinstance(api, TradingClient):
        logger_once.warning(
            "ALPACA_API_ADAPTER", key="alpaca_api_adapter"
        )
    return True


def list_open_orders(api: Any):
    """Return all open orders from ``api``.

    ``_validate_trading_api`` shims ``list_orders`` so that passing
    ``status="open"`` works across Alpaca SDK versions. Newer clients expect a
    ``GetOrdersRequest`` with ``QueryOrderStatus.OPEN`` (or ``OrderStatus.OPEN``
    on older SDKs) while legacy clients may accept the raw status string
    directly.
    """

    return api.list_orders(status="open")


# -- New helper: ensure context has an attached Alpaca client -----------------
def ensure_alpaca_attached(ctx) -> None:
    """Attach global trading client to the context if it's missing."""
    if getattr(ctx, "api", None) is not None:
        return
    try:
        _initialize_alpaca_clients()
    except COMMON_EXC as e:  # AI-AGENT-REF: surface init failure
        logger_once.error(
            "ALPACA_CLIENT_INIT_FAILED - %s",
            e,
            key="alpaca_client_init_failed",
        )
        if not is_shadow_mode():
            raise RuntimeError("Alpaca client initialization failed") from e
        return
    global trading_client
    if trading_client is None:
        logger_once.error(
            "ALPACA_CLIENT_MISSING after initialization", key="alpaca_client_missing"
        )
        if not is_shadow_mode():
            raise RuntimeError("Alpaca client missing after initialization")
        return
    if hasattr(ctx, "_ensure_initialized"):
        try:
            ctx._ensure_initialized()  # type: ignore[attr-defined]
        except COMMON_EXC:
            pass
    try:
        setattr(ctx, "api", trading_client)
    except COMMON_EXC:
        inner = getattr(ctx, "_context", None)
        if inner is not None and getattr(inner, "api", None) is None:
            try:
                setattr(inner, "api", trading_client)
            except COMMON_EXC:
                pass
    api = getattr(ctx, "api", None)
    if api is None:
        logger_once.error(
            "FAILED_TO_ATTACH_ALPACA_CLIENT", key="alpaca_attach_failed"
        )
        if not is_shadow_mode():
            raise RuntimeError("Failed to attach Alpaca client to context")
        return
    _validate_trading_api(api)

# Sentiment knobs used by tests
SENTIMENT_FAILURE_THRESHOLD: int = 25
_SENTIMENT_FAILURES: int = 0
_SENTIMENT_CACHE: dict[str, tuple[float, float]] = {}

from enum import Enum

from ai_trading.settings import (
    get_buy_threshold,
    get_capital_cap,
    get_conf_threshold,
    get_daily_loss_limit,
    get_disaster_dd_limit,
    get_dollar_risk_limit,
    get_max_portfolio_positions,
    get_max_trades_per_day,
    get_max_trades_per_hour,
    get_rebalance_interval_min,
    get_sector_exposure_cap,
    get_trade_cooldown_min,
    get_verbose_logging,
    get_volume_threshold,
)

__all__ = [
    "pre_trade_health_check",
    "run_all_trades_worker",
    "BotState",
    "BotMode",
    "SENTIMENT_API_KEY",
    "SENTIMENT_API_URL",
    "SENTIMENT_FAILURE_THRESHOLD",
    "_SENTIMENT_CACHE",
    "fetch_sentiment",
    "ALPACA_AVAILABLE",
    "_alpaca_available",
    "FINNHUB_AVAILABLE",
    "DataFetchError",
    "MockSignal",
    "MockContext",
    "StockHistoricalDataClient",
    "safe_get_stock_bars",
]
# AI-AGENT-REF: custom exception surfaced by fetch helpers


class DataFetchError(RuntimeError):
    """Raised when required market data is unavailable."""  # AI-AGENT-REF


COMMON_EXC = COMMON_EXC + (DataFetchError,)  # AI-AGENT-REF: broaden common exceptions


# AI-AGENT-REF: Track regime warnings to avoid spamming logs during market closed
# Using a mutable dict to avoid fragile `global` declarations inside functions.
_REGIME_INSUFFICIENT_DATA_WARNED = {"done": False}


class MockSignal:
    """Simple signal container for tests."""  # AI-AGENT-REF

    def __init__(self, score: float):
        self.score = score


class MockContext:
    """Lightweight context object for tests."""  # AI-AGENT-REF

    def __init__(self, **kw):
        self.__dict__.update(kw)


def pretrade_data_health(runtime, universe) -> None:  # AI-AGENT-REF: data gate
    """Validate that Alpaca returns data for benchmark and sample symbols."""
    symbols = ["SPY"] + list(universe[:5])
    errors: list[str] = []
    for sym in symbols:
        try:
            df = get_bars_df(
                sym,
                _bars.TimeFrame.Day,
                feed=os.getenv("ALPACA_DATA_FEED", "iex"),
            )  # AI-AGENT-REF: derive window & feed
            if df is None or df.empty:
                errors.append(f"{sym}:empty")
        except COMMON_EXC as exc:  # AI-AGENT-REF: narrow catch
            errors.append(f"{sym}:{exc}")
    if errors:
        feed = os.getenv("ALPACA_DATA_FEED", "iex")
        logger.critical(
            "DATA_HEALTH_FAIL",
            extra={"endpoint": "alpaca/bars", "feed": feed, "symbols": symbols, "errors": errors},
        )
        raise DataFetchError("data health check failed")


def data_check(symbols: Iterable[str], *, feed: str | None = None) -> dict[str, pd.DataFrame]:
    """Fetch daily bars for ``symbols`` and return mapping of symbol to DataFrame.

    Symbols returning empty data or raising :class:`ValueError` are skipped so
    callers can continue operating with the remaining symbols.  This gracefully
    degrades when certain feeds (e.g., SIP without authorization) provide no
    data.
    """

    results: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = get_bars_df(sym, _bars.TimeFrame.Day, feed=feed)
        except ValueError:
            continue
        if df is None or df.empty:
            continue
        results[sym] = df
    return results
import asyncio
import atexit
import hashlib  # AI-AGENT-REF: model hash helper
import importlib
import inspect
import io
import logging
from ai_trading.logging import get_logger
import math
import time
import traceback
import types
import uuid
import warnings
from collections.abc import Callable  # AI-AGENT-REF: now_provider hooks
from datetime import UTC, date, datetime, timedelta
from json import JSONDecodeError  # AI-AGENT-REF: narrow exception imports
from pathlib import Path
from typing import Any

try:  # AI-AGENT-REF: optional joblib import
    import joblib  # type: ignore
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore

from ai_trading.logging import (
    _get_metrics_logger,
    get_logger,  # AI-AGENT-REF: use sanitizing adapter
)
from ai_trading.logging.redact import redact as _redact  # AI-AGENT-REF: mask secrets
from ai_trading.utils import http
from ai_trading.utils.timing import (
    HTTP_TIMEOUT,
)  # AI-AGENT-REF: enforce request timeouts
from ai_trading.utils.http import clamp_request_timeout
from ai_trading.utils.prof import StageTimer
from ai_trading.guards.staleness import _ensure_data_fresh

# AI-AGENT-REF: optional pipeline import
try:
    pipeline = importlib.import_module("ai_trading.pipeline")  # type: ignore
except ImportError:  # pragma: no cover - optional (import resolution only)
    pipeline = None  # type: ignore  # AI-AGENT-REF: fallback when pipeline absent

logger = get_logger(__name__)  # AI-AGENT-REF: central logger adapter

# Deprecated legacy logger alias
_log = get_logger(__name__)


def _current_position_qty(ctx: Any, symbol: str) -> int:
    """Return current position quantity for *symbol*.

    Deprecated legacy helper retained for compatibility. Falls back to ``0``
    if the position is missing or an error occurs.
    """

    api = getattr(ctx, "api", None)
    if api is None:
        return 0
    try:
        position = api.get_position(symbol)
    except Exception:  # pragma: no cover - legacy safety net
        _log.debug("POSITION_LOOKUP_FAILED", exc_info=True, extra={"symbol": symbol})
        return 0
    qty = getattr(position, "qty", 0) if position is not None else 0
    try:
        return int(qty)
    except (TypeError, ValueError):  # pragma: no cover - legacy safety net
        return 0


def _mask_env_name(name: str) -> str:
    return f"{name[:8]}***"


def _get_env_str(key: str) -> str:
    try:
        return cast(str, get_env(key, required=True))
    except RuntimeError as e:  # noqa: BLE001
        masked = _mask_env_name(key)
        logger.error("Missing required environment variable: %s", masked)
        raise RuntimeError(f"Missing required environment variable: {masked}") from e


class BotEngine:
    """Minimal engine exposing memoized Alpaca clients."""

    def __init__(
        self,
        *,
        trading_client_cls: Any | None = None,
        data_client_cls: Any | None = None,
    ) -> None:
        self.logger = get_logger(__name__)
        # Expose the lazily-initialized global context through this engine
        self._ctx = get_ctx()
        self._trading_client_cls = trading_client_cls or get_trading_client_cls()
        self._data_client_cls = data_client_cls or get_data_client_cls()
        global APIError
        if getattr(APIError, "__module__", "") == __name__:
            APIError = get_api_error_cls()
        # Load universe tickers once and store on both engine and runtime
        self._tickers = load_universe()
        setattr(self._ctx, "tickers", self._tickers)

    @property
    def ctx(self):
        """Return the lazily-initialized bot context."""
        return self._ctx

    @cached_property
    def trading_client(self):
        """Alpaca TradingClient for order/trade ops."""
        base_url = (
            _get_env_str("ALPACA_API_URL")
            if os.getenv("ALPACA_API_URL")
            else _get_env_str("ALPACA_BASE_URL")
        )
        api_key = _get_env_str("ALPACA_API_KEY")
        secret_key = _get_env_str("ALPACA_SECRET_KEY")
        cls = self._trading_client_cls
        return cls(
            api_key=api_key,
            secret_key=secret_key,
            paper="paper" in base_url.lower(),
            url_override=base_url,
        )

    @cached_property
    def data_client(self):
        """Alpaca StockHistoricalDataClient for historical/market data."""
        cls = self._data_client_cls
        return cls(
            api_key=_get_env_str("ALPACA_API_KEY"),
            secret_key=_get_env_str("ALPACA_SECRET_KEY"),
        )

    @property
    def tickers(self) -> list[str]:
        """Return cached universe tickers."""
        return self._tickers

# AI-AGENT-REF: ensure FinBERT disabled message logged once
_finbert_logged = False

def _log_finbert_disabled() -> None:
    global _finbert_logged
    if not _finbert_logged:
        logger.info("FinBERT disabled by config; skipping model load.")
        _finbert_logged = True

# AI-AGENT-REF: normalize arbitrary inputs into DataFrames
from ai_trading.utils.lazy_imports import load_pandas
import logging

# Lazy pandas proxy
pd = load_pandas()
logger = get_logger(__name__)

def _alpaca_diag_info() -> dict[str, object]:
    """Collect Alpaca env & mode diagnostics for operator visibility."""
    # AI-AGENT-REF: structured diag for once-per-process logging
    try:
        key, secret, base_url = _resolve_alpaca_env()
        shadow = is_shadow_mode()
        paper = bool(base_url and ("paper" in base_url))
        return {
            "has_key": bool(key),
            "has_secret": bool(secret),
            "base_url": base_url or "",
            "paper": paper,
            "shadow_mode": shadow,
            "cwd": os.getcwd(),
        }
    except COMMON_EXC as e:  # pragma: no cover – diag never fatal
        return {"diag_error": str(e)}  # AI-AGENT-REF: narrowed diag exception

# --- path helpers (no imports of heavy deps) ---
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def abspath_safe(fname: str | Path | None) -> str:
    """Return absolute path or empty string for falsy inputs."""  # AI-AGENT-REF: guard None paths
    if not fname:
        return ""
    s = str(fname)
    if os.path.isabs(s):
        return s
    return os.path.join(BASE_DIR, s)


def default_trade_log_path() -> str:
    """Resolve trade log path from env vars or fallback."""  # AI-AGENT-REF: ensure trade log exists
    env_candidates = [
        os.getenv("TRADE_LOG_PATH"),
        os.getenv("AI_TRADING_TRADE_LOG_PATH"),
    ]
    for candidate in env_candidates:
        cand = abspath_safe(candidate)
        if cand:
            os.makedirs(os.path.dirname(cand), exist_ok=True)
            return cand

    fallback = os.path.join(BASE_DIR, "logs", "trades.jsonl")
    os.makedirs(os.path.dirname(fallback), exist_ok=True)
    return fallback


# Delayed import: not all environments have pandas-market-calendars
def _load_mcal():  # AI-AGENT-REF: lazy calendar import
    try:
        import pandas_market_calendars as _mcal  # type: ignore

        return _mcal
    except ImportError:
        return None


def _is_market_open_now(cfg=None) -> bool:
    """Check if market is currently open. Returns True if unable to determine (conservative)."""
    try:
        import pandas as pd
    except ImportError:
        return True
    mcal = _load_mcal()
    if not mcal:
        return True
    market_calendar = "XNYS"  # Default to NYSE
    if cfg is not None:
        market_calendar = getattr(cfg, "market_calendar", "XNYS")
    cal = mcal.get_calendar(market_calendar)
    # AI-AGENT-REF: use timezone-aware UTC now to avoid naive timestamps
    now = pd.Timestamp.now(tz="UTC")
    schedule = cal.schedule(start_date=now.date(), end_date=now.date())
    if schedule.empty:
        return False
    open_, close_ = (
        schedule.iloc[0]["market_open"],
        schedule.iloc[0]["market_close"],
    )
    return open_ <= now <= close_


# Import emit-once logger for startup banners
from ai_trading.data.universe import (
    load_universe,
)  # AI-AGENT-REF: packaged tickers loader
from ai_trading.indicators import (
    compute_atr as _compute_atr,  # AI-AGENT-REF: fail fast at import-time
)
from ai_trading.logging import (
    info_kv,
    logger_once,
    warning_kv,
)  # AI-AGENT-REF: structured logging helper
from ai_trading.utils.safe_cast import as_float, as_int
from ai_trading.utils.universe import load_universe as load_universe_from_path

from ai_trading.config.settings import (
    MODEL_PATH,
    TICKERS_FILE,
)

# RL: import the trader wrapper from the correct package
try:
    from ai_trading.rl_trading import (
        RLTrader,  # Provides .load() and .predict()  # AI-AGENT-REF: correct RL import path
    )
except ImportError as e:  # noqa: BLE001 - best-effort import; we log below.
    RLTrader = None  # type: ignore
    warning_kv(logger, "RL_IMPORT_FAILED", extra={"detail": str(e)})

logger = get_logger("ai_trading.core.bot_engine")

# AI-AGENT-REF: expose sentiment and Alpaca availability without import side effects
# GOOD: defer until explicitly initialized by runtime code


def maybe_init_brokers() -> None:
    """Initialize clients lazily, only when runtime needs them."""  # AI-AGENT-REF: lazy broker init
    global trading_client, data_client
    if trading_client is not None and data_client is not None:
        return
    if not ALPACA_AVAILABLE:
        return
    # actual initialization occurs elsewhere when Alpaca is available


# Simple cache exposed for tests


def fetch_sentiment(
    symbol_or_ctx, symbol: str | None = None, *, ttl_s: int = 300
) -> float:
    global _SENTIMENT_FAILURES
    """Fetch sentiment score with basic caching and failure tracking."""
    if symbol is None:
        symbol = symbol_or_ctx  # backward compat: first arg was context
    now = time.time()
    cached = _SENTIMENT_CACHE.get(symbol)
    if cached and now - cached[0] < ttl_s:
        return cached[1]
    if _SENTIMENT_FAILURES >= SENTIMENT_FAILURE_THRESHOLD or not SENTIMENT_API_KEY:
        return 0.0
    params = {"symbol": symbol, "apikey": SENTIMENT_API_KEY}
    try:
        # fmt: off
        resp = _HTTP_SESSION.get(
            SENTIMENT_API_URL, params=params, timeout=clamp_request_timeout(HTTP_TIMEOUT)
        )
        # fmt: on
        if resp.status_code in {429, 500, 502, 503, 504}:
            _SENTIMENT_FAILURES += 1
            _SENTIMENT_CACHE[symbol] = (now, 0.0)
            return 0.0
        resp.raise_for_status()
        data = resp.json()
        score = float(data.get("sentiment", 0.0))
        _SENTIMENT_CACHE[symbol] = (now, score)
        return score
    except COMMON_EXC:
        _SENTIMENT_FAILURES += 1
        if _SENTIMENT_FAILURES >= SENTIMENT_FAILURE_THRESHOLD:
            _SENTIMENT_CACHE[symbol] = (now, 0.0)
            return 0.0
        _SENTIMENT_CACHE[symbol] = (now, 0.0)
        return 0.0


def _sha256_file(path: str) -> str:
    """Compute sha256 digest for model artifacts."""  # AI-AGENT-REF: hash for model logging
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


_MODEL_CACHE: Any | None = None


def _load_required_model() -> Any:
    """Load ML model from path or module; fail fast if missing."""  # AI-AGENT-REF: strict model loader
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    path = os.getenv("AI_TRADING_MODEL_PATH")
    modname = os.getenv("AI_TRADING_MODEL_MODULE")

    if path and os.path.isfile(path):
        mdl = joblib.load(path)
        try:
            digest = _sha256_file(path)
        except OSError:  # hashing is best-effort; missing/perm issues shouldn't crash
            digest = "unknown"
        logger.info("MODEL_LOADED", extra={"source": "file", "path": path, "sha": digest})
        _MODEL_CACHE = mdl
        return mdl

    if modname:
        try:
            mod = importlib.import_module(modname)
        except COMMON_EXC as e:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to import AI_TRADING_MODEL_MODULE='{modname}': {e}"
            ) from e
        factory = getattr(mod, "get_model", None) or getattr(mod, "Model", None)
        if not factory:
            raise RuntimeError(
                f"Module '{modname}' missing get_model()/Model() factory."
            )
        mdl = factory() if callable(factory) else factory
        logger.info(
            "MODEL_LOADED",
            extra={
                "source": "module",
                "model_module": modname,
            },  # AI-AGENT-REF: avoid reserved key
        )
        _MODEL_CACHE = mdl
        return mdl

    msg = (
        "Model required but not configured. "
        "Set one of: "
        "AI_TRADING_MODEL_PATH=<abs path to .joblib/.pkl> "
        "or AI_TRADING_MODEL_MODULE=<import.path with get_model()/Model()>."
    )
    logger.error(
        "MODEL_CONFIG_MISSING",
            extra={
                "hint_paths": ["AI_TRADING_MODEL_PATH", "TradingConfig.ml_model_path"],
                "hint_modules": ["AI_TRADING_MODEL_MODULE", "TradingConfig.ml_model_module"],
            },
    )
    raise RuntimeError(msg)


# AI-AGENT-REF: emit-once helper and readiness gate for startup/runtime coordination
_EMITTED_KEYS: set[str] = set()


def _emit_once(logger: logging.Logger, key: str, level: int, msg: str) -> None:
    """Emit log message only once per key."""
    if key in _EMITTED_KEYS:
        return
    _EMITTED_KEYS.add(key)
    logger.log(level, msg)


_RUNTIME_READY: bool = False

# Guard to ensure configuration-loaded log only fires once per reload cycle
_CONFIG_LOGGED: bool = False


def _log_config_loaded() -> None:
    """Log that configuration settings have been loaded."""
    global _CONFIG_LOGGED
    if not _CONFIG_LOGGED:
        logger.info("Config settings loaded, validation deferred to runtime")
        _CONFIG_LOGGED = True


def _reset_config_log() -> None:
    global _CONFIG_LOGGED
    _CONFIG_LOGGED = False


@lru_cache(maxsize=1)
def _get_trading_config() -> TradingConfig:
    """Return cached ``TradingConfig`` and emit log once."""
    cfg = TradingConfig.from_env()
    _log_config_loaded()
    return cfg


def _reload_env(path: str | None = None, *, override: bool = True) -> str | None:
    """Reload environment variables and reset log guard."""
    result = config.reload_env(path, override=override)
    _get_trading_config.cache_clear()
    _reset_config_log()
    return result


def is_runtime_ready() -> bool:
    """Check if runtime context is fully initialized."""
    return _RUNTIME_READY


_HEALTH_CHECK_FAILURES = 0
_HEALTH_CHECK_FAIL_THRESHOLD = 3


def _initialize_bot_context_post_setup(ctx: Any) -> None:
    """
    Optional, non-fatal finishing steps after LazyBotContext builds its services.
    Never raise: any failure logs a warning and returns.
    """
    global _HEALTH_CHECK_FAILURES
    try:
        if "data_source_health_check" in globals() and "REGIME_SYMBOLS" in globals():
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    data_source_health_check(ctx, REGIME_SYMBOLS)  # type: ignore[name-defined]
                    logger.info("Post-setup data source health check completed.")
                    _HEALTH_CHECK_FAILURES = 0
                    break
                except APIError as e:
                    if attempt < max_attempts:
                        logger.warning(
                            "HEALTH_CHECK_FAILED",
                            extra={"cause": "APIError", "attempt": attempt, "detail": str(e)},
                        )
                        time.sleep(attempt)
                        continue
                    _HEALTH_CHECK_FAILURES += 1
                    level = (
                        logger.warning
                        if _HEALTH_CHECK_FAILURES < _HEALTH_CHECK_FAIL_THRESHOLD
                        else logger.error
                    )
                    level(
                        "HEALTH_CHECK_FAILED",
                        extra={
                            "cause": "APIError",
                            "detail": str(e),
                            "failures": _HEALTH_CHECK_FAILURES,
                        },
                    )
                    try:
                        ctx.data_fetcher = data_fetcher_module.build_fetcher(ctx)  # type: ignore[attr-defined]
                    except Exception:
                        pass
        else:
            logger.debug("Post-setup health check not available; skipping.")
    except (
        APIError,
        TimeoutError,
        ConnectionError,
        KeyError,
        ValueError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: tighten health probe error handling
        _HEALTH_CHECK_FAILURES += 1
        level = (
            logger.warning
            if _HEALTH_CHECK_FAILURES < _HEALTH_CHECK_FAIL_THRESHOLD
            else logger.error
        )
        level(
            "HEALTH_CHECK_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e), "failures": _HEALTH_CHECK_FAILURES},
        )


# AI-AGENT-REF: Track regime warnings to avoid spamming logs during market closed
# Using a mutable dict to avoid fragile `global` declarations inside functions.
_REGIME_INSUFFICIENT_DATA_WARNED = {"done": False}
import logging
import os


# AI-AGENT-REF: Memory optimization as optional feature
# (settings will be imported below with other config imports)
def _get_memory_optimization():
    """Initialize memory optimization based on settings."""
    from ai_trading.config.settings import get_settings

    S = get_settings()

    if S.enable_memory_optimization:
        try:
            from ai_trading.utils import (
                memory_optimizer,
            )  # AI-AGENT-REF: stable import path

        # Rate limit for Finnhub (calls/min); resolved at import time via settings
        except COMMON_EXC:

            def memory_profile(func):
                return func

            def optimize_memory():
                return {}

            def emergency_memory_cleanup():
                return {}

            return False, memory_profile, optimize_memory, emergency_memory_cleanup

        def memory_profile(func):
            return func

        def optimize_memory():
            return memory_optimizer.report_memory_use()

        def emergency_memory_cleanup():
            memory_optimizer.enable_low_memory_mode()
            return {}

        return True, memory_profile, optimize_memory, emergency_memory_cleanup

    # Fallback no-op decorators when memory optimization is disabled
    def memory_profile(func):
        return func

    def optimize_memory():
        return {}

    def emergency_memory_cleanup():
        return {}

    return False, memory_profile, optimize_memory, emergency_memory_cleanup


(
    MEMORY_OPTIMIZATION_AVAILABLE,
    memory_profile,
    optimize_memory,
    emergency_memory_cleanup,
) = _get_memory_optimization()


# AI-AGENT-REF: suppress noisy external library warnings
warnings.filterwarnings(
    "ignore", category=SyntaxWarning, message="invalid escape sequence"
)
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")

# Avoid failing under older Python versions during tests

from concurrent.futures import ThreadPoolExecutor, as_completed

from ai_trading import (
    paths,  # AI-AGENT-REF: Runtime paths for proper directory separation
)
from ai_trading.config import management as config
from ai_trading.config import get_settings
from ai_trading.settings import (
    _secret_to_str,
    get_news_api_key,
    get_seed_int,
)  # AI-AGENT-REF: runtime env settings

# Initialize settings once for global use
CFG = get_settings()
# Backward-compat constants for risk thresholds used throughout this module
# AI-AGENT-REF: restored weekly drawdown limit
WEEKLY_DRAWDOWN_LIMIT = getattr(CFG, "weekly_drawdown_limit", 0.10)
# AI-AGENT-REF: cached runtime settings for env aliases
S = CFG
SEED = get_seed_int()  # AI-AGENT-REF: deterministic seed from runtime settings


from ai_trading.market.calendars import ensure_final_bar

# AI-AGENT-REF: Import drawdown circuit breaker for real-time portfolio protection
from ai_trading.risk.circuit_breakers import DrawdownCircuitBreaker
from ai_trading.utils.timefmt import (
    utc_now_iso,  # AI-AGENT-REF: Import UTC timestamp utilities
)

# AI-AGENT-REF: lazy import expensive modules to speed up import for tests
if not os.getenv("PYTEST_RUNNING"):
    from ai_trading.model_loader import ML_MODELS  # AI-AGENT-REF: preloaded models
else:
    # AI-AGENT-REF: mock ML_MODELS for test environments to avoid slow imports
    ML_MODELS = {}


# AI-AGENT-REF: lazy numpy loader for improved import performance
# AI-AGENT-REF: numpy is a hard dependency - import directly
import numpy as np

# AI-AGENT-REF: logger configuration is handled upstream in ``ai_trading.main``
logger = get_logger(__name__)  # AI-AGENT-REF: define logger before use

# AI-AGENT-REF: import sanity signal for CI/ops
info_kv(
    logger,
    "INDICATOR_IMPORT_OK",
    extra={"compute_atr_is_function": bool(inspect.isfunction(_compute_atr))},
)

info_kv(
    logger,
    "RL_IMPORT_OK",
    extra={"available": bool(RLTrader)},
)  # AI-AGENT-REF: RL import self-check


# Handling missing portfolio weights function
def ensure_portfolio_weights(ctx, symbols):
    """Ensure portfolio weights are computed with fallback handling."""
    try:
        from ai_trading import portfolio

        if hasattr(portfolio, "compute_portfolio_weights"):
            return portfolio.compute_portfolio_weights(ctx, symbols)
        else:
            logger.warning("compute_portfolio_weights not found, using fallback method.")
            # Placeholder fallback: Evenly distribute portfolio weights
            return {symbol: 1.0 / len(symbols) for symbol in symbols}
    except (
        ZeroDivisionError,
        ValueError,
        KeyError,
    ) as e:  # AI-AGENT-REF: tighten portfolio sizing errors
        logger.error(
            "PORTFOLIO_WEIGHT_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        return {symbol: 1.0 / len(symbols) for symbol in symbols if symbols}


# Log Alpaca availability on startup (only once per process)
if _alpaca_available():
    _emit_once(logger, "alpaca_available", logging.INFO, "Alpaca SDK is available")
else:  # pragma: no cover - executed only when SDK missing
    _emit_once(
        logger,
        "alpaca_missing",
        logging.WARNING,
        "Alpaca SDK not installed",
    )
# Mirror config to maintain historical constant name
# REMOVED: MIN_CYCLE = CFG.scheduler_sleep_seconds
# AI-AGENT-REF: guard environment validation with explicit error logging
# AI-AGENT-REF: Move config validation to runtime to prevent import crashes
# Config validation moved to init_runtime_config()
# This ensures imports don't fail due to missing environment variables

try:
    # Only import config module, don't validate at import time
    from ai_trading.config.settings import get_settings
    _get_trading_config()
except (
    FileNotFoundError,
    PermissionError,
    IsADirectoryError,
    JSONDecodeError,
    ValueError,
    KeyError,
    TypeError,
    OSError,
) as e:  # AI-AGENT-REF: narrow exception
    logger.warning("Config settings import failed: %s", e)

# Provide a no-op ``profile`` decorator when line_profiler is not active.
try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - used only when kernprof is absent

    def profile(func):  # type: ignore[return-type]
        return func


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.critical(
        "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
    )
    # AI-AGENT-REF: flush and close log handlers to preserve logs on crash
    for h in logging.getLogger().handlers:
        try:
            h.flush()
            h.close()
        except (AttributeError, OSError) as e:
            # Log handler cleanup issues but continue shutdown process
            logger.warning("Failed to close logging handler: %s", e)
    logging.shutdown()


sys.excepthook = handle_exception

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*invalid escape sequence.*",
    category=SyntaxWarning,
    module="pandas_ta.*",
)


from ai_trading import utils

# AI-AGENT-REF: lazy import heavy feature computation modules to speed up import for tests
if not os.getenv("PYTEST_RUNNING"):
    from ai_trading.features.indicators import (
        compute_macd,
        compute_macds,
        compute_vwap,
        ensure_columns,
    )
    from ai_trading.indicators import compute_atr
else:
    # AI-AGENT-REF: mock feature functions for test environments to avoid slow imports
    def compute_macd(*args, **kwargs):
        return [0.0] * 20  # Mock MACD values

    def compute_atr(*args, **kwargs):
        return [1.0] * 20  # Mock ATR values

    def compute_vwap(*args, **kwargs):
        return [100.0] * 20  # Mock VWAP values

    def compute_macds(*args, **kwargs):
        return [0.0] * 20  # Mock MACD signal values

    def ensure_columns(*args, **kwargs):
        return args[0] if args else {}  # Mock column ensurer


try:
    from sklearn.exceptions import InconsistentVersionWarning
except (
    FileNotFoundError,
    PermissionError,
    IsADirectoryError,
    JSONDecodeError,
    ValueError,
    KeyError,
    TypeError,
    OSError,
    ImportError,
):  # pragma: no cover - sklearn optional  # AI-AGENT-REF: narrow exception

    class InconsistentVersionWarning(UserWarning):
        pass


warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings(
    "ignore",
    message="Converting to PeriodArray/Index representation will drop timezone information.*",
    category=UserWarning,
)

import os


# Refresh environment variables on startup for reliability
_reload_env()


# TRADING_MODE must be defined before any classes that reference it
# Define BotMode and a safe default at import time. Runtime may override later.
class BotMode(str, Enum):
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


# Import-time safe default; runtime code may overwrite this
TRADING_MODE = BotMode.BALANCED

import csv
import json
import logging
import random
import re
import signal
import sys
import threading
import time as pytime
from argparse import ArgumentParser
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC
from datetime import datetime as dt_
from datetime import time as dt_time
from threading import Lock, Semaphore, Thread
from zoneinfo import ZoneInfo

random.seed(SEED)
# AI-AGENT-REF: guard numpy random seed for test environments
if hasattr(np, "random"):
    np.random.seed(SEED)

# AI-AGENT-REF: throttle SKIP_COOLDOWN logs
_LAST_SKIP_CD_TIME = 0.0
_LAST_SKIP_SYMBOLS: frozenset[str] = frozenset()


# AI-AGENT-REF: optional heavy dependencies loaded lazily
def _lazy_import_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except COMMON_EXC:
        return False


def _lazy_import_hmmlearn() -> bool:
    try:
        import hmmlearn  # noqa: F401

        return True
    except COMMON_EXC:
        return False


def _seed_torch_if_available(seed: int) -> None:
    try:
        import torch

        torch.manual_seed(int(seed))
    except COMMON_EXC:
        pass


_rl_path = _secret_to_str(getattr(S, "rl_model_path", None))
RL_MODEL_PATH = (
    (Path(BASE_DIR) / _rl_path).resolve() if _rl_path else None
)  # AI-AGENT-REF: resolve RL model path
RL_AGENT: Any | None = None
if getattr(S, "use_rl_agent", False) and RL_MODEL_PATH:
    if RLTrader is not None:
        try:
            rl = RLTrader(RL_MODEL_PATH)
            rl.load()  # load PPO policy from zip path
            RL_AGENT = rl
            info_kv(logger, "RL_AGENT_READY", extra={"model": str(RL_MODEL_PATH)})
        except COMMON_EXC as e:  # noqa: BLE001
            warning_kv(logger, "RL_AGENT_INIT_FAILED", extra={"error": str(e)})
            RL_AGENT = None
    else:
        warning_kv(
            logger,
            "RL_TRADER_UNAVAILABLE",
            extra={"hint": "ai_trading.rl_trading import failed"},
        )

_DEFAULT_FEED = CFG.alpaca_data_feed or "iex"

# Ensure numpy.NaN exists for pandas_ta compatibility
# AI-AGENT-REF: guard numpy.NaN assignment for test environments
if hasattr(np, "nan"):
    np.NaN = np.nan

import pkgutil
from functools import cache


# AI-AGENT-REF: lazy load heavy modules when first accessed
class _LazyModule(types.ModuleType):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._module = None
        self.__name__ = name
        self._failed = False

    def _load(self):
        if self._module is None and not self._failed:
            try:
                self._module = importlib.import_module(self.__name__)
            except COMMON_EXC:
                self._failed = True
                logger.info(
                    f"{self.__name__.upper()}_MISSING",
                    extra={"hint": f"pip install {self.__name__}"},
                )

    def _create_fallback(self):
        """Create a fallback module object with common methods."""

        class FallbackModule:
            def ichimoku(self, *args, **kwargs):
                return pd.DataFrame(), {}

            def rsi(self, *args, **kwargs):
                # Return empty series for RSI
                return pd.Series()

            def atr(self, *args, **kwargs):
                return pd.Series()

            def vwap(self, *args, **kwargs):
                return pd.Series()

            def obv(self, *args, **kwargs):
                return pd.Series()

            def kc(self, *args, **kwargs):
                return pd.DataFrame()

            def bbands(self, *args, **kwargs):
                return pd.DataFrame()

            def adx(self, *args, **kwargs):
                return pd.Series()

            def cci(self, *args, **kwargs):
                return pd.Series()

            def mfi(self, *args, **kwargs):
                return pd.Series()

            def tema(self, *args, **kwargs):
                return pd.Series()

            def willr(self, *args, **kwargs):
                return pd.Series()

            def stochrsi(self, *args, **kwargs):
                return pd.DataFrame()

            def psar(self, *args, **kwargs):
                return pd.Series()

        return FallbackModule()

    def _bind_known_methods(self) -> None:
        """
        Bind a fixed set of known TA methods directly as attributes on this wrapper.
        This removes reliance on __getattr__ magic while preserving behavior.
        """
        self._load()
        target = self._module if self._module is not None else self._create_fallback()

        # List of known pandas_ta methods used in the codebase
        known_methods = [
            "ichimoku",
            "rsi",
            "atr",
            "vwap",
            "obv",
            "kc",
            "bbands",
            "adx",
            "cci",
            "mfi",
            "tema",
            "willr",
            "stochrsi",
            "psar",
        ]

        for method_name in known_methods:
            if hasattr(target, method_name):
                setattr(self, method_name, getattr(target, method_name))
            else:
                # Bind safe no-op if missing on target (future-proof)
                # Use closure to capture method_name correctly
                if method_name == "ichimoku":
                    setattr(
                        self,
                        method_name,
                        (lambda *args, **kwargs: (pd.DataFrame(), {})),
                    )
                elif method_name in ["kc", "bbands", "stochrsi"]:
                    setattr(self, method_name, (lambda *args, **kwargs: pd.DataFrame()))
                else:
                    setattr(self, method_name, (lambda *args, **kwargs: pd.Series()))


# AI-AGENT-REF: use our improved lazy loading instead of _LazyModule for pandas
# pd = _LazyModule("pandas")  # Commented out to use our LazyPandas implementation
ta = _LazyModule("pandas_ta")
# Bind known methods explicitly to avoid __getattr__ magic
ta._bind_known_methods()


def limits(*args, **kwargs):
    def decorator(func):
        def wrapped(*a, **k):
            from ratelimit import limits as _limits

            return _limits(*args, **kwargs)(func)(*a, **k)

        return wrapped

    return decorator


def sleep_and_retry(func):
    def wrapped(*a, **k):
        from ratelimit import sleep_and_retry as _sr

        return _sr(func)(*a, **k)

    return wrapped
# Retry helpers (optional Tenacity)
from ai_trading.utils.retry import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    RetryError,
)

# AI-AGENT-REF: lazy ichimoku setup to avoid pandas_ta import in tests
if not os.getenv("PYTEST_RUNNING"):
    ta.ichimoku = (
        ta.ichimoku if hasattr(ta, "ichimoku") else lambda *a, **k: (pd.DataFrame(), {})
    )
else:
    # AI-AGENT-REF: mock ichimoku for test environments
    def mock_ichimoku(*a, **k):
        return (pd.DataFrame(), {})

    ta.ichimoku = mock_ichimoku

_MARKET_SCHEDULE = None
NY = None  # Lazy-init to avoid optional dependency at import time


def get_market_schedule():
    global _MARKET_SCHEDULE
    cal = get_market_calendar()
    if _MARKET_SCHEDULE is None:
        if hasattr(cal, "schedule"):
            _MARKET_SCHEDULE = cal.schedule(
                start_date="2020-01-01", end_date="2030-12-31"
            )
        else:
            _MARKET_SCHEDULE = pd.DataFrame()
    return _MARKET_SCHEDULE


def get_market_calendar():
    """
    Return a pandas_market_calendars calendar, or a harmless dummy object
    if the optional dependency is unavailable. Memoized via module-global NY.
    """
    global NY
    if NY is not None:
        return NY
    mcal = _load_mcal()
    if not mcal:
        NY = types.SimpleNamespace(
            valid_days=lambda *a, **k: pd.DatetimeIndex([]),
            schedule=pd.DataFrame(),
            open_at_time=lambda *a, **k: False,
            is_session_open=lambda *a, **k: True,
        )
        return NY
    cal_name = "XNYS"
    try:
        NY = mcal.get_calendar(cal_name)
    except COMMON_EXC:
        NY = types.SimpleNamespace(
            valid_days=lambda *a, **k: pd.DatetimeIndex([]),
            schedule=pd.DataFrame(),
            open_at_time=lambda *a, **k: False,
            is_session_open=lambda *a, **k: True,
        )
    return NY


_FULL_DATETIME_RANGE = None


def get_full_datetime_range():
    global _FULL_DATETIME_RANGE
    if _FULL_DATETIME_RANGE is None:
        _FULL_DATETIME_RANGE = pd.date_range(start="09:30", end="16:00", freq="1T")
    return _FULL_DATETIME_RANGE


# AI-AGENT-REF: add simple timeout helper for API calls
@contextmanager
def timeout_protection(seconds: int = 30):
    """
    Context manager to enforce timeouts on operations.

    On CPython the ``signal`` module may only set alarms in the main thread.  When
    invoked from a worker thread (e.g. scheduler jobs), calling ``signal.alarm``
    raises a ``ValueError`` ("signal only works in main thread of the main interpreter").
    This wrapper checks whether it is running in the main thread and only installs
    an alarm in that case.  In all other contexts it simply yields without
    installing an alarm, preventing crashes in threaded environments.
    """
    import threading

    # Only install SIGALRM in the main thread if available
    if threading.current_thread() is threading.main_thread() and hasattr(
        signal, "SIGALRM"
    ):

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            # Disable alarm and restore handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Non-main thread or no SIGALRM support: no-op
        try:
            yield
        finally:
            pass


@cache
def is_holiday(ts: pd.Timestamp) -> bool:
    # Compare only dates, not full timestamps, to handle schedule timezones correctly
    dt = pd.Timestamp(ts).date()
    # Precompute set of valid trading dates (as dates) once
    trading_dates = {d.date() for d in get_market_schedule().index}
    return dt not in trading_dates


# FutureWarning now filtered globally in pytest.ini

# AI-AGENT-REF: portalocker is a hard dependency in pyproject.toml
import portalocker

# Bind HTTPError if available; fall back to generic Exception
try:  # pragma: no cover
    from requests.exceptions import HTTPError  # type: ignore
except ImportError:  # pragma: no cover  # AI-AGENT-REF: optional requests
    HTTPError = Exception

# AI-AGENT-REF: optional schedule dependency
try:  # pragma: no cover - optional dependency
    import schedule  # type: ignore
except ImportError:  # pragma: no cover - schedule may be absent in tests
    import types
    schedule = types.SimpleNamespace()

# AI-AGENT-REF: optional yfinance provider
from ai_trading.data.providers.yfinance_provider import get_yfinance, has_yfinance

YFINANCE_AVAILABLE = has_yfinance()  # AI-AGENT-REF: cached provider availability

# Alpaca SDK classes are imported lazily to keep module import light.  Runtime
# code must call ``_ensure_alpaca_classes()`` before referencing these symbols.
Quote: Any | None = None
Order: Any | None = None
OrderSide: Any | None = None
OrderStatus: Any | None = None
TimeInForce: Any | None = None
MarketOrderRequest: Any | None = None
LimitOrderRequest: Any | None = None
StockLatestQuoteRequest: Any | None = None
_ALPACA_IMPORT_ERROR: Exception | None = None


def _ensure_alpaca_classes() -> None:
    """Import Alpaca SDK types on demand."""
    global Quote, Order, OrderSide, OrderStatus, TimeInForce, MarketOrderRequest, LimitOrderRequest, StockLatestQuoteRequest, _ALPACA_IMPORT_ERROR
    if Quote is not None or _ALPACA_IMPORT_ERROR is not None:
        return
    try:  # pragma: no cover - independent imports with fallbacks
        from alpaca.data.requests import StockLatestQuoteRequest as _StockLatestQuoteRequest
    except Exception:
        from dataclasses import dataclass

        @dataclass
        class _StockLatestQuoteRequest:  # pragma: no cover - lightweight fallback
            symbol_or_symbols: Any
    try:
        from alpaca.trading.requests import (
            MarketOrderRequest as _MarketOrderRequest,
            LimitOrderRequest as _LimitOrderRequest,
        )
    except Exception:
        from dataclasses import dataclass

        @dataclass
        class _MarketOrderRequest:  # pragma: no cover - minimal request object
            symbol: str
            qty: int
            side: Any
            time_in_force: Any
            limit_price: float | None = None
            stop_price: float | None = None
            client_order_id: str | None = None

        @dataclass
        class _LimitOrderRequest:  # pragma: no cover - minimal request object
            symbol: str
            qty: int
            side: Any
            time_in_force: Any
            limit_price: float
            client_order_id: str | None = None
    try:
        from alpaca.trading.enums import (
            OrderSide as _OrderSide,
            OrderStatus as _OrderStatus,
            TimeInForce as _TimeInForce,
        )
    except Exception:
        from enum import Enum

        class _OrderSide(str, Enum):  # pragma: no cover - fallback enum
            BUY = "buy"
            SELL = "sell"

        class _OrderStatus(str, Enum):  # pragma: no cover - fallback enum
            NEW = "new"
            PARTIALLY_FILLED = "partially_filled"
            FILLED = "filled"
            CANCELED = "canceled"
            REJECTED = "rejected"
            EXPIRED = "expired"

        class _TimeInForce(str, Enum):  # pragma: no cover - fallback enum
            DAY = "day"
            GTC = "gtc"
    try:
        from alpaca.data.models import Quote as _Quote  # type: ignore
    except Exception:
        from dataclasses import dataclass

        @dataclass
        class _Quote:  # pragma: no cover - minimal quote stub
            bid_price: float | None = None
            ask_price: float | None = None
    try:
        from alpaca.trading.models import Order as _Order  # type: ignore
    except Exception:
        from dataclasses import dataclass

        @dataclass
        class _Order:  # pragma: no cover - minimal order stub
            id: str | None = None

    Quote = _Quote
    Order = _Order
    OrderSide = _OrderSide
    OrderStatus = _OrderStatus
    TimeInForce = _TimeInForce
    MarketOrderRequest = _MarketOrderRequest
    LimitOrderRequest = _LimitOrderRequest
    StockLatestQuoteRequest = _StockLatestQuoteRequest


# AI-AGENT-REF: beautifulsoup4 is a hard dependency in pyproject.toml
from bs4 import BeautifulSoup

# AI-AGENT-REF: flask is a hard dependency in pyproject.toml
from flask import Flask

# AI-AGENT-REF: lazy import to avoid import-time races and optional deps
def _alpaca_symbols():
    from ai_trading.alpaca_api import (
        alpaca_get as _alpaca_get,
        start_trade_updates_stream as _start,
    )
    return _alpaca_get, _start

if ALPACA_AVAILABLE:
    from ai_trading.rebalancer import (
        maybe_rebalance as original_rebalance,  # type: ignore
    )
else:  # pragma: no cover - no rebalance without Alpaca

    def original_rebalance(*args, **kwargs):
        return None


import pickle

# AI-AGENT-REF: Optional meta-learning — do not crash if unavailable
if not os.getenv("PYTEST_RUNNING") and ALPACA_AVAILABLE:
    try:
        from ai_trading.meta_learning import optimize_signals  # type: ignore
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as _e:  # AI-AGENT-REF: narrow exception
        logger.warning(
            "Meta-learning unavailable (%s); proceeding without signal optimization", _e
        )

        def optimize_signals(signals, *a, **k):  # type: ignore[no-redef]
            return signals

else:
    # AI-AGENT-REF: mock optimize_signals for test environments
    def optimize_signals(*args, **kwargs):
        return args[0] if args else []  # Return signals as-is


# AI-AGENT-REF: late import for model pipeline
def _import_model_pipeline():  # AI-AGENT-REF: import helper for tests
    try:  # pragma: no cover - import path resolution
        from ai_trading.pipeline import model_pipeline  # type: ignore

        return model_pipeline
    except ImportError as e:  # pragma: no cover  # AI-AGENT-REF: narrow import
        logger.error("model_pipeline import failed: %s", e)
        raise


# ML dependencies - sklearn is a hard dependency

from ai_trading.utils import log_warning, model_lock, safe_to_datetime, validate_ohlcv

# ai_trading/core/bot_engine.py:670 - Move retrain_meta_learner import to lazy location

validate_alpaca_credentials = getattr(config, "validate_alpaca_credentials", None)
TRADING_MODE_ENV = getattr(config, "TRADING_MODE", TRADING_MODE)
RUN_HEALTHCHECK = getattr(config, "RUN_HEALTHCHECK", None)


def _require_cfg(value: str | None, name: str) -> str:
    """Return value or load from config, retrying in production."""
    if value:
        return value

    # In testing mode, return a dummy value
    if CFG.testing:
        dummy_values = {
            "ALPACA_API_KEY": "test_api_key",
            "ALPACA_SECRET_KEY": "test_secret_key",
            "TRADING_MODE": "test",
        }
        return dummy_values.get(name, f"test_{name.lower()}")

    if TRADING_MODE_ENV == "production":
        while not value:
            logger.critical("Missing %s; retrying in 60s", name)
            time.sleep(60)
            _reload_env()
            import importlib

            importlib.reload(config)
            value = getattr(config, name, None)
        return str(value)
    raise RuntimeError(f"{name} must be defined in the configuration or environment")


# Defer credential checks to runtime (avoid import-time crashes before .env loads)
def _resolve_alpaca_env():
    # AI-AGENT-REF: ensure .env is loaded before resolving Alpaca environment
    try:
        from ai_trading.env import ensure_dotenv_loaded
        ensure_dotenv_loaded()
    except COMMON_EXC:
        pass
    key = cast(str | None, config.get_env("ALPACA_API_KEY"))
    secret = cast(str | None, config.get_env("ALPACA_SECRET_KEY"))
    base_url = cast(
        str,
        config.get_env(
            "ALPACA_BASE_URL",
            "https://paper-api.alpaca.markets",
        ),
    )
    return key, secret, base_url


def _ensure_alpaca_env_or_raise():
    """
    Contract: always return a (key, secret, base_url) tuple.
    - In SHADOW_MODE, we do not raise; we still return whatever is currently resolved.
    - Outside SHADOW_MODE, if missing key/secret, raise with a clear message.
    """
    k, s, b = _resolve_alpaca_env()
    # Check for shadow mode
    shadow_mode = is_shadow_mode()
    if shadow_mode:
        return k, s, b
    if not (k and s):
        logger.critical("Alpaca credentials missing – aborting client initialization")
        raise RuntimeError("Missing Alpaca API credentials")
    return k, s, b


def init_runtime_config():
    """Initialize runtime configuration and validate critical keys."""
    from ai_trading.config import get_settings

    cfg = get_settings()

    # Validate critical keys at runtime, not import time
    global ALPACA_API_KEY, ALPACA_SECRET_KEY, TRADING_MODE_ENV

    # Use the new credential resolution functions
    try:
        ALPACA_API_KEY, ALPACA_SECRET_KEY, _ = _ensure_alpaca_env_or_raise()
    except RuntimeError as e:
        if not CFG.testing:  # Allow missing credentials in test mode
            raise e
        # AI-AGENT-REF: Use environment variables even in test mode to avoid hardcoded secrets
        ALPACA_API_KEY = os.getenv("TEST_ALPACA_API_KEY", "")
        ALPACA_SECRET_KEY = os.getenv("TEST_ALPACA_SECRET_KEY", "")

    TRADING_MODE_ENV = _require_cfg(getattr(cfg, "TRADING_MODE", None), "TRADING_MODE")

    if not callable(validate_alpaca_credentials):
        raise RuntimeError("validate_alpaca_credentials not found in config")

    logger.info(
        "Runtime config initialized",
        extra={
            "alpaca_key_set": bool(ALPACA_API_KEY and len(ALPACA_API_KEY) > 8),
            "trading_mode": TRADING_MODE_ENV,
        },
    )
    return cfg


# Set module-level defaults that won't crash on import
ALPACA_API_KEY = None
ALPACA_SECRET_KEY = None
TRADING_MODE_ENV = "development"

# AI-AGENT-REF: optional pybreaker dependency
try:  # pragma: no cover - optional dependency
    import pybreaker  # type: ignore
except ImportError:  # pragma: no cover - fallback  # AI-AGENT-REF: optional pybreaker

    class pybreaker:  # type: ignore
            class CircuitBreaker:
                def __init__(self, *args, **kwargs):
                    pass
                def call(self, func):
                    def _wrapped(*a, **kw):
                        return func(*a, **kw)

                    return _wrapped

                def __call__(self, func):
                    return self.call(func)


# AI-AGENT-REF: optional prometheus_client dependency via adapter
from ai_trading.metrics import (
    PROMETHEUS_AVAILABLE,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    Summary,
    start_http_server,
)

# Prometheus metrics - lazy initialization to prevent duplicates
_METRICS_READY = False


def _init_metrics() -> None:
    """Create/register metrics once; tolerate partial imports & re-imports."""
    global _METRICS_READY, orders_total, order_failures, daily_drawdown, signals_evaluated
    global run_all_trades_duration, minute_cache_hit, minute_cache_miss, daily_cache_hit
    global daily_cache_miss, event_cooldown_hits, slippage_total, slippage_count
    global weekly_drawdown, skipped_duplicates, skipped_cooldown
    if _METRICS_READY:
        return
    try:
        orders_total = Counter("bot_orders_total", "Total orders sent")
        order_failures = Counter("bot_order_failures", "Order submission failures")
        daily_drawdown = Gauge("bot_daily_drawdown", "Current daily drawdown fraction")
        signals_evaluated = Counter(
            "bot_signals_evaluated_total", "Total signals evaluated"
        )
        run_all_trades_duration = Histogram(
            "run_all_trades_duration_seconds", "Time spent in run_all_trades"
        )
        minute_cache_hit = Counter("bot_minute_cache_hits", "Minute bar cache hits")
        minute_cache_miss = Counter(
            "bot_minute_cache_misses", "Minute bar cache misses"
        )
        daily_cache_hit = Counter("bot_daily_cache_hits", "Daily bar cache hits")
        daily_cache_miss = Counter("bot_daily_cache_misses", "Daily bar cache misses")
        event_cooldown_hits = Counter("bot_event_cooldown_hits", "Event cooldown hits")
        slippage_total = Counter("bot_slippage_total", "Cumulative slippage in cents")
        slippage_count = Counter(
            "bot_slippage_count", "Number of orders with slippage logged"
        )
        weekly_drawdown = Gauge(
            "bot_weekly_drawdown", "Current weekly drawdown fraction"
        )
        skipped_duplicates = Counter(
            "bot_skipped_duplicates",
            "Trades skipped due to open position",
        )
        skipped_cooldown = Counter(
            "bot_skipped_cooldown",
            "Trades skipped due to recent execution",
        )
    except ValueError:
        # Already registered (e.g., prior partial import). Reuse existing.
        # Accessing REGISTRY internals is stable in prometheus-client; safe fallback.
        if REGISTRY is not None:
            existing = getattr(REGISTRY, "_names_to_collectors", {})
            orders_total = existing.get("bot_orders_total")
            order_failures = existing.get("bot_order_failures")
            daily_drawdown = existing.get("bot_daily_drawdown")
            signals_evaluated = existing.get("bot_signals_evaluated_total")
            run_all_trades_duration = existing.get("run_all_trades_duration_seconds")
            minute_cache_hit = existing.get("bot_minute_cache_hits")
            minute_cache_miss = existing.get("bot_minute_cache_misses")
            daily_cache_hit = existing.get("bot_daily_cache_hits")
            daily_cache_miss = existing.get("bot_daily_cache_misses")
            event_cooldown_hits = existing.get("bot_event_cooldown_hits")
            slippage_total = existing.get("bot_slippage_total")
            slippage_count = existing.get("bot_slippage_count")
            weekly_drawdown = existing.get("bot_weekly_drawdown")
            skipped_duplicates = existing.get("bot_skipped_duplicates")
            skipped_cooldown = existing.get("bot_skipped_cooldown")
    _METRICS_READY = True


try:
    from ai_trading.execution import (
        ExecutionEngine,  # canonical import  # AI-AGENT-REF: fix ExecutionEngine import
    )
except ImportError:  # pragma: no cover - allow tests with stubbed module
    from ai_trading.core.enums import OrderSide

    class ExecutionEngine:
        """
        Fallback execution engine used when the real trade_execution module is
        unavailable.  Many parts of the trading logic expect an execution
        engine exposing ``execute_order`` as well as ``start_cycle`` and
        ``end_cycle`` hooks.  Without these methods the bot would raise
        AttributeError.  This stub logs invocation of each method and returns
        dummy order objects.
        """

        def __init__(self, *args, **kwargs) -> None:
            # Provide a logger specific to the stub
            self._logger = get_logger(__name__ + ".StubExecutionEngine")

        def logger(self, method: str, *args, **kwargs) -> None:
            self._logger.debug(
                "StubExecutionEngine.%s called with args=%s kwargs=%s",
                method,
                args,
                kwargs,
            )

        def execute_order(self, symbol: str, side: OrderSide, qty: int):
            """Simulate an order execution and return a dummy order object."""
            self.logger("execute_order", symbol, side, qty)
            # Return a simple namespace with an id attribute to mimic a real order
            return types.SimpleNamespace(id=None)

        # Provide empty hooks for cycle management used elsewhere in the code
        def start_cycle(self) -> None:
            self.logger("start_cycle")

        def end_cycle(self) -> None:
            self.logger("end_cycle")

        def check_trailing_stops(self) -> None:
            """Stub method for trailing stops check - used when real execution engine unavailable."""
            self.logger("check_trailing_stops")


try:
    from ai_trading.capital_scaling import CapitalScalingEngine
except (
    FileNotFoundError,
    PermissionError,
    IsADirectoryError,
    JSONDecodeError,
    ValueError,
    KeyError,
    TypeError,
    OSError,
):  # pragma: no cover - allow tests with stubbed module  # AI-AGENT-REF: narrow exception

    class CapitalScalingEngine:
        def __init__(self, *args, **kwargs):
            pass

        def scale_position(self, position):
            """Return ``position`` unchanged for smoke tests."""
            # AI-AGENT-REF: stub passthrough for unit tests
            return position

        def update(self, *args, **kwargs):  # AI-AGENT-REF: add missing update method
            """Update method for test compatibility."""


class StrategyAllocator:
    def __init__(self, *args, **kwargs):
        # Resolve StrategyAllocator from in-package modules only
        from ai_trading.utils.imports import resolve_strategy_allocator_cls

        cls = resolve_strategy_allocator_cls()
        if cls is None:
            raise RuntimeError(
                "StrategyAllocator not found. Please ensure that either "
                "ai_trading.strategy_allocator or ai_trading.strategies.performance_allocator is available."
            )
        self._alloc = cls(*args, **kwargs)

    def allocate_signals(self, *args, **kwargs):
        return self._alloc.allocate(*args, **kwargs)


# AI-AGENT-REF: lazy import heavy data_fetcher module to speed up import for tests
finnhub_client = None

# AI-AGENT-REF: Add cache size management to prevent memory leaks
_ML_MODEL_CACHE: dict[str, Any] = {}
_ML_MODEL_CACHE_MAX_SIZE = 100  # Limit cache size to prevent memory issues


def _cleanup_ml_model_cache():
    """Clean up ML model cache if it gets too large."""
    global _ML_MODEL_CACHE
    if len(_ML_MODEL_CACHE) > _ML_MODEL_CACHE_MAX_SIZE:
        # Keep only the most recently used items (simple LRU-like behavior)
        # For now, just clear half the cache when it gets too large
        items_to_remove = len(_ML_MODEL_CACHE) // 2
        keys_to_remove = list(_ML_MODEL_CACHE.keys())[:items_to_remove]
        for key in keys_to_remove:
            _ML_MODEL_CACHE.pop(key, None)
        logger.info("Cleaned up ML model cache, removed %d items", items_to_remove)


logger = get_logger(__name__)


# AI-AGENT-REF: helper for throttled SKIP_COOLDOWN logging
def log_skip_cooldown(
    symbols: Sequence[str] | str, state: BotState | None = None
) -> None:
    """Log SKIP_COOLDOWN once per unique set within 15 seconds."""
    global _LAST_SKIP_CD_TIME, _LAST_SKIP_SYMBOLS
    now = time.monotonic()
    sym_set = frozenset([symbols]) if isinstance(symbols, str) else frozenset(symbols)
    if sym_set != _LAST_SKIP_SYMBOLS or now - _LAST_SKIP_CD_TIME >= 15:
        logger.info("SKIP_COOLDOWN | %s", ", ".join(sorted(sym_set)))
        _LAST_SKIP_CD_TIME = now
        _LAST_SKIP_SYMBOLS = sym_set


def market_is_open(now: datetime | None = None) -> bool:
    from ai_trading.utils import is_market_open as utils_market_open

    """Return True if the market is currently open."""
    try:
        with timeout_protection(10):
            if os.getenv("FORCE_MARKET_OPEN", "false").lower() == "true":
                logger.info(
                    "FORCE_MARKET_OPEN is enabled; overriding market hours checks."
                )
                return True
            return utils_market_open(now)
    except TimeoutError:
        logger.error("Market status check timed out, assuming market closed")
        return False
    except (
        ImportError,
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.error("Market status check failed: %s", e)
        return False


# backward compatibility
is_market_open = market_is_open


# AI-AGENT-REF: snapshot live positions for debugging
PORTFOLIO_FILE = "portfolio_snapshot.json"


def save_portfolio_snapshot(
    portfolio: dict[str, int],
    now_provider: Callable[[], datetime] | None = None,
) -> None:
    from datetime import UTC, datetime

    from ai_trading.utils.timefmt import utc_now_iso_from

    now_fn = now_provider or (lambda: datetime.now(UTC))
    data = {
        "timestamp": utc_now_iso_from(now_fn()),  # AI-AGENT-REF: injectable clock
        "positions": portfolio,
    }
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_portfolio_snapshot() -> dict[str, int]:
    if not os.path.exists(PORTFOLIO_FILE):
        return {}
    with open(PORTFOLIO_FILE) as f:
        data = json.load(f)
    return data.get("positions", {})


def compute_current_positions(ctx: BotContext) -> dict[str, int]:
    try:
        if hasattr(ctx.api, "list_positions"):
            positions = ctx.api.list_positions()
        elif hasattr(ctx.api, "get_all_positions"):
            positions = ctx.api.get_all_positions()
        else:
            return {}
        logger.debug("Raw Alpaca positions: %s", positions)
        return {p.symbol: int(p.qty) for p in positions}
    except (AttributeError, ValueError, ConnectionError, TimeoutError) as e:
        logger.warning("compute_current_positions failed: %s", e, exc_info=True)
        return {}


def maybe_rebalance(ctx):
    portfolio = compute_current_positions(ctx)
    save_portfolio_snapshot(portfolio)
    return original_rebalance(ctx)


def get_latest_close(df: pd.DataFrame) -> float:
    """Return the last closing price or ``0.0`` if unavailable."""
    # AI-AGENT-REF: debug output to understand test failure
    logger.debug("get_latest_close called with df: %s", type(df).__name__)

    # AI-AGENT-REF: More robust check that works with different pandas instances
    if df is None:
        logger.debug("get_latest_close early return: df is None")
        return 0.0

    # Check if df has empty attribute and columns attribute (duck typing)
    try:
        is_empty = df.empty
        has_close = "close" in df.columns
    except (AttributeError, TypeError) as e:
        logger.debug("get_latest_close: DataFrame methods failed: %s", e)
        return 0.0

    if is_empty or not has_close:
        logger.debug(
            "get_latest_close early return: empty: %s, close in columns: %s",
            is_empty,
            has_close,
        )
        return 0.0

    try:
        last_valid_close = df["close"].dropna()
        logger.debug(
            "get_latest_close last_valid_close length: %d", len(last_valid_close)
        )

        if not last_valid_close.empty:
            price = last_valid_close.iloc[-1]
            logger.debug(
                "get_latest_close price from iloc[-1]: %s (type: %s)",
                price,
                type(price).__name__,
            )
        else:
            logger.critical("All NaNs in close column for get_latest_close")
            price = 0.0

        # More robust NaN check that works with different pandas instances
        if price is None or (hasattr(price, "__ne__") and price != price) or price <= 0:
            logger.debug("get_latest_close price is NaN or <= 0: price=%s", price)
            return 0.0

        result = float(price)
        logger.debug("get_latest_close returning: %s", result)
        return result

    except (
        ImportError,
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.warning("get_latest_close exception: %s", e)
        return 0.0


def compute_time_range(minutes: int) -> tuple[datetime, datetime]:
    """Return a UTC datetime range spanning the past ``minutes`` minutes."""
    # AI-AGENT-REF: provide timezone-aware datetimes
    now = datetime.now(UTC)
    start = now - timedelta(minutes=minutes)
    return start, now


def safe_price(price: float) -> float:
    """Defensively clamp ``price`` to a minimal positive value."""
    # AI-AGENT-REF: prevent invalid zero/negative prices
    return max(price, 1e-3)


# AI-AGENT-REF: utility to detect row drops during feature engineering
def assert_row_integrity(
    before_len: int, after_len: int, func_name: str, symbol: str
) -> None:
    if after_len < before_len:
        logger.warning(
            f"Row count dropped in {func_name} for {symbol}: {before_len} -> {after_len}"
        )


def _load_ml_model(symbol: str):
    """Return preloaded ML model from ``ML_MODELS`` cache."""

    # AI-AGENT-REF: Check cache size and cleanup if needed
    _cleanup_ml_model_cache()

    cached = _ML_MODEL_CACHE.get(symbol)
    if cached is not None:
        return cached

    model = ML_MODELS.get(symbol)
    if model is not None:
        _ML_MODEL_CACHE[symbol] = model
    return model


def fetch_minute_df_safe(symbol: str) -> pd.DataFrame:
    """Fetch the last day of minute bars and raise on empty."""
    # AI-AGENT-REF: raise on empty DataFrame
    now_utc = datetime.now(UTC)
    start_dt = now_utc - timedelta(days=1)

    # AI-AGENT-REF: Cache wrapper (optional around fetch)
    if hasattr(CFG, "market_cache_enabled") and CFG.market_cache_enabled:
        try:
            from ai_trading.market.cache import get_or_load as _get_or_load

            cache_key = f"minute:{symbol}:{start_dt.isoformat()}"
            df = _get_or_load(
                key=cache_key,
                loader=lambda: get_minute_df(symbol, start_dt, now_utc),
                ttl=getattr(S, "market_cache_ttl", 900),
            )
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.debug("Cache layer unavailable/failed: %s", e)
            df = get_minute_df(symbol, start_dt, now_utc)
    else:
        df = get_minute_df(symbol, start_dt, now_utc)

    if df.empty:
        msg = "Minute bars DataFrame is empty after fallbacks; market likely closed"  # AI-AGENT-REF
        logger.warning(
            "FETCH_MINUTE_EMPTY",
            extra={"reason": "empty", "context": "market_closed"},
        )
        raise DataFetchError(msg)

    # Check data freshness before proceeding with trading logic
    try:
        # Allow data up to 10 minutes old during market hours (600 seconds)
        _ensure_data_fresh(df, 600, symbol=symbol)
    except RuntimeError as e:
        logger.warning(f"Data staleness check failed for {symbol}: {e}")
        # Still return the data but log the staleness issue

    return df


def cancel_all_open_orders(runtime) -> None:
    """
    On startup or each run, cancel every Alpaca order whose status is 'open'.
    """
    if runtime.api is None:
        logger.warning("runtime.api is None - cannot cancel orders")
        return
    if not _validate_trading_api(runtime.api):
        return

    try:
        open_orders = list_open_orders(runtime.api)
        if not open_orders:
            return
        for od in open_orders:
            if getattr(od, "status", "").lower() == "open":
                try:
                    runtime.api.cancel_order(od.id)
                except APIError as exc:
                    # AI-AGENT-REF: narrow Alpaca API exceptions
                    logger.exception(
                        "Failed to cancel order %s",
                        getattr(od, "id", "unknown"),
                        exc_info=exc,
                        extra={"cause": exc.__class__.__name__},
                    )
    except APIError as exc:
        logger.warning(
            "Failed to cancel open orders: %s",
            exc,
            exc_info=True,
            extra={"cause": exc.__class__.__name__},
        )


def reconcile_positions(ctx: BotContext) -> None:
    """On startup, fetch all live positions and clear any in-memory stop/take targets for assets no longer held."""
    try:
        live_positions = {
            pos.symbol: int(pos.qty) for pos in ctx.api.list_positions()
        }
        with targets_lock:
            symbols_with_targets = list(ctx.stop_targets.keys()) + list(
                ctx.take_profit_targets.keys()
            )
            for symbol in symbols_with_targets:
                if symbol not in live_positions or live_positions[symbol] == 0:
                    ctx.stop_targets.pop(symbol, None)
                    ctx.take_profit_targets.pop(symbol, None)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        logger.exception("reconcile_positions failed", exc_info=exc)


import warnings

# ─── A. CONFIGURATION CONSTANTS ─────────────────────────────────────────────────
RUN_HEALTH = RUN_HEALTHCHECK == "1"

# Logging: set root logger to INFO, send to both stderr and a log file
get_logger("alpaca").setLevel(logging.WARNING)
get_logger("urllib3").setLevel(logging.WARNING)
get_logger("requests").setLevel(logging.WARNING)

# Suppress specific pandas_ta warnings
warnings.filterwarnings(
    "ignore", message=".*valid feature names.*", category=UserWarning
)

# ─── FINBERT SENTIMENT MODEL: LAZY SINGLETON LOADER ─────────────────────────────────
# FinBERT: lazy singleton loader to avoid startup RAM spike
import importlib.util

_finbert_tokenizer = None
_finbert_model = None


def ensure_finbert(cfg=None):
    """
    Load FinBERT on first use, if enabled in config.
    Returns (tokenizer, model) or (None, None) if disabled/unavailable.
    """
    global _finbert_tokenizer, _finbert_model
    # If already loaded (or deliberately disabled), short-circuit
    if _finbert_tokenizer is not None and _finbert_model is not None:
        return _finbert_tokenizer, _finbert_model
    try:
        # config gate: disable by default on low-RAM machines
        if cfg is not None:
            enabled = bool(getattr(cfg, "enable_finbert", False))
        else:
            # best-effort: try to read a default TradingConfig if none provided
            try:
                from ai_trading.config.management import TradingConfig

                enabled = bool(
                    getattr(TradingConfig.from_env(), "enable_finbert", False)
                )
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ):  # AI-AGENT-REF: narrow exception
                enabled = False
        if not enabled:
            _log_finbert_disabled()  # AI-AGENT-REF: reduce log spam
            return None, None

        # dependency presence check without ImportError guards
        if (
            importlib.util.find_spec("transformers") is None
            or importlib.util.find_spec("torch") is None
        ):
            logger.warning(
                "FinBERT requested but transformers/torch not installed; returning neutral sentiment."
            )
            return None, None

        import torch  # type: ignore
        import transformers  # type: ignore

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*_register_pytree_node.*",
                module="transformers.*",
            )
            tok = transformers.AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
            mdl = transformers.AutoModelForSequenceClassification.from_pretrained(
                "yiyanghkust/finbert-tone"
            )
            mdl.eval()
        _finbert_tokenizer, _finbert_model = tok, mdl
        _emit_once(logger, "finbert_loaded", logging.INFO, "FinBERT loaded successfully")
        return _finbert_tokenizer, _finbert_model
    except (
        ImportError,
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.error("FinBERT lazy-load failed: %s", e)
        return None, None


# REMOVED: module-scope get_disaster_dd_limit() = CFG.disaster_dd_limit
# Paths


def abspath(fname: str) -> str:
    """Return absolute path for model/flag files."""  # AI-AGENT-REF: prevent NoneType
    return str((Path(BASE_DIR) / str(fname)).resolve())


# AI-AGENT-REF: safe ML model path resolution
DEFAULT_MODEL_PATH = abspath_safe("trained_model.pkl")
env_model = os.getenv("AI_TRADING_MODEL_PATH")
MODEL_PATH = abspath_safe(env_model or getattr(S, "model_path", None))
if MODEL_PATH and os.path.exists(MODEL_PATH):
    USE_ML = True
elif MODEL_PATH and os.path.abspath(MODEL_PATH) != DEFAULT_MODEL_PATH:
    logger.warning("ML_MODEL_MISSING", extra={"path": MODEL_PATH})
    USE_ML = False
else:  # default path missing - no model required
    USE_ML = False

info_kv(
    logger,
    "RUNTIME_SETTINGS_RESOLVED",
    extra={
        "seed": getattr(S, "ai_trading_seed", 42),
        "model_path": MODEL_PATH or "",
        "interval_hint": "main.py",
    },
)


def abspath_repo_root(fname: str) -> str:
    """Path relative to repository root."""
    return str((Path(BASE_DIR) / fname).resolve())


def atomic_joblib_dump(obj, path: str) -> None:
    """Safely write joblib file using atomic replace."""
    import tempfile

    import joblib

    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def atomic_pickle_dump(obj, path: str) -> None:
    """Safely pickle object to path with atomic replace."""
    import tempfile

    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def get_git_hash() -> str:
    """Return current git commit short hash if available."""
    try:
        from ai_trading.utils import SUBPROCESS_TIMEOUT_DEFAULT, safe_subprocess_run

        cmd = ["git", "rev-parse", "--short", "HEAD"]
        out = safe_subprocess_run(cmd, timeout=SUBPROCESS_TIMEOUT_DEFAULT)
        return out.strip() or "unknown"
    except COMMON_EXC:
        return "unknown"


# AI-AGENT-REF: use centralized trade log path
TRADE_LOG_FILE = default_trade_log_path()
SIGNAL_WEIGHTS_FILE = str(paths.DATA_DIR / "signal_weights.csv")
EQUITY_FILE = str(paths.DATA_DIR / "last_equity.txt")
PEAK_EQUITY_FILE = str(paths.DATA_DIR / "peak_equity.txt")
HALT_FLAG_PATH = abspath(getattr(S, "halt_flag_path", "halt.flag"))  # AI-AGENT-REF: absolute halt flag path
SLIPPAGE_LOG_FILE = str(paths.LOG_DIR / "slippage.csv")
REWARD_LOG_FILE = str(paths.LOG_DIR / "reward_log.csv")
FEATURE_PERF_FILE = abspath_safe("feature_perf.csv")
INACTIVE_FEATURES_FILE = abspath_safe("inactive_features.json")

# Hyperparameter files (repo root, not core/)
HYPERPARAMS_FILE = abspath_repo_root("hyperparams.json")
BEST_HYPERPARAMS_FILE = abspath_repo_root("best_hyperparams.json")


def load_hyperparams() -> dict:
    """Load hyperparameters from best_hyperparams.json if present, else default."""
    path = (
        BEST_HYPERPARAMS_FILE
        if os.path.exists(BEST_HYPERPARAMS_FILE)
        else HYPERPARAMS_FILE
    )
    if not os.path.exists(path):
        logger.warning(f"Hyperparameter file {path} not found; using defaults")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load hyperparameters from %s: %s", path, exc)
        return {}


def _maybe_warm_cache(ctx: BotContext) -> None:
    """
    Warm up cache for the main universe symbols (daily + optional intraday).
    """
    settings = get_settings()
    if not getattr(settings, "data_cache_enable", False):
        return
    try:
        # Daily warm-up via batched fetch
        end_dt = datetime.now(UTC)
        start_dt = end_dt - timedelta(days=int(settings.data_warmup_lookback_days))
        if ctx.symbols:
            get_bars_batch(
                ctx.symbols,
                "1D",
                start_dt,
                end_dt,
                feed=getattr(ctx, "data_feed", None),
            )
        # Optional intraday warm-up
        if getattr(settings, "intraday_batch_enable", True):
            end_dt = datetime.now(UTC)
            start_dt = end_dt - timedelta(
                minutes=int(settings.intraday_lookback_minutes)
            )
            _fetch_intraday_bars_chunked(
                ctx.symbols,
                start=start_dt,
                end=end_dt,
                feed=getattr(ctx, "data_feed", None),
            )
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        logger.warning("Cache warm-up failed: %s", exc)


def _fetch_universe_bars(
    symbols: list[str],
    timeframe: str,
    start: datetime | str,
    end: datetime | str,
    feed: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch bars using batch API with safe fallback."""  # AI-AGENT-REF
    try:
        batch = get_bars_batch(symbols, timeframe, start, end, feed=feed)
        if isinstance(batch, dict):
            return batch
    except COMMON_EXC:
        pass
    out: Dict[str, pd.DataFrame] = {}

    # Fallback to per-symbol requests with bounded concurrency.
    settings = get_settings()
    max_workers = max(1, int(getattr(settings, "batch_fallback_workers", 4)))

    def _pull(sym: str) -> tuple[str, pd.DataFrame | None]:
        try:
            return sym, get_bars(sym, timeframe, start, end, feed=feed)
        except COMMON_EXC:
            return sym, None

    results: list[tuple[str, pd.DataFrame | None]] = []
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="fallback-bars"
    ) as ex:
        futures = [ex.submit(_pull, s) for s in symbols]
        for fut in as_completed(futures):
            results.append(fut.result())

    for sym, df in results:
        if df is not None and not getattr(df, "empty", False):
            out[sym] = df
    return out


def _fetch_universe_bars_chunked(
    symbols: list[str],
    timeframe: str,
    start: datetime | str,
    end: datetime | str,
    feed: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Chunked batched fetch for universe bars with safe fallback.
    """
    if not symbols:
        return {}
    settings = get_settings()
    batch_size = max(1, int(getattr(settings, "pretrade_batch_size", 50)))
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i : i + batch_size]
        out.update(_fetch_universe_bars(chunk, timeframe, start, end, feed))
    total_symbols = len(out)
    try:
        bars_loaded = sum(len(v) for v in out.values())
    except (TypeError, AttributeError):  # AI-AGENT-REF: bars summary fallback
        bars_loaded = 0
    first_symbol = next(iter(out.keys()), None)
    logger.info(
        "FETCH_SUMMARY",
        extra={
            "total_symbols": total_symbols,
            "bars_loaded": bars_loaded,
            "first_symbol": first_symbol,
        },
    )
    return out


def _fetch_intraday_bars_chunked(
    symbols: list[str],
    start: datetime | str,
    end: datetime | str,
    feed: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Chunked batched fetch for 1-minute bars with safe fallback.
    """
    if not symbols:
        return {}

    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)
    if end_dt <= start_dt:
        raise ValueError("end must be after start")

    max_span = timedelta(days=8)

    def _time_slices() -> Iterable[tuple[datetime, datetime]]:
        cur = start_dt
        while cur < end_dt:
            nxt = min(cur + max_span, end_dt)
            yield cur, nxt
            cur = nxt

    settings = get_settings()
    batch_size = max(1, int(getattr(settings, "intraday_batch_size", 40)))

    def _fetch_symbol(sym: str) -> pd.DataFrame:
        frames = []
        for s_dt, e_dt in _time_slices():
            if (e_dt - s_dt) > max_span:
                raise ValueError("time slice exceeds 8-day limit")
            df = get_minute_df(sym, s_dt, e_dt, feed=feed)
            if df is not None and not getattr(df, "empty", False):
                frames.append(df)
        return pd.concat(frames) if frames else pd.DataFrame()

    if not getattr(settings, "intraday_batch_enable", True):
        return {s: _fetch_symbol(s) for s in symbols}

    out: dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in symbols}
    for s_dt, e_dt in _time_slices():
        for i in range(0, len(symbols), batch_size):
            chunk = symbols[i : i + batch_size]
            try:
                got = get_bars_batch(chunk, str(_bars.TimeFrame.Minute), s_dt, e_dt, feed=feed)
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as exc:  # AI-AGENT-REF: narrow exception
                logger.warning(
                    "Intraday batch failed for chunk size %d: %s; falling back",
                    len(chunk),
                    exc,
                )
                got = {}
            missing = [
                s
                for s in chunk
                if s not in got or got.get(s) is None or getattr(got.get(s), "empty", False)
            ]
            if missing:
                max_workers = max(1, int(getattr(settings, "batch_fallback_workers", 4)))

                def _pull(sym: str):
                    try:
                        return sym, get_minute_df(sym, s_dt, e_dt, feed=feed)
                    except (
                        FileNotFoundError,
                        PermissionError,
                        IsADirectoryError,
                        JSONDecodeError,
                        ValueError,
                        KeyError,
                        TypeError,
                        OSError,
                    ) as one_exc:  # AI-AGENT-REF: narrow exception
                        logger.warning(
                            "Intraday per-symbol fallback failed for %s: %s", sym, one_exc
                        )
                        return sym, None

                with ThreadPoolExecutor(
                    max_workers=max_workers, thread_name_prefix="fallback-intraday"
                ) as ex:
                    for fut in as_completed([ex.submit(_pull, s) for s in missing]):
                        sym, df = fut.result()
                        if df is not None and not getattr(df, "empty", False):
                            got[sym] = df
            for sym, df in got.items():
                if df is not None and not getattr(df, "empty", False):
                    out[sym] = (
                        df
                        if out[sym].empty
                        else pd.concat([out[sym], df])
                    )

    total_symbols = len({k: v for k, v in out.items() if not v.empty})
    try:
        bars_loaded = sum(len(v) for v in out.values())
    except (TypeError, AttributeError):  # AI-AGENT-REF: bars summary fallback
        bars_loaded = 0
    first_symbol = next((k for k, v in out.items() if not v.empty), None)
    logger.info(
        "FETCH_SUMMARY",
        extra={
            "total_symbols": total_symbols,
            "bars_loaded": bars_loaded,
            "first_symbol": first_symbol,
        },
    )
    return {k: v for k, v in out.items() if not v.empty}


def _fetch_regime_bars(
    ctx: BotContext, start, end, timeframe="1D"
) -> dict[str, pd.DataFrame]:
    settings = get_settings()
    syms_csv = (getattr(settings, "regime_symbols_csv", None) or "SPY").strip()
    symbols = [s.strip() for s in syms_csv.split(",") if s.strip()]
    return _fetch_universe_bars_chunked(
        symbols, timeframe, start, end, getattr(ctx, "data_feed", None)
    )


def _build_regime_dataset(ctx: BotContext) -> pd.DataFrame:
    """
    Build regime dataset using a configurable basket via batched fetch.
    Returns a *wide* DataFrame: columns are symbols, rows are aligned by timestamp (index reset).
    """
    logger.info("Building regime dataset (batched)")
    try:
        end_dt = datetime.now(UTC)
        start_dt = end_dt - timedelta(
            days=max(30, int(getattr(ctx, "regime_lookback_days", 100)))
        )
        bundle = _fetch_regime_bars(ctx, start=start_dt, end=end_dt, timeframe="1D")
        if not bundle:
            return pd.DataFrame()
        cols = []
        for sym, df in bundle.items():
            if df is None or getattr(df, "empty", False):
                continue
            s = (
                df[["timestamp", "close"]]
                .rename(columns={"close": sym})
                .set_index("timestamp")
            )
            cols.append(s)
        if not cols:
            logger.warning(
                "Regime dataset empty after normalization; attempting SPY-only fallback"
            )
            try:
                spy_df = ctx.data_fetcher.fetch_bars("SPY", timeframe="1D", limit=180)
                if spy_df is not None and not getattr(spy_df, "empty", False):
                    s = (
                        spy_df[["timestamp", "close"]]
                        .rename(columns={"close": "SPY"})
                        .set_index("timestamp")
                    )
                    cols.append(s)
                else:
                    raise Exception("SPY data not available")
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as e:  # AI-AGENT-REF: narrow exception
                logger.error("SPY fallback failed: %s", e)
                logger.error(
                    "Not enough valid rows (0) to train regime model; using dummy fallback"
                )
                return pd.DataFrame()

        if not cols:  # Final check after SPY fallback attempt
            return pd.DataFrame()
        out = pd.concat(cols, axis=1).sort_index().reset_index()
        out.columns.name = None
        return out
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        logger.warning("REGIME bootstrap failed: %s", exc)
        return pd.DataFrame()


def _regime_basket_to_proxy_bars(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a wide close-price frame (timestamp + per-symbol columns) into a single proxy 'bars'
    DataFrame with at least ['timestamp','close'] for downstream model training.
    The proxy is an equal-weighted index of normalized closes.
    """
    if wide is None or getattr(wide, "empty", False):
        return pd.DataFrame()
    if "timestamp" not in wide.columns:
        return pd.DataFrame()
    close_cols = [c for c in wide.columns if c != "timestamp"]
    if not close_cols:
        return pd.DataFrame()
    df = wide.copy()
    # Normalize each series to 1.0 at first valid point to avoid scale bias
    base = df[close_cols].iloc[0]
    norm = df[close_cols] / base.replace(0, pd.NA)
    proxy_close = norm.mean(axis=1).astype(float)
    out = pd.DataFrame({"timestamp": df["timestamp"], "close": proxy_close})
    return out


# <-- NEW: marker file for daily retraining -->
RETRAIN_MARKER_FILE = abspath("last_retrain.txt")

# Main meta‐learner path: this is where retrain.py will dump the new sklearn model each day.
MODEL_RF_PATH = abspath_safe(getattr(CFG, "model_rf_path", None))
MODEL_XGB_PATH = abspath_safe(getattr(CFG, "model_xgb_path", None))
MODEL_LGB_PATH = abspath_safe(getattr(CFG, "model_lgb_path", None))

REGIME_MODEL_PATH = abspath_safe("regime_model.pkl")
# (We keep a separate meta‐model for signal‐weight learning, if you use Bayesian/Ridge, etc.)
META_MODEL_PATH = abspath_safe("meta_model.pkl")


# Strategy mode
class BotMode:
    def __init__(self, mode: str = "balanced") -> None:
        self.mode = mode.lower()
        # AI-AGENT-REF: canonical TradingConfig build
        self.config = _get_trading_config()
        params: dict[str, float] = {}
        from ai_trading import settings as S
        try:
            params["CONF_THRESHOLD"] = float(S.get_conf_threshold())
        except Exception:
            pass
        kf = getattr(S, "get_kelly_fraction", None)
        if callable(kf):
            try:
                params["KELLY_FRACTION"] = float(kf())
            except Exception:
                pass
        self.params = params

    def set_parameters(self) -> dict[str, float]:
        """Return trading parameters for the current mode.

        This method now delegates to the centralized configuration system
        for consistency and maintainability.
        """
        return self.params

    def get_config(self) -> dict[str, float]:
        cfg = config.TradingConfig.from_env()
        params = dict(self.params)
        from ai_trading import settings as S  # AI-AGENT-REF: authoritative getters
        try:
            params["CONF_THRESHOLD"] = float(S.get_conf_threshold())
        except Exception:
            pass
        kf = getattr(S, "get_kelly_fraction", None)
        if callable(kf):
            try:
                params["KELLY_FRACTION"] = float(kf())
            except Exception:
                pass
        return params


@dataclass
class BotState:
    """
    Central state management for the AI trading bot.

    This class maintains all critical bot state information including risk metrics,
    trading positions, performance tracking, and operational controls. It serves as
    the primary data structure for coordinating trading decisions and risk management
    across the entire system.

    Attributes:
        loss_streak (int): Current consecutive losing trade count for drawdown protection
        streak_halt_until (Optional[datetime]): Timestamp until which trading is halted due to streak
        day_start_equity (Optional[tuple[date, float]]): Daily equity baseline for performance tracking
        week_start_equity (Optional[tuple[date, float]]): Weekly equity baseline for performance tracking
        last_drawdown (float): Most recent portfolio drawdown percentage (-1.0 to 0.0)
        updates_halted (bool): Flag indicating if position updates are temporarily disabled
        running (bool): Current execution state of the trading loop
        current_regime (str): Detected market regime ('bull', 'bear', 'sideways', 'volatile')
        rolling_losses (list[float]): Sliding window of recent trade P&L for trend analysis
        mode_obj (BotMode): Trading mode configuration (conservative/balanced/aggressive)
        no_signal_events (int): Count of cycles with insufficient trading signals
        indicator_failures (int): Count of technical indicator calculation failures
        pdt_blocked (bool): Pattern Day Trader rule violation flag
        position_cache (Dict[str, int]): Cached broker positions to avoid redundant API calls
        long_positions (set[str]): Set of symbols with current long positions
        short_positions (set[str]): Set of symbols with current short positions
        last_run_at (Optional[datetime]): Timestamp of last trading cycle execution
        last_loop_duration (float): Duration in seconds of the previous trading cycle
        trade_cooldowns (Dict[str, datetime]): Per-symbol cooldown periods to prevent overtrading
        last_trade_direction (Dict[str, str]): Last trade direction per symbol ('buy'/'sell')
        skipped_cycles (int): Count of trading cycles skipped due to market/risk conditions

    Examples:
        >>> state = BotState()
        >>> state.running = True
        >>> state.current_regime = "bull"
        >>> state.position_cache['AAPL'] = 100  # 100 shares long
        >>> logging.info(f"Bot running: {state.running}, Regime: {state.current_regime}")
        Bot running: True, Regime: bull

    Note:
        This class uses dataclass fields with default factories to ensure proper
        initialization of mutable default values across instances.
    """

    # Risk Management State
    loss_streak: int = 0
    streak_halt_until: datetime | None = None
    day_start_equity: tuple[date, float] | None = None
    week_start_equity: tuple[date, float] | None = None
    last_drawdown: float = 0.0

    # Operational State
    updates_halted: bool = False
    running: bool = False
    current_regime: str = "sideways"
    rolling_losses: list[float] = field(default_factory=list)
    mode_obj: BotMode = field(default_factory=lambda: BotMode(TRADING_MODE))

    # Signal & Indicator State
    no_signal_events: int = 0
    indicator_failures: int = 0
    pdt_blocked: bool = False

    # Position Management
    position_cache: dict[str, int] = field(default_factory=dict)
    long_positions: set[str] = field(default_factory=set)
    short_positions: set[str] = field(default_factory=set)

    # Execution Timing
    last_run_at: datetime | None = None
    last_loop_duration: float = 0.0

    # Trade Management
    trade_cooldowns: dict[str, datetime] = field(default_factory=dict)
    last_trade_direction: dict[str, str] = field(default_factory=dict)
    skipped_cycles: int = 0

    # AI-AGENT-REF: Trade frequency tracking for overtrading prevention
    trade_history: list[tuple[str, datetime]] = field(
        default_factory=list
    )  # (symbol, timestamp)


class _LazyState:
    __slots__ = ("_inst",)  # AI-AGENT-REF: defer heavy state init

    def __init__(self) -> None:
        object.__setattr__(self, "_inst", None)

    def _ensure(self):
        inst = object.__getattribute__(self, "_inst")
        if inst is None:
            inst = BotState()
            object.__setattr__(self, "_inst", inst)
        return inst

    def __getattr__(self, name):
        return getattr(self._ensure(), name)

    def __setattr__(self, name, value):
        return setattr(self._ensure(), name, value)

    def __repr__(self) -> str:  # AI-AGENT-REF: debug helper
        return f"<Lazy(BotState) initialized={object.__getattribute__(self, '_inst') is not None}>"


state = _LazyState()
logger.info(f"Trading mode is set to '{state.mode_obj.mode}'")
params = state.mode_obj.get_config()
params.update(load_hyperparams())

TRAILING_FACTOR = params.get(
    "TRAILING_FACTOR",
    getattr(
        S, "trailing_factor", getattr(state.mode_obj.config, "trailing_factor", 1.0)
    ),
)
SECONDARY_TRAIL_FACTOR = 1.0
TAKE_PROFIT_FACTOR = params.get(
    "TAKE_PROFIT_FACTOR",
    getattr(
        S,
        "take_profit_factor",
        getattr(state.mode_obj.config, "take_profit_factor", 2.0),
    ),
)
SCALING_FACTOR = params.get(
    "SCALING_FACTOR",
    getattr(S, "scaling_factor", getattr(state.mode_obj.config, "scaling_factor", 1.0)),
)
ORDER_TYPE = "market"
LIMIT_ORDER_SLIPPAGE = params.get(
    "LIMIT_ORDER_SLIPPAGE",
    getattr(
        S,
        "limit_order_slippage",
        getattr(state.mode_obj.config, "limit_order_slippage", 0.001),
    ),
)
MAX_POSITION_SIZE = get_max_position_size(S, state.mode_obj.config)
SLICE_THRESHOLD = 50
POV_SLICE_PCT = params.get(
    "POV_SLICE_PCT",
    getattr(S, "pov_slice_pct", getattr(state.mode_obj.config, "pov_slice_pct", 0.05)),
)
DAILY_LOSS_LIMIT = params.get(
    "get_daily_loss_limit()",
    getattr(
        state.mode_obj.config, "daily_loss_limit", getattr(S, "daily_loss_limit", 0.05)
    ),
)
# Additional risk/sizing knobs aligned with Settings
KELLY_FRACTION = params.get(
    "KELLY_FRACTION",
    getattr(S, "kelly_fraction", getattr(state.mode_obj.config, "kelly_fraction", 0.0)),
)
STOP_LOSS = params.get(
    "STOP_LOSS",
    getattr(S, "stop_loss", getattr(state.mode_obj.config, "stop_loss", 0.05)),
)
TAKE_PROFIT = params.get(
    "TAKE_PROFIT",
    getattr(S, "take_profit", getattr(state.mode_obj.config, "take_profit", 0.10)),
)
LOOKBACK_DAYS = params.get(
    "LOOKBACK_DAYS",
    getattr(S, "lookback_days", getattr(state.mode_obj.config, "lookback_days", 60)),
)
MIN_SIGNAL_STRENGTH = params.get(
    "MIN_SIGNAL_STRENGTH",
    getattr(
        S,
        "min_signal_strength",
        getattr(state.mode_obj.config, "min_signal_strength", 0.1),
    ),
)
# AI-AGENT-REF: Increase default position limit from 10 to 20 for better portfolio utilization
# REMOVED: module-scope MAX_PORTFOLIO_POSITIONS = CFG.max_portfolio_positions
CORRELATION_THRESHOLD = 0.60
# REMOVED: SECTOR_EXPOSURE_CAP = CFG.sector_exposure_cap
# REMOVED: MAX_OPEN_POSITIONS = CFG.max_open_positions
# REMOVED: WEEKLY_DRAWDOWN_LIMIT = CFG.weekly_drawdown_limit
MARKET_OPEN = dt_time(6, 30)
MARKET_CLOSE = dt_time(13, 0)
# REMOVED: VOLUME_THRESHOLD = CFG.volume_threshold
ENTRY_START_OFFSET = timedelta(
    minutes=params.get(
        "ENTRY_START_OFFSET_MIN",
        getattr(
            S,
            "entry_start_offset_min",
            getattr(state.mode_obj.config, "entry_start_offset_min", 0),
        ),
    )
)
ENTRY_END_OFFSET = timedelta(
    minutes=params.get(
        "ENTRY_END_OFFSET_MIN",
        getattr(
            S,
            "entry_end_offset_min",
            getattr(state.mode_obj.config, "entry_end_offset_min", 0),
        ),
    )
)
REGIME_LOOKBACK = 14
REGIME_ATR_THRESHOLD = 20.0
RF_ESTIMATORS = 300

# AI-AGENT-REF: Initialize trading parameters from centralized configuration
RF_MAX_DEPTH = 3
RF_MIN_SAMPLES_LEAF = 5
ATR_LENGTH = 10
CONF_THRESHOLD = float(
    params.get(
        "CONF_THRESHOLD",
        getattr(state.mode_obj.config, "confidence_level", 0.75),
    )
)
CONFIRMATION_COUNT = int(params.get("CONFIRMATION_COUNT", 2))


def _env_float(default: float, *keys: str) -> float:
    for k in keys:
        v = os.getenv(k)
        if v is None or v == "":
            continue
        try:
            return float(v)
        except COMMON_EXC:
            logger.warning("ENV_COERCE_FLOAT_FAILED", extra={"key": k, "value": v})
    return default


CAPITAL_CAP = _env_float(0.04, "AI_TRADING_CAPITAL_CAP", "get_capital_cap()")
_cfg = TradingConfig.from_env()
_settings_drl = get_dollar_risk_limit()
DOLLAR_RISK_LIMIT = _settings_drl
if _cfg.dollar_risk_limit is not None and _cfg.dollar_risk_limit != _settings_drl:
    logger.debug(
        "DOLLAR_RISK_LIMIT_OVERRIDE",
        extra={
            "settings": _settings_drl,
            "trading_config": _cfg.dollar_risk_limit,
        },
    )
    DOLLAR_RISK_LIMIT = _cfg.dollar_risk_limit
_emit_once(
    logger,
    "dollar_risk_limit_resolved",
    logging.DEBUG,
    f"DOLLAR_RISK_LIMIT resolved to {DOLLAR_RISK_LIMIT:.3f}",
)
BUY_THRESHOLD = params.get(
    "get_buy_threshold()",
    getattr(state.mode_obj.config, "buy_threshold", get_buy_threshold()),
)


# Coerce MAX_*SIZE to bounded integers to avoid noisy "invalid" logs
def _as_int(v, default, min_v=1, max_v=1_000_000):
    try:
        x = int(float(v))
        return min(max(x, min_v), max_v)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ):  # AI-AGENT-REF: narrow exception
        return default


# AI-AGENT-REF: Add comprehensive validation for critical trading parameters
def validate_trading_parameters():
    """Validate critical trading parameters and log warnings for invalid values."""
    global get_capital_cap, DOLLAR_RISK_LIMIT, MAX_POSITION_SIZE, get_conf_threshold, get_buy_threshold

    # Validate get_capital_cap() (should be between 0.01 and 0.5)
    if not isinstance(get_capital_cap(), int | float) or not (
        0.01 <= get_capital_cap() <= 0.5
    ):
        logger.error(
            "Invalid get_capital_cap() %s, using default 0.25", get_capital_cap()
        )
        CAPITAL_CAP = 0.25

    # Validate DOLLAR_RISK_LIMIT (should be between 0.005 and 0.1)
    if not isinstance(DOLLAR_RISK_LIMIT, int | float) or not (
        0.005 <= DOLLAR_RISK_LIMIT <= 0.1
    ):
        logger.error(
            "Invalid DOLLAR_RISK_LIMIT %s, using default 0.05",
            DOLLAR_RISK_LIMIT,
        )
        DOLLAR_RISK_LIMIT = 0.05

    # Validate MAX_POSITION_SIZE (should be between 1 and 10000)
    MAX_POSITION_SIZE = _as_int(MAX_POSITION_SIZE, 8000, min_v=1, max_v=10000)

    # Validate get_conf_threshold() (should be between 0.5 and 0.95)
    if not isinstance(get_conf_threshold(), int | float) or not (
        0.5 <= get_conf_threshold() <= 0.95
    ):
        logger.error(
            "Invalid get_conf_threshold() %s, using default 0.75", get_conf_threshold()
        )
        CONF_THRESHOLD = 0.75

    # Validate get_buy_threshold() (should be between 0.1 and 0.9)
    if not isinstance(get_buy_threshold(), int | float) or not (
        0.1 <= get_buy_threshold() <= 0.9
    ):
        logger.error(
            "Invalid get_buy_threshold() %s, using default 0.2", get_buy_threshold()
        )
        BUY_THRESHOLD = 0.2

    logger.info(
        "TRADING_PARAMS_VALIDATED",
        extra={
            "get_capital_cap()": f"{get_capital_cap():.3f}",
            "dollar_risk_limit": f"{DOLLAR_RISK_LIMIT:.3f}",
            "MAX_POSITION_SIZE": MAX_POSITION_SIZE,
        },
    )


# AI-AGENT-REF: Defer parameter validation in testing environments to prevent import blocking
# Validate parameters after loading
if not os.getenv("TESTING"):
    validate_trading_parameters()

PACIFIC = ZoneInfo("America/Los_Angeles")
PDT_DAY_TRADE_LIMIT = params.get("PDT_DAY_TRADE_LIMIT", 3)
PDT_EQUITY_THRESHOLD = params.get("PDT_EQUITY_THRESHOLD", 25_000.0)

# Regime symbols (makes SPY configurable)
REGIME_SYMBOLS = ["SPY"]

# ─── THREAD-SAFETY LOCKS & CIRCUIT BREAKER ─────────────────────────────────────
cache_lock = Lock()
targets_lock = Lock()
vol_lock = Lock()
sentiment_lock = Lock()
slippage_lock = Lock()
meta_lock = Lock()
run_lock = Lock()
# AI-AGENT-REF: Add thread-safe locking for trade cooldown state
trade_cooldowns_lock = Lock()

# AI-AGENT-REF: Enhanced circuit breaker configuration for external services
breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60)

# AI-AGENT-REF: Specific circuit breakers for different external services
alpaca_breaker = pybreaker.CircuitBreaker(
    fail_max=3,  # Alpaca should be more reliable, fail after 3 attempts
    reset_timeout=30,  # Shorter reset timeout for trading API
    name="alpaca_api",
)

data_breaker = pybreaker.CircuitBreaker(
    fail_max=5,  # Data services can be less reliable
    reset_timeout=120,  # Longer timeout for data recovery
    name="data_services",
)

finnhub_breaker = pybreaker.CircuitBreaker(
    fail_max=3,  # External data API
    reset_timeout=300,  # 5 minutes for external services
    name="finnhub_api",
)

# AI-AGENT-REF: lazy executor init avoids import-time settings
executor: ThreadPoolExecutor | None = None
prediction_executor: ThreadPoolExecutor | None = None


def _ensure_executors() -> None:
    """Create thread pool executors on demand."""  # AI-AGENT-REF: deferred init
    global executor, prediction_executor
    if executor is not None and prediction_executor is not None:
        return
    from ai_trading.config.settings import get_settings

    cpu = os.cpu_count() or 2
    s = get_settings()
    exec_fn = getattr(s, "effective_executor_workers", None)
    exec_workers = exec_fn(cpu) if callable(exec_fn) else cpu
    pred_fn = getattr(s, "effective_prediction_workers", None)
    pred_workers = pred_fn(cpu) if callable(pred_fn) else cpu
    executor = ThreadPoolExecutor(max_workers=exec_workers)
    prediction_executor = ThreadPoolExecutor(max_workers=pred_workers)


# AI-AGENT-REF: Add proper cleanup with atexit handlers for ThreadPoolExecutor resource leak
def cleanup_executors():
    """Cleanup ThreadPoolExecutor resources to prevent resource leaks."""
    try:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            logger.debug("Main executor shutdown successfully")
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.warning("Error shutting down main executor: %s", e)

    try:
        if prediction_executor is not None:
            prediction_executor.shutdown(wait=True, cancel_futures=True)
            logger.debug("Prediction executor shutdown successfully")
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.warning("Error shutting down prediction executor: %s", e)


atexit.register(cleanup_executors)

# EVENT cooldown
_LAST_EVENT_TS = {}
EVENT_COOLDOWN = 15.0  # seconds
# AI-AGENT-REF: hold time now configurable; default to 0 for pure signal holding
REBALANCE_HOLD_SECONDS = int(os.getenv("REBALANCE_HOLD_SECONDS", "0"))
RUN_INTERVAL_SECONDS = 60  # don't run trading loop more often than this
TRADE_COOLDOWN_MIN = get_trade_cooldown_min()  # minutes
TRADE_COOLDOWN = getattr(S, "trade_cooldown", timedelta(minutes=TRADE_COOLDOWN_MIN))  # AI-AGENT-REF: validated cooldown

# AI-AGENT-REF: Enhanced overtrading prevention with frequency limits
MAX_TRADES_PER_HOUR = get_max_trades_per_hour()  # limit high-frequency trading
MAX_TRADES_PER_DAY = (
    get_max_trades_per_day()
)  # daily limit to prevent excessive trading
TRADE_FREQUENCY_WINDOW_HOURS = 1  # rolling window for hourly limits

# Loss streak kill-switch (managed via BotState)

# Volatility stats (for SPY ATR mean/std)
_VOL_STATS = {"mean": None, "std": None, "last_update": None, "last": None}

# Slippage logs (in-memory for quick access)
_slippage_log: list[tuple[str, float, float, datetime]] = (
    []
)  # (symbol, expected, actual, timestamp)
# Ensure persistent slippage log file exists
if not os.path.exists(SLIPPAGE_LOG_FILE):
    try:
        os.makedirs(os.path.dirname(SLIPPAGE_LOG_FILE) or ".", exist_ok=True)
        with open(SLIPPAGE_LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "symbol", "expected", "actual", "slippage_cents"]
            )
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.warning(f"Could not create slippage log {SLIPPAGE_LOG_FILE}: {e}")

# Sector cache for portfolio exposure calculations
_SECTOR_CACHE: dict[str, str] = {}


def _log_health_diagnostics(runtime, reason: str) -> None:
    """Log detailed diagnostics used for halt decisions."""
    try:
        cash = float(runtime.api.get_account().cash)
        positions = len(runtime.api.list_positions())
    except (AttributeError, APIError) as e:
        cash = -1.0
        positions = -1
        logger.debug(
            "health_diagnostics_account_error",
            extra={"cause": e.__class__.__name__},
        )
    try:
        df = runtime.data_fetcher.get_minute_df(
            runtime, REGIME_SYMBOLS[0], lookback_minutes=CFG.min_health_rows
        )
        rows = len(df)
        last_time = df.index[-1].isoformat() if not df.empty else "n/a"
    except (AttributeError, ValueError, KeyError, APIError) as e:
        rows = 0
        last_time = "n/a"
        logger.debug(
            "health_diagnostics_data_error",
            extra={"cause": e.__class__.__name__},
        )
    vol = _VOL_STATS.get("last")
    sentiment = getattr(runtime, "last_sentiment", 0.0)
    logger.debug(
        "Health diagnostics: rows=%s, last_time=%s, vol=%s, sent=%s, cash=%s, positions=%s, reason=%s",
        rows,
        last_time,
        vol,
        sentiment,
        cash,
        positions,
        reason,
    )


# ─── TYPED EXCEPTION ─────────────────────────────────────────────────────────
class DataFetchError(Exception):
    """Raised when expected market data is missing or unusable."""


class OrderExecutionError(Exception):
    """Raised when an Alpaca order fails after submission."""


# ─── B. CLIENTS & SINGLETONS ─────────────────────────────────────────────────


def ensure_alpaca_credentials() -> None:
    """Verify Alpaca credentials are present before starting."""
    validate_alpaca_credentials()


def log_circuit_breaker_status():
    """Log the status of all circuit breakers for monitoring."""
    try:
        breakers = {
            "main": breaker,
            "alpaca": alpaca_breaker,
            "data": data_breaker,
            "finnhub": finnhub_breaker,
        }

        for name, cb in breakers.items():
            if hasattr(cb, "state") and hasattr(cb, "fail_counter"):
                logger.info(
                    "CIRCUIT_BREAKER_STATUS",
                    extra={
                        "breaker": name,
                        "state": cb.state,
                        "failures": cb.fail_counter,
                        "last_failure": getattr(cb, "last_failure", None),
                    },
                )
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.debug(f"Circuit breaker status logging failed: {e}")


def get_circuit_breaker_health() -> dict:
    """Get health status of all circuit breakers."""
    try:
        breakers = {
            "main": breaker,
            "alpaca": alpaca_breaker,
            "data": data_breaker,
            "finnhub": finnhub_breaker,
        }

        health = {}
        for name, cb in breakers.items():
            if hasattr(cb, "state"):
                health[name] = {
                    "state": str(cb.state),
                    "healthy": cb.state != "open",
                    "failures": getattr(cb, "fail_counter", 0),
                }
            else:
                health[name] = {"state": "unknown", "healthy": True, "failures": 0}

        return health
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.error(f"Failed to get circuit breaker health: {e}")
        return {}


# IMPORTANT: Alpaca credentials will be validated at runtime when needed.
# Do not validate at import time to prevent crashes during module loading.


# Prometheus-safe account fetch with circuit breaker protection
@alpaca_breaker
def safe_alpaca_get_account(ctx: BotContext) -> object | None:
    """Safely get Alpaca account; returns None when unavailable or on failure."""
    ensure_alpaca_attached(ctx)
    if ctx.api is None:
        # Log once per process to avoid per-cycle noise when creds are missing
        logger_once.error(
            "ctx.api is None - Alpaca trading client unavailable",
            key="alpaca_unavailable",
        )  # AI-AGENT-REF: unify key to dedupe across call sites
        return None
    try:
        return ctx.api.get_account()
    except (
        APIError,
        TimeoutError,
        ConnectionError,
    ) as e:  # AI-AGENT-REF: explicit error logging for account fetch
        logger.warning(
            "HEALTH_ACCOUNT_FETCH_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        return None  # AI-AGENT-REF: normalized None return on failure


# ─── C. HELPERS ────────────────────────────────────────────────────────────────
def chunked(iterable: Sequence, n: int):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def ttl_seconds() -> int:
    """Configurable TTL for minute-bar cache (default 60s)."""
    return CFG.minute_cache_ttl


def asset_class_for(symbol: str) -> str:
    """Very small heuristic to map tickers to asset classes."""
    sym = symbol.upper()
    if sym.endswith("USD") and len(sym) == 6:
        return "forex"
    if sym.startswith(("BTC", "ETH")):
        return "crypto"
    return "equity"


MAX_VOL_FETCH_RETRIES = 3
# Default lookback for daily bar requests (~1.5 years to cover 252 trading days)
DEFAULT_DAILY_LOOKBACK_DAYS = 400


def compute_spy_vol_stats(runtime) -> None:
    """Compute daily ATR mean/std on SPY for the past 1 year."""
    today = date.today()
    with vol_lock:
        if _VOL_STATS["last_update"] == today:
            return

    symbol = REGIME_SYMBOLS[0]
    required_rows = 252 + ATR_LENGTH
    df = None

    fetcher = getattr(runtime, "data_fetcher", None)
    cached = None
    if fetcher is not None:
        with cache_lock:
            cached = getattr(fetcher, "_daily_cache", {}).get(symbol)
    if cached is not None and len(cached) >= required_rows:
        df = cached
        logger.info(
            "SPY_VOL_FETCH_SKIP",
            extra={"symbol": symbol, "reason": "cache", "rows": len(cached)},
        )
    else:
        logger.info("SPY_VOL_FETCH_REQUEST", extra={"symbol": symbol})
        backoff = 1
        for attempt in range(1, MAX_VOL_FETCH_RETRIES + 1):
            try:
                df = runtime.data_fetcher.get_daily_df(runtime, symbol)
                row_count = 0 if df is None else len(df)
                if row_count < required_rows:
                    logger.error(
                        "SPY_VOL_FETCH_INSUFFICIENT",
                        extra={
                            "symbol": symbol,
                            "required": required_rows,
                            "received": row_count,
                        },
                    )
                    break
                break
            except DataFetchError as e:
                logger.warning(
                    "SPY_VOL_FETCH_RETRY",
                    extra={"attempt": attempt, "symbol": symbol, "cause": str(e)},
                )
                if attempt == MAX_VOL_FETCH_RETRIES:
                    logger.error(
                        "SPY_VOL_DATA_UNAVAILABLE",
                        extra={
                            "symbol": symbol,
                            "attempts": attempt,
                            "hint": "acquire more historical data",
                        },
                    )
                    halt_mgr = getattr(runtime, "halt_manager", None)
                    if halt_mgr is not None:
                        try:
                            halt_mgr.manual_halt_trading("volatility data unavailable")
                        except Exception as hm_exc:  # noqa: BLE001
                            logger.error(
                                "HALT_MANAGER_ERROR", extra={"cause": str(hm_exc)}
                            )
                    with vol_lock:
                        if _VOL_STATS["mean"] is None:
                            _VOL_STATS["mean"] = 0.0
                        if _VOL_STATS["std"] is None:
                            _VOL_STATS["std"] = 0.0
                        if _VOL_STATS["last"] is None:
                            _VOL_STATS["last"] = 0.0
                    return
                time.sleep(backoff)
                backoff *= 2

    row_count = 0 if df is None else len(df)
    logger.info(
        "SPY_VOL_ROW_COUNTS",
        extra={"symbol": symbol, "required": required_rows, "received": row_count},
    )
    if row_count < required_rows:
        raise DataFetchError(
            f"insufficient data for SPY volatility stats; have {row_count} rows, need {required_rows}. "
            "Acquire more historical data manually."
        )

    # Compute ATR series for last 252 trading days
    atr_series = ta.atr(df["high"], df["low"], df["close"], length=ATR_LENGTH).dropna()
    if len(atr_series) < 252:
        raise DataFetchError("insufficient ATR series for SPY")

    recent = atr_series.iloc[-252:]
    mean_val = float(recent.mean())
    std_val = float(recent.std())
    last_val = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

    with vol_lock:
        _VOL_STATS["mean"] = mean_val
        _VOL_STATS["std"] = std_val
        _VOL_STATS["last_update"] = today
        _VOL_STATS["last"] = last_val

    logger.info(
        "SPY_VOL_STATS_UPDATED",
        extra={"mean": mean_val, "std": std_val, "atr": last_val},
    )


def is_high_vol_thr_spy() -> bool:
    """Return True if SPY ATR > mean + 2*std."""
    with vol_lock:
        mean = _VOL_STATS["mean"]
        std = _VOL_STATS["std"]
    if mean is None or std is None:
        return False

    with cache_lock:
        spy_df = data_fetcher._daily_cache.get(REGIME_SYMBOLS[0])
    if spy_df is None or len(spy_df) < ATR_LENGTH:
        return False

    atr_series = ta.atr(
        spy_df["high"], spy_df["low"], spy_df["close"], length=ATR_LENGTH
    )
    if atr_series.empty:
        return False

    current_atr = float(atr_series.iloc[-1])
    return (current_atr - mean) / std >= 2


def is_high_vol_regime() -> bool:
    """
    Wrapper for is_high_vol_thr_spy to be used inside update_trailing_stop and execute_entry.
    Returns True if SPY is in a high-volatility regime (ATR > mean + 2*std).
    """
    return is_high_vol_thr_spy()


_last_fh_prefetch_date: date | None = None
@dataclass
class DataFetcher:
    def __post_init__(self):
        self._daily_cache: dict[str, pd.DataFrame | None] = {}
        self._minute_cache: dict[str, pd.DataFrame | None] = {}
        self._minute_timestamps: dict[str, datetime] = {}
        # AI-AGENT-REF: rate-limit repeated warnings per (symbol, timeframe)
        self._warn_seen: dict[str, int] = {}

        # Verify required Alpaca configuration
        has_key = bool(getattr(CFG, "alpaca_api_key", ""))
        has_secret = bool(getattr(CFG, "alpaca_secret_key_plain", ""))
        if not (has_key and has_secret):
            logger.error(
                "ALPACA_CREDENTIALS_MISSING",
                extra={"has_key": has_key, "has_secret": has_secret},
            )
        logger.debug(
            "ALPACA_DATA_CONFIG",
            extra={
                "feed": getattr(CFG, "alpaca_data_feed", None),
                "adjustment": getattr(CFG, "alpaca_adjustment", None),
            },
        )

    def _warn_once(self, key: str, msg: str) -> None:
        """Log a warning at most once per minute for a given key."""
        try:
            import time as _time

            bucket = int(_time.time() // 60)
            if self._warn_seen.get(key) == bucket:
                return
            self._warn_seen[key] = bucket
        except (ImportError, OSError, ValueError):  # AI-AGENT-REF: narrow warn_once
            pass
        logger.warning(msg)

    def get_daily_df(self, ctx: BotContext, symbol: str) -> pd.DataFrame | None:
        symbol = symbol.upper()
        now_utc = datetime.now(UTC)

        # AI-AGENT-REF: normalize daily window to previous trading day when closed
        try:  # lazy import to avoid hard dependency at import time
            from ai_trading.market.calendars import is_trading_day as _is_trading_day
        except ImportError:  # pragma: no cover  # AI-AGENT-REF: narrow import
            _is_trading_day = None  # type: ignore

        ref_date = now_utc.date()
        if not is_market_open():
            for _ in range(10):  # safety bound
                ref_date = ref_date - timedelta(days=1)
                if _is_trading_day is None:
                    if ref_date.weekday() < 5:
                        break
                else:
                    try:
                        if _is_trading_day(symbol, datetime.combine(ref_date, dt_time(0, 0), tzinfo=UTC)):
                            break
                    except COMMON_EXC:
                        if ref_date.weekday() < 5:
                            break

        end_ts = datetime.combine(ref_date, dt_time(0, 0), tzinfo=UTC) + timedelta(days=1)
        start_ts = end_ts - timedelta(days=DEFAULT_DAILY_LOOKBACK_DAYS)

        logger.info(
            "DAILY_FETCH_REQUEST",
            extra={
                "symbol": symbol,
                "timeframe": str(_bars.TimeFrame.Day),
                "start": start_ts.isoformat(),
                "end": end_ts.isoformat(),
            },
        )

        with cache_lock:
            if symbol in self._daily_cache:
                if daily_cache_hit:
                    try:
                        daily_cache_hit.inc()
                    except (
                        FileNotFoundError,
                        PermissionError,
                        IsADirectoryError,
                        JSONDecodeError,
                        ValueError,
                        KeyError,
                        TypeError,
                        OSError,
                    ) as exc:  # AI-AGENT-REF: narrow exception
                        logger.exception("bot.py unexpected", exc_info=exc)
                        raise
                cached = self._daily_cache[symbol]
                logger.info(
                    "DAILY_FETCH_RESULT",
                    extra={
                        "symbol": symbol,
                        "timeframe": str(_bars.TimeFrame.Day),
                        "start": start_ts.isoformat(),
                        "end": end_ts.isoformat(),
                        "rows": 0 if cached is None else len(cached),
                        "cache": True,
                    },
                )
                return cached

        settings = get_settings()
        api_key = settings.alpaca_api_key
        api_secret = settings.alpaca_secret_key_plain or get_alpaca_secret_key_plain()
        # AI-AGENT-REF: use plain secret string
        if not api_key or not api_secret:
            logger.error(f"Missing Alpaca credentials for {symbol}")
            return None

        client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret,
        )

        def _minute_resample() -> pd.DataFrame | None:  # AI-AGENT-REF: minute fallback helper
            try:
                m_req = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=_bars.TimeFrame.Minute,
                    start=start_ts,
                    end=end_ts,
                    feed="iex",
                )
                mdf = safe_get_stock_bars(client, m_req, symbol, "FALLBACK MINUTE")
                if mdf is None or mdf.empty:
                    return None
                if isinstance(mdf.columns, pd.MultiIndex):
                    mdf = mdf.xs(symbol, level=0, axis=1)
                else:
                    mdf = mdf.drop(columns=["symbol"], errors="ignore")
                idx_vals = mdf.index
                try:
                    idx = safe_to_datetime(idx_vals, context=f"minute fallback {symbol}")
                except ValueError:
                    return None
                mdf.index = idx
                mdf = mdf.rename(columns=lambda c: c.lower())
                return _resample_minutes_to_daily(mdf)
            except (ValueError, TypeError):  # noqa: BLE001
                return None

        try:
            req = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=_bars.TimeFrame.Day,
                start=start_ts,
                end=end_ts,
                feed=_DEFAULT_FEED,
            )

            # AI-AGENT-REF: pre-sanitize to avoid TypeError on callables
            _today_utc = datetime.now(UTC)
            _prev_day = _prev_bus_day(
                _today_utc.astimezone(ZoneInfo("America/New_York")).date()
            )
            _default_start_u, _default_end_u = _nyse_session_utc(_prev_day)

            _was_callable = False

            def _sanitize_pre(x, default):
                nonlocal _was_callable
                if callable(x):
                    _was_callable = True
                try:
                    return _ensure_utc_dt(x, allow_callables=True)
                except (ValueError, TypeError):
                    return default

            req.start = _sanitize_pre(
                getattr(req, "start", start_ts), _default_start_u
            )
            req.end = _sanitize_pre(
                getattr(req, "end", end_ts), _default_end_u
            )
            if _was_callable:
                logger.debug("DAILY_BARS_INPUT_SANITIZED", extra={"symbol": symbol})

            # AI-AGENT-REF: safety net retry with downgraded log level
            try:
                bars = safe_get_stock_bars(client, req, symbol, "DAILY")
            except TypeError as te:
                msg = str(te)
                if "datetime argument was callable" in msg:
                    logger.debug(
                        f"DAILY_BARS_RETRY_SANITIZE {symbol} due to: {msg}"
                    )
                    req.start = _sanitize_pre(
                        getattr(req, "start", start_ts), _default_start_u
                    )
                    req.end = _sanitize_pre(
                        getattr(req, "end", end_ts), _default_end_u
                    )
                    bars = safe_get_stock_bars(client, req, symbol, "DAILY")
                else:
                    raise
            if bars is None or bars.empty:
                logger.warning(
                    "DAILY_BARS_EMPTY",
                    extra={
                        "symbols": req.symbol_or_symbols,
                        "feed": req.feed,
                        "response": None if bars is None else bars.to_dict(),
                    },
                )
                bars = safe_get_stock_bars(client, req, symbol, "DAILY_RETRY")
                if bars is None or bars.empty:
                    bars = _minute_resample()
                    if bars is None or bars.empty:
                        raise DataFetchError(f"no daily data for {symbol}")
            if isinstance(bars.columns, pd.MultiIndex):
                bars = bars.xs(symbol, level=0, axis=1)
            else:
                bars = bars.drop(columns=["symbol"], errors="ignore")
            if bars.empty:
                logger.warning(
                    "DAILY_BARS_EMPTY",
                    extra={
                        "symbols": req.symbol_or_symbols,
                        "feed": req.feed,
                        "response": bars.to_dict(),
                    },
                )
                bars = _minute_resample()
                if bars is None or bars.empty:
                    raise DataFetchError(f"no daily data for {symbol}")
            if len(bars.index) and isinstance(bars.index[0], tuple):
                idx_vals = [t[1] for t in bars.index]
            else:
                idx_vals = bars.index
            try:
                idx = safe_to_datetime(idx_vals, context=f"daily {symbol}")
            except ValueError as e:
                reason = "empty data" if bars.empty else "unparseable timestamps"
                logger.warning(
                    f"Invalid daily index for {symbol}; skipping. {reason} | {e}"
                )
                raise DataFetchError(f"invalid daily index for {symbol}")
            bars.index = idx
            df = bars.rename(columns=lambda c: c.lower()).drop(
                columns=["symbol"], errors="ignore"
            )
        except APIError as e:
            err_msg = str(e).lower()
            if "subscription does not permit querying recent sip data" in err_msg:
                logger.warning(f"ALPACA SUBSCRIPTION ERROR for {symbol}: {repr(e)}")
                logger.info(f"ATTEMPTING IEX-DELAYERED DATA FOR {symbol}")
                try:
                    req.feed = "iex"
                    df_iex = safe_get_stock_bars(client, req, symbol, "IEX DAILY")
                    if df_iex is None:
                        raise DataFetchError(f"no IEX data for {symbol}")
                    if isinstance(df_iex.columns, pd.MultiIndex):
                        df_iex = df_iex.xs(symbol, level=0, axis=1)
                    else:
                        df_iex = df_iex.drop(columns=["symbol"], errors="ignore")
                    if len(df_iex.index) and isinstance(df_iex.index[0], tuple):
                        idx_vals = [t[1] for t in df_iex.index]
                    else:
                        idx_vals = df_iex.index
                    try:
                        idx = safe_to_datetime(idx_vals, context=f"IEX daily {symbol}")
                    except ValueError as e:
                        reason = (
                            "empty data" if df_iex.empty else "unparseable timestamps"
                        )
                        logger.warning(
                            f"Invalid IEX daily index for {symbol}; skipping. {reason} | {e}"
                        )
                        raise DataFetchError(f"invalid IEX daily index for {symbol}")
                    df_iex.index = idx
                    df = df_iex.rename(columns=lambda c: c.lower())
                except (
                    FileNotFoundError,
                    PermissionError,
                    IsADirectoryError,
                    JSONDecodeError,
                    ValueError,
                    KeyError,
                    TypeError,
                    OSError,
                ) as iex_err:  # AI-AGENT-REF: narrow exception
                    logger.warning(f"ALPACA IEX ERROR for {symbol}: {repr(iex_err)}")
                    rdf = _minute_resample()
                    if rdf is not None and not rdf.empty:
                        df = rdf
                    else:
                        logger.info(
                            f"INSERTING DUMMY DAILY FOR {symbol} ON {end_ts.date().isoformat()}"
                        )
                        ts = pd.to_datetime(end_ts, utc=True, errors="coerce")
                        if ts is None:
                            ts = pd.Timestamp.now(tz="UTC")
                        dummy_date = ts
                        df = pd.DataFrame(
                            [
                                {
                                    "open": 0.0,
                                    "high": 0.0,
                                    "low": 0.0,
                                    "close": 0.0,
                                    "volume": 0,
                                }
                            ],
                            index=[dummy_date],
                        )
            else:
                logger.warning(f"ALPACA DAILY FETCH ERROR for {symbol}: {repr(e)}")
                rdf = _minute_resample()
                if rdf is not None and not rdf.empty:
                    df = rdf
                else:
                    ts2 = pd.to_datetime(end_ts, utc=True, errors="coerce")
                    if ts2 is None:
                        ts2 = pd.Timestamp.now(tz="UTC")
                    dummy_date = ts2
                    df = pd.DataFrame(
                        [{"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0}],
                        index=[dummy_date],
                    )
        except (NameError, AttributeError) as e:
            # Handle pandas schema errors (like missing _RealMultiIndex) gracefully
            logger.error(
                "DATA_SOURCE_SCHEMA_ERROR", extra={"symbol": symbol, "cause": str(e)}
            )
            return _create_empty_bars_dataframe()
        except (KeyError, ValueError) as e:
            logger.error(f"DATA_VALIDATION_ERROR for {symbol}: {repr(e)}")
            return _create_empty_bars_dataframe()
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.error(f"Failed to fetch daily data for {symbol}: {repr(e)}")
            raise DataFetchError(f"failed to fetch daily data for {symbol}") from e

        with cache_lock:
            self._daily_cache[symbol] = df
        logger.info(
            "DAILY_FETCH_RESULT",
            extra={
                "symbol": symbol,
                "timeframe": str(_bars.TimeFrame.Day),
                "start": start_ts.isoformat(),
                "end": end_ts.isoformat(),
                "rows": 0 if df is None else len(df),
                "cache": False,
            },
        )
        return df

    def get_minute_df(
        self, ctx: BotContext, symbol: str, lookback_minutes: int = 30
    ) -> pd.DataFrame | None:
        symbol = symbol.upper()
        now_utc = datetime.now(UTC)
        last_closed_minute = now_utc.replace(second=0, microsecond=0) - timedelta(
            minutes=1
        )
        start_minute = last_closed_minute - timedelta(minutes=lookback_minutes)

        with cache_lock:
            last_ts = self._minute_timestamps.get(symbol)
            if last_ts and last_ts > now_utc - timedelta(seconds=ttl_seconds()):
                if minute_cache_hit:
                    try:
                        minute_cache_hit.inc()
                    except (
                        FileNotFoundError,
                        PermissionError,
                        IsADirectoryError,
                        JSONDecodeError,
                        ValueError,
                        KeyError,
                        TypeError,
                        OSError,
                    ) as exc:  # AI-AGENT-REF: narrow exception
                        logger.exception("bot.py unexpected", exc_info=exc)
                        raise
                return self._minute_cache[symbol]

        if minute_cache_miss:
            try:
                minute_cache_miss.inc()
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as exc:  # AI-AGENT-REF: narrow exception
                logger.exception("bot.py unexpected", exc_info=exc)
                raise
        settings = get_settings()
        api_key = settings.alpaca_api_key
        api_secret = settings.alpaca_secret_key_plain or get_alpaca_secret_key_plain()
        # AI-AGENT-REF: use plain secret string
        if not api_key or not api_secret:
            raise RuntimeError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set for data fetching"
            )

        client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret,
        )

        try:
            req = _bars.StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=_bars.TimeFrame.Minute,
                start=start_minute,
                end=last_closed_minute,
                feed=_DEFAULT_FEED,
            )
            bars = _bars.safe_get_stock_bars(client, req, symbol, "MINUTE")
            if bars is None:
                return None
            if isinstance(bars.columns, pd.MultiIndex):
                bars = bars.xs(symbol, level=0, axis=1)
            else:
                bars = bars.drop(columns=["symbol"], errors="ignore")
            if bars.empty:
                logger.warning(
                    f"No minute bars returned for {symbol}. Possible market holiday or API outage"
                )
                return None
            if len(bars.index) and isinstance(bars.index[0], tuple):
                idx_vals = [t[1] for t in bars.index]
            else:
                idx_vals = bars.index
            try:
                idx = safe_to_datetime(idx_vals, context=f"minute {symbol}")
            except ValueError as e:
                reason = "empty data" if bars.empty else "unparseable timestamps"
                logger.warning(
                    f"Invalid minute index for {symbol}; skipping. {reason} | {e}"
                )
                return None
            bars.index = idx
            df = bars.rename(columns=lambda c: c.lower()).drop(
                columns=["symbol"], errors="ignore"
            )[["open", "high", "low", "close", "volume"]]
        except APIError as e:
            err_msg = str(e)
            if (
                "subscription does not permit querying recent sip data"
                in err_msg.lower()
            ):
                logger.warning(f"ALPACA SUBSCRIPTION ERROR for {symbol}: {repr(e)}")
                logger.info(f"ATTEMPTING IEX-DELAYERED DATA FOR {symbol}")
                try:
                    req.feed = "iex"
                    df_iex = _bars.safe_get_stock_bars(client, req, symbol, "IEX MINUTE")
                    if df_iex is None:
                        return None
                    if isinstance(df_iex.columns, pd.MultiIndex):
                        df_iex = df_iex.xs(symbol, level=0, axis=1)
                    else:
                        df_iex = df_iex.drop(columns=["symbol"], errors="ignore")
                    if len(df_iex.index) and isinstance(df_iex.index[0], tuple):
                        idx_vals = [t[1] for t in df_iex.index]
                    else:
                        idx_vals = df_iex.index
                    try:
                        idx = safe_to_datetime(idx_vals, context=f"IEX minute {symbol}")
                    except ValueError as _e:
                        reason = (
                            "empty data" if df_iex.empty else "unparseable timestamps"
                        )
                        logger.warning(
                            f"Invalid IEX minute index for {symbol}; skipping. {reason} | {_e}"
                        )
                        df = pd.DataFrame()
                    else:
                        df_iex.index = idx
                        df = df_iex.rename(columns=lambda c: c.lower())[
                            "open", "high", "low", "close", "volume"
                        ]
                except (
                    FileNotFoundError,
                    PermissionError,
                    IsADirectoryError,
                    JSONDecodeError,
                    ValueError,
                    KeyError,
                    TypeError,
                    OSError,
                ) as iex_err:  # AI-AGENT-REF: narrow exception
                    logger.warning(f"ALPACA IEX ERROR for {symbol}: {repr(iex_err)}")
                    logger.info(f"NO ALTERNATIVE MINUTE DATA FOR {symbol}")
                    df = pd.DataFrame()
            else:
                logger.warning(f"ALPACA MINUTE FETCH ERROR for {symbol}: {repr(e)}")
                df = pd.DataFrame()
        except (NameError, AttributeError) as e:
            # Handle pandas schema errors (like missing _RealMultiIndex) gracefully
            logger.error(
                "DATA_SOURCE_SCHEMA_ERROR", extra={"symbol": symbol, "cause": str(e)}
            )
            df = _create_empty_bars_dataframe()
        except (KeyError, ValueError) as e:
            logger.warning(f"DATA_VALIDATION_ERROR for minute data {symbol}: {repr(e)}")
            df = _create_empty_bars_dataframe()
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.warning(f"ALPACA MINUTE FETCH ERROR for {symbol}: {repr(e)}")
            df = pd.DataFrame()

        with cache_lock:
            self._minute_cache[symbol] = df
            self._minute_timestamps[symbol] = now_utc
        return df

    def get_historical_minute(
        self,
        ctx: BotContext,  # ← still needs ctx here, per retrain.py
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame | None:
        """
        Fetch all minute bars for `symbol` between start_date and end_date (inclusive),
        by calling Alpaca’s get_bars for each calendar day. Returns a DataFrame
        indexed by naive Timestamps, or None if no data was returned at all.
        """
        all_days: list[pd.DataFrame] = []
        current_day = start_date

        while current_day <= end_date:
            day_start = datetime.combine(current_day, dt_time.min, UTC)
            day_end = datetime.combine(current_day, dt_time.max, UTC)
            if isinstance(day_start, tuple):
                day_start, _tmp = day_start
            if isinstance(day_end, tuple):
                _, day_end = day_end

            try:
                bars_req = _bars.StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=_bars.TimeFrame.Minute,
                    start=day_start,
                    end=day_end,
                    limit=10000,
                    feed=_DEFAULT_FEED,
                )
                try:
                    bars_day = _bars.safe_get_stock_bars(
                        ctx.data_client, bars_req, symbol, "INTRADAY"
                    )
                    if bars_day is None:
                        return []
                except APIError as e:
                    if (
                        "subscription does not permit" in str(e).lower()
                        and _DEFAULT_FEED != "iex"
                    ):
                        logger.warning(
                            (
                                "[historic_minute] subscription error for %s %s-%s: %s; "
                                "retrying with IEX"
                            ),
                            symbol,
                            day_start,
                            day_end,
                            e,
                        )
                        bars_req.feed = "iex"
                        bars_day = _bars.safe_get_stock_bars(
                            ctx.data_client, bars_req, symbol, "IEX INTRADAY"
                        )
                        if bars_day is None:
                            return []
                    else:
                        raise
                if isinstance(bars_day.columns, pd.MultiIndex):
                    bars_day = bars_day.xs(symbol, level=0, axis=1)
                else:
                    bars_day = bars_day.drop(columns=["symbol"], errors="ignore")
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as e:  # AI-AGENT-REF: narrow exception
                logger.warning(
                    f"[historic_minute] failed for {symbol} {day_start}-{day_end}: {e}"
                )
                bars_day = None

            if bars_day is not None and not bars_day.empty:
                if "symbol" in bars_day.columns:
                    bars_day = bars_day.drop(columns=["symbol"], errors="ignore")

                try:
                    idx = safe_to_datetime(
                        bars_day.index, context=f"historic minute {symbol}"
                    )
                except ValueError as e:
                    reason = (
                        "empty data" if bars_day.empty else "unparseable timestamps"
                    )
                    logger.warning(
                        f"Invalid minute index for {symbol}; skipping day {day_start}. {reason} | {e}"
                    )
                    bars_day = None
                else:
                    bars_day.index = idx
                    bars_day = bars_day.rename(columns=lambda c: c.lower())[
                        ["open", "high", "low", "close", "volume"]
                    ]
                    all_days.append(bars_day)

            current_day += timedelta(days=1)

        if not all_days:
            return None

        combined = pd.concat(all_days, axis=0)
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.sort_index()
        return combined


# Helper to prefetch daily data in bulk with Alpaca, handling SIP subscription
# issues and falling back to IEX delayed feed per symbol if needed.
def prefetch_daily_data(
    symbols: list[str], start_date: date, end_date: date
) -> dict[str, pd.DataFrame]:
    settings = get_settings()
    alpaca_key = settings.alpaca_api_key
    alpaca_secret = settings.alpaca_secret_key_plain or get_alpaca_secret_key_plain()
    # AI-AGENT-REF: use plain secret string
    if not alpaca_key or not alpaca_secret:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set for data fetching"
        )

    client = StockHistoricalDataClient(
        api_key=alpaca_key,
        secret_key=alpaca_secret,
    )

    try:
        req = _bars.StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=_bars.TimeFrame.Day,
            start=start_date,
            end=end_date,
            feed=_DEFAULT_FEED,
        )
        bars = _bars.safe_get_stock_bars(client, req, str(symbols), "BULK DAILY")
        if bars is None:
            return {}
        if isinstance(bars.columns, pd.MultiIndex):
            grouped_raw = {
                sym: bars.xs(sym, level=0, axis=1)
                for sym in symbols
                if sym in bars.columns.get_level_values(0)
            }
        else:
            grouped_raw = dict(bars.groupby("symbol"))
        grouped = {}
        for sym, df in grouped_raw.items():
            df = df.drop(columns=["symbol"], errors="ignore")
            try:
                idx = safe_to_datetime(df.index, context=f"bulk {sym}")
            except ValueError as e:
                logger.warning(f"Invalid bulk index for {sym}; skipping | {e}")
                continue
            df.index = idx
            df = df.rename(columns=lambda c: c.lower())
            grouped[sym] = df
        return grouped
    except APIError as e:
        err_msg = str(e).lower()
        if "subscription does not permit querying recent sip data" in err_msg:
            logger.warning(f"ALPACA SUBSCRIPTION ERROR in bulk for {symbols}: {repr(e)}")
            logger.info(f"ATTEMPTING IEX-DELAYERED BULK FETCH FOR {symbols}")
            try:
                req.feed = "iex"
                bars_iex = _bars.safe_get_stock_bars(
                    client, req, str(symbols), "IEX BULK DAILY"
                )
                if bars_iex is None:
                    return {}
                if isinstance(bars_iex.columns, pd.MultiIndex):
                    grouped_raw = {
                        sym: bars_iex.xs(sym, level=0, axis=1)
                        for sym in symbols
                        if sym in bars_iex.columns.get_level_values(0)
                    }
                else:
                    grouped_raw = dict(bars_iex.groupby("symbol"))
                grouped = {}
                for sym, df in grouped_raw.items():
                    df = df.drop(columns=["symbol"], errors="ignore")
                    try:
                        idx = safe_to_datetime(df.index, context=f"IEX bulk {sym}")
                    except ValueError as e:
                        logger.warning(
                            f"Invalid IEX bulk index for {sym}; skipping | {e}"
                        )
                        continue
                    df.index = idx
                    df = df.rename(columns=lambda c: c.lower())
                    grouped[sym] = df
                return grouped
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as iex_err:  # AI-AGENT-REF: narrow exception
                logger.warning(f"ALPACA IEX BULK ERROR for {symbols}: {repr(iex_err)}")
                daily_dict = {}
                for sym in symbols:
                    try:
                        req_sym = _bars.StockBarsRequest(
                            symbol_or_symbols=[sym],
                            timeframe=_bars.TimeFrame.Day,
                            start=start_date,
                            end=end_date,
                            feed=_DEFAULT_FEED,
                        )
                        df_sym = _bars.safe_get_stock_bars(
                            client, req_sym, sym, "FALLBACK DAILY"
                        )
                        if df_sym is None:
                            continue
                        df_sym = df_sym.drop(columns=["symbol"], errors="ignore")
                        try:
                            idx = safe_to_datetime(
                                df_sym.index, context=f"fallback bulk {sym}"
                            )
                        except ValueError as _e:
                            logger.warning(
                                f"Invalid fallback bulk index for {sym}; skipping | {_e}"
                            )
                            continue
                        df_sym.index = idx
                        df_sym = df_sym.rename(columns=lambda c: c.lower())
                        daily_dict[sym] = df_sym
                    except (
                        FileNotFoundError,
                        PermissionError,
                        IsADirectoryError,
                        JSONDecodeError,
                        ValueError,
                        KeyError,
                        TypeError,
                        OSError,
                    ) as indiv_err:  # AI-AGENT-REF: narrow exception
                        logger.warning(f"ALPACA IEX ERROR for {sym}: {repr(indiv_err)}")
                        logger.info(
                            f"INSERTING DUMMY DAILY FOR {sym} ON {end_date.isoformat()}"
                        )
                        tsd = pd.to_datetime(end_date, utc=True, errors="coerce")
                        if tsd is None:
                            tsd = pd.Timestamp.now(tz="UTC")
                        dummy_date = tsd
                        dummy_df = pd.DataFrame(
                            [
                                {
                                    "open": 0.0,
                                    "high": 0.0,
                                    "low": 0.0,
                                    "close": 0.0,
                                    "volume": 0,
                                }
                            ],
                            index=[dummy_date],
                        )
                        daily_dict[sym] = dummy_df
                return daily_dict
        else:
            logger.warning(f"ALPACA BULK FETCH UNKNOWN ERROR for {symbols}: {repr(e)}")
            daily_dict = {}
            for sym in symbols:
                t2 = pd.to_datetime(end_date, utc=True, errors="coerce")
                if t2 is None:
                    t2 = pd.Timestamp.now(tz="UTC")
                dummy_date = t2
                dummy_df = pd.DataFrame(
                    [{"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0}],
                    index=[dummy_date],
                )
                daily_dict[sym] = dummy_df
            return daily_dict
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.warning(f"ALPACA BULK FETCH EXCEPTION for {symbols}: {repr(e)}")
        daily_dict = {}
        for sym in symbols:
            t3 = pd.to_datetime(end_date, utc=True, errors="coerce")
            if t3 is None:
                t3 = pd.Timestamp.now(tz="UTC")
            dummy_date = t3
            dummy_df = pd.DataFrame(
                [{"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0}],
                index=[dummy_date],
            )
            daily_dict[sym] = dummy_df
        return daily_dict


# ─── E. TRADE LOGGER ───────────────────────────────────────────────────────────
class TradeLogger:
    def __init__(self, path: str | Path | None = None, *args, **kwargs):
        # AI-AGENT-REF: sanitize and default trade log path
        resolved = abspath_safe(path)
        if not resolved:
            resolved = default_trade_log_path()
        parent = os.path.dirname(resolved) or BASE_DIR
        os.makedirs(parent, exist_ok=True)

        self.path = resolved
        if not os.path.exists(resolved):
            try:
                with open(resolved, "w") as f:
                    portalocker.lock(f, portalocker.LOCK_EX)
                    try:
                        csv.writer(f).writerow(
                            [
                                "symbol",
                                "entry_time",
                                "entry_price",
                                "exit_time",
                                "exit_price",
                                "qty",
                                "side",
                                "strategy",
                                "classification",
                                "signal_tags",
                                "confidence",
                                "reward",
                            ]
                        )
                    finally:
                        portalocker.unlock(f)
            except PermissionError:
                logger.debug("TradeLogger init path not writable: %s", path)
        if not os.path.exists(REWARD_LOG_FILE):
            try:
                os.makedirs(os.path.dirname(REWARD_LOG_FILE) or ".", exist_ok=True)
                with open(REWARD_LOG_FILE, "w", newline="") as rf:
                    csv.writer(rf).writerow(
                        [
                            "timestamp",
                            "symbol",
                            "reward",
                            "pnl",
                            "confidence",
                            "band",
                        ]
                    )
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as e:  # AI-AGENT-REF: narrow exception
                logger.warning(f"Failed to create reward log: {e}")

    def log_entry(
        self,
        symbol: str,
        price: float,
        qty: int,
        side: str,
        strategy: str,
        signal_tags: str = "",
        confidence: float = 0.0,
    ) -> None:
        now_iso = utc_now_iso()
        try:
            with open(self.path, "a") as f:
                portalocker.lock(f, portalocker.LOCK_EX)
                try:
                    csv.writer(f).writerow(
                        [
                            symbol,
                            now_iso,
                            price,
                            "",
                            "",
                            qty,
                            side,
                            strategy,
                            "",
                            signal_tags,
                            confidence,
                            "",
                        ]
                    )
                finally:
                    portalocker.unlock(f)
        except PermissionError:
            logger.debug("TradeLogger entry log skipped; path not writable")

    def log_exit(self, state: BotState, symbol: str, exit_price: float) -> None:
        try:
            with open(self.path, "r+") as f:
                portalocker.lock(f, portalocker.LOCK_EX)
                try:
                    rows = list(csv.reader(f))
                    header, data = rows[0], rows[1:]
                    pnl = 0.0
                    conf = 0.0
                    for row in data:
                        if row[0] == symbol and row[3] == "":
                            entry_t = datetime.fromisoformat(row[1])
                            days = (datetime.now(UTC) - entry_t).days
                            cls = (
                                "day_trade"
                                if days == 0
                                else "swing_trade" if days < 5 else "long_trade"
                            )
                            row[3], row[4], row[8] = (
                                utc_now_iso(),
                                exit_price,
                                cls,
                            )
                            # Compute PnL
                            entry_price = float(row[2])
                            pnl = (exit_price - entry_price) * (
                                1 if row[6] == "buy" else -1
                            )
                            if len(row) >= 11:
                                try:
                                    conf = float(row[10])
                                except (
                                    FileNotFoundError,
                                    PermissionError,
                                    IsADirectoryError,
                                    JSONDecodeError,
                                    ValueError,
                                    KeyError,
                                    TypeError,
                                    OSError,
                                ):  # AI-AGENT-REF: narrow exception
                                    conf = 0.0
                            if len(row) >= 12:
                                row[11] = pnl * conf
                            else:
                                row.append(conf)
                                row.append(pnl * conf)
                            break
                    f.seek(0)
                    f.truncate()
                    w = csv.writer(f)
                    w.writerow(header)
                    w.writerows(data)
                finally:
                    portalocker.unlock(f)
        except PermissionError:
            logger.debug("TradeLogger exit log skipped; path not writable")
            return

        # log reward
        try:
            with open(REWARD_LOG_FILE, "a", newline="") as rf:
                csv.writer(rf).writerow(
                    [
                        utc_now_iso(),
                        symbol,
                        pnl * conf,
                        pnl,
                        conf,
                        state.capital_band,
                    ]
                )
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as exc:  # AI-AGENT-REF: narrow exception
            logger.exception("bot.py unexpected", exc_info=exc)
            raise

        # Update streak-based kill-switch
        if pnl < 0:
            state.loss_streak += 1
        else:
            state.loss_streak = 0
        if state.loss_streak >= 3:
            state.streak_halt_until = datetime.now(UTC).astimezone(PACIFIC) + timedelta(
                minutes=60
            )
            logger.warning(
                "STREAK_HALT_TRIGGERED",
                extra={
                    "loss_streak": state.loss_streak,
                    "halt_until": state.streak_halt_until,
                },
            )

        # AI-AGENT-REF: ai_trading/core/bot_engine.py:2960 - Convert import guard to settings-gated import
        from ai_trading.config import get_settings

        settings = get_settings()
        if settings.enable_sklearn:  # Meta-learning requires sklearn
            from ai_trading.meta_learning import (
                validate_trade_data_quality,
            )

            quality_report = validate_trade_data_quality(self.path)

            # If we have audit format rows, trigger conversion for meta-learning
            if quality_report.get("audit_format_rows", 0) > 0:
                logger.info(
                    "METALEARN_TRIGGER_CONVERSION: Converting audit format to meta-learning format"
                )
                # The conversion will be handled by the meta-learning system when it reads the log
        else:
            logger.debug("Meta-learning disabled, skipping conversion")


def _read_trade_log(
    path: str = TRADE_LOG_FILE,
    usecols: list[str] | None = None,
    dtype: dict | str | None = None,
):
    """Safely load the trade log CSV or return ``None`` if unavailable.

    Emits a warning when the log file is missing or empty with guidance on
    initializing the trade log via :func:`get_trade_logger`.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        logger.warning(
            "TRADE_LOG_MISSING_OR_EMPTY | path=%s | hint=call get_trade_logger() to initialize",
            path,
        )
        return None
    try:  # AI-AGENT-REF: gate heavy import
        import pandas as pd  # type: ignore
    except ImportError:
        logger.warning("pandas not available; cannot load trade log at %s", path)
        return None
    try:
        df = pd.read_csv(
            path,
            on_bad_lines="skip",
            engine="python",
            usecols=usecols,
            dtype=dtype,
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger.warning(
            "Failed to parse trade log %s: %s | hint=call get_trade_logger() to initialize",
            path,
            e,
        )
        return None
    if df.empty:
        logger.warning(
            "Trade log %s parsed but contains no rows | hint=call get_trade_logger() to initialize",
            path,
        )
        return None
    return df


def _parse_local_positions() -> dict[str, int]:
    """Return current local open positions from the trade logger."""
    # Ensure the trade log file exists with headers before attempting to read
    # from it.  ``get_trade_logger`` will create the file and write the header
    # row on first use, which prevents later reads from failing due to a missing
    # or empty file.
    get_trade_logger()

    positions: dict[str, int] = {}
    df = _read_trade_log(
        TRADE_LOG_FILE, usecols=["symbol", "qty", "side", "exit_time"], dtype=str
    )
    if df is None:
        return positions
    for _, row in df.iterrows():
        if str(row.get("exit_time", "")) != "":
            continue
        qty = int(row.qty)
        qty = qty if row.side == "buy" else -qty
        positions[row.symbol] = positions.get(row.symbol, 0) + qty
    positions = {k: v for k, v in positions.items() if v != 0}
    return positions


def audit_positions(ctx) -> None:
    """
    Fetch local vs. broker positions and submit market orders to correct any mismatch.
    """
    # 1) Read local open positions from the trade log
    local = _parse_local_positions()

    # 2) Fetch remote (broker) positions
    try:
        remote = {p.symbol: int(p.qty) for p in ctx.api.get_all_positions()}
    except APIError as e:
        logger = get_logger(__name__)
        logger.exception(
            "bot_engine: failed to fetch remote positions from broker",
            exc_info=e,
            extra={"cause": e.__class__.__name__},
        )
        return

    max_order_size = _as_int(os.getenv("MAX_ORDER_SIZE", "1000"), 1000)

    # 3) For any symbol in remote whose remote_qty != local_qty, correct via market order
    for sym, rq in remote.items():
        lq = local.get(sym, 0)
        if lq != rq:
            diff = rq - lq
            if diff > 0:
                # Broker has more shares than local: sell off the excess
                try:
                    req = MarketOrderRequest(
                        symbol=sym,
                        qty=min(abs(diff), max_order_size),
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                    safe_submit_order(ctx.api, req)
                except APIError as exc:
                    logger.exception(
                        "bot.py unexpected",
                        exc_info=exc,
                        extra={"cause": exc.__class__.__name__},
                    )
                    raise
            else:
                # Broker has fewer shares than local: buy back the missing shares
                try:
                    req = MarketOrderRequest(
                        symbol=sym,
                        qty=min(abs(diff), max_order_size),
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                    )
                    safe_submit_order(ctx.api, req)
                except APIError as exc:
                    logger.exception(
                        "bot.py unexpected",
                        exc_info=exc,
                        extra={"cause": exc.__class__.__name__},
                    )
                    raise

    # 4) For any symbol in local that is not in remote, submit order matching the local side
    for sym, lq in local.items():
        if sym not in remote:
            # AI-AGENT-REF: prevent oversize orders on unmatched locals
            if abs(lq) > max_order_size:
                logger.warning(
                    "Order size %d exceeds maximum %d for %s",
                    abs(lq),
                    max_order_size,
                    sym,
                )
                continue
            try:
                side = OrderSide.BUY if lq > 0 else OrderSide.SELL
                req = MarketOrderRequest(
                    symbol=sym,
                    qty=abs(lq),
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
                safe_submit_order(ctx.api, req)
            except APIError as exc:
                logger.exception(
                    "bot.py unexpected",
                    exc_info=exc,
                    extra={"cause": exc.__class__.__name__},
                )
                raise


def validate_open_orders(ctx: BotContext) -> None:
    local = _parse_local_positions()
    if not local:
        get_logger(__name__).debug(
            "No local positions parsed; skipping open-order audit"
        )
        return
    try:
        open_orders = list_open_orders(ctx.api)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger = get_logger(__name__)
        logger.exception(
            "bot_engine: failed to fetch open orders from broker", exc_info=e
        )
        return

    now = datetime.now(UTC)
    for od in open_orders:
        created = pd.to_datetime(getattr(od, "created_at", now))
        age = (now - created).total_seconds() / 60.0

        if age > 5 and getattr(od, "status", "").lower() in {"new", "accepted"}:
            try:
                ctx.api.cancel_order(od.id)
                qty = int(getattr(od, "qty", 0))
                side = getattr(od, "side", "")
                if qty > 0 and side in {"buy", "sell"}:
                    req = MarketOrderRequest(
                        symbol=od.symbol,
                        qty=qty,
                        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                    safe_submit_order(ctx.api, req)
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as exc:  # AI-AGENT-REF: narrow exception
                logger.exception("bot.py unexpected", exc_info=exc)
                raise

    # After canceling/replacing any stuck orders, fix any position mismatches
    audit_positions(ctx)


# ─── F. SIGNAL MANAGER & HELPER FUNCTIONS ─────────────────────────────────────
_LAST_PRICE: dict[str, float] = {}
PRICE_TTL_PCT = 0.005  # only fetch sentiment if price moved > 0.5%
SENTIMENT_TTL_SEC = 600  # 10 minutes
# AI-AGENT-REF: Enhanced sentiment caching for rate limiting
SENTIMENT_RATE_LIMITED_TTL_SEC = 3600  # 1 hour cache when rate limited
_SENTIMENT_CIRCUIT_BREAKER = {
    "failures": 0,
    "last_failure": 0,
    "state": "closed",
    "next_retry": 0,
}  # closed, open, half-open
# AI-AGENT-REF: Enhanced sentiment circuit breaker thresholds for better resilience
SENTIMENT_RECOVERY_TIMEOUT = (
    1800  # Extended to 30 minutes (1800s) for better recovery per problem statement
)
SENTIMENT_BASE_DELAY = 5


class SignalManager:
    def __init__(self) -> None:
        self.momentum_lookback = 5
        self.mean_rev_lookback = 20
        self.mean_rev_zscore_threshold = 2.0
        self.regime_volatility_threshold = REGIME_ATR_THRESHOLD
        self.last_components: list[tuple[int, float, str]] = []

    def signal_momentum(self, df: pd.DataFrame, model=None) -> tuple[int, float, str]:
        if df is None or len(df) <= self.momentum_lookback:
            return -1, 0.0, "momentum"
        try:
            df["momentum"] = df["close"].pct_change(
                self.momentum_lookback, fill_method=None
            )
            val = df["momentum"].iloc[-1]
            s = 1 if val > 0 else -1 if val < 0 else -1
            w = min(abs(val) * 10, 1.0)
            return s, w, "momentum"
        except (KeyError, ValueError, TypeError, IndexError):
            logger.exception("Error in signal_momentum")
            return -1, 0.0, "momentum"

    def signal_mean_reversion(
        self, df: pd.DataFrame, model=None
    ) -> tuple[int, float, str]:
        if df is None or len(df) < self.mean_rev_lookback:
            return -1, 0.0, "mean_reversion"
        try:
            ma = df["close"].rolling(self.mean_rev_lookback).mean()
            sd = df["close"].rolling(self.mean_rev_lookback).std()
            df["zscore"] = (df["close"] - ma) / sd
            val = df["zscore"].iloc[-1]
            s = (
                -1
                if val > self.mean_rev_zscore_threshold
                else 1 if val < -self.mean_rev_zscore_threshold else -1
            )
            w = min(abs(val) / 3, 1.0)
            return s, w, "mean_reversion"
        except (KeyError, ValueError, TypeError, IndexError):
            logger.exception("Error in signal_mean_reversion")
            return -1, 0.0, "mean_reversion"

    def signal_stochrsi(self, df: pd.DataFrame, model=None) -> tuple[int, float, str]:
        if df is None or "stochrsi" not in df or df["stochrsi"].dropna().empty:
            return -1, 0.0, "stochrsi"
        try:
            val = df["stochrsi"].iloc[-1]
            s = 1 if val < 0.2 else -1 if val > 0.8 else -1
            return s, 0.3, "stochrsi"
        except (KeyError, ValueError, TypeError, IndexError):
            logger.exception("Error in signal_stochrsi")
            return -1, 0.0, "stochrsi"

    def signal_obv(self, df: pd.DataFrame, model=None) -> tuple[int, float, str]:
        if df is None or len(df) < 6:
            return -1, 0.0, "obv"
        try:
            obv = pd.Series(ta.obv(df["close"], df["volume"]).values)
            if len(obv) < 5:
                return -1, 0.0, "obv"
            slope = np.polyfit(range(5), obv.tail(5), 1)[0]
            s = 1 if slope > 0 else -1 if slope < 0 else -1
            w = min(abs(slope) / 1e6, 1.0)
            return s, w, "obv"
        except (KeyError, ValueError, TypeError, IndexError):
            logger.exception("Error in signal_obv")
            return -1, 0.0, "obv"

    def signal_vsa(self, df: pd.DataFrame, model=None) -> tuple[int, float, str]:
        if df is None or len(df) < 20:
            return -1, 0.0, "vsa"
        try:
            body = abs(df["close"] - df["open"])
            vsa = df["volume"] * body
            score = vsa.iloc[-1]
            roll = vsa.rolling(20).mean()
            avg = roll.iloc[-1] if not roll.empty else 0.0
            s = (
                1
                if df["close"].iloc[-1] > df["open"].iloc[-1]
                else -1 if df["close"].iloc[-1] < df["open"].iloc[-1] else -1
            )
            # AI-AGENT-REF: Fix division by zero in VSA signal calculation
            if avg > 0:
                w = min(score / avg, 1.0)
            else:
                w = 0.0  # Safe fallback when average is zero
            return s, w, "vsa"
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ):  # AI-AGENT-REF: narrow exception
            logger.exception("Error in signal_vsa")
            return -1, 0.0, "vsa"

    def signal_ml(
        self, df: pd.DataFrame, model: Any | None = None, symbol: str | None = None
    ) -> tuple[int, float, str] | None:
        """Machine learning prediction signal with probability logging."""
        if model is None and symbol is not None:
            model = _load_ml_model(symbol)
        if model is None or not (
            hasattr(model, "predict") and hasattr(model, "predict_proba")
        ):
            logger.debug(
                "ML_PREDICT_SKIPPED", extra={"reason": "model_missing_or_invalid"}
            )  # AI-AGENT-REF: guard absent model
            return None
        try:
            if hasattr(model, "feature_names_in_"):
                feat = list(model.feature_names_in_)
            else:
                feat = ["rsi", "macd", "atr", "vwap", "sma_50", "sma_200"]

            X = df[feat].iloc[-1].values.reshape(1, -1)
            try:
                pred = model.predict(X)[0]
                proba = float(model.predict_proba(X)[0][pred])
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as e:  # AI-AGENT-REF: narrow exception
                logger.error("signal_ml predict failed: %s", e)
                return -1, 0.0, "ml"
            s = 1 if pred == 1 else -1
            logger.info(
                "ML_SIGNAL", extra={"prediction": int(pred), "probability": proba}
            )
            return s, proba, "ml"
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.exception(f"signal_ml failed: {e}")
            return -1, 0.0, "ml"

    def signal_sentiment(
        self, ctx: BotContext, ticker: str, df: pd.DataFrame = None, model: Any = None
    ) -> tuple[int, float, str]:
        """
        Only fetch sentiment if price has moved > PRICE_TTL_PCT; otherwise, return cached/neutral.
        """
        if df is None or df.empty:
            return -1, 0.0, "sentiment"

        latest_close = float(get_latest_close(df))
        with sentiment_lock:
            prev_close = _LAST_PRICE.get(ticker)

        # If price hasn’t moved enough, return cached or neutral
        if (
            prev_close is not None
            and abs(latest_close - prev_close) / prev_close < PRICE_TTL_PCT
        ):
            with sentiment_lock:
                cached = _SENTIMENT_CACHE.get(ticker)
                if cached and (pytime.time() - cached[0] < SENTIMENT_TTL_SEC):
                    score = cached[1]
                else:
                    score = 0.0
                    _SENTIMENT_CACHE[ticker] = (pytime.time(), score)
        else:
            # Price moved enough → fetch fresh sentiment
            try:
                score = _fetch_sentiment_ctx(ctx, ticker)
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as e:  # AI-AGENT-REF: narrow exception
                logger.warning(f"[signal_sentiment] {ticker} error: {e}")
                score = 0.0

        # Update last‐seen price & cache
        with sentiment_lock:
            _LAST_PRICE[ticker] = latest_close
            _SENTIMENT_CACHE[ticker] = (pytime.time(), score)

        score = max(-1.0, min(1.0, score))
        s = 1 if score > 0 else -1 if score < 0 else -1
        weight = abs(score)
        if is_high_vol_regime():
            weight *= 1.5
        return s, weight, "sentiment"

    def signal_regime(
        self, runtime: BotContext, state: BotState, df: pd.DataFrame, model=None
    ) -> tuple[int, float, str]:
        # AI-AGENT-REF: propagate runtime into regime signal evaluation
        ok = check_market_regime(runtime, state)
        s = 1 if ok else -1
        return s, 1.0, "regime"

    def load_signal_weights(self) -> dict[str, float]:
        if not os.path.exists(SIGNAL_WEIGHTS_FILE):
            return {}
        try:
            df = pd.read_csv(
                SIGNAL_WEIGHTS_FILE,
                on_bad_lines="skip",
                engine="python",
                usecols=["signal_name", "weight"],
            )
            if df.empty:
                logger.warning(
                    "Loaded DataFrame from %s is empty after parsing/fallback",
                    SIGNAL_WEIGHTS_FILE,
                )
                return {}
            return {row["signal_name"]: row["weight"] for _, row in df.iterrows()}
        except ValueError as e:
            if "usecols" in str(e).lower():
                logger.warning(
                    "Signal weights CSV missing expected columns, trying fallback read"
                )
                try:
                    # Fallback: read all columns and try to map
                    df = pd.read_csv(
                        SIGNAL_WEIGHTS_FILE, on_bad_lines="skip", engine="python"
                    )
                    if "signal" in df.columns:
                        # Old format with 'signal' column
                        return {
                            row["signal"]: row["weight"] for _, row in df.iterrows()
                        }
                    elif "signal_name" in df.columns:
                        # New format with 'signal_name' column
                        return {
                            row["signal_name"]: row["weight"]
                            for _, row in df.iterrows()
                        }
                    else:
                        logger.error(
                            "Signal weights CSV has unexpected format: %s",
                            df.columns.tolist(),
                        )
                        return {}
                except (
                    FileNotFoundError,
                    PermissionError,
                    IsADirectoryError,
                    JSONDecodeError,
                    ValueError,
                    KeyError,
                    TypeError,
                    OSError,
                ) as fallback_e:  # AI-AGENT-REF: narrow exception
                    logger.error(
                        "Failed to load signal weights with fallback: %s", fallback_e
                    )
                    return {}
            else:
                logger.error("Failed to load signal weights: %s", e)
                return {}

    def evaluate(
        self,
        ctx: BotContext,
        state: BotState,
        df: pd.DataFrame,
        ticker: str,
        model: Any,
    ) -> tuple[int, float, str]:
        """Evaluate all active signals and return a combined decision.

        Parameters
        ----------
        ctx : BotContext
            Global bot context with API clients and configuration.
        state : BotState
            Current mutable bot state used by some signals.
        df : pandas.DataFrame
            DataFrame containing indicator columns for ``ticker``.
        ticker : str
            Symbol being evaluated.
        model : Any
            Optional machine learning model for ``signal_ml``.

        Returns
        -------
        tuple[int, float, str]
            ``(signal, confidence, label)`` where ``signal`` is -1, 0 or 1.
        """
        signals: list[tuple[int, float, str]] = []
        performance_data = load_global_signal_performance()

        # AI-AGENT-REF: Graceful degradation when no meta-learning data exists
        if performance_data is None:
            # For new deployments, allow all signal types with warning
            logger.info(
                "METALEARN_FALLBACK | No trade history - allowing all signals for new deployment"
            )
            allowed_tags = None  # None means allow all tags
        else:
            allowed_tags = set(performance_data.keys())
            if not allowed_tags:
                logger.warning(
                    "METALEARN_NO_QUALIFIED_SIGNALS | No signals meet performance criteria - using basic signals"
                )
                # Use a basic set of reliable signal types as fallback
                allowed_tags = {"sma_cross", "bb_squeeze", "rsi_oversold", "momentum"}

        self.load_signal_weights()

        # Track total signals evaluated
        if signals_evaluated:
            try:
                signals_evaluated.inc()
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as exc:  # AI-AGENT-REF: narrow exception
                logger.exception("bot.py unexpected", exc_info=exc)
                raise

        # simple moving averages
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["sma_200"] = df["close"].rolling(window=200).mean()

        raw = [
            self.signal_momentum(df, model),
            self.signal_mean_reversion(df, model),
            self.signal_ml(df, model, ticker),
            self.signal_sentiment(ctx, ticker, df, model),
            self.signal_regime(ctx, state, df, model),
            self.signal_stochrsi(df, model),
            self.signal_obv(df, model),
            self.signal_vsa(df, model),
        ]
        # drop skipped signals
        signals = [s for s in raw if s is not None]
        if not signals:
            return 0.0, 0.0, "no_signals"
        self.last_components = signals
        score = sum(s * w for s, w, _ in signals)
        confidence = sum(w for _, w, _ in signals)
        labels = "+".join(label for _, _, label in signals)
        return math.copysign(1, score), confidence, labels


# ─── G. BOT CONTEXT ───────────────────────────────────────────────────────────
@dataclass
class BotContext:
    # Trading client using the new Alpaca SDK
    api: Any
    # Separate market data client
    data_client: Any
    data_fetcher: DataFetcher
    signal_manager: SignalManager
    trade_logger: TradeLogger
    sem: Semaphore
    volume_threshold: int
    entry_start_offset: timedelta
    entry_end_offset: timedelta
    market_open: dt_time
    market_close: dt_time
    regime_lookback: int
    regime_atr_threshold: float
    daily_loss_limit: float
    kelly_fraction: float
    capital_scaler: CapitalScalingEngine
    adv_target_pct: float
    max_position_dollars: float
    params: dict
    sector_cap: float = get_sector_exposure_cap()
    correlation_limit: float = CORRELATION_THRESHOLD
    capital_band: str = "small"
    confirmation_count: dict[str, int] = field(default_factory=dict)
    trailing_extremes: dict[str, float] = field(default_factory=dict)
    take_profit_targets: dict[str, float] = field(default_factory=dict)
    stop_targets: dict[str, float] = field(default_factory=dict)
    portfolio_weights: dict[str, float] = field(default_factory=dict)
    tickers: list[str] = field(default_factory=list)
    rebalance_buys: dict[str, datetime] = field(default_factory=dict)
    # AI-AGENT-REF: track client_order_id base for INITIAL_REBALANCE orders
    rebalance_ids: dict[str, str] = field(default_factory=dict)
    rebalance_attempts: dict[str, int] = field(default_factory=dict)
    trailing_stop_data: dict[str, Any] = field(default_factory=dict)
    risk_engine: RiskEngine | None = None
    allocator: StrategyAllocator | None = None
    strategies: list[Any] = field(default_factory=list)
    execution_engine: ExecutionEngine | None = None
    # AI-AGENT-REF: Add drawdown circuit breaker for real-time protection
    drawdown_circuit_breaker: DrawdownCircuitBreaker | None = None
    logger: logging.Logger = logger

    # AI-AGENT-REF: Add backward compatibility property for alpaca_client
    @property
    def alpaca_client(self):
        """Backward compatibility property for accessing the trading API client."""
        return getattr(self, "api", None)


data_fetcher: DataFetcher | None = None
signal_manager = SignalManager()
# AI-AGENT-REF: Lazy initialization for trade logger to speed up imports in testing
_TRADE_LOGGER_SINGLETON = None


def get_trade_logger() -> TradeLogger:
    """Get trade logger singleton and ensure CSV headers exist."""  # AI-AGENT-REF: ensure default path resolution

    global _TRADE_LOGGER_SINGLETON
    if _TRADE_LOGGER_SINGLETON is None:
        # AI-AGENT-REF: respect configured trade log path
        _TRADE_LOGGER_SINGLETON = TradeLogger(TRADE_LOG_FILE)

    path = _TRADE_LOGGER_SINGLETON.path
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        try:
            with open(path, "w", newline="") as f:
                portalocker.lock(f, portalocker.LOCK_EX)
                try:
                    csv.writer(f).writerow(
                        [
                            "symbol",
                            "entry_time",
                            "entry_price",
                            "exit_time",
                            "exit_price",
                            "qty",
                            "side",
                            "strategy",
                            "classification",
                            "signal_tags",
                            "confidence",
                            "reward",
                        ]
                    )
                finally:
                    portalocker.unlock(f)
        except PermissionError:
            logger.debug("TradeLogger init path not writable: %s", path)
    return _TRADE_LOGGER_SINGLETON


risk_engine = None
allocator = None
strategies = None


from ai_trading.utils.imports import (
    resolve_risk_engine_cls,
    resolve_strategy_allocator_cls,
)

logger = get_logger(__name__)


def get_risk_engine():
    """Get risk engine with fallback to RiskManager from ai_trading.risk.manager."""
    global risk_engine
    if risk_engine is None:
        cls = resolve_risk_engine_cls()
        if cls is None:
            try:
                from ai_trading.risk.manager import (
                    RiskManager as _RM,
                )  # in-package fallback

                _emit_once(
                    logger,
                    "risk_engine_fallback",
                    logging.INFO,
                    "Risk engine: RiskManager (in-package fallback)",
                )
                risk_engine = _RM()
            except COMMON_EXC as e:  # noqa: BLE001 - propagate after logging
                logger.error("RISK_ENGINE_IMPORT_FAILED", extra={"detail": str(e)})
                raise ImportError("Risk engine unavailable") from e
        else:
            _emit_once(
                logger,
                "risk_engine_resolved",
                logging.INFO,
                f"Risk engine: {cls.__module__}.{cls.__name__}",
            )
            risk_engine = cls()
    return risk_engine


def get_allocator():
    global allocator
    if allocator is None:
        cls = resolve_strategy_allocator_cls()
        if cls is None:
            logger.error(
                "StrategyAllocator not found (ai_trading.strategy_allocator, ai_trading.strategies.performance_allocator)."
            )
            raise ImportError("StrategyAllocator unavailable")
        allocator = cls()
    return allocator


def _is_concrete_strategy(obj, BaseStrategy):
    return (
        inspect.isclass(obj)
        and issubclass(obj, BaseStrategy)
        and obj is not BaseStrategy
        and not inspect.isabstract(obj)
    )


def _import_all_strategy_submodules(pkg_name: str):
    """
    Import all submodules under pkg_name so strategy classes defined anywhere under
    ai_trading/strategies become visible without manual re-exports.
    Never raises: logs errors and returns best-effort imported package.
    """
    try:
        pkg = importlib.import_module(pkg_name)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.error("Failed to import %s: %s", pkg_name, e)
        return None
    path = getattr(pkg, "__path__", None)
    if not path:
        return pkg
    for modinfo in pkgutil.walk_packages(path, pkg_name + "."):
        name = modinfo.name
        try:
            submodule = importlib.import_module(name)
            # After importing submodule, scan it for strategy classes and add them to pkg namespace
            for attr_name, attr_obj in vars(submodule).items():
                if (
                    inspect.isclass(attr_obj)
                    and hasattr(attr_obj, "__module__")
                    and attr_obj.__module__ == name
                    and attr_name not in ["BaseStrategy"]
                ):  # Don't re-add BaseStrategy
                    # Add strategy classes to the main package namespace
                    setattr(pkg, attr_name, attr_obj)
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            # Keep going; one bad module shouldn't hide others.
            logger.error("Failed to import strategy module %s: %s", name, e)
    return pkg


def get_strategies():
    """Load configured strategies with safe defaults."""  # AI-AGENT-REF: robust strategy selection
    wanted: list[str] | None = None  # AI-AGENT-REF: ensure variable always defined
    try:
        from ai_trading.config.settings import get_settings

        S = get_settings()
        raw = getattr(S, "STRATEGIES_WANTED", None) or getattr(
            S, "strategies_wanted", None
        )
        if raw:
            if isinstance(raw, str):
                wanted = [raw.lower()]
            else:
                try:
                    wanted = [str(s).lower() for s in raw]
                except TypeError:  # not iterable
                    wanted = [str(raw).lower()]
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ):
        wanted = None

    if not wanted:
        env_raw = os.getenv("STRATEGIES")
        if env_raw:
            wanted = [
                part.strip().lower() for part in env_raw.split(",") if part.strip()
            ]
    if not wanted:
        wanted = ["momentum"]

    try:
        from ai_trading.strategies import (
            REGISTRY,  # type: ignore  # AI-AGENT-REF: lazy import avoids cycles
        )
    except COMMON_EXC as e:  # noqa: BLE001 - best-effort import
        REGISTRY = {}
        logger.error("Failed to import strategy registry: %s", e)

    selected: list[object] = []
    names: list[str] = []
    for name in wanted:
        cls = REGISTRY.get(name)
        if cls is None:
            logger.warning("Unknown strategy %s; skipping.", name)
            continue
        try:
            inst = cls()
            selected.append(inst)
            names.append(cls.__name__)
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:
            logger.error("Failed to instantiate strategy %s: %s", cls.__name__, e)

    if not selected:
        default_cls = REGISTRY.get("momentum")
        if default_cls is not None:
            selected = [default_cls()]
            names = [default_cls.__name__]

    if names:
        _strategies = ", ".join(sorted(names))
        logger_once.info(
            f"Loaded strategies: {_strategies}",
            key="loaded_strategies_once",
        )  # AI-AGENT-REF: pre-format and provide key for logger_once

    return selected


# AI-AGENT-REF: Defer credential validation to runtime instead of import-time
# This prevents crashes during import when environment variables are missing
stream = None


def _initialize_alpaca_clients() -> bool:
    """Initialize Alpaca trading clients lazily to avoid import delays.

    Returns ``True`` on success, ``False`` if initialization fails or is skipped.
    """
    global trading_client, data_client, stream
    if trading_client is not None:
        return True  # Already initialized
    for attempt in (1, 2):
        try:
            key, secret, base_url = _ensure_alpaca_env_or_raise()
        except Exception as e:  # AI-AGENT-REF: surface env resolution failures
            logger.error(
                "ALPACA_ENV_RESOLUTION_FAILED", extra={"error": str(e)}
            )
            if attempt == 1:
                try:
                    config.reload_env(override=False)
                except COMMON_EXC:
                    pass
                continue
            logger_once.error(
                "ALPACA_CLIENT_INIT_FAILED - env", key="alpaca_client_init_failed"
            )
            trading_client = None
            data_client = None
            raise
        if not (key and secret):
            # In SHADOW_MODE we may not have creds; skip client init
            diag = _alpaca_diag_info()
            # AI-AGENT-REF: surface skip reason for tests via standard logger
            logger.info(
                "Shadow mode or missing credentials: skipping Alpaca client initialization"
            )
            logger_once.warning(
                "ALPACA_INIT_SKIPPED - shadow mode or missing credentials",
                key="alpaca_init_skipped",
            )
            logger.info("ALPACA_DIAG", extra=_redact(diag))
            return False
        try:
            AlpacaREST = get_trading_client_cls()
            stock_client_cls = get_data_client_cls()

            logger.debug("Successfully imported Alpaca SDK class")
        except COMMON_EXC as e:
            logger.error(
                "ALPACA_CLIENT_IMPORT_FAILED", extra={"error": str(e)}
            )
            if attempt == 1:
                time.sleep(1)
                continue
            logger_once.error(
                "ALPACA_CLIENT_INIT_FAILED - import", key="alpaca_client_init_failed"
            )
            trading_client = None
            data_client = None
            return False
        try:
            trading_client = AlpacaREST(
                api_key=key,
                secret_key=secret,
                url_override=base_url,
            )
            data_client = stock_client_cls(
                api_key=key,
                secret_key=secret,
            )
        except Exception as e:  # AI-AGENT-REF: expose network or auth issues
            logger.error(
                "ALPACA_CLIENT_INIT_FAILED", extra={"error": str(e)}
            )
            logger_once.error(
                "ALPACA_CLIENT_INIT_FAILED - client", key="alpaca_client_init_failed"
            )
            trading_client = None
            data_client = None
            return False
        logger.info(
            "ALPACA_DIAG", extra=_redact({"initialized": True, **_alpaca_diag_info()})
        )
        stream = None  # initialize stream lazily elsewhere if/when required
        return True
    return False


# IMPORTANT: do not initialize Alpaca clients at import time.
# They will be initialized on-demand by the functions that need them.


init_alpaca_clients = _initialize_alpaca_clients


async def on_trade_update(event):
    """Handle order status updates from the Alpaca stream."""
    try:
        symbol = event.order.symbol
        status = event.order.status
    except AttributeError:
        # Fallback for dict-like event objects
        symbol = event.order.get("symbol") if isinstance(event.order, dict) else "?"
        status = event.order.get("status") if isinstance(event.order, dict) else "?"
    logger.info(f"Trade update for {symbol}: {status}")


# AI-AGENT-REF: Global context and engine will be initialized lazily
_ctx = None
_exec_engine = None


class LazyBotContext:
    """Wrapper that initializes the bot context lazily on first access."""

    def __init__(self):
        self._initialized = False
        self._context = None
        # Do NOT initialize eagerly - that's the whole point of being lazy

    def _ensure_initialized(self):
        """Ensure the context is initialized."""
        _init_metrics()
        global _ctx, _exec_engine, data_fetcher

        if self._initialized and self._context is not None:
            return

        # Initialize Alpaca clients first if needed
        _initialize_alpaca_clients()

        # AI-AGENT-REF: add null check for stream to handle Alpaca unavailable gracefully
        if stream and hasattr(stream, "subscribe_trade_updates"):
            try:
                stream.subscribe_trade_updates(on_trade_update)
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as e:  # AI-AGENT-REF: narrow exception
                logger.warning("Failed to subscribe to trade updates: %s", e)

        fetcher = data_fetcher_module.build_fetcher(params)
        self._context = BotContext(
            api=trading_client,
            data_client=data_client,
            data_fetcher=fetcher,
            signal_manager=signal_manager,
            trade_logger=get_trade_logger(),
            sem=Semaphore(4),
            volume_threshold=get_volume_threshold(),
            entry_start_offset=ENTRY_START_OFFSET,
            entry_end_offset=ENTRY_END_OFFSET,
            market_open=MARKET_OPEN,
            market_close=MARKET_CLOSE,
            regime_lookback=REGIME_LOOKBACK,
            regime_atr_threshold=REGIME_ATR_THRESHOLD,
            daily_loss_limit=get_daily_loss_limit(),
            kelly_fraction=params.get("KELLY_FRACTION", 0.6),
            capital_scaler=CapitalScalingEngine(params),
            adv_target_pct=0.002,
            max_position_dollars=10_000,
            params=params,
            confirmation_count={},
            trailing_extremes={},
            take_profit_targets={},
            stop_targets={},
            portfolio_weights={},
            rebalance_buys={},
            risk_engine=get_risk_engine(),
            allocator=get_allocator(),
            strategies=get_strategies(),
            # AI-AGENT-REF: Initialize drawdown circuit breaker for real-time protection
            drawdown_circuit_breaker=(
                DrawdownCircuitBreaker(
                    max_drawdown=CFG.max_drawdown_threshold, recovery_threshold=0.8
                )
                if DrawdownCircuitBreaker
                else None
            ),
        )
        try:
            ensure_alpaca_attached(self._context)
        except COMMON_EXC:
            pass
        # ExecutionEngine does not accept slippage/metrics kwargs.
        # Metrics remain tracked in bot_engine via Prometheus counters.
        _exec_engine = ExecutionEngine(
            self._context
        )  # AI-AGENT-REF: remove unsupported kwargs
        self._context.execution_engine = _exec_engine
        data_fetcher = fetcher
        # One-time, mandatory model load
        if getattr(self._context, "model", None) is None:
            if _MODEL_CACHE is None:
                self._context.model = _load_required_model()
            else:
                self._context.model = _MODEL_CACHE

        # Propagate the capital_scaler to the risk engine so that position_size
        self._context.risk_engine.capital_scaler = self._context.capital_scaler

        # Complete context setup (only in non-test environments)
        if not (os.getenv("PYTEST_RUNNING") or os.getenv("TESTING")):
            try:
                _initialize_bot_context_post_setup(self._context)
            except NameError:
                logger.debug("_initialize_bot_context_post_setup not present; skipping.")

        _ctx = self._context
        self._initialized = True

        # AI-AGENT-REF: Mark runtime as ready after context is fully initialized
        global _RUNTIME_READY
        _RUNTIME_READY = True

    def __setattr__(self, name, value):
        """Delegate attribute setting to the underlying context."""
        if name.startswith("_") or name in ("_initialized", "_context"):
            super().__setattr__(name, value)
        else:
            self._ensure_initialized()
            setattr(self._context, name, value)

    # Explicit properties for commonly accessed attributes
    @property
    def api(self):
        self._ensure_initialized()
        return self._context.api

    @property
    def data_client(self):
        self._ensure_initialized()
        return self._context.data_client

    @property
    def data_fetcher(self):
        self._ensure_initialized()
        return self._context.data_fetcher

    @property
    def signal_manager(self):
        self._ensure_initialized()
        return self._context.signal_manager

    @property
    def risk_engine(self):
        self._ensure_initialized()
        return self._context.risk_engine

    @property
    def capital_scaler(self):
        self._ensure_initialized()
        return self._context.capital_scaler

    @property
    def execution_engine(self):
        self._ensure_initialized()
        return self._context.execution_engine

    @property
    def stop_targets(self):
        self._ensure_initialized()
        return self._context.stop_targets

    @property
    def take_profit_targets(self):
        self._ensure_initialized()
        return self._context.take_profit_targets

    @property
    def confirmation_count(self):
        self._ensure_initialized()
        return self._context.confirmation_count

    @property
    def symbols(self):
        self._ensure_initialized()
        return getattr(self._context, "symbols", [])

    @property
    def sem(self):
        self._ensure_initialized()
        return self._context.sem

    @property
    def volume_threshold(self):
        self._ensure_initialized()
        return self._context.volume_threshold

    @property
    def kelly_fraction(self):
        self._ensure_initialized()
        return self._context.kelly_fraction

    @property
    def max_position_dollars(self):
        self._ensure_initialized()
        return self._context.max_position_dollars

    @property
    def trailing_extremes(self):
        self._ensure_initialized()
        return self._context.trailing_extremes

    @property
    def params(self):
        self._ensure_initialized()
        return self._context.params

    # Allow setting attributes by delegating to context
    def __setattr__(self, name, value):
        if name.startswith("_") or name in ("_initialized", "_context"):
            super().__setattr__(name, value)
        else:
            self._ensure_initialized()
            setattr(self._context, name, value)


# AI-AGENT-REF: No module-level context creation to prevent import-time side effects
# Context will be created when first accessed via get_ctx() or _get_bot_context()
_global_ctx = None


def get_ctx():
    """Get the global bot context (backwards compatibility)."""
    global _global_ctx
    if _global_ctx is None:
        _global_ctx = LazyBotContext()
    return _global_ctx


# AI-AGENT-REF: Defer context initialization to prevent expensive operations during import
# The context will be created when first accessed via get_ctx() or _get_bot_context()


# Central place to acquire the runtime context from the lazy singleton
def _get_runtime_context_or_none():
    """Get runtime context safely, returning None if not ready."""
    try:
        lbc = get_ctx()
        lbc._ensure_initialized()
        return lbc._context
    except Exception as e:  # AI-AGENT-REF: generic catch for safety
        _log.warning("Runtime context unavailable for risk exposure update: %s", e)
        return None


def _emit_periodic_metrics():
    """
    Emit periodic metrics if enabled and runtime is ready.

    AI-AGENT-REF: Periodic lightweight metrics emission via existing metrics_logger.
    """
    if not hasattr(S, "metrics_enabled") or not CFG.metrics_enabled:
        return

    if not is_runtime_ready():
        logger.debug("Skipping metrics emission: runtime not ready")
        return

    runtime = _get_runtime_context_or_none()
    if runtime is None:
        return

    try:
        from ai_trading.monitoring import metrics as _metrics

        account = getattr(runtime, "api", None)
        if account:
            account_obj = account.get_account()
            positions = account.list_positions()
            _metrics.emit_account_health(account_obj, positions)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.debug("Metrics emission failed: %s", e)


def _update_risk_engine_exposure():
    """
    Called by the scheduler/loop with runtime readiness gate.

    AI-AGENT-REF: Enhanced with readiness gate to prevent early context access warnings.
    """
    runtime = _get_runtime_context_or_none()
    if runtime is None:
        return

    risk_engine = getattr(runtime, "risk_engine", None)
    if not risk_engine:
        _log.debug("No risk_engine on runtime context; skipping exposure update.")
        return

    try:
        risk_engine.update_exposure(runtime)
        risk_engine.wait_for_exposure_update(timeout=0.5)
    except Exception as e:  # AI-AGENT-REF: generic catch for safety
        _log.warning("Risk engine exposure update failed: %s", e)


def data_source_health_check(ctx: BotContext, symbols: Sequence[str]) -> None:
    """Log warnings if no market data is available on startup."""
    missing: list[str] = []
    for sym in symbols:
        df = ctx.data_fetcher.get_daily_df(ctx, sym)
        if df is None or df.empty:
            missing.append(sym)
    if not symbols:
        return
    if len(missing) == len(symbols):
        try:
            # AI-AGENT-REF: compute prior session from aware UTC now
            session = last_market_session(pd.Timestamp.now(tz="UTC"))  # AI-AGENT-REF: prior session window
            if session is not None:
                start_ts = session.open.tz_convert("UTC")
                end_ts = session.close.tz_convert("UTC")
                df_min = get_minute_df("SPY", start_ts, end_ts, feed="iex")  # AI-AGENT-REF: robust minute fetch
                if df_min.empty:
                    df_min = get_minute_df("SPY", start_ts, end_ts, feed="sip")
                if df_min.empty:
                    logger.warning(
                        "DATA_HEALTH_CHECK: minute fallback still empty (rows=0)"
                    )
                else:
                    logger.info(
                        "DATA_HEALTH_CHECK: minute fallback ok",
                        extra={"rows": int(df_min.shape[0])},
                    )
                    return
            else:
                logger.info(
                    "DATA_HEALTH_CHECK: no prior market session (weekend/holiday); skip minute fallback"
                )
        except (ValueError, TypeError) as e:  # pragma: no cover - defensive
            logger.warning("DATA_HEALTH_CHECK: minute fallback exception: %s", e)
        if not is_market_open():
            logger.info(
                "DATA_SOURCE_HEALTH_CHECK: No data for any symbol (market closed; health check deferred)."
            )
        else:
            logger.warning(
                "DATA_SOURCE_HEALTH_CHECK: No data for any symbol. Possible API outage or market holiday."
            )
    elif missing:
        logger.info(
            "DATA_SOURCE_HEALTH_CHECK: missing daily data for %s",
            ", ".join(missing),
        )
# AI-AGENT-REF: Module-level health check removed to prevent NameError: ctx
# Health check now happens safely in _initialize_bot_context_post_setup() after context creation


@memory_profile  # AI-AGENT-REF: Monitor memory usage during health checks
def pre_trade_health_check(
    ctx: BotContext, symbols: Sequence[str], min_rows: int | None = None
) -> dict:
    """
    Validate symbol data sufficiency, required columns, and timezone sanity using chunked batch.

    Robust min_rows resolution:
      1) explicit param
      2) ctx.min_rows if present
      3) default 120
    Avoids `'BotContext' object has no attribute 'min_rows'` hard failures.
    """
    # Robust min_rows resolution with precedence
    if min_rows is None:
        min_rows = getattr(ctx, "min_rows", 120)
    min_rows = int(min_rows)

    results = {
        "checked": 0,
        "failures": [],
        "insufficient_rows": [],
        "missing_columns": [],
        "timezone_issues": [],
    }
    if not symbols:
        return results
    # Compute start/end with fallbacks so this function is safe to call early in the loop
    settings = get_settings()
    _now = datetime.now(UTC)
    _fallback_days = int(getattr(settings, "pretrade_lookback_days", 120))
    _start = getattr(ctx, "lookback_start", _now - timedelta(days=_fallback_days))
    _end = getattr(ctx, "lookback_end", _now)
    frames = _fetch_universe_bars_chunked(
        symbols=symbols,
        timeframe="1D",
        start=_start,
        end=_end,
        feed=getattr(ctx, "data_feed", None),
    )
    for sym in symbols:
        df = frames.get(sym)
        if df is None or getattr(df, "empty", False):
            results["failures"].append((sym, "no_data"))
            continue
        results["checked"] += 1
        try:
            # Use the function parameter, not a non-existent ctx attribute
            if len(df) < min_rows:
                results["insufficient_rows"].append(sym)
                continue
            _validate_columns(
                df,
                required=["timestamp", "open", "high", "low", "close", "volume"],
                results=results,
                symbol=sym,
            )
            _validate_timezones(df, results, sym)
        except (
            APIError,
            TimeoutError,
            ConnectionError,
            KeyError,
            ValueError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: tighten health probe error handling
            results["failures"].append((sym, str(e)))
            logger.warning(
                "HEALTH_CHECK_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
    return results


def _validate_columns(df, required, results, symbol):
    """Helper to validate required columns are present."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        results["missing_columns"].append(symbol)


def _validate_timezones(df, results, symbol):
    """Helper to validate timezone information."""
    if hasattr(df, "index") and hasattr(df.index, "tz") and df.index.tz is None:
        results["timezone_issues"].append(symbol)


# ─── H. MARKET HOURS GUARD ────────────────────────────────────────────────────


def in_trading_hours(ts: pd.Timestamp) -> bool:
    if is_holiday(ts):
        logger.warning(
            f"No NYSE market schedule for {ts.date()}; skipping market open/close check."
        )
        return False
    cal = get_market_calendar()
    try:
        return cal.open_at_time(get_market_schedule(), ts)
    except (AttributeError, ValueError) as exc:
        logger.warning(f"Invalid schedule time {ts}: {exc}; assuming market closed")
        return False


# ─── I. SENTIMENT & EVENTS ────────────────────────────────────────────────────
@sleep_and_retry
@limits(calls=30, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
)
def get_sec_headlines(ctx: BotContext, ticker: str) -> str:
    with ctx.sem:
        r = http.get(
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={ticker}&type=8-K&count=5",
            headers={"User-Agent": "AI Trading Bot"},
            timeout=clamp_request_timeout(HTTP_TIMEOUT),  # AI-AGENT-REF: explicit timeout
        )
        r.raise_for_status()

    try:
        soup = BeautifulSoup(r.content, "lxml")
        texts = []
        for a in soup.find_all("a", string=re.compile(r"8[- ]?K")):
            tr = a.find_parent("tr")
            tds = tr.find_all("td") if tr else []
            if len(tds) >= 4:
                texts.append(tds[-1].get_text(strip=True))
        return " ".join(texts)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.warning(f"[get_sec_headlines] parse failed for {ticker}: {e}")
        return ""


def _check_sentiment_circuit_breaker() -> bool:
    """Check if sentiment circuit breaker allows requests."""
    global _SENTIMENT_CIRCUIT_BREAKER
    now = pytime.time()
    cb = _SENTIMENT_CIRCUIT_BREAKER

    if cb["state"] == "open":
        if now - cb["last_failure"] > SENTIMENT_RECOVERY_TIMEOUT:
            cb["state"] = "half-open"
            logger.info("Sentiment circuit breaker moved to half-open state")
            return True
        return False
    if now < cb.get("next_retry", 0):
        logger.debug("Sentiment retry delayed %.1fs", cb["next_retry"] - now)
        return False
    return True


def _record_sentiment_success():
    """Record successful sentiment API call."""
    global _SENTIMENT_CIRCUIT_BREAKER
    _SENTIMENT_CIRCUIT_BREAKER["failures"] = 0
    _SENTIMENT_CIRCUIT_BREAKER["next_retry"] = 0
    if _SENTIMENT_CIRCUIT_BREAKER["state"] == "half-open":
        _SENTIMENT_CIRCUIT_BREAKER["state"] = "closed"
        logger.info("Sentiment circuit breaker closed - service recovered")


def _record_sentiment_failure():
    """Record failed sentiment API call and update circuit breaker."""
    global _SENTIMENT_CIRCUIT_BREAKER
    cb = _SENTIMENT_CIRCUIT_BREAKER
    cb["failures"] += 1
    cb["last_failure"] = pytime.time()
    delay = min(
        SENTIMENT_BASE_DELAY * (2 ** (cb["failures"] - 1)),
        SENTIMENT_RECOVERY_TIMEOUT,
    )
    cb["next_retry"] = cb["last_failure"] + delay
    if cb["failures"] >= SENTIMENT_FAILURE_THRESHOLD:
        cb["state"] = "open"
        logger.warning(
            f"Sentiment circuit breaker opened after {cb['failures']} failures"
        )
    else:
        logger.debug(
            "Sentiment failure %s; next retry in %.1fs", cb["failures"], delay
        )


@retry(
    stop=stop_after_attempt(
        2
    ),  # Reduced from 3 to avoid hitting rate limits too quickly
    wait=wait_exponential(multiplier=1, min=2, max=10),  # Increased delays
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
)
def _fetch_sentiment_ctx(ctx: BotContext, ticker: str) -> float:
    """
    Fetch sentiment via NewsAPI + FinBERT + Form 4 signal.
    Uses a simple in-memory TTL cache to avoid hitting NewsAPI too often.
    If FinBERT isn’t available, return neutral 0.0.
    """
    from ai_trading.config.settings import get_settings  # AI-AGENT-REF: modern settings source

    settings = get_settings()
    # AI-AGENT-REF: guard missing attributes across older Settings versions
    api_key = (
        getattr(settings, "sentiment_api_key", None)
        or getattr(settings, "azure_language_key", None)
        or getattr(settings, "news_api_key", None)
        or get_news_api_key()
    )
    if not api_key:
        logger.debug(
            "No sentiment API key configured (checked settings.sentiment_api_key, azure_language_key, news_api_key; env fallback)"
        )
        return 0.0

    now_ts = pytime.time()

    # AI-AGENT-REF: Enhanced caching with longer TTL during rate limiting
    with sentiment_lock:
        cached = _SENTIMENT_CACHE.get(ticker)
        if cached:
            last_ts, last_score = cached
            # Use longer cache during circuit breaker open state
            cache_ttl = (
                SENTIMENT_RATE_LIMITED_TTL_SEC
                if _SENTIMENT_CIRCUIT_BREAKER["state"] == "open"
                else SENTIMENT_TTL_SEC
            )
            if now_ts - last_ts < cache_ttl:
                logger.debug(
                    f"Sentiment cache hit for {ticker} (age: {(now_ts - last_ts) / 60:.1f}m)"
                )
                return last_score

    # Cache miss or stale → fetch fresh
    # AI-AGENT-REF: Circuit breaker pattern for graceful degradation
    if not _check_sentiment_circuit_breaker():
        logger.info(
            f"Sentiment circuit breaker open, returning cached/neutral for {ticker}"
        )
        with sentiment_lock:
            # Try to use any existing cache, even if stale
            cached = _SENTIMENT_CACHE.get(ticker)
            if cached:
                _, last_score = cached
                logger.debug(f"Using stale cached sentiment {last_score} for {ticker}")
                return last_score
            # No cache available, store and return neutral
            _SENTIMENT_CACHE[ticker] = (now_ts, 0.0)
            return 0.0

    try:
        # 1) Fetch NewsAPI articles using configurable URL
        url = (
            f"{settings.sentiment_api_url}?"
            f"q={ticker}&sortBy=publishedAt&language=en&pageSize=5"
            f"&apiKey={api_key}"
        )
        resp = http.get(url, timeout=clamp_request_timeout(HTTP_TIMEOUT))  # AI-AGENT-REF: explicit timeout

        if resp.status_code == 429:
            # AI-AGENT-REF: Enhanced rate limiting handling
            logger.warning(
                f"fetch_sentiment({ticker}) rate-limited → caching neutral with extended TTL"
            )
            _record_sentiment_failure()
            with sentiment_lock:
                # Cache neutral score with extended TTL during rate limiting
                _SENTIMENT_CACHE[ticker] = (now_ts, 0.0)
            return 0.0

        resp.raise_for_status()

        payload = resp.json()
        articles = payload.get("articles", [])
        scores = []
        if articles:
            for art in articles:
                text = (art.get("title") or "") + ". " + (art.get("description") or "")
                if text.strip():
                    scores.append(predict_text_sentiment(text))
        news_score = float(sum(scores) / len(scores)) if scores else 0.0

        # 2) Fetch Form 4 data (insider trades) - with error handling
        form4_score = 0.0
        try:
            form4 = fetch_form4_filings(ticker)
            # If any insider buy in last 7 days > $50k, boost sentiment
            for filing in form4:
                if filing["type"] == "buy" and filing["dollar_amount"] > 50_000:
                    form4_score += 0.1
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.debug(
                f"Form4 fetch failed for {ticker}: {e}"
            )  # Reduced to debug level

        final_score = 0.8 * news_score + 0.2 * form4_score
        final_score = max(-1.0, min(1.0, final_score))

        # AI-AGENT-REF: Record success and update cache
        _record_sentiment_success()
        with sentiment_lock:
            _SENTIMENT_CACHE[ticker] = (now_ts, final_score)
        return final_score

    except requests.exceptions.RequestException as e:
        logger.warning(f"Sentiment API request failed for {ticker}: {e}")
        _record_sentiment_failure()

        # AI-AGENT-REF: Fallback to cached data or neutral if no cache
        with sentiment_lock:
            cached = _SENTIMENT_CACHE.get(ticker)
            if cached:
                _, last_score = cached
                logger.debug(f"Using cached sentiment fallback {last_score} for {ticker}")
                return last_score
            # No cache available, return neutral
            _SENTIMENT_CACHE[ticker] = (now_ts, 0.0)
            return 0.0
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.error(f"Unexpected error fetching sentiment for {ticker}: {e}")
        _record_sentiment_failure()
        with sentiment_lock:
            _SENTIMENT_CACHE[ticker] = (now_ts, 0.0)
        return 0.0


def predict_text_sentiment(text: str, cfg=None) -> float:
    """
    Uses FinBERT (if available) to assign a sentiment score ∈ [–1, +1].
    FinBERT-backed sentiment; neutral fallback if disabled/unavailable.
    Returns positive probability (or 0.0 neutral when disabled/missing).
    """
    pair = ensure_finbert(cfg)
    if not pair or pair[0] is None or pair[1] is None:
        return 0.0  # neutral fallback
    tokenizer, model = pair
    import torch  # safe: ensure_finbert verified presence

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]  # shape = (3,)
            probs = torch.softmax(logits, dim=0)  # [p_neg, p_neu, p_pos]

        neg, neu, pos = probs.tolist()
        return float(pos - neg)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.warning(
            f"[predict_text_sentiment] FinBERT inference failed ({e}); returning neutral"
        )
        return 0.0


def analyze_sentiment(text: str, cfg=None) -> float:
    """
    FinBERT-backed sentiment; neutral fallback if disabled/unavailable.
    Returns positive probability or a sentiment score.
    """
    pair = ensure_finbert(cfg)
    if not pair or pair[0] is None or pair[1] is None:
        return 0.0  # neutral fallback
    tok, mdl = pair
    import torch  # safe: ensure_finbert verified presence

    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = mdl(**inputs).logits
    # assuming index 2 = positive per finbert-tone convention
    return float(logits.softmax(dim=-1)[0, 2].item())


def fetch_form4_filings(ticker: str) -> list[dict]:
    """
    Scrape SEC Form 4 filings for insider trade info.
    Returns a list of dicts: {"date": datetime, "type": "buy"/"sell", "dollar_amount": float}.
    """
    url = f"https://www.sec.gov/cgi-bin/own-disp?action=getowner&CIK={ticker}&type=4"
    r = http.get(
        url,
        headers={"User-Agent": "AI Trading Bot"},
        timeout=clamp_request_timeout(HTTP_TIMEOUT),  # AI-AGENT-REF: explicit timeout
    )
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "lxml")
    filings = []
    # Parse table rows (approximate)
    table = soup.find("table", {"class": "tableFile2"})
    if not table:
        return filings
    rows = table.find_all("tr")[1:]  # skip header
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 6:
            continue
        date_str = cols[3].get_text(strip=True)
        try:
            fdate = datetime.strptime(date_str, "%Y-%m-%d")
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ):  # AI-AGENT-REF: narrow exception
            continue
        txn_type = cols[4].get_text(strip=True).lower()  # "purchase" or "sale"
        amt_str = cols[5].get_text(strip=True).replace("$", "").replace(",", "")
        try:
            amt = float(amt_str)
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ):  # AI-AGENT-REF: narrow exception
            amt = 0.0
        filings.append(
            {
                "date": fdate,
                "type": ("buy" if "purchase" in txn_type else "sell"),
                "dollar_amount": amt,
            }
        )
    return filings


def _can_fetch_events(symbol: str) -> bool:
    now_ts = pytime.time()
    last_ts = _LAST_EVENT_TS.get(symbol, 0)
    if now_ts - last_ts < EVENT_COOLDOWN:
        if event_cooldown_hits:
            try:
                event_cooldown_hits.inc()
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as exc:  # AI-AGENT-REF: narrow exception
                logger.exception("bot.py unexpected", exc_info=exc)
                raise
        return False
    _LAST_EVENT_TS[symbol] = now_ts
    return True


_calendar_cache: dict[str, pd.DataFrame] = {}
_calendar_last_fetch: dict[str, date] = {}


# AI-AGENT-REF: optional yfinance calendar fetch
def _fetch_calendar_via_yf(symbol: str) -> pd.DataFrame:
    yf = get_yfinance() if YFINANCE_AVAILABLE else None
    if yf is None:
        logger.warning(
            "YF_PROVIDER_UNAVAILABLE", extra={"provider": "yfinance", "symbol": symbol}
        )
        return pd.DataFrame()
    try:
        cal = yf.Ticker(symbol).calendar
    except HTTPError as e:
        logger.warning(
            "YF_CALENDAR_HTTP_ERROR",
            extra={
                "provider": "yfinance",
                "symbol": symbol,
                "cause": e.__class__.__name__,
                "detail": str(e),
            },
        )
        return pd.DataFrame()
    except COMMON_EXC as e:  # noqa: BLE001
        logger.warning(
            "YF_CALENDAR_FAILED",
            extra={
                "provider": "yfinance",
                "symbol": symbol,
                "cause": e.__class__.__name__,
                "detail": str(e),
            },
        )
        return pd.DataFrame()
    if cal is None or getattr(cal, "empty", False):
        logger.warning(
            "YF_CALENDAR_EMPTY", extra={"provider": "yfinance", "symbol": symbol}
        )
        return pd.DataFrame()
    return cal


def get_calendar_safe(symbol: str) -> pd.DataFrame:
    today_date = date.today()
    if symbol in _calendar_cache and _calendar_last_fetch.get(symbol) == today_date:
        return _calendar_cache[symbol]
    cal = _fetch_calendar_via_yf(symbol)
    _calendar_cache[symbol] = cal
    _calendar_last_fetch[symbol] = today_date
    return cal


def is_near_event(symbol: str, days: int = 3) -> bool:
    cal = get_calendar_safe(symbol)
    if not hasattr(cal, "empty") or cal.empty:
        return False
    try:
        dates = []
        for col in cal.columns:
            if "Value" not in cal.index or col not in cal.columns:
                continue
            raw = cal.at["Value", col]
            if isinstance(raw, list | tuple):
                raw = raw[0]
            dates.append(pd.to_datetime(raw))
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ):  # AI-AGENT-REF: narrow exception
        logger.debug(
            f"[Events] Malformed calendar for {symbol}, columns={getattr(cal, 'columns', None)}"
        )
        return False
    today_ts = pd.Timestamp.now().normalize()
    cutoff = today_ts + pd.Timedelta(days=days)
    return any(today_ts <= d <= cutoff for d in dates)


# ─── J. RISK & GUARDS ─────────────────────────────────────────────────────────


@sleep_and_retry
@limits(calls=200, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(APIError),
)
def check_daily_loss(ctx: BotContext, state: BotState) -> bool:
    acct = safe_alpaca_get_account(ctx)
    if acct is None:
        logger.warning("Daily loss check skipped - Alpaca account unavailable")
        return False
    equity = float(acct.equity)
    today_date = date.today()
    limit = params.get("get_daily_loss_limit()", 0.07)

    if state.day_start_equity is None or state.day_start_equity[0] != today_date:
        if state.last_drawdown >= 0.05:
            limit = 0.03
        state.last_drawdown = (
            (state.day_start_equity[1] - equity) / state.day_start_equity[1]
            if state.day_start_equity
            else 0.0
        )
        state.day_start_equity = (today_date, equity)
        daily_drawdown.set(0.0)
        return False

    loss = (state.day_start_equity[1] - equity) / state.day_start_equity[1]
    daily_drawdown.set(loss)
    if loss > 0.05:
        logger.warning("[WARNING] Daily drawdown = %.2f%%", loss * 100)
    return loss >= limit


def check_weekly_loss(ctx: BotContext, state: BotState) -> bool:
    """Weekly portfolio drawdown guard."""
    acct = safe_alpaca_get_account(ctx)
    if acct is None:
        logger.warning("Weekly loss check skipped - Alpaca account unavailable")
        return False
    equity = float(acct.equity)
    today_date = date.today()
    week_start = today_date - timedelta(days=today_date.weekday())

    if state.week_start_equity is None or state.week_start_equity[0] != week_start:
        state.week_start_equity = (week_start, equity)
        weekly_drawdown.set(0.0)
        return False

    loss = (state.week_start_equity[1] - equity) / state.week_start_equity[1]
    weekly_drawdown.set(loss)
    return loss >= WEEKLY_DRAWDOWN_LIMIT


def count_day_trades() -> int:
    try:  # AI-AGENT-REF: gate heavy import
        import pandas as pd  # type: ignore
    except ImportError:
        return 0
    df = _read_trade_log(TRADE_LOG_FILE, usecols=["entry_time", "exit_time"])
    if df is None:
        return 0
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    df = df.dropna(subset=["entry_time", "exit_time"])
    today_ts = pd.Timestamp.now().normalize()
    bdays = pd.bdate_range(end=today_ts, periods=5)
    df["entry_date"] = df["entry_time"].dt.normalize()
    df["exit_date"] = df["exit_time"].dt.normalize()
    mask = (
        (df["entry_date"].isin(bdays))
        & (df["exit_date"].isin(bdays))
        & (df["entry_date"] == df["exit_date"])
    )
    return int(mask.sum())


@sleep_and_retry
@limits(calls=200, period=60)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(APIError),
)
def check_pdt_rule(runtime) -> bool:
    """Check PDT rule with graceful degradation when Alpaca is unavailable.

    Returns False when Alpaca is unavailable, allowing the bot to continue
    operating in simulation mode.
    """
    ensure_alpaca_attached(runtime)
    acct = safe_alpaca_get_account(runtime)

    # If account is unavailable (Alpaca not available), assume no PDT blocking
    if acct is None:  # AI-AGENT-REF: contract now explicit; keep None-check
        # Log once to prevent per-cycle PDT spam
        logger_once.info(
            "PDT_CHECK_SKIPPED - Alpaca unavailable, assuming no PDT restrictions",
            key="pdt_check_skipped",
        )  # AI-AGENT-REF: rate-limit PDT skip message
        return False

    try:
        equity = float(acct.equity)
    except (AttributeError, TypeError, ValueError):
        logger.warning(
            "PDT_CHECK_FAILED - Invalid equity value, assuming no PDT restrictions"
        )
        return False

    # AI-AGENT-REF: Improve API null value handling for PDT checks
    api_day_trades_raw = getattr(acct, "pattern_day_trades", None) or getattr(
        acct, "pattern_day_trades_count", None
    )
    api_buying_pw_raw = (
        getattr(acct, "daytrade_buying_power", None)
        or getattr(acct, "day_trade_buying_power", None)
        or getattr(acct, "buying_power", None)
    )
    api_day_trades = as_int(api_day_trades_raw, 0)
    api_buying_pw = as_float(api_buying_pw_raw, 0.0)

    logger.info(
        "PDT_CHECK",
        extra={
            "equity": equity,
            "api_day_trades": api_day_trades,
            "api_buying_pw": api_buying_pw,
        },
    )

    if api_day_trades is not None and api_day_trades >= PDT_DAY_TRADE_LIMIT:
        logger.info("SKIP_PDT_RULE", extra={"api_day_trades": api_day_trades})
        return True

    if equity < PDT_EQUITY_THRESHOLD:
        if api_buying_pw and float(api_buying_pw) > 0:
            logger.warning(
                "PDT_EQUITY_LOW", extra={"equity": equity, "buying_pw": api_buying_pw}
            )
        else:
            logger.warning(
                "PDT_EQUITY_LOW_NO_BP",
                extra={"equity": equity, "buying_pw": api_buying_pw},
            )
            return True

    return False


def set_halt_flag(reason: str) -> None:
    """Persist a halt flag with the provided reason."""
    try:
        with open(HALT_FLAG_PATH, "w") as f:
            f.write(f"{reason} " + dt_.now(UTC).isoformat())
        logger.info(f"TRADING_HALTED set due to {reason}")
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # pragma: no cover - disk issues  # AI-AGENT-REF: narrow exception
        logger.error(f"Failed to write halt flag: {exc}")


def check_halt_flag(runtime) -> bool:
    """
    Determine whether the trading loop should halt for safety/ops.
    Priority:
      1) Env var AI_TRADING_HALT=1
      2) Config-defined halt file with truthy content
      3) runtime.halt boolean attribute
    """
    import os

    # AI-AGENT-REF: thread runtime into halt checks and drop global ctx
    # 1) Environment override
    if os.getenv("AI_TRADING_HALT", "").strip() in {"1", "true", "True"}:
        return True

    # 2) Config file flag (if provided)
    halt_file = getattr(getattr(runtime, "cfg", None), "halt_file", None)
    if isinstance(halt_file, str) and halt_file:
        try:
            if os.path.exists(halt_file):
                with open(halt_file, encoding="utf-8", errors="ignore") as fh:
                    content = fh.read().strip()
                if content and content not in {"0", "false", "False"}:
                    return True
        except OSError as e:
            # AI-AGENT-REF: log read issues without raising
            logger.info(
                "HALT_FLAG_READ_ISSUE", extra={"halt_file": halt_file, "error": str(e)}
            )

    # 3) Runtime attribute
    if bool(getattr(runtime, "halt", False)):
        return True

    return False


def too_many_positions(ctx: BotContext, symbol: str | None = None) -> bool:
    """Check if there are too many positions, with allowance for rebalancing."""
    try:
        current_positions = ctx.api.list_positions()
        position_count = len(current_positions)

        # If we're not at the limit, allow new positions
        if position_count < get_max_portfolio_positions():
            return False

        # If we're at the limit, check if this is a rebalancing opportunity
        if symbol and position_count >= get_max_portfolio_positions():
            # Allow trades for symbols we already have positions in (rebalancing)
            existing_symbols = {pos.symbol for pos in current_positions}
            if symbol in existing_symbols:
                logger.info(
                    f"ALLOW_REBALANCING | symbol={symbol} existing_positions={position_count}"
                )
                return False

            # For new symbols at position limit, check if we can close underperforming positions
            # This implements intelligent position management
            logger.info(
                f"POSITION_LIMIT_REACHED | current={position_count} max={get_max_portfolio_positions()} new_symbol={symbol}"
            )

        return position_count >= get_max_portfolio_positions()

    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.warning(f"[too_many_positions] Could not fetch positions: {e}")
        return False


def too_correlated(ctx: BotContext, sym: str) -> bool:
    try:  # AI-AGENT-REF: gate heavy import
        import pandas as pd  # type: ignore
    except ImportError:
        return False
    df = _read_trade_log(TRADE_LOG_FILE, usecols=["symbol", "exit_time"])
    if df is None or "exit_time" not in df.columns or "symbol" not in df.columns:
        return False
    open_syms = df.loc[df.exit_time == "", "symbol"].unique().tolist() + [sym]
    rets: dict[str, pd.Series] = {}
    for s in open_syms:
        d = ctx.data_fetcher.get_daily_df(ctx, s)
        if d is None or d.empty:
            continue
        # Handle DataFrame with MultiIndex columns (symbol, field) or single-level
        if isinstance(d.columns, pd.MultiIndex):
            if (s, "close") in d.columns:
                series = d[(s, "close")].pct_change(fill_method=None).dropna()
            else:
                continue
        else:
            series = d["close"].pct_change(fill_method=None).dropna()
        if not series.empty:
            rets[s] = series

    if len(rets) < 2:
        return False
    min_len = min(len(r) for r in rets.values())
    if min_len < 1:
        return False
    good_syms = [s for s, r in rets.items() if len(r) >= min_len]
    idx = rets[good_syms[0]].tail(min_len).index
    mat = pd.DataFrame({s: rets[s].tail(min_len).values for s in good_syms}, index=idx)
    corr_matrix = mat.corr().abs()
    avg_corr = corr_matrix.where(~np.eye(len(good_syms), dtype=bool)).stack().mean()
    limit = getattr(ctx, "correlation_limit", CORRELATION_THRESHOLD)
    return avg_corr > limit


# AI-AGENT-REF: optional yfinance sector fetch
def _fetch_sector_via_yf(symbol: str) -> str | None:
    yf = get_yfinance() if YFINANCE_AVAILABLE else None
    if yf is None:
        logger.warning(
            "YF_PROVIDER_UNAVAILABLE", extra={"provider": "yfinance", "symbol": symbol}
        )
        return None
    try:
        info = yf.Ticker(symbol).info
    except COMMON_EXC as e:  # noqa: BLE001
        logger.warning(
            "YF_SECTOR_FAILED",
            extra={
                "provider": "yfinance",
                "symbol": symbol,
                "cause": e.__class__.__name__,
                "detail": str(e),
            },
        )
        return None
    sector = info.get("sector")
    if not sector or sector == "Unknown":
        logger.warning(
            "YF_SECTOR_EMPTY", extra={"provider": "yfinance", "symbol": symbol}
        )
        return None
    return sector


def get_sector(symbol: str) -> str:
    """
    Get sector classification for a stock symbol.
    Uses yfinance API with fallback to hardcoded mappings for common stocks.
    """
    if symbol in _SECTOR_CACHE:
        return _SECTOR_CACHE[symbol]

    # AI-AGENT-REF: Fallback sector mappings for common stocks when yfinance fails
    SECTOR_MAPPINGS = {
        # Technology
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Technology",
        "GOOG": "Technology",
        "AMZN": "Technology",
        "TSLA": "Technology",
        "META": "Technology",
        "NVDA": "Technology",
        "NFLX": "Technology",
        "AMD": "Technology",
        "INTC": "Technology",
        "ORCL": "Technology",
        "CRM": "Technology",
        "ADBE": "Technology",
        "PYPL": "Technology",
        "UBER": "Technology",
        "SQ": "Technology",
        "SHOP": "Technology",
        "TWLO": "Technology",
        "ZM": "Technology",
        "PLTR": "Technology",  # AI-AGENT-REF: Added PLTR to Technology sector per problem statement
        "BABA": "Technology",  # AI-AGENT-REF: Added BABA to Technology sector per problem statement
        "JD": "Technology",
        "PDD": "Technology",
        "TCEHY": "Technology",  # Additional Chinese tech stocks
        # Financial Services
        "JPM": "Financial Services",
        "BAC": "Financial Services",
        "WFC": "Financial Services",
        "GS": "Financial Services",
        "MS": "Financial Services",
        "C": "Financial Services",
        "V": "Financial Services",
        "MA": "Financial Services",
        "BRK-B": "Financial Services",
        "AXP": "Financial Services",
        # Healthcare
        "JNJ": "Healthcare",
        "PFE": "Healthcare",
        "ABBV": "Healthcare",
        "MRK": "Healthcare",
        "UNH": "Healthcare",
        "TMO": "Healthcare",
        "MDT": "Healthcare",
        "ABT": "Healthcare",
        "LLY": "Healthcare",
        "BMY": "Healthcare",
        "AMGN": "Healthcare",
        "GILD": "Healthcare",
        # Consumer Cyclical
        "AMZN": "Consumer Cyclical",
        "HD": "Consumer Cyclical",
        "NKE": "Consumer Cyclical",
        "MCD": "Consumer Cyclical",
        "SBUX": "Consumer Cyclical",
        "DIS": "Consumer Cyclical",
        "LOW": "Consumer Cyclical",
        "TGT": "Consumer Cyclical",
        # Consumer Defensive
        "PG": "Consumer Defensive",
        "KO": "Consumer Defensive",
        "PEP": "Consumer Defensive",
        "WMT": "Consumer Defensive",
        "COST": "Consumer Defensive",
        "CL": "Consumer Defensive",
        # Communication Services
        "GOOGL": "Communication Services",
        "GOOG": "Communication Services",
        "META": "Communication Services",
        "NFLX": "Communication Services",
        "DIS": "Communication Services",
        "VZ": "Communication Services",
        "T": "Communication Services",
        "CMCSA": "Communication Services",
        # Energy
        "XOM": "Energy",
        "CVX": "Energy",
        "COP": "Energy",
        "EOG": "Energy",
        "SLB": "Energy",
        # Industrials
        "BA": "Industrials",
        "CAT": "Industrials",
        "GE": "Industrials",
        "MMM": "Industrials",
        "UPS": "Industrials",
        "HON": "Industrials",
        "LMT": "Industrials",
        "RTX": "Industrials",
        # Utilities
        "NEE": "Utilities",
        "DUK": "Utilities",
        "SO": "Utilities",
        "D": "Utilities",
        # Real Estate
        "AMT": "Real Estate",
        "CCI": "Real Estate",
        "EQIX": "Real Estate",
        "PSA": "Real Estate",
        # Materials
        "LIN": "Basic Materials",
        "APD": "Basic Materials",
        "ECL": "Basic Materials",
        "DD": "Basic Materials",
        # ETFs - treat as diversified
        "SPY": "Diversified",
        "QQQ": "Technology",
        "IWM": "Diversified",
        "VTI": "Diversified",
        "VOO": "Diversified",
        "VEA": "Diversified",
        "VWO": "Diversified",
        "BND": "Fixed Income",
        "TLT": "Fixed Income",
        "GLD": "Commodities",
        "SLV": "Commodities",
    }

    # First try fallback mapping
    if symbol in SECTOR_MAPPINGS:
        sector = SECTOR_MAPPINGS[symbol]
        _SECTOR_CACHE[symbol] = sector
        logger.debug(f"Using fallback sector mapping for {symbol}: {sector}")
        return sector

    sector = _fetch_sector_via_yf(symbol)
    if sector:
        _SECTOR_CACHE[symbol] = sector
        logger.debug(
            "YF_SECTOR_SUCCESS",
            extra={"provider": "yfinance", "symbol": symbol, "sector": sector},
        )
        return sector

    # Default to Unknown if all methods fail
    sector = "Unknown"
    _SECTOR_CACHE[symbol] = sector
    logger.warning("YF_SECTOR_UNKNOWN", extra={"provider": "yfinance", "symbol": symbol})
    return sector


def sector_exposure(ctx: BotContext) -> dict[str, float]:
    """Return current portfolio exposure by sector as fraction of equity."""
    try:
        positions = ctx.api.list_positions()
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ):  # AI-AGENT-REF: narrow exception
        return {}
    try:
        total = float(ctx.api.get_account().portfolio_value)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ):  # AI-AGENT-REF: narrow exception
        total = 0.0
    exposure: dict[str, float] = {}
    for pos in positions:
        qty = abs(int(getattr(pos, "qty", 0)))
        price = float(
            getattr(pos, "current_price", 0) or getattr(pos, "avg_entry_price", 0) or 0
        )
        sec = get_sector(getattr(pos, "symbol", ""))
        val = qty * price
        exposure[sec] = exposure.get(sec, 0.0) + val
    if total <= 0:
        return dict.fromkeys(exposure, 0.0)
    return {k: v / total for k, v in exposure.items()}


def sector_exposure_ok(ctx: BotContext, symbol: str, qty: int, price: float) -> bool:
    """Return True if adding qty*price of symbol keeps sector exposure within cap."""
    sec = get_sector(symbol)
    exposures = sector_exposure(ctx)
    try:
        total = float(ctx.api.get_account().portfolio_value)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.warning(
            f"SECTOR_EXPOSURE_PORTFOLIO_ERROR: Failed to get portfolio value for {symbol}: {e}"
        )
        total = 0.0

    # Calculate trade value and exposure metrics
    trade_value = qty * price
    current_sector_exposure = exposures.get(sec, 0.0)
    projected_exposure = (
        current_sector_exposure + (trade_value / total) if total > 0 else 0.0
    )
    cap = getattr(ctx, "sector_cap", get_sector_exposure_cap())

    # AI-AGENT-REF: Enhanced sector cap logic with clear reasoning
    if total <= 0:
        # For empty portfolios, allow initial positions as they can't exceed sector caps
        logger.info(
            f"SECTOR_EXPOSURE_EMPTY_PORTFOLIO: Allowing initial position for {symbol} (sector: {sec})"
        )
        return True

    # AI-AGENT-REF: Special handling for "Unknown" sector to prevent false concentration
    if sec == "Unknown":
        # Use a higher cap for Unknown sector since it's a catch-all category
        # and may contain diversified stocks that couldn't be classified
        unknown_cap = min(
            cap * 2.0, 0.8
        )  # Allow up to 2x normal cap or 80%, whichever is lower
        logger.debug(
            f"SECTOR_EXPOSURE_UNKNOWN: Using relaxed cap {unknown_cap:.1%} for Unknown sector"
        )
        cap = unknown_cap

    # Log detailed exposure analysis
    exposure_pct = current_sector_exposure * 100
    projected_pct = projected_exposure * 100
    cap_pct = cap * 100

    # AI-AGENT-REF: Enhanced debugging for sector exposure analysis
    logger.info(
        f"SECTOR_EXPOSURE_DEBUG: {symbol} analysis - "
        f"Sector: {sec}, Trade Value: ${trade_value:,.2f}, "
        f"Portfolio Value: ${total:,.2f}, "
        f"Current Sector Exposure: {exposure_pct:.1f}%, "
        f"Projected Exposure: {projected_pct:.1f}%, "
        f"Sector Cap: {cap_pct:.1f}%"
    )

    logger.debug(
        f"SECTOR_EXPOSURE_ANALYSIS: {symbol} (sector: {sec}) - "
        f"Current: {exposure_pct:.1f}%, Projected: {projected_pct:.1f}%, Cap: {cap_pct:.1f}%"
    )

    if projected_exposure <= cap:
        logger.debug(
            f"SECTOR_EXPOSURE_OK: {symbol} trade approved - projected exposure {projected_pct:.1f}% within {cap_pct:.1f}% cap"
        )
        return True
    else:
        # Provide clear reasoning for sector cap rejection
        excess_pct = (projected_exposure - cap) * 100
        logger.warning(
            f"SECTOR_EXPOSURE_EXCEEDED: {symbol} trade rejected - "
            f"projected exposure {projected_pct:.1f}% exceeds {cap_pct:.1f}% cap by {excess_pct:.1f}%",
            extra={
                "symbol": symbol,
                "sector": sec,
                "current_exposure_pct": exposure_pct,
                "projected_exposure_pct": projected_pct,
                "cap_pct": cap_pct,
                "excess_pct": excess_pct,
                "trade_value": trade_value,
                "portfolio_value": total,
                "reason": "sector_concentration_risk",
            },
        )
        return False


# ─── K. SIZING & EXECUTION HELPERS ─────────────────────────────────────────────
def is_within_entry_window(ctx: BotContext, state: BotState) -> bool:
    """Return True if current time is during regular Eastern trading hours."""
    now_et = datetime.now(UTC).astimezone(ZoneInfo("America/New_York"))
    start = dt_time(9, 30)
    end = dt_time(16, 0)
    if not (start <= now_et.time() <= end):
        logger.info(
            "SKIP_ENTRY_WINDOW",
            extra={"start": start, "end": end, "now": now_et.time()},
        )
        return False
    if (
        state.streak_halt_until
        and datetime.now(UTC).astimezone(PACIFIC) < state.streak_halt_until
    ):
        logger.info("SKIP_STREAK_HALT", extra={"until": state.streak_halt_until})
        return False
    return True


def scaled_atr_stop(
    entry_price: float,
    atr: float,
    now: datetime,
    market_open: datetime,
    market_close: datetime,
    max_factor: float = 2.0,
    min_factor: float = 0.5,
) -> tuple[float, float]:
    """Calculate scaled ATR stop-loss and take-profit with comprehensive validation."""
    try:
        # AI-AGENT-REF: Add comprehensive input validation for stop-loss calculation

        # Validate entry price
        if not isinstance(entry_price, int | float) or entry_price <= 0:
            logger.error("Invalid entry price for ATR stop: %s", entry_price)
            return (
                entry_price * 0.95,
                entry_price * 1.05,
            )  # Return conservative defaults

        # Validate ATR
        if not isinstance(atr, int | float) or atr < 0:
            logger.error("Invalid ATR for stop calculation: %s", atr)
            return entry_price * 0.95, entry_price * 1.05

        if atr == 0:
            logger.warning("ATR is zero, using 1% stop/take levels")
            return entry_price * 0.99, entry_price * 1.01

        # Validate datetime inputs
        if not all(isinstance(dt, datetime) for dt in [now, market_open, market_close]):
            logger.error("Invalid datetime inputs for ATR stop calculation")
            return entry_price * 0.95, entry_price * 1.05

        # Validate market times make sense
        if market_close <= market_open:
            logger.error(
                "Invalid market times: close=%s <= open=%s", market_close, market_open
            )
            return entry_price * 0.95, entry_price * 1.05

        # Validate factors
        if not isinstance(max_factor, int | float) or max_factor <= 0:
            logger.warning("Invalid max_factor %s, using default 2.0", max_factor)
            max_factor = 2.0

        if not isinstance(min_factor, int | float) or min_factor < 0:
            logger.warning("Invalid min_factor %s, using default 0.5", min_factor)
            min_factor = 0.5

        if min_factor > max_factor:
            logger.warning(
                "min_factor %s > max_factor %s, swapping", min_factor, max_factor
            )
            min_factor, max_factor = max_factor, min_factor

        # Calculate time-based scaling factor
        total = (market_close - market_open).total_seconds()
        elapsed = (now - market_open).total_seconds()

        # Handle edge cases
        if total <= 0:
            logger.warning("Invalid market session duration: %s seconds", total)
            α = 0.5  # Use middle factor
        else:
            α = max(0, min(1, 1 - elapsed / total))

        factor = min_factor + α * (max_factor - min_factor)

        # Validate factor is reasonable
        if factor <= 0 or factor > 10:  # Sanity check - no more than 10x ATR
            logger.warning("Calculated factor %s out of bounds, capping", factor)
            factor = max(0.1, min(factor, 10.0))

        stop = entry_price - factor * atr
        take = entry_price + factor * atr

        # Validate calculated levels are reasonable
        if stop < 0:
            logger.warning("Calculated stop price %s is negative, adjusting", stop)
            stop = entry_price * 0.5  # Minimum 50% stop

        if take <= entry_price:
            logger.warning(
                "Calculated take profit %s <= entry price %s, adjusting",
                take,
                entry_price,
            )
            take = entry_price * 1.1  # Minimum 10% profit target

        # Ensure stop is below entry and take is above entry
        if stop >= entry_price:
            logger.warning(
                "Stop price %s >= entry price %s, adjusting", stop, entry_price
            )
            stop = entry_price * 0.95

        if take <= entry_price:
            logger.warning(
                "Take profit %s <= entry price %s, adjusting", take, entry_price
            )
            take = entry_price * 1.05

        logger.debug(
            "ATR stop calculation: entry=%s, atr=%s, factor=%s, stop=%s, take=%s",
            entry_price,
            atr,
            factor,
            stop,
            take,
        )

        return stop, take

    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.error("Error in ATR stop calculation: %s", e)
        # Return conservative defaults on error
        return entry_price * 0.95, entry_price * 1.05


def liquidity_factor(ctx: BotContext, symbol: str) -> float:
    try:
        df = fetch_minute_df_safe(symbol)
    except DataFetchError:
        logger.warning("[liquidity_factor] no data for %s", symbol)
        return 0.0
    if df is None or df.empty:
        return 0.0
    if "volume" not in df.columns:
        return 0.0
    avg_vol = df["volume"].tail(30).mean()
    try:
        req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        quote: Quote = ctx.data_client.get_stock_latest_quote(req)
        spread = (
            (quote.ask_price - quote.bid_price)
            if quote.ask_price and quote.bid_price
            else 0.0
        )
    except APIError as e:
        logger.warning(f"[liquidity_factor] Alpaca quote failed for {symbol}: {e}")
        spread = 0.0
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ):  # AI-AGENT-REF: narrow exception
        spread = 0.0
    vol_score = min(1.0, avg_vol / ctx.volume_threshold) if avg_vol else 0.0

    # AI-AGENT-REF: More reasonable spread scoring to reduce excessive retries
    # Dynamic spread threshold based on volume - high volume stocks can handle wider spreads
    base_spread_threshold = 0.05
    volume_adjusted_threshold = base_spread_threshold * (
        1 + min(1.0, avg_vol / 1000000)
    )
    spread_score = max(
        0.2, 1 - spread / volume_adjusted_threshold
    )  # Min 0.2 instead of 0.0

    # Combine scores with less aggressive penalization
    final_score = (vol_score * 0.7) + (
        spread_score * 0.3
    )  # Weight volume more than spread

    return max(0.1, min(1.0, final_score))  # Min 0.1 to avoid complete blocking


def fractional_kelly_size(
    ctx: BotContext,
    balance: float,
    price: float,
    atr: float,
    win_prob: float,
    payoff_ratio: float = 1.5,
) -> int:
    """Calculate position size using fractional Kelly criterion with comprehensive validation."""
    # AI-AGENT-REF: Add comprehensive input validation for Kelly calculation
    try:
        # Validate inputs
        if not isinstance(balance, int | float) or balance <= 0:
            logger.error("Invalid balance for Kelly calculation: %s", balance)
            return 0

        if not isinstance(price, int | float) or price <= 0:
            logger.error("Invalid price for Kelly calculation: %s", price)
            return 0

        if not isinstance(atr, int | float) or atr < 0:
            logger.warning(
                "Invalid ATR for Kelly calculation: %s, using minimum position", atr
            )
            return 1

        # AI-AGENT-REF: Normalize confidence values to valid probability range
        if not isinstance(win_prob, int | float):
            logger.error(
                "Invalid win probability type for Kelly calculation: %s", win_prob
            )
            return 0

        # Handle confidence values that exceed 1.0 by normalizing them
        if win_prob > 1.0:
            logger.debug("Normalizing confidence value %s to probability", win_prob)
            # Use sigmoid function to map confidence to probability range [0,1]
            # This preserves the relative ordering while constraining to valid range
            win_prob = 1.0 / (1.0 + math.exp(-win_prob + 1.0))
            logger.debug("Normalized win probability: %s", win_prob)
        elif win_prob < 0:
            logger.warning("Negative confidence value %s, using 0.0", win_prob)
            win_prob = 0.0

        if not isinstance(payoff_ratio, int | float) or payoff_ratio <= 0:
            logger.error("Invalid payoff ratio for Kelly calculation: %s", payoff_ratio)
            return 0

        # Validate ctx object and its attributes
        if not hasattr(ctx, "kelly_fraction") or not isinstance(
            ctx.kelly_fraction, int | float
        ):
            logger.error("Invalid kelly_fraction in context")
            return 0

        if not hasattr(ctx, "max_position_dollars") or not isinstance(
            ctx.max_position_dollars, int | float
        ):
            logger.error("Invalid max_position_dollars in context")
            return 0

        # AI-AGENT-REF: adaptive kelly fraction based on historical peak equity
        if os.path.exists(PEAK_EQUITY_FILE):
            try:
                with open(PEAK_EQUITY_FILE, "r+") as lock:
                    portalocker.lock(lock, portalocker.LOCK_EX)
                    try:
                        try:
                            data = lock.read()
                        except io.UnsupportedOperation:
                            logger.warning(
                                "Cannot read peak equity file, using current balance"
                            )
                            return 0
                        prev_peak = float(data) if data else balance
                        if prev_peak <= 0:
                            logger.warning(
                                "Invalid peak equity %s, using current balance",
                                prev_peak,
                            )
                            prev_peak = balance
                    finally:
                        portalocker.unlock(lock)
            except (OSError, ValueError) as e:
                logger.warning(
                    "Error reading peak equity file: %s, using current balance", e
                )
                prev_peak = balance
        else:
            prev_peak = balance

        update_if_present(ctx, balance)
        base_frac = ctx.kelly_fraction * capital_scale(ctx)

        # Validate base_frac
        if not isinstance(base_frac, int | float) or base_frac < 0 or base_frac > 1:
            logger.error("Invalid base fraction calculated: %s", base_frac)
            return 0

        drawdown = (prev_peak - balance) / prev_peak if prev_peak > 0 else 0

        # Apply drawdown-based risk reduction
        if drawdown > 0.10:
            frac = 0.3
        elif drawdown > 0.05:
            frac = 0.45
        else:
            frac = base_frac

        # Apply volatility-based risk reduction
        try:
            if is_high_vol_thr_spy():
                frac *= 0.5
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.warning("Error checking SPY volatility: %s", e)

        cap_scale = frac / base_frac if base_frac > 0 else 1.0

        # Calculate Kelly edge with validation
        # AI-AGENT-REF: Fix division by zero in Kelly criterion calculation
        if payoff_ratio <= 0:
            logger.warning(
                "Invalid payoff_ratio %s for Kelly calculation, using zero position",
                payoff_ratio,
            )
            edge = 0
            kelly = 0
        else:
            edge = win_prob - (1 - win_prob) / payoff_ratio
            kelly = max(edge / payoff_ratio, 0) * frac

        # Validate Kelly fraction is reasonable
        if kelly < 0 or kelly > 1:
            logger.warning("Kelly fraction %s out of bounds, capping", kelly)
            kelly = max(0, min(kelly, 1))

        dollars_to_risk = kelly * balance

        if atr <= 0:
            logger.warning("ATR is zero or negative, using minimum position size")
            try:
                new_peak = max(balance, prev_peak)
                with open(PEAK_EQUITY_FILE, "w") as lock:
                    portalocker.lock(lock, portalocker.LOCK_EX)
                    try:
                        lock.write(str(new_peak))
                    finally:
                        portalocker.unlock(lock)
            except OSError as e:
                logger.warning("Error updating peak equity file: %s", e)
            return 1

        # Calculate position sizes with multiple caps
        raw_pos = dollars_to_risk / atr if atr > 0 else 0
        cap_pos = (balance * get_capital_cap() * cap_scale) / price if price > 0 else 0
        risk_cap = (balance * get_dollar_risk_limit()) / atr if atr > 0 else raw_pos
        dollar_cap = ctx.max_position_dollars / price if price > 0 else raw_pos

        # Apply all limits
        size = int(
            round(min(raw_pos, cap_pos, risk_cap, dollar_cap, MAX_POSITION_SIZE))
        )
        size = max(size, 1)  # Ensure minimum position size

        # Validate final size is reasonable
        if size > MAX_POSITION_SIZE:
            logger.warning("Position size %s exceeds maximum, capping", size)
            size = MAX_POSITION_SIZE

        # Update peak equity
        try:
            new_peak = max(balance, prev_peak)
            with open(PEAK_EQUITY_FILE, "w") as lock:
                portalocker.lock(lock, portalocker.LOCK_EX)
                try:
                    lock.write(str(new_peak))
                finally:
                    portalocker.unlock(lock)
        except OSError as e:
            logger.warning("Error updating peak equity file: %s", e)

        logger.debug(
            "Kelly calculation: balance=%s, price=%s, atr=%s, win_prob=%s, size=%s",
            balance,
            price,
            atr,
            win_prob,
            size,
        )

        return size

    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.error("Error in Kelly calculation: %s", e)
        return 0

    return size


def vol_target_position_size(
    cash: float, price: float, returns: np.ndarray, target_vol: float = 0.02
) -> int:
    sigma = np.std(returns)
    if sigma <= 0 or price <= 0:
        return 1
    dollar_alloc = cash * (target_vol / sigma)
    qty = int(round(dollar_alloc / price))
    return max(qty, 1)


def compute_kelly_scale(vol: float, sentiment: float) -> float:
    """Return basic Kelly scaling factor."""
    base = 1.0
    if vol > 0.05:
        base *= 0.5
    if sentiment < 0:
        base *= 0.5
    return max(base, 0.1)


def adjust_position_size(position, scale: float) -> None:
    """Placeholder for adjusting position quantity."""
    try:
        position.qty = str(int(int(position.qty) * scale))
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ):  # AI-AGENT-REF: narrow exception
        logger.debug("adjust_position_size no-op")


def adjust_trailing_stop(position, new_stop: float) -> None:
    """Placeholder for adjusting trailing stop price."""
    logger.debug("adjust_trailing_stop %s -> %.2f", position.symbol, new_stop)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(APIError),
)
def submit_order(ctx: BotContext, symbol: str, qty: int, side: str) -> Order | None:
    """Submit an order using the institutional execution engine."""
    if not market_is_open():
        logger.warning("MARKET_CLOSED_ORDER_SKIP", extra={"symbol": symbol})
        return None

    # AI-AGENT-REF: Add validation for execution engine initialization
    if _exec_engine is None:
        logger.error(
            "EXEC_ENGINE_NOT_INITIALIZED",
            extra={"symbol": symbol, "qty": qty, "side": side},
        )
        raise RuntimeError("Execution engine not initialized. Cannot execute orders.")

    # AI-AGENT-REF: Liquidity checks before order submission (gated by flag)
    if hasattr(S, "liquidity_checks_enabled") and CFG.liquidity_checks_enabled:
        try:
            from ai_trading.execution.liquidity import LiquidityManager

            lm = LiquidityManager()
            lm.pre_trade_check(
                {"symbol": symbol, "qty": qty, "side": side},
                getattr(ctx, "market_data", None),
            )
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.warning("Liquidity checks failed open-loop: %s", e)

    try:
        return _exec_engine.execute_order(symbol, OrderSide(side.lower()), qty)
    except (APIError, TimeoutError, ConnectionError) as e:
        logger.error(
            "BROKER_OP_FAILED",
            extra={
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "cause": e.__class__.__name__,
                "detail": str(e),
            },
        )
        raise


def safe_submit_order(api: Any, req) -> Order | None:
    if not market_is_open():
        logger.warning(
            "MARKET_CLOSED_ORDER_SKIP", extra={"symbol": getattr(req, "symbol", "")}
        )
        return None

    def _req_to_args(r):
        side = getattr(r, "side", "")
        side = getattr(side, "value", str(side)).lower()
        tif = getattr(r, "time_in_force", "day")
        tif = getattr(tif, "value", str(tif)).lower()
        order_type_map = {
            "MarketOrderRequest": "market",
            "LimitOrderRequest": "limit",
            "StopOrderRequest": "stop",
            "StopLimitOrderRequest": "stop_limit",
        }
        otype = order_type_map.get(r.__class__.__name__, "market")
        args = {
            "symbol": getattr(r, "symbol", ""),
            "qty": getattr(r, "qty", 0),
            "side": side,
            "type": otype,
            "time_in_force": tif,
            "client_order_id": getattr(r, "client_order_id", None),
        }
        if hasattr(r, "limit_price"):
            args["limit_price"] = r.limit_price
        if hasattr(r, "stop_price"):
            args["stop_price"] = r.stop_price
        return args

    order_args = _req_to_args(req)

    for attempt in range(2):
        try:
            try:
                acct = api.get_account()
            except (APIError, TimeoutError, ConnectionError):
                acct = None
            if acct and order_args.get("side") == "buy":
                price = order_args.get("limit_price") or order_args.get("notional", 0)
                need = float(price or 0) * float(order_args.get("qty", 0))
                if need > float(getattr(acct, "buying_power", 0)):
                    logger.warning(
                        "insufficient buying power for %s: requested %s, available %s",
                        order_args.get("symbol"),
                        order_args.get("qty"),
                        getattr(acct, "buying_power", 0),
                    )
                    return None
            if order_args.get("side") == "sell":
                try:
                    positions = api.list_positions()
                except (APIError, TimeoutError, ConnectionError):
                    positions = []
                avail = next(
                    (
                        float(p.qty)
                        for p in positions
                        if p.symbol == order_args.get("symbol")
                    ),
                    0.0,
                )
                if float(order_args.get("qty", 0)) > avail:
                    logger.warning(
                        f"insufficient qty available for {order_args.get('symbol')}: requested {order_args.get('qty')}, available {avail}"
                    )
                    return None

            try:
                order = api.submit_order(**order_args)
            except APIError as e:
                if getattr(e, "code", None) == 40310000:
                    available = int(
                        getattr(e, "_raw_errors", [{}])[0].get("available", 0)
                    )
                    if available > 0:
                        logger.info(
                            f"Adjusting order for {order_args.get('symbol')} to available qty={available}"
                        )
                        order_args["qty"] = available
                        order = api.submit_order(**order_args)
                    else:
                        logger.warning(
                            f"Skipping {order_args.get('symbol')}, no available qty"
                        )
                        continue
                else:
                    raise

            start_ts = time.monotonic()
            while getattr(order, "status", None) == OrderStatus.PENDING_NEW:
                if time.monotonic() - start_ts > 1:
                    logger.warning(
                        f"Order stuck in PENDING_NEW: {order_args.get('symbol')}, retrying or monitoring required."
                    )
                    break
                time.sleep(0.1)  # AI-AGENT-REF: avoid busy polling
                order = api.get_order(order.id)
            logger.info(
                f"Order status for {order_args.get('symbol')}: {getattr(order, 'status', '')}"
            )
            status = getattr(order, "status", "")
            filled_qty = getattr(order, "filled_qty", "0")
            if status == "filled":
                logger.info(
                    "ORDER_ACK",
                    extra={
                        "symbol": order_args.get("symbol"),
                        "order_id": getattr(order, "id", ""),
                    },
                )
            elif status == "partially_filled":
                logger.warning(
                    f"Order partially filled for {order_args.get('symbol')}: {filled_qty}/{order_args.get('qty', 0)}"
                )
            elif status in ("rejected", "canceled"):
                logger.error(
                    f"Order for {order_args.get('symbol')} was {status}: {getattr(order, 'reject_reason', '')}"
                )
                raise OrderExecutionError(
                    f"Buy failed for {order_args.get('symbol')}: {status}"
                )
            elif status == OrderStatus.NEW:
                logger.info(f"Order for {order_args.get('symbol')} is NEW; awaiting fill")
            else:
                logger.error(
                    f"Order for {order_args.get('symbol')} status={status}: {getattr(order, 'reject_reason', '')}"
                )
            return order
        except APIError as e:
            if "insufficient qty" in str(e).lower():
                logger.warning(
                    f"insufficient qty available for {order_args.get('symbol')}: {e}"
                )
                return None
            time.sleep(1)
            if attempt == 1:
                logger.error(
                    "BROKER_OP_FAILED",
                    extra={
                        "cause": e.__class__.__name__,
                        "detail": str(e),
                        "op": "submit",
                        "symbol": order_args.get("symbol", ""),
                    },
                )
                raise
        except (TimeoutError, ConnectionError) as e:
            time.sleep(1)
            if attempt == 1:
                logger.error(
                    "BROKER_OP_FAILED",
                    extra={
                        "cause": e.__class__.__name__,
                        "detail": str(e),
                        "op": "submit",
                        "symbol": getattr(req, "symbol", ""),
                    },
                )
                raise
    return None


def poll_order_fill_status(ctx: BotContext, order_id: str, timeout: int = 120) -> None:
    """Poll Alpaca for order fill status until it is no longer open."""
    start = pytime.time()
    while pytime.time() - start < timeout:
        try:
            od = ctx.api.get_order(order_id)
            status = getattr(od, "status", "")
            filled = getattr(od, "filled_qty", "0")
            if status not in {"new", "accepted", "partially_filled"}:
                logger.info(
                    "ORDER_FINAL_STATUS",
                    extra={
                        "order_id": order_id,
                        "status": status,
                        "filled_qty": filled,
                    },
                )
                return
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.warning(f"[poll_order_fill_status] failed for {order_id}: {e}")
            return
        pytime.sleep(3)


def send_exit_order(
    ctx: BotContext,
    symbol: str,
    exit_qty: int,
    price: float,
    reason: str,
    raw_positions: list | None = None,
) -> None:
    logger.info(
        f"EXIT_SIGNAL | symbol={symbol}  reason={reason}  exit_qty={exit_qty}  price={price}"
    )
    if raw_positions is not None and not any(
        getattr(p, "symbol", "") == symbol for p in raw_positions
    ):
        logger.info("SKIP_NO_POSITION", extra={"symbol": symbol})
        return
    try:
        pos = ctx.api.get_position(symbol)
        held_qty = int(pos.qty)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ):  # AI-AGENT-REF: narrow exception
        held_qty = 0

    if held_qty < exit_qty:
        logger.warning(
            f"No shares available to exit for {symbol} (requested {exit_qty}, have {held_qty})"
        )
        return

    if price <= 0.0:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=exit_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = safe_submit_order(ctx.api, req)
        if order is not None:
            from ai_trading.strategies.base import StrategySignal as TradeSignal

            try:
                acct = ctx.api.get_account()
                eq = float(getattr(acct, "equity", 0) or 0)
                wt = (exit_qty * price) / eq if eq > 0 else 0.0
                ctx.risk_engine.register_fill(
                    TradeSignal(
                        symbol=symbol,
                        side="sell",
                        confidence=1.0,
                        strategy="exit",
                        weight=abs(wt),
                        asset_class="equity",
                    )
                )
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ):  # AI-AGENT-REF: narrow exception
                logger.debug("register_fill exit failed", exc_info=True)
        return

    limit_order = safe_submit_order(
        ctx.api,
        LimitOrderRequest(
            symbol=symbol,
            qty=exit_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=price,
        ),
    )
    if limit_order is not None:
        from ai_trading.strategies.base import StrategySignal as TradeSignal

        try:
            acct = ctx.api.get_account()
            eq = float(getattr(acct, "equity", 0) or 0)
            wt = (exit_qty * price) / eq if eq > 0 else 0.0
            ctx.risk_engine.register_fill(
                TradeSignal(
                    symbol=symbol,
                    side="sell",
                    confidence=1.0,
                    strategy="exit",
                    weight=abs(wt),
                    asset_class="equity",
                )
            )
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ):  # AI-AGENT-REF: narrow exception
            logger.debug("register_fill exit failed", exc_info=True)
    pytime.sleep(5)
    try:
        o2 = ctx.api.get_order(limit_order.id)
        if getattr(o2, "status", "") in {"new", "accepted", "partially_filled"}:
            ctx.api.cancel_order(limit_order.id)
            safe_submit_order(
                ctx.api,
                MarketOrderRequest(
                    symbol=symbol,
                    qty=exit_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                ),
            )
    except (APIError, TimeoutError, ConnectionError) as e:
        logger.error(
            "BROKER_OP_FAILED",
            extra={
                "cause": e.__class__.__name__,
                "detail": str(e),
                "op": "cancel",
                "order_id": getattr(limit_order, "id", ""),
            },
        )
        raise


def twap_submit(
    ctx: BotContext,
    symbol: str,
    total_qty: int,
    side: str,
    window_secs: int = 600,
    n_slices: int = 10,
) -> None:
    slice_qty = total_qty // n_slices
    wait_secs = window_secs / n_slices
    for i in range(n_slices):
        try:
            submit_order(ctx, symbol, slice_qty, side)
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error(
                "BROKER_OP_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
            raise
        pytime.sleep(wait_secs)


def vwap_pegged_submit(
    ctx: BotContext, symbol: str, total_qty: int, side: str, duration: int = 300
) -> None:
    start_time = pytime.time()
    placed = 0
    while placed < total_qty and pytime.time() - start_time < duration:
        try:
            df = fetch_minute_df_safe(symbol)
        except DataFetchError:
            logger.error("[VWAP] no minute data for %s", symbol)
            break
        if df is None or df.empty:
            logger.warning(
                "[VWAP] missing bars, aborting VWAP slice", extra={"symbol": symbol}
            )
            break
        vwap_price = ta.vwap(df["high"], df["low"], df["close"], df["volume"]).iloc[-1]
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quote: Quote = ctx.data_client.get_stock_latest_quote(req)
            spread = (
                (quote.ask_price - quote.bid_price)
                if quote.ask_price and quote.bid_price
                else 0.0
            )
        except APIError as e:
            logger.warning(f"[vwap_slice] Alpaca quote failed for {symbol}: {e}")
            spread = 0.0
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ):  # AI-AGENT-REF: narrow exception
            spread = 0.0
        if spread > 0.05:
            slice_qty = max(1, int((total_qty - placed) * 0.5))
        else:
            slice_qty = min(max(1, total_qty // 10), total_qty - placed)
        order = None
        for attempt in range(3):
            try:
                logger.info(
                    "ORDER_SENT",
                    extra={
                        "timestamp": utc_now_iso(),
                        "symbol": symbol,
                        "side": side,
                        "qty": slice_qty,
                        "order_type": "limit",
                    },
                )
                order = safe_submit_order(
                    ctx.api,
                    LimitOrderRequest(
                        symbol=symbol,
                        qty=slice_qty,
                        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                        time_in_force=TimeInForce.IOC,
                        limit_price=round(vwap_price, 2),
                    ),
                )
                logger.info(
                    "ORDER_ACK",
                    extra={
                        "symbol": symbol,
                        "order_id": getattr(order, "id", ""),
                        "status": getattr(order, "status", ""),
                    },
                )
                Thread(
                    target=poll_order_fill_status,
                    args=(ctx, getattr(order, "id", "")),
                    daemon=True,
                ).start()
                fill_price = float(getattr(order, "filled_avg_price", 0) or 0)
                if fill_price > 0:
                    slip = (fill_price - vwap_price) * 100
                    if slippage_total:
                        try:
                            slippage_total.inc(abs(slip))
                        except (
                            FileNotFoundError,
                            PermissionError,
                            IsADirectoryError,
                            JSONDecodeError,
                            ValueError,
                            KeyError,
                            TypeError,
                            OSError,
                        ) as exc:  # AI-AGENT-REF: narrow exception
                            logger.exception("bot.py unexpected", exc_info=exc)
                            raise
                    if slippage_count:
                        try:
                            slippage_count.inc()
                        except (
                            FileNotFoundError,
                            PermissionError,
                            IsADirectoryError,
                            JSONDecodeError,
                            ValueError,
                            KeyError,
                            TypeError,
                            OSError,
                        ) as exc:  # AI-AGENT-REF: narrow exception
                            logger.exception("bot.py unexpected", exc_info=exc)
                            raise
                    _slippage_log.append(
                        (
                            symbol,
                            vwap_price,
                            fill_price,
                            datetime.now(UTC),
                        )
                    )
                    with slippage_lock:
                        try:
                            with open(SLIPPAGE_LOG_FILE, "a", newline="") as sf:
                                csv.writer(sf).writerow(
                                    [
                                        utc_now_iso(),
                                        symbol,
                                        vwap_price,
                                        fill_price,
                                        slip,
                                    ]
                                )
                        except (
                            FileNotFoundError,
                            PermissionError,
                            IsADirectoryError,
                            JSONDecodeError,
                            ValueError,
                            KeyError,
                            TypeError,
                            OSError,
                        ) as e:  # AI-AGENT-REF: narrow exception
                            logger.warning(f"Failed to append slippage log: {e}")
                if orders_total:
                    try:
                        orders_total.inc()
                    except (
                        FileNotFoundError,
                        PermissionError,
                        IsADirectoryError,
                        JSONDecodeError,
                        ValueError,
                        KeyError,
                        TypeError,
                        OSError,
                    ) as exc:  # AI-AGENT-REF: narrow exception
                        logger.exception("bot.py unexpected", exc_info=exc)
                        raise
                break
            except APIError as e:
                logger.warning(f"[VWAP] APIError attempt {attempt + 1} for {symbol}: {e}")
                pytime.sleep(attempt + 1)
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as e:  # AI-AGENT-REF: narrow exception
                logger.exception(f"[VWAP] slice attempt {attempt + 1} failed: {e}")
                pytime.sleep(attempt + 1)
        if order is None:
            break
        placed += slice_qty
        pytime.sleep(duration / 10)


@dataclass(frozen=True)
class SliceConfig:
    pct: float = 0.1
    sleep_interval: int = 60
    max_retries: int = 3
    backoff_factor: float = 2.0
    max_backoff_interval: int = 300


DEFAULT_SLICE_CFG = SliceConfig(
    pct=POV_SLICE_PCT,
    sleep_interval=60,
    max_retries=3,
    backoff_factor=2.0,
    max_backoff_interval=300,
)


def pov_submit(
    ctx: BotContext,
    symbol: str,
    total_qty: int,
    side: str,
    cfg: SliceConfig = DEFAULT_SLICE_CFG,
) -> bool:
    placed = 0
    retries = 0
    interval = cfg.sleep_interval
    while placed < total_qty:
        try:
            df = fetch_minute_df_safe(symbol)
        except DataFetchError:
            retries += 1
            if retries > cfg.max_retries:
                logger.warning(
                    f"[pov_submit] no minute data after {cfg.max_retries} retries, aborting",
                    extra={"symbol": symbol},
                )
                return False
            logger.warning(
                f"[pov_submit] missing bars, retry {retries}/{cfg.max_retries} in {interval:.1f}s",
                extra={"symbol": symbol},
            )
            sleep_time = interval * (0.8 + 0.4 * random.random())
            pytime.sleep(sleep_time)
            interval = min(interval * cfg.backoff_factor, cfg.max_backoff_interval)
            continue
        if df is None or df.empty:
            retries += 1
            if retries > cfg.max_retries:
                logger.warning(
                    f"[pov_submit] no minute data after {cfg.max_retries} retries, aborting",
                    extra={"symbol": symbol},
                )
                return False
            logger.warning(
                f"[pov_submit] missing bars, retry {retries}/{cfg.max_retries} in {interval:.1f}s",
                extra={"symbol": symbol},
            )
            sleep_time = interval * (0.8 + 0.4 * random.random())
            pytime.sleep(sleep_time)
            interval = min(interval * cfg.backoff_factor, cfg.max_backoff_interval)
            continue
        retries = 0
        interval = cfg.sleep_interval

        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quote: Quote = ctx.data_client.get_stock_latest_quote(req)
            spread = (
                (quote.ask_price - quote.bid_price)
                if quote.ask_price and quote.bid_price
                else 0.0
            )
        except APIError as e:
            logger.warning(f"[pov_submit] Alpaca quote failed for {symbol}: {e}")
            spread = 0.0
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ):  # AI-AGENT-REF: narrow exception
            spread = 0.0

        vol = df["volume"].iloc[-1]

        # AI-AGENT-REF: Dynamic spread threshold based on market conditions
        # Instead of fixed 0.05, use dynamic threshold based on symbol characteristics
        dynamic_spread_threshold = min(0.10, max(0.02, vol / 1000000 * 0.05))

        if spread > dynamic_spread_threshold:
            # Less aggressive reduction - only 25% instead of 50%
            slice_qty = min(int(vol * cfg.pct * 0.75), total_qty - placed)
            logger.debug(
                "[pov_submit] High spread detected, reducing slice by 25%",
                extra={
                    "symbol": symbol,
                    "spread": spread,
                    "threshold": dynamic_spread_threshold,
                    "reduced_slice_qty": slice_qty,
                },
            )
        else:
            slice_qty = min(int(vol * cfg.pct), total_qty - placed)

        if slice_qty < 1:
            logger.debug(
                f"[pov_submit] slice_qty<1 (vol={vol}), waiting",
                extra={"symbol": symbol},
            )
            pytime.sleep(cfg.sleep_interval * (0.8 + 0.4 * random.random()))
            continue
        try:
            # AI-AGENT-REF: Fix order slicing to track actual filled quantities
            order = submit_order(ctx, symbol, slice_qty, side)
            if order is None:
                logger.warning(
                    "[pov_submit] submit_order returned None for slice, skipping",
                    extra={"symbol": symbol, "slice_qty": slice_qty},
                )
                continue

            # Track actual filled quantity, not intended quantity
            actual_filled = int(getattr(order, "filled_qty", "0") or "0")

            # For partially filled orders, the filled_qty might be less than slice_qty
            if actual_filled < slice_qty:
                logger.warning(
                    "POV_SLICE_PARTIAL_FILL",
                    extra={
                        "symbol": symbol,
                        "intended_qty": slice_qty,
                        "actual_filled": actual_filled,
                        "order_id": getattr(order, "id", ""),
                        "status": getattr(order, "status", ""),
                    },
                )

            placed += actual_filled  # Use actual filled, not intended

        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.exception(
                f"[pov_submit] submit_order failed on slice, aborting: {e}",
                extra={"symbol": symbol},
            )
            return False

        logger.info(
            "POV_SLICE_PLACED",
            extra={
                "symbol": symbol,
                "slice_qty": slice_qty,
                "actual_filled": (
                    actual_filled if "actual_filled" in locals() else slice_qty
                ),
                "total_placed": placed,
            },
        )
        pytime.sleep(cfg.sleep_interval * (0.8 + 0.4 * random.random()))
    logger.info("POV_SUBMIT_COMPLETE", extra={"symbol": symbol, "placed": placed})
    return True


def maybe_pyramid(
    ctx: BotContext,
    symbol: str,
    entry_price: float,
    current_price: float,
    atr: float,
    prob: float,
):
    """Add to a winning position when probability remains high."""
    profit = (current_price - entry_price) if entry_price else 0
    if profit > 2 * atr and prob >= 0.75:
        try:
            pos = ctx.api.get_position(symbol)
            qty = int(abs(int(pos.qty)) * 0.5)
            if qty > 0:
                submit_order(ctx, symbol, qty, "buy")
                logger.info("PYRAMIDED", extra={"symbol": symbol, "qty": qty})
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.exception(f"[maybe_pyramid] failed for {symbol}: {e}")


def update_trailing_stop(
    ctx: BotContext,
    ticker: str,
    price: float,
    qty: int,
    atr: float,
) -> str:
    factor = 1.0 if is_high_vol_regime() else TRAILING_FACTOR
    te = ctx.trailing_extremes
    if qty > 0:
        with targets_lock:
            te[ticker] = max(te.get(ticker, price), price)
        if price < te[ticker] - factor * atr:
            return "exit_long"
    elif qty < 0:
        with targets_lock:
            te[ticker] = min(te.get(ticker, price), price)
        if price > te[ticker] + factor * atr:
            return "exit_short"
    return "hold"


def calculate_entry_size(
    ctx: BotContext, symbol: str, price: float, atr: float, win_prob: float
) -> int:
    """Calculate entry size based on account balance and risk parameters."""

    if ctx.api is None:
        logger.warning("ctx.api is None - using default entry size")
        return 1

    try:
        cash = float(ctx.api.get_account().cash)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        logger.warning("Failed to get cash for entry size calculation: %s", exc)
        return 1

    cap_pct = ctx.params.get("get_capital_cap()", get_capital_cap())
    cap_sz = int(round((cash * cap_pct) / price)) if price > 0 else 0
    df = ctx.data_fetcher.get_daily_df(ctx, symbol)
    rets = (
        df["close"].pct_change(fill_method=None).dropna().values
        if df is not None and not df.empty
        else np.array([0.0])
    )
    kelly_sz = fractional_kelly_size(ctx, cash, price, atr, win_prob)
    vol_sz = vol_target_position_size(cash, price, rets, target_vol=0.02)
    dollar_cap = ctx.max_position_dollars / price if price > 0 else kelly_sz
    base = int(round(min(kelly_sz, vol_sz, cap_sz, dollar_cap, MAX_POSITION_SIZE)))
    factor = max(0.5, min(1.5, 1 + (win_prob - 0.5)))
    liq = liquidity_factor(ctx, symbol)
    # AI-AGENT-REF: Fix zero quantity from low liquidity - use minimum viable size
    if liq < 0.2:
        # If we have significant cash, still allow minimum position
        if cash > 5000:
            logger.info(
                f"Low liquidity for {symbol} (factor={liq:.3f}), using minimum position size"
            )
            return max(1, int(1000 / price)) if price > 0 else 1
        return 0
    size = int(round(base * factor * liq))
    return max(size, 1)


def execute_entry(ctx: BotContext, symbol: str, qty: int, side: str) -> None:
    """Execute entry order."""

    if ctx.api is None:
        logger.warning("ctx.api is None - cannot execute entry")
        return

    try:
        buying_pw = float(ctx.api.get_account().buying_power)
        if buying_pw <= 0:
            logger.info("NO_BUYING_POWER", extra={"symbol": symbol})
            return
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        logger.warning("Failed to get buying power for %s: %s", symbol, exc)
        return
    if qty is None or qty <= 0 or not np.isfinite(qty):
        logger.error(
            f"Invalid order quantity for {symbol}: {qty}. Skipping order and logging input data."
        )
        # Optionally, log signal, price, and input features here for debug
        return
    if POV_SLICE_PCT > 0 and qty > SLICE_THRESHOLD:
        logger.info("POV_SLICE_ENTRY", extra={"symbol": symbol, "qty": qty})
        pov_submit(ctx, symbol, qty, side)
    elif qty > SLICE_THRESHOLD:
        logger.info("VWAP_SLICE_ENTRY", extra={"symbol": symbol, "qty": qty})
        vwap_pegged_submit(ctx, symbol, qty, side)
    else:
        logger.info("MARKET_ENTRY", extra={"symbol": symbol, "qty": qty})
        submit_order(ctx, symbol, qty, side)

    try:
        raw = fetch_minute_df_safe(symbol)
    except DataFetchError:
        logger.warning("NO_MINUTE_BARS_POST_ENTRY", extra={"symbol": symbol})
        return
    if raw is None or raw.empty:
        logger.warning("NO_MINUTE_BARS_POST_ENTRY", extra={"symbol": symbol})
        return
    try:
        df_ind = prepare_indicators(raw)
        if df_ind is None:
            logger.warning("INSUFFICIENT_INDICATORS_POST_ENTRY", extra={"symbol": symbol})
            return
    except ValueError as exc:
        logger.warning(f"Indicator preparation failed for {symbol}: {exc}")
        return
    if df_ind.empty:
        logger.warning("INSUFFICIENT_INDICATORS_POST_ENTRY", extra={"symbol": symbol})
        return
    entry_price = get_latest_close(df_ind)
    ctx.trade_logger.log_entry(symbol, entry_price, qty, side, "", "", confidence=0.5)

    now_pac = datetime.now(UTC).astimezone(PACIFIC)
    mo = datetime.combine(now_pac.date(), ctx.market_open, PACIFIC)
    mc = datetime.combine(now_pac.date(), ctx.market_close, PACIFIC)
    tp_factor = TAKE_PROFIT_FACTOR * 1.1 if is_high_vol_regime() else TAKE_PROFIT_FACTOR
    stop, take = scaled_atr_stop(
        entry_price,
        df_ind["atr"].iloc[-1],
        now_pac,
        mo,
        mc,
        max_factor=tp_factor,
        min_factor=0.5,
    )
    with targets_lock:
        ctx.stop_targets[symbol] = stop
        ctx.take_profit_targets[symbol] = take


def execute_exit(ctx: BotContext, state: BotState, symbol: str, qty: int) -> None:
    if qty is None or not np.isfinite(qty) or qty <= 0:
        logger.warning(f"Skipping {symbol}: computed qty <= 0")
        return
    try:
        raw = fetch_minute_df_safe(symbol)
    except DataFetchError:
        logger.warning("NO_MINUTE_BARS_POST_EXIT", extra={"symbol": symbol})
        raw = pd.DataFrame()
    exit_price = get_latest_close(raw) if raw is not None else 1.0
    send_exit_order(ctx, symbol, qty, exit_price, "manual_exit")
    ctx.trade_logger.log_exit(state, symbol, exit_price)
    on_trade_exit_rebalance(ctx)
    with targets_lock:
        ctx.take_profit_targets.pop(symbol, None)
        ctx.stop_targets.pop(symbol, None)


def exit_all_positions(ctx: BotContext) -> None:
    raw_positions = ctx.api.list_positions()
    for pos in raw_positions:
        qty = abs(int(pos.qty))
        if qty:
            send_exit_order(
                ctx, pos.symbol, qty, 0.0, "eod_exit", raw_positions=raw_positions
            )
            logger.info("EOD_EXIT", extra={"symbol": pos.symbol, "qty": qty})


def _liquidate_all_positions(runtime: BotContext) -> None:
    """Helper to liquidate every open position."""
    # AI-AGENT-REF: existing exit_all_positions wrapper for emergency liquidation
    exit_all_positions(runtime)


def liquidate_positions_if_needed(runtime: BotContext) -> None:
    """Liquidate all positions when certain risk conditions trigger."""
    if check_halt_flag(runtime):
        # Modified: DO NOT liquidate positions on halt flag.
        logger.info(
            "TRADING_HALTED_VIA_FLAG is active: NOT liquidating positions, holding open positions."
        )
        return

    # normal liquidation logic would go here (stub)


# ─── L. SIGNAL & TRADE LOGIC ───────────────────────────────────────────────────
def signal_and_confirm(
    ctx: BotContext, state: BotState, symbol: str, df: pd.DataFrame, model
) -> tuple[int, float, str]:
    """Wrapper that evaluates signals and checks confidence threshold."""
    sig, conf, strat = ctx.signal_manager.evaluate(ctx, state, df, symbol, model)
    if sig == -1 or conf < get_conf_threshold():
        logger.debug(
            "SKIP_LOW_SIGNAL", extra={"symbol": symbol, "sig": sig, "conf": conf}
        )
        return -1, 0.0, ""
    return sig, conf, strat


def pre_trade_checks(
    ctx: BotContext, state: BotState, symbol: str, balance: float, regime_ok: bool
) -> bool:
    if getattr(CFG, "force_trades", False):
        logger.warning("FORCE_TRADES override active: ignoring all pre-trade halts.")
        return True
    # Streak kill-switch check
    if (
        state.streak_halt_until
        and datetime.now(UTC).astimezone(PACIFIC) < state.streak_halt_until
    ):
        logger.info(
            "SKIP_STREAK_HALT",
            extra={"symbol": symbol, "until": state.streak_halt_until},
        )
        _log_health_diagnostics(ctx, "streak")
        return False
    if getattr(state, "pdt_blocked", False):
        logger.info("SKIP_PDT_RULE", extra={"symbol": symbol})
        _log_health_diagnostics(ctx, "pdt")
        return False
    if check_halt_flag(ctx):
        logger.info("SKIP_HALT_FLAG", extra={"symbol": symbol})
        _log_health_diagnostics(ctx, "halt_flag")
        return False
    if check_daily_loss(ctx, state):
        logger.info("SKIP_DAILY_LOSS", extra={"symbol": symbol})
        _log_health_diagnostics(ctx, "daily_loss")
        return False
    if check_weekly_loss(ctx, state):
        logger.info("SKIP_WEEKLY_LOSS", extra={"symbol": symbol})
        _log_health_diagnostics(ctx, "weekly_loss")
        return False
    if too_many_positions(ctx, symbol):
        logger.info("SKIP_TOO_MANY_POSITIONS", extra={"symbol": symbol})
        _log_health_diagnostics(ctx, "positions")
        return False
    if too_correlated(ctx, symbol):
        logger.info("SKIP_HIGH_CORRELATION", extra={"symbol": symbol})
        _log_health_diagnostics(ctx, "correlation")
        return False
    return ctx.data_fetcher.get_daily_df(ctx, symbol) is not None


def should_enter(
    ctx: BotContext, state: BotState, symbol: str, balance: float, regime_ok: bool
) -> bool:
    return pre_trade_checks(ctx, state, symbol, balance, regime_ok)


def should_exit(
    ctx: BotContext, symbol: str, price: float, atr: float
) -> tuple[bool, int, str]:
    current_qty = _current_qty(ctx, symbol)  # AI-AGENT-REF: derive qty safely

    # AI-AGENT-REF: remove time-based rebalance hold logic
    if symbol in ctx.rebalance_buys:
        ctx.rebalance_buys.pop(symbol, None)

    stop = ctx.stop_targets.get(symbol)
    if stop is not None:
        if current_qty > 0 and price <= stop:
            return True, abs(current_qty), "stop_loss"
        if current_qty < 0 and price >= stop:
            return True, abs(current_qty), "stop_loss"

    tp = ctx.take_profit_targets.get(symbol)
    if current_qty > 0 and tp and price >= tp:
        exit_qty = max(int(abs(current_qty) * SCALING_FACTOR), 1)
        return True, exit_qty, "take_profit"
    if current_qty < 0 and tp and price <= tp:
        exit_qty = max(int(abs(current_qty) * SCALING_FACTOR), 1)
        return True, exit_qty, "take_profit"

    action = update_trailing_stop(ctx, symbol, price, current_qty, atr)
    if (action == "exit_long" and current_qty > 0) or (
        action == "exit_short" and current_qty < 0
    ):
        return True, abs(current_qty), "trailing_stop"

    return False, 0, ""


def _safe_trade(
    ctx: BotContext,
    state: BotState,
    symbol: str,
    balance: float,
    model: Any,
    regime_ok: bool,
    side: OrderSide | None = None,
) -> bool:
    try:
        # Real-time position check to prevent buy/sell flip-flops
        if side is not None:
            try:
                live_positions = {
                    p.symbol: int(p.qty) for p in ctx.api.list_positions()
                }
                if side == OrderSide.BUY and symbol in live_positions:
                    logger.info(f"REALTIME_SKIP | {symbol} already held. Skipping BUY.")
                    return False
                elif side == OrderSide.SELL and symbol not in live_positions:
                    logger.info(f"REALTIME_SKIP | {symbol} not held. Skipping SELL.")
                    return False
            except (
                APIError,
                TimeoutError,
                ConnectionError,
            ) as e:  # AI-AGENT-REF: tighten live position check errors
                logger.warning(
                    "REALTIME_CHECK_FAIL",
                    extra={
                        "symbol": symbol,
                        "cause": e.__class__.__name__,
                        "detail": str(e),
                    },
                )
        return trade_logic(ctx, state, symbol, balance, model, regime_ok)
    except RetryError as e:
        logger.warning(
            f"[trade_logic] retries exhausted for {symbol}: {e}",
            extra={"symbol": symbol},
        )
        return False
    except APIError as e:
        msg = str(e).lower()
        if "insufficient buying power" in msg or "potential wash trade" in msg:
            logger.warning(
                f"[trade_logic] skipping {symbol} due to APIError: {e}",
                extra={"symbol": symbol},
            )
            return False
        else:
            logger.exception(f"[trade_logic] APIError for {symbol}: {e}")
            return False
    except (
        APIError,
        TimeoutError,
        ConnectionError,
        ValueError,
        KeyError,
        TypeError,
    ) as e:  # AI-AGENT-REF: tighten trade_logic errors
        logger.exception(
            "[trade_logic] unhandled exception",
            extra={"symbol": symbol, "cause": e.__class__.__name__, "detail": str(e)},
        )
        return False


def _fetch_feature_data(
    ctx: BotContext,
    state: BotState,
    symbol: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, bool | None]:
    """Fetch raw price data and compute indicators.

    Returns ``(raw_df, feat_df, skip_flag)``. When data is missing returns
    ``(None, None, False)``; when indicators are insufficient returns
    ``(raw_df, None, True)``.
    """
    def _halt(reason: str) -> None:
        logger.error("DATA_FETCH_EMPTY", extra={"symbol": symbol, "reason": reason})
        halt_mgr = getattr(ctx, "halt_manager", None)
        if halt_mgr is not None:
            try:
                halt_mgr.manual_halt_trading(f"{symbol}:{reason}")
            except Exception as hm_exc:  # noqa: BLE001
                logger.error("HALT_MANAGER_ERROR", extra={"cause": str(hm_exc)})

    try:
        raw_df = fetch_minute_df_safe(symbol)
    except DataFetchError:
        _halt("minute_data_unavailable")
        return None, None, False
    except APIError as e:
        msg = str(e).lower()
        if "subscription does not permit querying recent sip data" in msg:
            logger.debug(f"{symbol}: minute fetch failed, falling back to daily.")
            raw_df = ctx.data_fetcher.get_daily_df(ctx, symbol)
            if raw_df is None or raw_df.empty:
                logger.debug(f"{symbol}: no daily data either; skipping.")
                _halt("daily_data_unavailable")
                return None, None, False
        else:
            raise
    if raw_df is None or raw_df.empty:
        _halt("empty_frame")
        return None, None, False

    # Guard: validate OHLCV shape before feature engineering
    try:
        validate_ohlcv(raw_df)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.warning("OHLCV validation failed for %s: %s; skipping symbol", symbol, e)
        return raw_df, pd.DataFrame(), True

    df = raw_df.copy()

    # AI-AGENT-REF: Data sanitize integration (gated by flag)
    if hasattr(S, "data_sanitize_enabled") and CFG.data_sanitize_enabled:
        try:
            from ai_trading.data.sanitize import clean as _clean

            df = _clean(df)
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.warning("Data sanitize failed: %s", e)

    # AI-AGENT-REF: Corporate actions adjustment (gated by flag)
    if hasattr(S, "corp_actions_enabled") and CFG.corp_actions_enabled:
        try:
            from ai_trading.data.corp_actions import adjust as _adjust

            df = _adjust(df, symbol)
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.warning("Corp actions adjust failed: %s", e)

    # AI-AGENT-REF: log initial dataframe and monitor row drops
    logger.debug(f"Initial tail data for {symbol}:\n{df.tail(5)}")
    initial_len = len(df)

    df = compute_macd(df)
    assert_row_integrity(initial_len, len(df), "compute_macd", symbol)
    logger.debug(f"[{symbol}] Post MACD: last closes:\n{df[['close']].tail(5)}")

    df = compute_atr(df)
    assert_row_integrity(initial_len, len(df), "compute_atr", symbol)
    logger.debug(f"[{symbol}] Post ATR: last closes:\n{df[['close']].tail(5)}")

    df = compute_vwap(df)
    assert_row_integrity(initial_len, len(df), "compute_vwap", symbol)
    logger.debug(f"[{symbol}] Post VWAP: last closes:\n{df[['close']].tail(5)}")

    df = compute_macds(df)
    logger.debug(f"{symbol} dataframe columns after indicators: {df.columns.tolist()}")
    df = ensure_columns(df, ["macd", "atr", "vwap", "macds"], symbol)
    if df.empty and raw_df is not None:
        df = raw_df.copy()

    try:
        feat_df = prepare_indicators(df)
        if feat_df is None:
            return raw_df, None, True
        # AI-AGENT-REF: fallback to raw data when feature engineering drops all rows
        if feat_df.empty:
            logger.warning("Parsed feature DataFrame is empty; falling back to raw data")
            feat_df = raw_df.copy()
    except ValueError as exc:
        logger.warning(f"Indicator preparation failed for {symbol}: {exc}")
        return raw_df, None, True
    if feat_df.empty:
        logger.debug(f"SKIP_INSUFFICIENT_FEATURES | symbol={symbol}")
        return raw_df, None, True
    return raw_df, feat_df, None


def _model_feature_names(model) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return [
        "rsi",
        "macd",
        "atr",
        "vwap",
        "macds",
        "ichimoku_conv",
        "ichimoku_base",
        "stochrsi",
    ]


def _should_hold_position(df: pd.DataFrame) -> bool:
    """Return True if trend indicators favor staying in the trade."""
    from ai_trading.indicators import rsi  # type: ignore

    try:
        close = df["close"].astype(float)
        ema_fast = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema_slow = close.ewm(span=50, adjust=False).mean().iloc[-1]
        rsi_val = rsi(tuple(close), 14).iloc[-1]
        return close.iloc[-1] > ema_fast > ema_slow and rsi_val >= 55
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ):  # AI-AGENT-REF: narrow exception
        return False


def _exit_positions_if_needed(
    ctx: BotContext,
    state: BotState,
    symbol: str,
    feat_df: pd.DataFrame,
    final_score: float,
    conf: float,
    current_qty: int,
) -> bool:
    if final_score < 0 and current_qty > 0 and abs(conf) >= get_conf_threshold():
        if _should_hold_position(feat_df):
            logger.info("HOLD_SIGNAL_ACTIVE", extra={"symbol": symbol})
        else:
            price = get_latest_close(feat_df)
            logger.info(
                f"SIGNAL_REVERSAL_EXIT | symbol={symbol}  final_score={final_score:.4f}  confidence={conf:.4f}"
            )
            send_exit_order(ctx, symbol, current_qty, price, "reversal")
            ctx.trade_logger.log_exit(state, symbol, price)
            with targets_lock:
                ctx.stop_targets.pop(symbol, None)
                ctx.take_profit_targets.pop(symbol, None)
            return True

    if final_score > 0 and current_qty < 0 and abs(conf) >= get_conf_threshold():
        price = get_latest_close(feat_df)
        logger.info(
            f"SIGNAL_BULLISH_EXIT | symbol={symbol}  final_score={final_score:.4f}  confidence={conf:.4f}"
        )
        send_exit_order(ctx, symbol, abs(current_qty), price, "reversal")
        ctx.trade_logger.log_exit(state, symbol, price)
        with targets_lock:
            ctx.stop_targets.pop(symbol, None)
            ctx.take_profit_targets.pop(symbol, None)
        return True
    return False


def _enter_long(
    ctx: BotContext,
    state: BotState,
    symbol: str,
    balance: float,
    feat_df: pd.DataFrame,
    final_score: float,
    conf: float,
    strat: str,
) -> bool:
    current_price = get_latest_close(feat_df)
    logger.debug(f"Latest 5 rows for {symbol}:\n{feat_df.tail(5)}")
    logger.debug(f"Computed price for {symbol}: {current_price}")
    if current_price <= 0 or pd.isna(current_price):
        logger.critical(f"Invalid price computed for {symbol}: {current_price}")
        return True

    # AI-AGENT-REF: Get target weight with sensible fallback for signal-based trading
    target_weight = ctx.portfolio_weights.get(symbol, 0.0)
    if target_weight == 0.0:
        # If no portfolio weight exists (e.g., new signal), calculate a reasonable default
        # Based on confidence and ensuring we don't exceed exposure limits
        confidence_weight = conf * 0.15  # Max 15% for high confidence signals
        exposure_cap = (
            getattr(ctx.config, "exposure_cap_aggressive", 0.88)
            if hasattr(ctx, "config")
            else 0.88
        )

        # Get current total exposure to avoid exceeding cap
        try:
            positions = ctx.api.list_positions()
            current_exposure = sum(
                abs(float(p.market_value)) for p in positions
            ) / float(ctx.api.get_account().equity)
            available_exposure = max(0, exposure_cap - current_exposure)
            target_weight = min(
                confidence_weight, available_exposure, 0.15
            )  # Cap at 15%
            logger.info(
                f"Computed weight for {symbol}: {target_weight:.3f} (confidence={conf:.3f}, available_exposure={available_exposure:.3f})"
            )
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.warning(
                f"Could not compute dynamic weight for {symbol}: {e}, using confidence-based weight"
            )
            target_weight = min(confidence_weight, 0.10)  # Conservative 10% fallback

    raw_qty = int(balance * target_weight / current_price) if current_price > 0 else 0

    # AI-AGENT-REF: Position sizing integration (gated by flag)
    if hasattr(S, "sizing_enabled") and CFG.sizing_enabled:
        try:
            from ai_trading.portfolio import sizing as _sizing

            account_equity = float(ctx.api.get_account().equity) if ctx.api else balance
            optimized_qty = _sizing.position_size(
                symbol,
                final_score,
                account_equity,
                getattr(S, "risk_level", "moderate"),
            )
            optimized_qty = min(optimized_qty, getattr(S, "max_position_size", 1000))
            if optimized_qty > 0:
                raw_qty = optimized_qty
                logger.debug("Sizing decided qty=%s for %s", raw_qty, symbol)
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.warning("Sizing failed; falling back to default sizing: %s", e)

    # AI-AGENT-REF: Fix zero quantity calculations - ensure minimum position size when cash available
    if raw_qty is None or not np.isfinite(raw_qty) or raw_qty <= 0:
        # If we have significant cash available and a valid signal, use minimum position size
        if balance > 1000 and target_weight > 0.001 and current_price > 0:
            raw_qty = max(1, int(1000 / current_price))  # Minimum $1000 position
            logger.info(
                f"Using minimum position size for {symbol}: {raw_qty} shares (balance=${balance:.0f})"
            )
        else:
            logger.warning(
                f"Skipping {symbol}: computed qty <= 0 (balance=${balance:.0f}, weight={target_weight:.4f})"
            )
            return True
    logger.info(
        f"SIGNAL_BUY | symbol={symbol}  final_score={final_score:.4f}  confidence={conf:.4f}  qty={raw_qty}"
    )
    if not sector_exposure_ok(ctx, symbol, raw_qty, current_price):
        logger.info(
            "SKIP_SECTOR_CAP | Buy order skipped due to sector exposure limits",
            extra={
                "symbol": symbol,
                "side": "buy",
                "qty": raw_qty,
                "price": current_price,
            },
        )
        return True
    order = submit_order(ctx, symbol, raw_qty, "buy")
    if order is None:
        logger.debug(f"TRADE_LOGIC_NO_ORDER | symbol={symbol}")
    else:
        logger.debug(f"TRADE_LOGIC_ORDER_PLACED | symbol={symbol}  order_id={order.id}")
        ctx.trade_logger.log_entry(
            symbol,
            current_price,
            raw_qty,
            "buy",
            strat,
            signal_tags=strat,
            confidence=conf,
        )
        now_pac = datetime.now(UTC).astimezone(PACIFIC)
        mo = datetime.combine(now_pac.date(), ctx.market_open, PACIFIC)
        mc = datetime.combine(now_pac.date(), ctx.market_close, PACIFIC)
        tp_factor = (
            TAKE_PROFIT_FACTOR * 1.1 if is_high_vol_regime() else TAKE_PROFIT_FACTOR
        )
        stop, take = scaled_atr_stop(
            entry_price=current_price,
            atr=feat_df["atr"].iloc[-1],
            now=now_pac,
            market_open=mo,
            market_close=mc,
            max_factor=tp_factor,
            min_factor=0.5,
        )
        with targets_lock:
            ctx.stop_targets[symbol] = stop
            ctx.take_profit_targets[symbol] = take
        # AI-AGENT-REF: Add thread-safe locking for trade cooldown modifications
        with trade_cooldowns_lock:
            state.trade_cooldowns[symbol] = datetime.now(UTC)
        state.last_trade_direction[symbol] = "buy"

        # AI-AGENT-REF: Record trade in frequency tracker for overtrading prevention
        _record_trade_in_frequency_tracker(state, symbol, datetime.now(UTC))
    return True


def _enter_short(
    ctx: BotContext,
    state: BotState,
    symbol: str,
    feat_df: pd.DataFrame,
    final_score: float,
    conf: float,
    strat: str,
) -> bool:
    current_price = get_latest_close(feat_df)
    logger.debug(f"Latest 5 rows for {symbol}:\n{feat_df.tail(5)}")
    logger.debug(f"Computed price for {symbol}: {current_price}")
    if current_price <= 0 or pd.isna(current_price):
        logger.critical(f"Invalid price computed for {symbol}: {current_price}")
        return True
    atr = feat_df["atr"].iloc[-1]
    qty = calculate_entry_size(ctx, symbol, current_price, atr, conf)
    try:
        asset = ctx.api.get_asset(symbol)
        if hasattr(asset, "shortable") and not asset.shortable:
            logger.info(f"SKIP_NOT_SHORTABLE | symbol={symbol}")
            return True
        avail = getattr(asset, "shortable_shares", None)
        if avail is not None:
            qty = min(qty, int(avail))
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        logger.exception("bot.py unexpected", exc_info=exc)
        raise
    if qty is None or not np.isfinite(qty) or qty <= 0:
        logger.warning(f"Skipping {symbol}: computed qty <= 0")
        return True
    logger.info(
        f"SIGNAL_SHORT | symbol={symbol}  final_score={final_score:.4f}  confidence={conf:.4f}  qty={qty}"
    )
    if not sector_exposure_ok(ctx, symbol, qty, current_price):
        logger.info(
            "SKIP_SECTOR_CAP | Short order skipped due to sector exposure limits",
            extra={
                "symbol": symbol,
                "side": "sell_short",
                "qty": qty,
                "price": current_price,
            },
        )
        return True
    order = submit_order(
        ctx, symbol, qty, "sell_short"
    )  # AI-AGENT-REF: Use sell_short for short signals
    if order is None:
        logger.debug(f"TRADE_LOGIC_NO_ORDER | symbol={symbol}")
    else:
        logger.debug(f"TRADE_LOGIC_ORDER_PLACED | symbol={symbol}  order_id={order.id}")
        ctx.trade_logger.log_entry(
            symbol,
            current_price,
            qty,
            "sell",
            strat,
            signal_tags=strat,
            confidence=conf,
        )
        now_pac = datetime.now(UTC).astimezone(PACIFIC)
        mo = datetime.combine(now_pac.date(), ctx.market_open, PACIFIC)
        mc = datetime.combine(now_pac.date(), ctx.market_close, PACIFIC)
        tp_factor = (
            TAKE_PROFIT_FACTOR * 1.1 if is_high_vol_regime() else TAKE_PROFIT_FACTOR
        )
        long_stop, long_take = scaled_atr_stop(
            entry_price=current_price,
            atr=atr,
            now=now_pac,
            market_open=mo,
            market_close=mc,
            max_factor=tp_factor,
            min_factor=0.5,
        )
        stop, take = long_take, long_stop
        with targets_lock:
            ctx.stop_targets[symbol] = stop
            ctx.take_profit_targets[symbol] = take
        # AI-AGENT-REF: Add thread-safe locking for trade cooldown modifications
        with trade_cooldowns_lock:
            state.trade_cooldowns[symbol] = datetime.now(UTC)
        state.last_trade_direction[symbol] = "sell"

        # AI-AGENT-REF: Record trade in frequency tracker for overtrading prevention
        _record_trade_in_frequency_tracker(state, symbol, datetime.now(UTC))
    return True


def _manage_existing_position(
    ctx: BotContext,
    state: BotState,
    symbol: str,
    feat_df: pd.DataFrame,
    conf: float,
    atr: float,
    current_qty: int,
) -> bool:
    price = get_latest_close(feat_df)
    logger.debug(f"Latest 5 rows for {symbol}:\n{feat_df.tail(5)}")
    logger.debug(f"Computed price for {symbol}: {price}")
    if price <= 0 or pd.isna(price):
        logger.critical(f"Invalid price computed for {symbol}: {price}")
        return False
    # AI-AGENT-REF: always rely on indicator-driven exits
    should_exit_flag, exit_qty, reason = should_exit(ctx, symbol, price, atr)
    if should_exit_flag and exit_qty > 0:
        logger.info(
            f"EXIT_SIGNAL | symbol={symbol}  reason={reason}  exit_qty={exit_qty}  price={price:.4f}"
        )
        send_exit_order(ctx, symbol, exit_qty, price, reason)
        if reason == "stop_loss":
            # AI-AGENT-REF: Add thread-safe locking for trade cooldown modifications
            with trade_cooldowns_lock:
                state.trade_cooldowns[symbol] = datetime.now(UTC)
            state.last_trade_direction[symbol] = "sell"

            # AI-AGENT-REF: Record trade in frequency tracker for overtrading prevention
            _record_trade_in_frequency_tracker(state, symbol, datetime.now(UTC))
        ctx.trade_logger.log_exit(state, symbol, price)
        try:
            pos_after = ctx.api.get_position(symbol)
            if int(pos_after.qty) == 0:
                with targets_lock:
                    ctx.stop_targets.pop(symbol, None)
                    ctx.take_profit_targets.pop(symbol, None)
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as exc:  # AI-AGENT-REF: narrow exception
            logger.exception("bot.py unexpected", exc_info=exc)
            raise
    else:
        try:
            pos = ctx.api.get_position(symbol)
            entry_price = float(pos.avg_entry_price)
            maybe_pyramid(ctx, symbol, entry_price, price, atr, conf)
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as exc:  # AI-AGENT-REF: narrow exception
            logger.exception("bot.py unexpected", exc_info=exc)
            raise
    return True


def _evaluate_trade_signal(
    ctx: BotContext, state: BotState, feat_df: pd.DataFrame, symbol: str, model: Any
) -> tuple[float, float, str]:
    """Return ``(final_score, confidence, strategy)`` for ``symbol``."""

    sig, conf, strat = ctx.signal_manager.evaluate(ctx, state, feat_df, symbol, model)
    comp_list = [
        {"signal": lab, "flag": s, "weight": w}
        for s, w, lab in ctx.signal_manager.last_components
    ]
    logger.debug("COMPONENTS | symbol=%s  components=%r", symbol, comp_list)
    final_score = sum(s * w for s, w, _ in ctx.signal_manager.last_components)
    logger.info(
        "SIGNAL_RESULT | symbol=%s  final_score=%.4f  confidence=%.4f",
        symbol,
        final_score,
        conf,
    )
    if final_score is None or not np.isfinite(final_score) or final_score == 0:
        raise ValueError("Invalid or empty signal")
    return final_score, conf, strat


def _current_qty(ctx, symbol: str) -> int:
    pos = ctx.position_map.get(symbol) if hasattr(ctx, "position_map") else None  # AI-AGENT-REF: safe lookup
    if pos is None:
        return 0
    qty = getattr(pos, "qty", 0) or 0
    try:
        return int(qty)
    except (TypeError, ValueError):
        return 0


def _recent_rebalance_flag(ctx: BotContext, symbol: str) -> bool:
    """Return ``False`` and clear any rebalance timestamp."""
    if symbol in ctx.rebalance_buys:
        ctx.rebalance_buys.pop(symbol, None)
    return False


def trade_logic(
    ctx: BotContext,
    state: BotState,
    symbol: str,
    balance: float,
    model: Any,
    regime_ok: bool,
    now_provider: Callable[[], datetime] | None = None,
) -> bool:
    """
    Core per-symbol logic: fetch data, compute features, evaluate signals, enter/exit orders.
    """
    logger.info(f"PROCESSING_SYMBOL | symbol={symbol}")

    if not pre_trade_checks(ctx, state, symbol, balance, regime_ok):
        logger.debug("SKIP_PRE_TRADE_CHECKS", extra={"symbol": symbol})
        return False

    raw_df, feat_df, skip_flag = _fetch_feature_data(ctx, state, symbol)
    if feat_df is None:
        return skip_flag if skip_flag is not None else False

    for col in ["macd", "atr", "vwap", "macds"]:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    feature_names = _model_feature_names(model)
    missing = [f for f in feature_names if f not in feat_df.columns]
    if missing:
        logger.debug(
            f"Feature snapshot for {symbol}: macd={feat_df['macd'].iloc[-1]}, atr={feat_df['atr'].iloc[-1]}, vwap={feat_df['vwap'].iloc[-1]}, macds={feat_df['macds'].iloc[-1]}"
        )
        logger.info("SKIP_MISSING_FEATURES | symbol=%s  missing=%s", symbol, missing)
        return True

    try:
        final_score, conf, strat = _evaluate_trade_signal(
            ctx, state, feat_df, symbol, model
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return True
    if pd.isna(final_score) or pd.isna(conf):
        logger.warning(f"Skipping {symbol}: model returned NaN prediction")
        return True

    current_qty = _current_qty(ctx, symbol)

    from datetime import UTC, datetime

    now_fn = now_provider or (lambda: datetime.now(UTC))
    now = now_fn()  # AI-AGENT-REF: injectable clock

    signal = "buy" if final_score > 0 else "sell" if final_score < 0 else "hold"

    if _exit_positions_if_needed(
        ctx, state, symbol, feat_df, final_score, conf, current_qty
    ):
        return True

    # AI-AGENT-REF: Add thread-safe locking for trade cooldown access
    with trade_cooldowns_lock:
        cd_ts = state.trade_cooldowns.get(symbol)
    if cd_ts and (now - cd_ts).total_seconds() < get_trade_cooldown_min() * 60:
        prev = state.last_trade_direction.get(symbol)
        if prev and (
            (prev == "buy" and signal == "sell") or (prev == "sell" and signal == "buy")
        ):
            logger.info("SKIP_REVERSED_SIGNAL", extra={"symbol": symbol})
            return True
        logger.debug("SKIP_COOLDOWN", extra={"symbol": symbol})
        return True

    # AI-AGENT-REF: Enhanced overtrading prevention - check frequency limits
    if _check_trade_frequency_limits(state, symbol, now):
        logger.info("SKIP_FREQUENCY_LIMIT", extra={"symbol": symbol})
        return True

    if final_score > 0 and conf >= get_buy_threshold() and current_qty == 0:
        if symbol in state.long_positions:
            held = state.position_cache.get(symbol, 0)
            logger.info(
                f"Skipping BUY for {symbol} — position already LONG {held} shares"
            )
            return True
        return _enter_long(
            ctx, state, symbol, balance, feat_df, final_score, conf, strat
        )

    if final_score < 0 and conf >= get_buy_threshold() and current_qty == 0:
        if symbol in state.short_positions:
            held = abs(state.position_cache.get(symbol, 0))
            logger.info(
                f"Skipping SELL for {symbol} — position already SHORT {held} shares"
            )
            return True
        return _enter_short(ctx, state, symbol, feat_df, final_score, conf, strat)

    # If holding, check for stops/take/trailing
    if current_qty != 0:
        atr = feat_df["atr"].iloc[-1]
        return _manage_existing_position(
            ctx, state, symbol, feat_df, conf, atr, current_qty
        )

    # Else hold / no action
    logger.info(
        f"SKIP_LOW_OR_NO_SIGNAL | symbol={symbol}  "
        f"final_score={final_score:.4f}  confidence={conf:.4f}"
    )
    return True


def compute_portfolio_weights(ctx: BotContext, symbols: list[str]) -> dict[str, float]:
    """Delegate to ai_trading.portfolio.compute_portfolio_weights with correct ctx."""
    from ai_trading.portfolio import compute_portfolio_weights as _cpw

    return _cpw(ctx, symbols)


def on_trade_exit_rebalance(ctx: BotContext) -> None:
    from ai_trading import portfolio
    from ai_trading.utils import portfolio_lock

    try:
        positions = ctx.api.list_positions()
        symbols = [p.symbol for p in positions]
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ):  # AI-AGENT-REF: narrow exception
        symbols = []
    current = portfolio.compute_portfolio_weights(ctx, symbols)
    old = ctx.portfolio_weights
    drift = max(abs(current[s] - old.get(s, 0)) for s in current) if current else 0
    if drift <= 0.1:
        return True
    with portfolio_lock:  # FIXED: protect shared portfolio state
        ctx.portfolio_weights = current
    total_value = float(ctx.api.get_account().portfolio_value)
    for sym, w in current.items():
        target_dollar = w * total_value
        try:
            raw = fetch_minute_df_safe(sym)
        except DataFetchError:
            logger.warning("REBALANCE_NO_DATA | %s", sym)
            continue
        price = get_latest_close(raw) if raw is not None else 1.0
        if price <= 0:
            continue
        target_shares = int(round(target_dollar / price))
        try:
            submit_order(
                ctx,
                sym,
                abs(target_shares),
                "buy" if target_shares > 0 else "sell",
            )
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ):  # AI-AGENT-REF: narrow exception
            logger.exception(f"Rebalance failed for {sym}")
    logger.info("PORTFOLIO_REBALANCED")


def pair_trade_signal(ctx: BotContext, sym1: str, sym2: str) -> tuple[str, int]:
    from statsmodels.tsa.stattools import coint

    df1 = ctx.data_fetcher.get_daily_df(ctx, sym1)
    df2 = ctx.data_fetcher.get_daily_df(ctx, sym2)
    if not hasattr(df1, "loc") or "close" not in df1.columns:
        raise ValueError(
            f"pair_trade_signal: df1 for {sym1} is invalid or missing 'close'"
        )
    if not hasattr(df2, "loc") or "close" not in df2.columns:
        raise ValueError(
            f"pair_trade_signal: df2 for {sym2} is invalid or missing 'close'"
        )
    df = pd.concat([df1["close"], df2["close"]], axis=1).dropna()
    if df.empty:
        return ("no_signal", 0)
    t_stat, p_value, _ = coint(df.iloc[:, 0], df.iloc[:, 1])
    if p_value < 0.05:
        beta = np.polyfit(df.iloc[:, 1], df.iloc[:, 0], 1)[0]
        spread = df.iloc[:, 0] - beta * df.iloc[:, 1]
        z = (spread - spread.mean()) / spread.std()
        z0 = z.iloc[-1]
        if z0 > 2:
            return ("short_spread", 1)
        elif z0 < -2:
            return ("long_spread", 1)
    return ("no_signal", 0)


# ─── M. UTILITIES ─────────────────────────────────────────────────────────────
def fetch_data(
    ctx: BotContext, symbols: list[str], period: str, interval: str
) -> pd.DataFrame | None:
    if not os.getenv("FINNHUB_API_KEY"):
        logger.debug("Skipping Finnhub fetch; FINNHUB_API_KEY not set")
        return None
    if not FINNHUB_AVAILABLE:
        logger.debug("Skipping Finnhub fetch; finnhub-python not installed")
        return None

    finnhub = importlib.import_module("finnhub")
    global finnhub_client
    if finnhub_client is None:
        finnhub_client = finnhub.Client(os.getenv("FINNHUB_API_KEY"))

    frames: list[pd.DataFrame] = []
    now = datetime.now(UTC)
    if period.endswith("d"):
        delta = timedelta(days=int(period[:-1]))
    elif period.endswith("mo"):
        delta = timedelta(days=30 * int(period[:-2]))
    elif period.endswith("y"):
        delta = timedelta(days=365 * int(period[:-1]))
    else:
        delta = timedelta(days=7)
    unix_to = int(now.timestamp())
    unix_from = int((now - delta).timestamp())

    for batch in chunked(symbols, 3):
        for sym in batch:
            try:
                ohlc = finnhub_client.stock_candle(
                    sym, resolution=interval, _from=unix_from, to=unix_to
                )
            except finnhub.FinnhubAPIException as e:
                logger.debug("FINNHUB_FETCH_FAILED", extra={"symbol": sym, "err": str(e)})
                continue

            if not ohlc or ohlc.get("s") != "ok":
                continue

            idx = safe_to_datetime(ohlc.get("t", []), context=f"prefetch {sym}")
            df_sym = pd.DataFrame(
                {
                    "open": ohlc.get("o", []),
                    "high": ohlc.get("h", []),
                    "low": ohlc.get("l", []),
                    "close": ohlc.get("c", []),
                    "volume": ohlc.get("v", []),
                },
                index=idx,
            )

            df_sym.columns = pd.MultiIndex.from_product([[sym], df_sym.columns])
            frames.append(df_sym)

        pytime.sleep(random.uniform(2, 5))

    if not frames:
        return None

    return pd.concat(frames, axis=1)


class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        probs = [m.predict_proba(X) for m in self.models]
        return np.mean(probs, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def load_model(path: str = MODEL_PATH) -> dict | EnsembleModel | None:
    """Load a model from ``path`` supporting both single and ensemble files."""
    import joblib

    if not path or not os.path.exists(path):
        return None  # AI-AGENT-REF: handle disabled ML gracefully

    loaded = joblib.load(path)
    # if this is a plain dict, return it directly
    if isinstance(loaded, dict):
        logger.info("MODEL_LOADED")
        return loaded

    # AI-AGENT-REF: use isfile checks for optional ensemble components
    rf_exists = bool(MODEL_RF_PATH) and os.path.isfile(MODEL_RF_PATH)
    xgb_exists = bool(MODEL_XGB_PATH) and os.path.isfile(MODEL_XGB_PATH)
    lgb_exists = bool(MODEL_LGB_PATH) and os.path.isfile(MODEL_LGB_PATH)
    if rf_exists and xgb_exists and lgb_exists:
        models = []
        for p in [MODEL_RF_PATH, MODEL_XGB_PATH, MODEL_LGB_PATH]:
            try:
                models.append(joblib.load(p))
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as e:  # AI-AGENT-REF: narrow exception
                logger.exception("MODEL_LOAD_FAILED: %s", e)
                return None
        logger.info(
            "MODEL_LOADED",
            extra={"path": f"{MODEL_RF_PATH}, {MODEL_XGB_PATH}, {MODEL_LGB_PATH}"},
        )
        return EnsembleModel(models)

    try:
        if isinstance(loaded, list):
            model = EnsembleModel(loaded)
            logger.info("MODEL_LOADED")
            return model
        logger.info("MODEL_LOADED")
        return loaded
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.exception("MODEL_LOAD_FAILED: %s", e)
        return None


def online_update(state: BotState, symbol: str, X_new, y_new) -> None:
    y_new = np.clip(y_new, -0.05, 0.05)
    if state.updates_halted:
        return
    with model_lock:
        try:
            _import_model_pipeline().partial_fit(X_new, y_new)
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.error(f"Online update failed for {symbol}: {e}")
            return
    pred = _import_model_pipeline().predict(X_new)
    online_error = float(np.mean((pred - y_new) ** 2))
    ml = _get_metrics_logger()  # AI-AGENT-REF: lazy metrics import
    ml.log_metrics(
        {
            "timestamp": utc_now_iso(),
            "type": "online_update",
            "symbol": symbol,
            "error": online_error,
        }
    )
    state.rolling_losses.append(online_error)
    if len(state.rolling_losses) >= 20 and sum(state.rolling_losses[-20:]) > 0.02:
        state.updates_halted = True
        logger.warning("Halting online updates due to 20-trade rolling loss >2%")


def update_signal_weights() -> None:
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return
    try:
        df = _read_trade_log(
            TRADE_LOG_FILE,
            usecols=[
                "entry_price",
                "exit_price",
                "signal_tags",
                "side",
                "confidence",
                "exit_time",
            ],
        )
        if df is None:
            logger.warning("No trades log found; skipping weight update.")
            return
        df = df.dropna(subset=["entry_price", "exit_price", "signal_tags"])
        direction = np.where(df["side"] == "buy", 1, -1)
        df["pnl"] = (df["exit_price"] - df["entry_price"]) * direction
        df["confidence"] = df.get("confidence", 0.5)
        df["reward"] = df["pnl"] * df["confidence"]
        optimize_signals(df, config)
        recent_cut = pd.to_datetime(df["exit_time"], errors="coerce")
        recent_mask = recent_cut >= (datetime.now(UTC) - timedelta(days=30))
        df_recent = df[recent_mask]

        df_tags = df.assign(tag=df["signal_tags"].str.split("+")).explode("tag")
        df_recent_tags = df_recent.assign(
            tag=df_recent["signal_tags"].str.split("+")
        ).explode("tag")
        stats_all = df_tags.groupby("tag")["reward"].agg(list).to_dict()
        stats_recent = df_recent_tags.groupby("tag")["reward"].agg(list).to_dict()

        new_weights = {}
        for tag, pnls in stats_all.items():
            overall_wr = np.mean([1 if p > 0 else 0 for p in pnls]) if pnls else 0.0
            recent_wr = (
                np.mean([1 if p > 0 else 0 for p in stats_recent.get(tag, [])])
                if stats_recent.get(tag)
                else overall_wr
            )
            weight = 0.7 * recent_wr + 0.3 * overall_wr
            if recent_wr < 0.4:
                weight *= 0.5
            new_weights[tag] = round(weight, 3)

        ALPHA = 0.2
        if os.path.exists(SIGNAL_WEIGHTS_FILE):
            try:
                old_df = pd.read_csv(
                    SIGNAL_WEIGHTS_FILE,
                    on_bad_lines="skip",
                    engine="python",
                    usecols=["signal_name", "weight"],
                )
                if old_df.empty:
                    logger.debug(
                        "Loaded DataFrame from %s is empty after parsing/fallback",
                        SIGNAL_WEIGHTS_FILE,
                    )
                    old = {}
                else:
                    old = old_df.set_index("signal_name")["weight"].to_dict()
            except ValueError as e:
                if "usecols" in str(e).lower():
                    logger.warning(
                        "Signal weights CSV missing expected columns, trying fallback read"
                    )
                    try:
                        # Fallback: read all columns and try to map
                        old_df = pd.read_csv(
                            SIGNAL_WEIGHTS_FILE, on_bad_lines="skip", engine="python"
                        )
                        if "signal" in old_df.columns:
                            # Old format with 'signal' column
                            old = old_df.set_index("signal")["weight"].to_dict()
                        elif "signal_name" in old_df.columns:
                            # New format with 'signal_name' column
                            old = old_df.set_index("signal_name")["weight"].to_dict()
                        else:
                            logger.error(
                                "Signal weights CSV has unexpected format: %s",
                                old_df.columns.tolist(),
                            )
                            old = {}
                    except (
                        FileNotFoundError,
                        PermissionError,
                        IsADirectoryError,
                        JSONDecodeError,
                        ValueError,
                        KeyError,
                        TypeError,
                        OSError,
                    ) as fallback_e:  # AI-AGENT-REF: narrow exception
                        logger.error(
                            "Failed to load signal weights with fallback: %s",
                            fallback_e,
                        )
                        old = {}
                else:
                    logger.error("Failed to load signal weights: %s", e)
                    old = {}
        else:
            old = {}
        merged = {
            tag: round(ALPHA * w + (1 - ALPHA) * old.get(tag, w), 3)
            for tag, w in new_weights.items()
        }
        out_df = pd.DataFrame.from_dict(
            merged, orient="index", columns=["weight"]
        ).reset_index()
        out_df.columns = ["signal_name", "weight"]
        out_df.to_csv(SIGNAL_WEIGHTS_FILE, index=False)
        logger.info("SIGNAL_WEIGHTS_UPDATED", extra={"count": len(merged)})
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.exception(f"update_signal_weights failed: {e}")


def run_meta_learning_weight_optimizer(
    trade_log_path: str = TRADE_LOG_FILE,
    output_path: str = SIGNAL_WEIGHTS_FILE,
    alpha: float = 1.0,
):
    if not meta_lock.acquire(blocking=False):
        logger.warning("METALEARN_SKIPPED_LOCKED")
        return
    try:
        df = _read_trade_log(
            trade_log_path,
            usecols=["entry_price", "exit_price", "signal_tags", "side", "confidence"],
        )
        if df is None:
            logger.warning("METALEARN_NO_TRADES")
            return
        df = df.dropna(subset=["entry_price", "exit_price", "signal_tags"])
        if df.empty:
            logger.warning("METALEARN_NO_VALID_ROWS")
            return

        direction = np.where(df["side"] == "buy", 1, -1)
        df["pnl"] = (df["exit_price"] - df["entry_price"]) * direction
        df["confidence"] = df.get("confidence", 0.5)
        df["reward"] = df["pnl"] * df["confidence"]
        df["outcome"] = (df["pnl"] > 0).astype(int)

        tags = sorted({tag for row in df["signal_tags"] for tag in row.split("+")})
        X = np.array(
            [[int(tag in row.split("+")) for tag in tags] for row in df["signal_tags"]]
        )
        y = df["outcome"].values

        if len(y) < len(tags):
            logger.warning("METALEARN_TOO_FEW_SAMPLES")
            return

        sample_w = df["reward"].abs() + 1e-3
        model = _ridge()(alpha=alpha, fit_intercept=True)
        if X.empty:
            logger.warning("META_MODEL_TRAIN_SKIPPED_EMPTY")
            return
        model.fit(X, y, sample_weight=sample_w)
        atomic_joblib_dump(model, META_MODEL_PATH)
        logger.info("META_MODEL_TRAINED", extra={"samples": len(y)})
        _get_metrics_logger().log_metrics(  # AI-AGENT-REF: lazy metrics import
            {
                "timestamp": utc_now_iso(),
                "type": "meta_model_train",
                "samples": len(y),
                "hyperparams": json.dumps({"alpha": alpha}),
                "seed": SEED,
                "model": "Ridge",
                "git_hash": get_git_hash(),
            }
        )

        weights = {
            tag: round(max(0, min(1, w)), 3)
            for tag, w in zip(tags, model.coef_, strict=False)
        }
        out_df = pd.DataFrame(list(weights.items()), columns=["signal_name", "weight"])
        out_df.to_csv(output_path, index=False)
        logger.info("META_WEIGHTS_UPDATED", extra={"weights": weights})
    finally:
        meta_lock.release()


def run_bayesian_meta_learning_optimizer(
    trade_log_path: str = TRADE_LOG_FILE, output_path: str = SIGNAL_WEIGHTS_FILE
):
    if not meta_lock.acquire(blocking=False):
        logger.warning("METALEARN_SKIPPED_LOCKED")
        return
    try:
        df = _read_trade_log(
            trade_log_path,
            usecols=["entry_price", "exit_price", "signal_tags", "side"],
        )
        if df is None:
            logger.warning("METALEARN_NO_TRADES")
            return
        df = df.dropna(subset=["entry_price", "exit_price", "signal_tags"])
        if df.empty:
            logger.warning("METALEARN_NO_VALID_ROWS")
            return

        direction = np.where(df["side"] == "buy", 1, -1)
        df["pnl"] = (df["exit_price"] - df["entry_price"]) * direction
        df["outcome"] = (df["pnl"] > 0).astype(int)

        tags = sorted({tag for row in df["signal_tags"] for tag in row.split("+")})
        X = np.array(
            [[int(tag in row.split("+")) for tag in tags] for row in df["signal_tags"]]
        )
        y = df["outcome"].values

        if len(y) < len(tags):
            logger.warning("METALEARN_TOO_FEW_SAMPLES")
            return

        model = _bayesian_ridge()(fit_intercept=True, normalize=True)
        if X.size == 0:
            logger.warning("BAYES_MODEL_TRAIN_SKIPPED_EMPTY")
            return
        model.fit(X, y)
        atomic_joblib_dump(model, abspath("meta_model_bayes.pkl"))
        logger.info("META_MODEL_BAYESIAN_TRAINED", extra={"samples": len(y)})
        _get_metrics_logger().log_metrics(  # AI-AGENT-REF: lazy metrics import
            {
                "timestamp": utc_now_iso(),
                "type": "meta_model_bayes_train",
                "samples": len(y),
                "seed": SEED,
                "model": "BayesianRidge",
                "git_hash": get_git_hash(),
            }
        )

        weights = {
            tag: round(max(0, min(1, w)), 3)
            for tag, w in zip(tags, model.coef_, strict=False)
        }
        out_df = pd.DataFrame(list(weights.items()), columns=["signal_name", "weight"])
        out_df.to_csv(output_path, index=False)
        logger.info("META_WEIGHTS_UPDATED", extra={"weights": weights})
    finally:
        meta_lock.release()


def load_global_signal_performance(
    min_trades: int | None = None, threshold: float | None = None
) -> dict[str, float] | None:
    """Load global signal performance with enhanced error handling and configurable thresholds."""
    # AI-AGENT-REF: Use configurable meta-learning parameters from environment
    # Reduced requirements to allow meta-learning to activate more easily
    if min_trades is None:
        min_trades = int(os.getenv("METALEARN_MIN_TRADES", "2"))  # Reduced from 3 to 2
    if threshold is None:
        threshold = float(
            os.getenv("METALEARN_PERFORMANCE_THRESHOLD", "0.3")
        )  # Reduced from 0.4 to 0.3
    try:
        df = _read_trade_log(
            TRADE_LOG_FILE,
            usecols=["exit_price", "entry_price", "signal_tags", "side"],
        )
        if df is None:
            logger.info("METALEARN_NO_HISTORY | Using defaults for new deployment")
            return None
        df = df.dropna(subset=["exit_price", "entry_price", "signal_tags"])

        if df.empty:
            logger.warning("METALEARN_EMPTY_TRADE_LOG - No valid trades found")
            return {}

        # Enhanced data validation and cleaning
        import pandas as pd  # type: ignore
        df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")
        df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
        df["signal_tags"] = df["signal_tags"].astype(str)

        # Remove rows with invalid price data
        df = df.dropna(subset=["exit_price", "entry_price"])
        if df.empty:
            logger.warning(
                "METALEARN_INVALID_PRICES - No trades with valid prices. "
                "This suggests price data corruption or insufficient trading history. "
                "Using default signal weights.",
                extra={
                    "trade_log": TRADE_LOG_FILE,
                    "suggestion": "Check price data format and trade logging",
                },
            )
            return {}

        # Calculate PnL with validation
        direction = np.where(df.side == "buy", 1, -1)
        df["pnl"] = (df.exit_price - df.entry_price) * direction

        # Enhanced signal tag processing
        df_tags = df.assign(tag=df.signal_tags.str.split("+")).explode("tag")
        df_tags = df_tags[
            df_tags["tag"].notna() & (df_tags["tag"] != "")
        ]  # Remove empty tags

        if df_tags.empty:
            logger.warning("METALEARN_NO_SIGNAL_TAGS - No valid signal tags found")
            return {}

        # Calculate win rates with minimum trade validation
        win_rates = {}
        tag_groups = df_tags.groupby("tag")

        for tag, group in tag_groups:
            if len(group) >= min_trades:
                win_rate = (group["pnl"] > 0).mean()
                win_rates[tag] = round(win_rate, 3)

        if not win_rates:
            logger.warning(
                "METALEARN_INSUFFICIENT_TRADES - No signals meet minimum trade requirement (%d)",
                min_trades,
            )
            return {}

        # Filter by performance threshold
        filtered = {tag: wr for tag, wr in win_rates.items() if wr >= threshold}

        # Enhanced logging with more details
        logger.info(
            "METALEARN_FILTERED_SIGNALS",
            extra={
                "signals": list(filtered.keys()) or [],
                "total_signals_analyzed": len(win_rates),
                "signals_above_threshold": len(filtered),
                "threshold": threshold,
                "min_trades": min_trades,
                "total_trades": len(df),
            },
        )

        if not filtered:
            logger.warning(
                "METALEARN_NO_SIGNALS_ABOVE_THRESHOLD - No signals above threshold %.3f",
                threshold,
            )
            # Return best performing signals even if below threshold, with reduced weight
            if win_rates:
                best_signal = max(win_rates.items(), key=lambda x: x[1])
                logger.info(
                    "METALEARN_FALLBACK_SIGNAL - Using best signal: %s (%.3f)",
                    best_signal[0],
                    best_signal[1],
                )
                return {best_signal[0]: best_signal[1] * 0.5}  # Reduced confidence

        return filtered

    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.error(
            "METALEARN_PROCESSING_ERROR - Failed to process signal performance: %s",
            e,
            exc_info=True,
        )
        return {}


def _normalize_index(data: pd.DataFrame) -> pd.DataFrame:
    """Return ``data`` with a clean UTC index named ``date``."""
    if data.index.name:
        data = data.reset_index().rename(columns={data.index.name: "date"})
    else:
        data = data.reset_index().rename(columns={"index": "date"})
    data["date"] = pd.to_datetime(data["date"], utc=True)
    data = data.sort_values("date").set_index("date")
    if data.index.tz is not None:
        data.index = data.index.tz_convert("UTC").tz_localize(None)
    return data


def _add_basic_indicators(
    df: pd.DataFrame, symbol: str, state: BotState | None
) -> None:
    """Add VWAP, RSI, ATR and simple moving averages."""
    try:
        df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        log_warning("INDICATOR_VWAP_FAIL", exc=exc, extra={"symbol": symbol})
        if state:
            state.indicator_failures += 1
        df["vwap"] = np.nan
    try:
        df["rsi"] = ta.rsi(df["close"], length=14)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        log_warning("INDICATOR_RSI_FAIL", exc=exc, extra={"symbol": symbol})
        if state:
            state.indicator_failures += 1
        df["rsi"] = np.nan
    try:
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        log_warning("INDICATOR_ATR_FAIL", exc=exc, extra={"symbol": symbol})
        if state:
            state.indicator_failures += 1
        df["atr"] = np.nan
    if "close" not in df.columns:
        raise KeyError("'close' column missing for SMA calculations")
    close = df["close"].dropna()
    if close.empty:
        raise ValueError("No close price data available for SMA calculations")
    df["sma_50"] = close.astype(float).rolling(window=50).mean()
    df["sma_200"] = close.astype(float).rolling(window=200).mean()


def _add_macd(df: pd.DataFrame, symbol: str, state: BotState | None) -> None:
    """Add MACD indicators using the defensive helper."""
    # ai_trading/core/bot_engine.py:7337 - Convert import guard to hard import (internal module)
    from ai_trading.signals import (
        calculate_macd as signals_calculate_macd,  # type: ignore
    )

    try:
        if "close" not in df.columns:
            raise KeyError("'close' column missing for MACD calculation")
        close_series = df["close"].dropna()
        if close_series.empty:
            raise ValueError("No close price data available for MACD")
        macd_df = signals_calculate_macd(close_series)
        if macd_df is None:
            logger.warning("MACD returned None for %s", symbol)
            raise ValueError("MACD calculation returned None")
        macd_col = macd_df.get("macd")
        signal_col = macd_df.get("signal")
        if macd_col is None or signal_col is None:
            raise KeyError("MACD dataframe missing required columns")
        df["macd"] = macd_col.astype(float)
        df["macds"] = signal_col.astype(float)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        log_warning(
            "INDICATOR_MACD_FAIL",
            exc=exc,
            extra={"symbol": symbol, "snapshot": df["close"].tail(5).to_dict()},
        )
        if state:
            state.indicator_failures += 1
        df["macd"] = np.nan
        df["macds"] = np.nan


def _add_additional_indicators(
    df: pd.DataFrame, symbol: str, state: BotState | None
) -> None:
    """Add a suite of secondary technical indicators."""
    # dedupe any duplicate timestamps
    df = df[~df.index.duplicated(keep="first")]
    try:
        kc = ta.kc(df["high"], df["low"], df["close"], length=20)
        df["kc_lower"] = kc.iloc[:, 0]
        df["kc_mid"] = kc.iloc[:, 1]
        df["kc_upper"] = kc.iloc[:, 2]
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        log_warning("INDICATOR_KC_FAIL", exc=exc, extra={"symbol": symbol})
        if state:
            state.indicator_failures += 1
        df["kc_lower"] = np.nan
        df["kc_mid"] = np.nan
        df["kc_upper"] = np.nan

    df["atr_band_upper"] = df["close"] + 1.5 * df["atr"]
    df["atr_band_lower"] = df["close"] - 1.5 * df["atr"]
    df["avg_vol_20"] = df["volume"].rolling(20).mean()
    df["dow"] = df.index.dayofweek

    try:
        bb = ta.bbands(df["close"], length=20)
        df["bb_upper"] = bb["BBU_20_2.0"]
        df["bb_lower"] = bb["BBL_20_2.0"]
        df["bb_percent"] = bb["BBP_20_2.0"]
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        log_warning("INDICATOR_BBANDS_FAIL", exc=exc, extra={"symbol": symbol})
        if state:
            state.indicator_failures += 1
        df["bb_upper"] = np.nan
        df["bb_lower"] = np.nan
        df["bb_percent"] = np.nan

    try:
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["adx"] = adx["ADX_14"]
        df["dmp"] = adx["DMP_14"]
        df["dmn"] = adx["DMN_14"]
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        log_warning("INDICATOR_ADX_FAIL", exc=exc, extra={"symbol": symbol})
        if state:
            state.indicator_failures += 1
        df["adx"] = np.nan
        df["dmp"] = np.nan
        df["dmn"] = np.nan

    try:
        df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        log_warning("INDICATOR_CCI_FAIL", exc=exc, extra={"symbol": symbol})
        if state:
            state.indicator_failures += 1
        df["cci"] = np.nan

    df[["high", "low", "close", "volume"]] = df[
        ["high", "low", "close", "volume"]
    ].astype(float)
    try:
        mfi_vals = ta.mfi(df.high, df.low, df.close, df.volume, length=14)
        df["+mfi"] = mfi_vals
    except ValueError:
        logger.warning("Skipping MFI: insufficient or duplicate data")

    try:
        df["tema"] = ta.tema(df["close"], length=10)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        log_warning("INDICATOR_TEMA_FAIL", exc=exc, extra={"symbol": symbol})
        if state:
            state.indicator_failures += 1
        df["tema"] = np.nan

    try:
        df["willr"] = ta.willr(df["high"], df["low"], df["close"], length=14)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        log_warning("INDICATOR_WILLR_FAIL", exc=exc, extra={"symbol": symbol})
        if state:
            state.indicator_failures += 1
        df["willr"] = np.nan

    try:
        psar = ta.psar(df["high"], df["low"], df["close"])
        df["psar_long"] = psar["PSARl_0.02_0.2"]
        df["psar_short"] = psar["PSARs_0.02_0.2"]
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        log_warning("INDICATOR_PSAR_FAIL", exc=exc, extra={"symbol": symbol})
        if state:
            state.indicator_failures += 1
        df["psar_long"] = np.nan
        df["psar_short"] = np.nan

    try:
        # compute_ichimoku returns the indicator dataframe and the signal dataframe
        ich_df, ich_signal_df = compute_ichimoku(df["high"], df["low"], df["close"])
        for col in ich_df.columns:
            df[f"ich_{col}"] = ich_df[col]
        for col in ich_signal_df.columns:
            df[f"ichi_signal_{col}"] = ich_signal_df[col]
    except (KeyError, IndexError):
        logger.warning("Skipping Ichimoku: empty or irregular index")

    try:
        st = ta.stochrsi(df["close"])
        df["stochrsi"] = st["STOCHRSIk_14_14_3_3"]
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        log_warning("INDICATOR_STOCHRSI_FAIL", exc=exc, extra={"symbol": symbol})
        if state:
            state.indicator_failures += 1
        df["stochrsi"] = np.nan


def _add_multi_timeframe_features(
    df: pd.DataFrame, symbol: str, state: BotState | None
) -> None:
    """Add multi-timeframe and lag-based features."""
    try:
        df["ret_5m"] = df["close"].pct_change(5, fill_method=None)
        df["ret_1h"] = df["close"].pct_change(60, fill_method=None)
        df["ret_d"] = df["close"].pct_change(390, fill_method=None)
        df["ret_w"] = df["close"].pct_change(1950, fill_method=None)
        df["vol_norm"] = (
            df["volume"].rolling(60).mean() / df["volume"].rolling(5).mean()
        )
        df["5m_vs_1h"] = df["ret_5m"] - df["ret_1h"]
        df["vol_5m"] = df["close"].pct_change(fill_method=None).rolling(5).std()
        df["vol_1h"] = df["close"].pct_change(fill_method=None).rolling(60).std()
        df["vol_d"] = df["close"].pct_change(fill_method=None).rolling(390).std()
        df["vol_w"] = df["close"].pct_change(fill_method=None).rolling(1950).std()
        df["vol_ratio"] = df["vol_5m"] / df["vol_1h"]
        df["mom_agg"] = df["ret_5m"] + df["ret_1h"] + df["ret_d"]
        df["lag_close_1"] = df["close"].shift(1)
        df["lag_close_3"] = df["close"].shift(3)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        log_warning("INDICATOR_MULTITF_FAIL", exc=exc, extra={"symbol": symbol})
        if state:
            state.indicator_failures += 1
        df["ret_5m"] = df["ret_1h"] = df["ret_d"] = df["ret_w"] = np.nan
        df["vol_norm"] = df["5m_vs_1h"] = np.nan
        df["vol_5m"] = df["vol_1h"] = df["vol_d"] = df["vol_w"] = np.nan
        df["vol_ratio"] = df["mom_agg"] = df["lag_close_1"] = df["lag_close_3"] = np.nan


def _drop_inactive_features(df: pd.DataFrame) -> None:
    """Remove features listed in ``INACTIVE_FEATURES_FILE`` if present."""
    if os.path.exists(INACTIVE_FEATURES_FILE):
        try:
            with open(INACTIVE_FEATURES_FILE) as f:
                inactive = set(json.load(f))
            df.drop(
                columns=[c for c in inactive if c in df.columns],
                inplace=True,
                errors="ignore",
            )
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as exc:  # pragma: no cover - unexpected I/O  # AI-AGENT-REF: narrow exception
            logger.exception("bot.py unexpected", exc_info=exc)
            raise


@profile
def prepare_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    # Calculate RSI and assign to both rsi and rsi_14
    frame["rsi"] = ta.rsi(frame["close"], length=14)
    frame["rsi_14"] = frame["rsi"]

    # Ichimoku conversion and base lines
    frame["ichimoku_conv"] = (
        frame["high"].rolling(window=9).max() + frame["low"].rolling(window=9).min()
    ) / 2
    frame["ichimoku_base"] = (
        frame["high"].rolling(window=26).max() + frame["low"].rolling(window=26).min()
    ) / 2

    # Stochastic RSI calculation
    rsi_min = frame["rsi_14"].rolling(window=14).min()
    rsi_max = frame["rsi_14"].rolling(window=14).max()
    frame["stochrsi"] = (frame["rsi_14"] - rsi_min) / (rsi_max - rsi_min)

    # Guarantee all required columns exist
    required = ["ichimoku_conv", "ichimoku_base", "stochrsi"]
    for col in required:
        if col not in frame.columns:
            frame[col] = np.nan

    # Only drop rows where all are missing
    frame.dropna(subset=required, how="all", inplace=True)

    return frame


# --- Back-compat path for tests that expect this symbol at module scope ---
def _compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute regime features; tolerate proxy bars that only include 'close'."""
    # 1) Canonicalize columns (o/h/l/c/v mapping, lowercase)
    try:
        from ai_trading.utils.ohlcv import standardize_ohlcv

        df = standardize_ohlcv(df)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        # OHLCV standardization failed - log warning but continue with raw data
        logger.warning("Failed to standardize OHLCV data: %s", e)

    # 2) Synthesize missing OHLC from 'close' when needed (proxy baskets)
    if "close" in df.columns:
        if "high" not in df.columns:
            df["high"] = df["close"].rolling(3, min_periods=1).max()
        if "low" not in df.columns:
            df["low"] = df["close"].rolling(3, min_periods=1).min()
        if "open" not in df.columns:
            df["open"] = df["close"].shift(1).fillna(df["close"])
        if "volume" not in df.columns:
            df["volume"] = 0.0

    # 3) Optional MACD import (keep failures soft)
    try:
        from ai_trading.signals import (
            calculate_macd as signals_calculate_macd,  # type: ignore
        )
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ):  # AI-AGENT-REF: narrow exception
        logger.warning("signals module not available for regime features")
        signals_calculate_macd = None

    # 4) Build features with fallbacks
    feat = pd.DataFrame(index=df.index)
    try:
        feat["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ):  # AI-AGENT-REF: narrow exception
        # Fallback ATR proxy from close-to-close movement
        tr = (df["close"].diff().abs()).fillna(0.0)
        feat["atr"] = tr.rolling(14, min_periods=1).mean()
    feat["rsi"] = ta.rsi(df["close"], length=14)
    if signals_calculate_macd:
        try:
            macd_df = signals_calculate_macd(df["close"])
            feat["macd"] = (
                macd_df["macd"] if macd_df is not None and "macd" in macd_df else np.nan
            )
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.warning("Regime MACD calculation failed: %s", e)
            feat["macd"] = np.nan
    else:
        feat["macd"] = np.nan
    feat["vol"] = (
        df["close"].pct_change(fill_method=None).rolling(14, min_periods=1).std()
    )
    return feat.dropna(how="all")


def detect_regime(df: pd.DataFrame) -> str:
    """Simple SMA-based market regime detection."""
    if df is None or df.empty or "close" not in df:
        return "chop"
    close = df["close"].astype(float)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    if sma50.iloc[-1] > sma200.iloc[-1]:
        return "bull"
    if sma50.iloc[-1] < sma200.iloc[-1]:
        return "bear"
    return "chop"


def _initialize_regime_model(ctx=None):
    """Initialize regime model - load existing or train new one."""
    # Train or load regime model - skip in test environment
    if os.getenv("TESTING") == "1" or os.getenv("PYTEST_RUNNING"):
        logger.info("Skipping regime model training in test environment")
        return _rf_class()(n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH)
    elif os.path.exists(REGIME_MODEL_PATH):
        try:
            model_path = Path(REGIME_MODEL_PATH)
            return safe_pickle_load(model_path, [model_path.parent])
        except RuntimeError as e:  # AI-AGENT-REF: path-validated load
            logger.warning("Failed to load regime model: %s", e)
            return _rf_class()(n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH)
    else:
        if ctx is None:
            logger.warning(
                "No context provided for regime model training; using fallback"
            )
            return _rf_class()(n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH)

        # --- Regime training uses basket-based proxy now ---
        wide = _build_regime_dataset(ctx)
        if wide is None or getattr(wide, "empty", False):
            logger.warning("Regime basket is empty; skipping model train")
            bars = pd.DataFrame()
        else:
            bars = _regime_basket_to_proxy_bars(wide)

        # Normalize to a DatetimeIndex robustly (proxy has 'timestamp' column)
        try:
            if "timestamp" in getattr(bars, "columns", []):
                idx = safe_to_datetime(bars["timestamp"], context="regime data")
                bars = bars.drop(columns=["timestamp"])
                bars.index = idx
            # Final conversion (idempotent for Timestamps)
            bars.index = safe_to_datetime(bars.index, context="regime data")
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.warning("REGIME index normalization failed: %s", e)
            bars = pd.DataFrame()
        bars = bars.rename(columns=lambda c: c.lower())
        feats = _compute_regime_features(bars)
        labels = (
            (bars["close"] > bars["close"].rolling(200).mean())
            .loc[feats.index]
            .astype(int)
            .rename("label")
        )
        training = feats.join(labels, how="inner").dropna()

        # Add validation for training data quality
        if training.empty:
            logger.warning(
                "Regime training dataset is empty after joining features and labels"
            )
            if not _REGIME_INSUFFICIENT_DATA_WARNED["done"]:
                logger.warning("No valid training data for regime model; using fallback")
                _REGIME_INSUFFICIENT_DATA_WARNED["done"] = True
            return _rf_class()(n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH)

        # Import settings for regime configuration
        from ai_trading.config.settings import get_settings

        settings = get_settings()

        logger.debug(
            "Regime training data validation: %d rows available, minimum required: %d",
            len(training),
            settings.REGIME_MIN_ROWS,
        )

        if len(training) >= settings.REGIME_MIN_ROWS:
            X = training[["atr", "rsi", "macd", "vol"]]
            y = training["label"]
            regime_model = _rf_class()(
                n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH
            )
            if X.empty:
                logger.warning("REGIME_MODEL_TRAIN_SKIPPED_EMPTY")
            else:
                regime_model.fit(X, y)
            try:
                atomic_pickle_dump(regime_model, REGIME_MODEL_PATH)
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as e:  # AI-AGENT-REF: narrow exception
                logger.warning(f"Failed to save regime model: {e}")
            else:
                logger.info("REGIME_MODEL_TRAINED", extra={"rows": len(training)})
            return regime_model
        else:
            # Log once at WARNING level; avoid noisy ERROR during closed market.
            if not _REGIME_INSUFFICIENT_DATA_WARNED["done"]:
                logger.warning(
                    "Insufficient rows (%d < %d) for regime model; using fallback",
                    len(training),
                    settings.REGIME_MIN_ROWS,
                )
                _REGIME_INSUFFICIENT_DATA_WARNED["done"] = True
            return _rf_class()(n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH)


# Initialize regime model lazily
regime_model = None


def _market_breadth(ctx: BotContext) -> float:
    syms = load_tickers(TICKERS_FILE)[:20]
    up = 0
    total = 0
    for sym in syms:
        df = ctx.data_fetcher.get_daily_df(ctx, sym)
        if df is None or len(df) < 2:
            continue
        total += 1
        if df["close"].iloc[-1] > df["close"].iloc[-2]:
            up += 1
    return up / total if total else 0.5


def detect_regime_state(ctx: BotContext) -> str:
    """
    Inspect recent returns/volatility/volume breadth to classify the regime.
    NOTE: Previously this used a free/global `ctx`. We now pass the explicit
    runtime.
    """
    try:
        df = ctx.data_fetcher.get_daily_df(ctx, REGIME_SYMBOLS[0])
        if df is None or len(df) < 200:
            return "sideways"
        atr14 = ta.atr(df["high"], df["low"], df["close"], length=14).iloc[-1]
        atr50 = ta.atr(df["high"], df["low"], df["close"], length=50).iloc[-1]
        high_vol = atr50 > 0 and atr14 / atr50 > 1.5
        sma50 = df["close"].rolling(50).mean().iloc[-1]
        sma200 = df["close"].rolling(200).mean().iloc[-1]
        trend = sma50 - sma200
        breadth = _market_breadth(ctx)
        if high_vol:
            return "high_volatility"
        if abs(trend) / sma200 < 0.005:
            return "sideways"
        if trend > 0 and breadth > 0.55:
            return "trending"
        if trend < 0 and breadth < 0.45:
            return "mean_reversion"
        return "sideways"
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(
            "REGIME_DETECT_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        return "sideways"


def check_market_regime(runtime: BotContext, state: BotState) -> bool:
    """
    Evaluate the current market regime and update state.current_regime.
    Returns True/False indicating whether trading is allowed under this regime.
    """
    try:
        # AI-AGENT-REF: pass runtime explicitly into regime detection
        state.current_regime = detect_regime_state(runtime)
        return bool(getattr(state.current_regime, "allow_trading", True))
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(
            "REGIME_DETECT_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        return False


_SCREEN_CACHE: dict[str, float] = {}


def _validate_market_data_quality(df: pd.DataFrame, symbol: str) -> dict:
    """
    Comprehensive market data validation to prevent trading with insufficient or poor quality data.

    AI-AGENT-REF: Critical fix for market data validation problems from problem statement.
    Implements circuit breakers for poor data conditions and minimum data requirements.

    Returns:
        dict: Validation result with valid flag, reason, and detailed message
    """
    try:
        # Basic existence check
        if df is None:
            return {
                "valid": False,
                "reason": "no_data",
                "message": "No data available",
                "details": {"symbol": symbol, "data_source": "missing"},
            }

        # Minimum rows requirement
        min_rows_required = max(
            ATR_LENGTH, 20
        )  # Ensure enough for technical indicators
        if len(df) < min_rows_required:
            return {
                "valid": False,
                "reason": f"insufficient_data_{len(df)}_rows",
                "message": f"Insufficient data ({len(df)} rows, need {min_rows_required})",
                "details": {
                    "symbol": symbol,
                    "rows_available": len(df),
                    "rows_required": min_rows_required,
                },
            }

        # Data completeness checks
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {
                "valid": False,
                "reason": "missing_columns",
                "message": f"Missing required columns: {missing_columns}",
                "details": {
                    "symbol": symbol,
                    "missing_columns": missing_columns,
                    "available_columns": list(df.columns),
                },
            }

        # Data quality checks
        recent_data = df.tail(min(50, len(df)))  # Check last 50 rows or all available

        # Check for excessive NaN values
        for col in required_columns:
            nan_count = recent_data[col].isna().sum()
            nan_percentage = (nan_count / len(recent_data)) * 100
            if nan_percentage > 10:  # More than 10% NaN values
                return {
                    "valid": False,
                    "reason": f"excessive_nan_{col}",
                    "message": f"Excessive NaN values in {col} column ({nan_percentage:.1f}%)",
                    "details": {
                        "symbol": symbol,
                        "column": col,
                        "nan_percentage": nan_percentage,
                    },
                }

        # Check for price data anomalies
        close_prices = recent_data["close"].dropna()
        if len(close_prices) < 5:
            return {
                "valid": False,
                "reason": "insufficient_price_data",
                "message": "Less than 5 valid close prices in recent data",
                "details": {"symbol": symbol, "valid_close_prices": len(close_prices)},
            }

        # Check for zero or negative prices
        if (close_prices <= 0).any():
            return {
                "valid": False,
                "reason": "invalid_prices",
                "message": "Found zero or negative prices",
                "details": {"symbol": symbol, "min_price": float(close_prices.min())},
            }

        # Check for unrealistic price volatility (circuit breaker)
        price_changes = close_prices.pct_change().dropna()
        if len(price_changes) > 0:
            extreme_moves = (abs(price_changes) > 0.5).sum()  # 50%+ single-day moves
            if (
                extreme_moves > len(price_changes) * 0.1
            ):  # More than 10% of days have extreme moves
                logger.warning(
                    "DATA_QUALITY_EXTREME_VOLATILITY",
                    extra={
                        "symbol": symbol,
                        "extreme_moves": extreme_moves,
                        "total_days": len(price_changes),
                        "percentage": round(
                            (extreme_moves / len(price_changes)) * 100, 1
                        ),
                        "note": "Potential data quality issue - consider excluding from trading",
                    },
                )

        # Check volume data quality
        volume_data = recent_data["volume"].dropna()
        if len(volume_data) > 0:
            # Check for suspiciously low volume
            median_volume = volume_data.median()
            if median_volume < 10000:  # Very low liquidity threshold
                return {
                    "valid": False,
                    "reason": "low_liquidity",
                    "message": f"Median volume too low ({median_volume:,.0f})",
                    "details": {
                        "symbol": symbol,
                        "median_volume": median_volume,
                        "threshold": 10000,
                    },
                }

            # Check for zero volume days
            zero_volume_days = (volume_data == 0).sum()
            if (
                zero_volume_days > len(volume_data) * 0.2
            ):  # More than 20% zero volume days
                return {
                    "valid": False,
                    "reason": "excessive_zero_volume",
                    "message": f"Too many zero volume days ({zero_volume_days}/{len(volume_data)})",
                    "details": {
                        "symbol": symbol,
                        "zero_volume_days": zero_volume_days,
                        "total_days": len(volume_data),
                    },
                }

        # All checks passed
        return {
            "valid": True,
            "reason": "passed_validation",
            "message": "Data quality validation passed",
            "details": {
                "symbol": symbol,
                "rows_validated": len(df),
                "recent_rows_checked": len(recent_data),
                "validation_checks": [
                    "existence",
                    "completeness",
                    "quality",
                    "anomalies",
                    "volume",
                ],
            },
        }

    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.error(
            "DATA_VALIDATION_ERROR",
            extra={"symbol": symbol, "error": str(e), "error_type": type(e).__name__},
        )
        return {
            "valid": False,
            "reason": "validation_error",
            "message": f"Data validation failed with error: {e}",
            "details": {"symbol": symbol, "error": str(e)},
        }

_screen_lock = Lock()
_screening_in_progress = False

# Hard-coded fallback symbols when no watchlist is provided
FALLBACK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


def screen_universe(
    candidates: Sequence[str],
    runtime,
) -> list[str]:
    global _screening_in_progress
    if not _screen_lock.acquire(blocking=False):
        logger.info("SCREENER_SKIP_REENTRANT")
        return []
    try:
        if _screening_in_progress:
            logger.info("SCREENER_SKIP_ALREADY_RUNNING")
            return []
        _screening_in_progress = True
        try:
            top_n = 20  # AI-AGENT-REF: maintain top N selection
            cand_set = set(candidates)
            logger.info(
                f"[SCREEN_UNIVERSE] Starting screening of {len(cand_set)} candidates: {sorted(cand_set)}"
            )

            spy_df = runtime.data_fetcher.get_daily_df(runtime, "SPY")
            if spy_df is None or spy_df.empty:
                time.sleep(0.25)
                spy_df = runtime.data_fetcher.get_daily_df(runtime, "SPY")
            if spy_df is None or spy_df.empty:
                if not is_market_open():
                    logger.info(
                        "DATA_FEED_UNAVAILABLE",
                        extra={
                            "provider": "alpaca",
                            "status": "empty",
                            "context": "market closed; health check deferred",
                        },
                    )
                else:
                    logger.warning(
                        "DATA_FEED_UNAVAILABLE",
                        extra={"provider": "alpaca", "status": "empty"},
                    )
                return []

            for sym in list(_SCREEN_CACHE):
                if sym not in cand_set:
                    _SCREEN_CACHE.pop(sym, None)

            new_syms = cand_set - _SCREEN_CACHE.keys()
            filtered_out = {}  # Track reasons for filtering
            tried = valid = empty = failed = 0

            for sym in new_syms:
                tried += 1
                df = runtime.data_fetcher.get_daily_df(runtime, sym)
                if not is_valid_ohlcv(df):
                    empty += 1
                    filtered_out[sym] = "no_data"
                    logger.debug(f"[SCREEN_UNIVERSE] {sym}: returned empty dataframe")
                    time.sleep(0.25)
                    continue

                # AI-AGENT-REF: Enhanced market data validation for critical trading decisions
                validation_result = _validate_market_data_quality(df, sym)
                if not validation_result["valid"]:
                    failed += 1
                    filtered_out[sym] = validation_result["reason"]
                    logger.debug(f"[SCREEN_UNIVERSE] {sym}: {validation_result['message']}")
                    time.sleep(0.25)
                    continue

                original_len = len(df)
                df = df[df["volume"] > 100_000]
                if df.empty:
                    filtered_out[sym] = "low_volume"
                    logger.debug(
                        f"[SCREEN_UNIVERSE] {sym}: Filtered out due to low volume (original: {original_len} rows)"
                    )
                    time.sleep(0.25)
                    continue

                def _calc_atr(df_in: pd.DataFrame) -> pd.Series:
                    ser: pd.Series | None = None
                    ta_available = hasattr(ta, "atr") and not getattr(ta, "_failed", False)
                    if ta_available:
                        try:
                            ser = ta.atr(
                                df_in["high"], df_in["low"], df_in["close"], length=ATR_LENGTH
                            )
                        except Exception:  # pragma: no cover - fall back below
                            ser = None
                    if ser is None or not hasattr(ser, "empty") or ser.empty:
                        try:
                            from ai_trading.indicators import atr as _atr

                            ser = _atr(
                                df_in["high"], df_in["low"], df_in["close"], period=ATR_LENGTH
                            )
                        except Exception:
                            ser = pd.Series()
                    return ser if isinstance(ser, pd.Series) else pd.Series()

                series = _calc_atr(df)
                if series.empty or series.dropna().empty:
                    try:
                        from datetime import UTC, datetime, timedelta
                        from ai_trading.data.fetch import get_daily_df as _fetch_daily_df

                        end = datetime.now(UTC)
                        start = end - timedelta(days=DEFAULT_DAILY_LOOKBACK_DAYS * 2)
                        df2 = _fetch_daily_df(sym, start, end)
                        df2 = df2[df2["volume"] > 100_000]
                        series = _calc_atr(df2)
                        if series.empty or series.dropna().empty:
                            filtered_out[sym] = "atr_insufficient_data"
                            logger.warning(
                                f"[SCREEN_UNIVERSE] {sym}: ATR unavailable after extended fetch"
                            )
                            time.sleep(0.25)
                            continue
                        df = df2
                    except Exception:
                        filtered_out[sym] = "atr_fetch_failed"
                        logger.warning(
                            f"[SCREEN_UNIVERSE] {sym}: ATR extended fetch failed"
                        )
                        time.sleep(0.25)
                        continue

                atr_val = series.iloc[-1]
                if not pd.isna(atr_val):
                    _SCREEN_CACHE[sym] = float(atr_val)
                    logger.debug(f"[SCREEN_UNIVERSE] {sym}: ATR = {atr_val:.4f}")
                    valid += 1
                else:
                    filtered_out[sym] = "atr_nan"
                    logger.debug(f"[SCREEN_UNIVERSE] {sym}: ATR value is NaN")
                time.sleep(0.25)

            atrs = {sym: _SCREEN_CACHE[sym] for sym in cand_set if sym in _SCREEN_CACHE}
            ranked = sorted(atrs.items(), key=lambda kv: kv[1], reverse=True)
            selected = [sym for sym, _ in ranked[:top_n]]

            logger.info(
                f"[SCREEN_UNIVERSE] Selected {len(selected)} of {len(cand_set)} candidates. "
                f"Selected: {selected}. "
                f"Filtered out: {len(filtered_out)} symbols: {filtered_out}"
            )
            logger.info(
                "SCREEN_SUMMARY",
                extra={
                    "tried": tried,
                    "valid": valid,
                    "empty": empty,
                    "failed": failed,
                },
            )

            return selected
        except (KeyError, ValueError, TypeError) as e:
            logger.error(
                "SCREENING_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
            raise
        finally:
            _screening_in_progress = False
    finally:
        _screen_lock.release()


def screen_candidates(runtime, candidates, *, fallback_symbols=None) -> list[str]:
    """Run screening on provided candidate tickers using runtime."""
    try:
        if not candidates:
            symbols = list(fallback_symbols or FALLBACK_SYMBOLS)
            logger.info(
                "SCREEN_FALLBACK_USED",
                extra={"tickers": symbols},
            )
            return symbols
        return screen_universe(candidates, runtime)
    except (KeyError, ValueError, TypeError) as e:
        logger.error(
            "SCREENING_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        raise


# Fix for handling missing tickers.csv file
def get_stock_bars_safe(api, symbol, timeframe):
    """Safely fetch stock bars using Alpaca's request object."""

    import pandas as pd

    try:
        req = _bars.StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=_parse_timeframe(timeframe),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Malformed StockBarsRequest") from exc

    get_stock_bars_fn = getattr(api, "get_stock_bars", None)
    if callable(get_stock_bars_fn):
        try:
            resp = get_stock_bars_fn(req)
        except TypeError as exc:
            raise RuntimeError(
                "Incompatible Alpaca SDK: expected get_stock_bars(request)"
            ) from exc
    else:
        get_bars_fn = getattr(api, "get_bars", None)
        if callable(get_bars_fn):
            return get_bars_fn(symbol, timeframe)
        raise AttributeError("API missing get_stock_bars/get_bars")

    df = getattr(resp, "df", resp)
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Unexpected get_stock_bars response") from exc
    return df


def load_tickers(path: str = TICKERS_FILE) -> list[str]:
    """Load tickers from file or CSV list; no fallback."""
    return load_universe_from_path(path)


def load_candidate_universe(runtime, tickers: list[str] | None = None) -> list[str]:
    """Load tickers for screening from cached runtime list.

    Raises
    ------
    RuntimeError
        If no tickers are available after loading.
    """  # AI-AGENT-REF: use packaged universe loader
    if tickers is not None:
        setattr(runtime, "tickers", tickers)
    tickers = getattr(runtime, "tickers", None)
    if not tickers:
        tickers = load_tickers()
        setattr(runtime, "tickers", tickers)
    if not tickers:
        raise RuntimeError("No tickers available")
    logger.debug(
        "CANDIDATE_UNIVERSE_LOADED",
        extra={"count": len(tickers)},
    )
    return tickers


def daily_summary() -> None:
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return
    try:
        df = _read_trade_log(
            TRADE_LOG_FILE, usecols=["entry_price", "exit_price", "side"]
        )
        if df is None:
            logger.info("DAILY_SUMMARY_NO_TRADES")
            return
        df = df.dropna(subset=["entry_price", "exit_price"])
        direction = np.where(df["side"] == "buy", 1, -1)
        df["pnl"] = (df.exit_price - df.entry_price) * direction
        total_trades = len(df)
        win_rate = (df.pnl > 0).mean() if total_trades else 0
        total_pnl = df.pnl.sum()
        max_dd = (df.pnl.cumsum().cummax() - df.pnl.cumsum()).max()
        logger.info(
            "DAILY_SUMMARY",
            extra={
                "trades": total_trades,
                "win_rate": f"{win_rate:.2%}",
                "pnl": total_pnl,
                "max_drawdown": max_dd,
            },
        )
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.exception(f"daily_summary failed: {e}")


# ─── PCA-BASED PORTFOLIO ADJUSTMENT ─────────────────────────────────────────────
def run_daily_pca_adjustment(ctx: BotContext) -> None:
    from ai_trading.utils import portfolio_lock

    if not SKLEARN_AVAILABLE:
        return
    from sklearn.decomposition import PCA  # lazy import

    """
    Once per day, run PCA on last 90-day returns of current universe.
    If top PC explains >40% variance and portfolio loads heavily,
    reduce those weights by 20%.
    """
    universe = list(ctx.portfolio_weights.keys())
    if not universe:
        return
    returns_df = pd.DataFrame()
    for sym in universe:
        df = ctx.data_fetcher.get_daily_df(ctx, sym)
        if df is None or len(df) < 90:
            continue
        rts = df["close"].pct_change(fill_method=None).tail(90).reset_index(drop=True)
        returns_df[sym] = rts
    returns_df = returns_df.dropna(axis=1, how="any")
    if returns_df.shape[1] < 2:
        return
    pca = PCA(n_components=3)
    if returns_df.empty:
        logger.warning("PCA_SKIPPED_EMPTY_RETURNS")
        return
    pca.fit(returns_df.values)
    var_explained = pca.explained_variance_ratio_[0]
    if var_explained < 0.4:
        return
    top_loadings = pd.Series(pca.components_[0], index=returns_df.columns).abs()
    # Identify symbols loading > median loading
    median_load = top_loadings.median()
    high_load_syms = top_loadings[top_loadings > median_load].index.tolist()
    if not high_load_syms:
        return
    # Reduce those weights by 20%
    with portfolio_lock:  # FIXED: protect shared portfolio state
        for sym in high_load_syms:
            old = ctx.portfolio_weights.get(sym, 0.0)
            ctx.portfolio_weights[sym] = round(old * 0.8, 4)
        # Re-normalize to sum to 1
        total = sum(ctx.portfolio_weights.values())
        if total > 0:
            for sym in ctx.portfolio_weights:
                ctx.portfolio_weights[sym] = round(
                    ctx.portfolio_weights[sym] / total, 4
                )
    logger.info(
        "PCA_ADJUSTMENT_APPLIED",
        extra={"var_explained": round(var_explained, 3), "adjusted": high_load_syms},
    )


def daily_reset(state: BotState) -> None:
    """Reset daily counters and in-memory slippage logs."""
    try:
        _reload_env()
        _slippage_log.clear()
        state.loss_streak = 0
        logger.info("DAILY_STATE_RESET")
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.exception(f"daily_reset failed: {e}")


def _average_reward(n: int = 20) -> float:
    if not os.path.exists(REWARD_LOG_FILE):
        return 0.0
    df = pd.read_csv(
        REWARD_LOG_FILE,
        on_bad_lines="skip",
        engine="python",
        usecols=["reward"],
    ).tail(n)
    if df.empty:
        if _is_market_open_now():
            logger.debug(
                "Loaded DataFrame from %s is empty after parsing/fallback",
                REWARD_LOG_FILE,
            )
        else:
            logger.debug(
                "Loaded DataFrame from %s is empty (market closed)",
                REWARD_LOG_FILE,
            )
    if df.empty or "reward" not in df.columns:
        return 0.0
    return float(df["reward"].mean())


def _current_drawdown() -> float:
    try:
        with open(PEAK_EQUITY_FILE) as pf:
            peak = float(pf.read().strip() or 0)
        with open(EQUITY_FILE) as ef:
            eq = float(ef.read().strip() or 0)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ):  # AI-AGENT-REF: narrow exception
        return 0.0
    if peak <= 0:
        return 0.0
    return max(0.0, (peak - eq) / peak)


def update_bot_mode(state: BotState) -> None:
    try:
        avg_r = _average_reward()
        dd = _current_drawdown()
        regime = state.current_regime
        if dd > 0.05 or avg_r < -0.01:
            new_mode = "conservative"
        elif avg_r > 0.05 and regime == "trending":
            new_mode = "aggressive"
        else:
            new_mode = "balanced"
        if new_mode != state.mode_obj.mode:
            state.mode_obj = BotMode(new_mode)
            state.params.update(state.mode_obj.get_config())
            state.kelly_fraction = state.params.get("KELLY_FRACTION", 0.6)
            logger.info(
                "MODE_SWITCH",
                extra={
                    "new_mode": new_mode,
                    "avg_reward": avg_r,
                    "drawdown": dd,
                    "regime": regime,
                },
            )
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.exception(f"update_bot_mode failed: {e}")


def adaptive_risk_scaling(ctx: BotContext) -> None:
    """Adjust risk parameters based on volatility, rewards and drawdown."""
    try:
        vol = _VOL_STATS.get("mean", 0)
        spy_atr = _VOL_STATS.get("last", 0)
        avg_r = _average_reward(30)
        dd = _current_drawdown()
        try:
            equity = float(ctx.api.get_account().equity)
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ):  # AI-AGENT-REF: narrow exception
            equity = 0.0
        update_if_present(ctx, equity)
        params["get_capital_cap()"] = ctx.params["get_capital_cap()"]
        frac = params.get("KELLY_FRACTION", 0.6)
        if spy_atr and vol and spy_atr > vol * 1.5:
            frac *= 0.5
        if avg_r < -0.02:
            frac *= 0.7
        if dd > 0.1:
            frac *= 0.5
        ctx.kelly_fraction = round(max(0.2, min(frac, 1.0)), 2)
        params["get_capital_cap()"] = round(
            max(0.02, min(0.1, params.get("get_capital_cap()", 0.25) * (1 - dd))), 3
        )
        logger.info(
            "RISK_SCALED",
            extra={
                "kelly_fraction": ctx.kelly_fraction,
                "dd": dd,
                "atr": spy_atr,
                "avg_reward": avg_r,
            },
        )
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.exception(f"adaptive_risk_scaling failed: {e}")


def check_disaster_halt() -> None:
    try:
        dd = _current_drawdown()
        if dd >= get_disaster_dd_limit():
            set_halt_flag(f"DISASTER_DRAW_DOWN_{dd:.2%}")
            logger.error("DISASTER_HALT_TRIGGERED", extra={"drawdown": dd})
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.exception(f"check_disaster_halt failed: {e}")


# retrain_meta_learner is imported above if available


def load_or_retrain_daily(ctx: BotContext) -> Any:
    """
    1. Check RETRAIN_MARKER_FILE for last retrain date (YYYY-MM-DD).
    2. If missing or older than today, call retrain_meta_learner(ctx, symbols) and update marker.
    3. Then load the (new) model from MODEL_PATH.
    """
    today_str = (
        datetime.now(UTC).astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    )
    marker = RETRAIN_MARKER_FILE

    need_to_retrain = True
    if CFG.disable_daily_retrain:
        logger.info("Daily retraining disabled via DISABLE_DAILY_RETRAIN")
        need_to_retrain = False
    if os.path.isfile(marker):
        with open(marker) as f:
            last_date = f.read().strip()
        if last_date == today_str:
            need_to_retrain = False

    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        logger.warning(
            "MODEL_PATH missing; forcing initial retrain.",
            extra={"path": MODEL_PATH or ""},
        )
        need_to_retrain = True  # AI-AGENT-REF: guard for missing model path

    if need_to_retrain:
        if not callable(globals().get("retrain_meta_learner")):
            logger.warning(
                "Daily retraining requested, but retrain_meta_learner is unavailable."
            )
        else:
            if not meta_lock.acquire(blocking=False):
                logger.warning("METALEARN_SKIPPED_LOCKED")
            else:
                try:
                    symbols = load_tickers(TICKERS_FILE)
                    logger.info(
                        f"RETRAINING START for {today_str} on {len(symbols)} tickers"
                    )
                    valid_symbols = []
                    for symbol in symbols:
                        try:
                            df_min = fetch_minute_df_safe(symbol)
                        except DataFetchError:
                            logger.info(
                                f"{symbol} returned no minute data; skipping symbol."
                            )
                            continue
                        if df_min is None or df_min.empty:
                            logger.info(
                                f"{symbol} returned no minute data; skipping symbol."
                            )
                            continue
                        valid_symbols.append(symbol)
                    if not valid_symbols:
                        logger.warning(
                            "No symbols returned valid minute data; skipping retraining entirely."
                        )
                    else:
                        force_train = (not MODEL_PATH) or (
                            not os.path.exists(MODEL_PATH)
                        )  # AI-AGENT-REF: guard for missing model path
                        if is_market_open():
                            from ai_trading.meta_learning import (
                                retrain_meta_learner,  # AI-AGENT-REF: lazy import
                            )

                            success = retrain_meta_learner(
                                trade_log_path=TRADE_LOG_FILE,
                                model_path=MODEL_PATH or "meta_model.pkl",
                            )
                        else:
                            logger.info(
                                "[retrain_meta_learner] Outside market hours; skipping"
                            )
                            success = False
                        if success:
                            try:
                                with open(marker, "w") as f:
                                    f.write(today_str)
                            except (
                                FileNotFoundError,
                                PermissionError,
                                IsADirectoryError,
                                JSONDecodeError,
                                ValueError,
                                KeyError,
                                TypeError,
                                OSError,
                            ) as e:  # AI-AGENT-REF: narrow exception
                                logger.warning(
                                    f"Failed to write retrain marker file: {e}"
                                )
                        else:
                            logger.warning(
                                "Retraining failed; continuing with existing model."
                            )
                finally:
                    meta_lock.release()

    df_train = ctx.data_fetcher.get_daily_df(ctx, REGIME_SYMBOLS[0])
    if df_train is not None and not df_train.empty:
        X_train = (
            df_train[["open", "high", "low", "close", "volume"]]
            .astype(float)
            .iloc[:-1]
            .values
        )
        y_train = (
            df_train["close"]
            .pct_change(fill_method=None)
            .shift(-1)
            .fillna(0)
            .values[:-1]
        )
        mp = _import_model_pipeline()
        with model_lock:
            try:
                if len(X_train) == 0:
                    logger.warning("DAILY_MODEL_TRAIN_SKIPPED_EMPTY")
                else:
                    mp.fit(X_train, y_train)
                    mse = float(np.mean((mp.predict(X_train) - y_train) ** 2))
                    logger.info("TRAIN_METRIC", extra={"mse": mse})
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as e:  # AI-AGENT-REF: narrow exception
                logger.error(f"Daily retrain failed: {e}")

        date_str = datetime.now(UTC).strftime("%Y%m%d_%H%M")
        os.makedirs("models", exist_ok=True)
        path = f"models/sgd_{date_str}.pkl"
        atomic_joblib_dump(mp, path)
        logger.info(f"Model checkpoint saved: {path}")

        for f in os.listdir("models"):
            if f.endswith(".pkl"):
                dt = datetime.strptime(f.split("_")[1].split(".")[0], "%Y%m%d").replace(
                    tzinfo=UTC
                )
                if datetime.now(UTC) - dt > timedelta(days=30):
                    os.remove(os.path.join("models", f))

        batch_mse = float(np.mean((mp.predict(X_train) - y_train) ** 2))
        _get_metrics_logger().log_metrics(  # AI-AGENT-REF: lazy metrics import
            {
                "timestamp": utc_now_iso(),
                "type": "daily_retrain",
                "batch_mse": batch_mse,
                "hyperparams": json.dumps(utils.to_serializable(CFG.sgd_params)),
                "seed": SEED,
                "model": "SGDRegressor",
                "git_hash": get_git_hash(),
            }
        )
        state.updates_halted = False
        state.rolling_losses.clear()

    return mp


def on_market_close() -> None:
    """Trigger daily retraining after the market closes."""
    now_est = dt_.now(UTC).astimezone(ZoneInfo("America/New_York"))
    if market_is_open(now_est):
        logger.info("RETRAIN_SKIP_MARKET_OPEN")
        return
    if now_est.time() < dt_time(16, 0):
        logger.info("RETRAIN_SKIP_EARLY", extra={"time": now_est.isoformat()})
        return
    try:
        load_or_retrain_daily(get_ctx())
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        logger.exception(f"on_market_close failed: {exc}")


# ─── M. MAIN LOOP & SCHEDULER ─────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/health", methods=["GET"])
@app.route("/health_check", methods=["GET"])
def health() -> str:
    """Health endpoint exposing basic system metrics."""
    try:
        runtime = (
            _get_runtime_context_or_none()
        )  # AI-AGENT-REF: runtime-aware health check
        if runtime is None:
            raise RuntimeError("runtime not ready")
        pre_trade_health_check(runtime, runtime.tickers or REGIME_SYMBOLS)
        status = "ok"
    except (
        APIError,
        TimeoutError,
        ConnectionError,
        KeyError,
        ValueError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: tighten health probe error handling
        status = f"degraded: {e}"
        logger.warning(
            "HEALTH_CHECK_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
    summary = {
        "status": status,
        "no_signal_events": state.no_signal_events,
        "indicator_failures": state.indicator_failures,
    }
    from flask import jsonify

    return jsonify(summary), 200


def start_healthcheck() -> None:
    port = CFG.healthcheck_port
    try:
        app.run(host="0.0.0.0", port=port)
    except OSError as e:
        logger.warning(f"Healthcheck port {port} in use: {e}. Skipping health-endpoint.")
    except (
        APIError,
        TimeoutError,
        ConnectionError,
        KeyError,
        ValueError,
        TypeError,
    ) as e:  # AI-AGENT-REF: tighten health probe error handling
        logger.warning(
            "HEALTH_CHECK_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )


def start_metrics_server(default_port: int = 9200) -> None:
    """Start Prometheus metrics server handling port conflicts."""
    if not PROMETHEUS_AVAILABLE:
        logger.debug("Prometheus not available; skipping metrics server start")
        return
    try:
        start_http_server(default_port)
        logger.debug("Metrics server started on %d", default_port)
        return
    except OSError as exc:
        if "Address already in use" in str(exc):
            try:
                resp = http.get(
                    f"http://localhost:{default_port}",
                    timeout=clamp_request_timeout(2),
                )
                if resp.ok:
                    logger.info("Metrics port %d already serving; reusing", default_port)
                    return
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as e:  # AI-AGENT-REF: narrow exception
                # Metrics server connectivity check failed - continue with port search
                logger.debug(
                    "Metrics server check failed on port %d: %s", default_port, e
                )
            # Avoid NameError (no 'utils' alias imported here)
            from ai_trading.utils import (
                get_free_port,
            )  # AI-AGENT-REF: local import avoids cycles

            port = get_free_port(default_port + 1, default_port + 50)
            if port is None:
                logger.warning("No free port available for metrics server")
                return
            logger.warning("Metrics port %d busy; using %d", default_port, port)
            try:
                start_http_server(port)
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as exc2:  # AI-AGENT-REF: narrow exception
                logger.warning("Failed to start metrics server on %d: %s", port, exc2)
        else:
            logger.warning("Failed to start metrics server on %d: %s", default_port, exc)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # pragma: no cover - unexpected error  # AI-AGENT-REF: narrow exception
        logger.warning("Failed to start metrics server on %d: %s", default_port, exc)


def run_multi_strategy(ctx) -> None:
    """Execute all modular strategies via allocator and risk engine."""
    signals_by_strategy: dict[str, list[TradeSignal]] = {}
    for strat in ctx.strategies:
        try:
            gen = getattr(strat, "generate", None)
            # AI-AGENT-REF: support generate() and generate_signals()
            if callable(gen):
                sigs = gen(ctx)
            else:
                gs = getattr(strat, "generate_signals", None)
                if callable(gs):
                    sigs = gs(getattr(ctx, "market_data", ctx))
                else:
                    logger.error(
                        "Strategy %s has neither `generate` nor `generate_signals`; skipping",
                        type(strat).__name__,
                    )
                    continue
            signals_by_strategy[strat.name] = sigs
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.warning(f"Strategy {strat.name} failed: {e}")
    # Optionally augment strategy signals with reinforcement learning signals.
    if RL_AGENT:
        try:
            # Determine the set of symbols that currently have signals from other strategies
            all_symbols: list[str] = []
            for sigs in signals_by_strategy.values():
                for sig in sigs:
                    sym = getattr(sig, "symbol", None)
                    if sym and sym not in all_symbols:
                        all_symbols.append(sym)
            if all_symbols:
                # Compute meaningful feature vectors for each symbol
                import numpy as _np  # AI-AGENT-REF: alias to avoid shadowing global np

                from ai_trading.rl_trading.features import compute_features

                states: list[_np.ndarray] = []
                for sym in all_symbols:
                    df = None
                    try:
                        df = ctx.data_fetcher.get_daily_df(ctx, sym)
                    except (
                        FileNotFoundError,
                        PermissionError,
                        IsADirectoryError,
                        JSONDecodeError,
                        ValueError,
                        KeyError,
                        TypeError,
                        OSError,
                    ):
                        df = None
                    if df is None or getattr(df, "empty", True):
                        try:
                            df = ctx.data_fetcher.get_minute_df(ctx, sym)
                        except (
                            FileNotFoundError,
                            PermissionError,
                            IsADirectoryError,
                            JSONDecodeError,
                            ValueError,
                            KeyError,
                            TypeError,
                            OSError,
                        ):
                            df = None
                    state_vec = compute_features(df, window=10)
                    states.append(state_vec)
                if states:
                    state_mat = _np.stack(states).astype(_np.float32)
                    rl_sigs = RL_AGENT.predict(state_mat, symbols=all_symbols)
                    if rl_sigs:
                        signals_by_strategy["rl"] = (
                            rl_sigs if isinstance(rl_sigs, list) else [rl_sigs]
                        )
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as exc:
            logger.error("RL_AGENT_ERROR", extra={"exc": str(exc)})

    # AI-AGENT-REF: Add position holding logic to reduce churn
    try:
        # Get current positions
        current_positions = ctx.api.list_positions()

        # Generate hold signals for existing positions
        # ai_trading/core/bot_engine.py:8588 - Convert import guard to hard import (internal module)
        from ai_trading.signals import (  # type: ignore
            enhance_signals_with_position_logic,
            generate_position_hold_signals,
        )

        hold_signals = generate_position_hold_signals(ctx, current_positions)

        # Apply position holding logic to all strategy signals
        enhanced_signals_by_strategy = {}
        for strategy_name, strategy_signals in signals_by_strategy.items():
            enhanced_signals = enhance_signals_with_position_logic(
                strategy_signals, ctx, hold_signals
            )
            enhanced_signals_by_strategy[strategy_name] = enhanced_signals

        # Log the effect of position holding
        original_count = sum(len(sigs) for sigs in signals_by_strategy.values())
        enhanced_count = sum(
            len(sigs) for sigs in enhanced_signals_by_strategy.values()
        )
        logger.info(
            "POSITION_HOLD_FILTER",
            extra={
                "original_signals": original_count,
                "enhanced_signals": enhanced_count,
                "filtered_out": original_count - enhanced_count,
                "hold_signals_count": len(hold_signals),
            },
        )

        # Use enhanced signals for allocation
        signals_by_strategy = enhanced_signals_by_strategy

    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        logger.warning("Position holding logic failed, using original signals: %s", exc)
    if getattr(ctx, "allocator", None) is None:  # AI-AGENT-REF: ensure allocator
        ctx.allocator = get_allocator()

    all_signals = [s for sigs in signals_by_strategy.values() for s in sigs]
    if not all_signals:
        logger.info("No signals produced this cycle; skipping allocation and execution")
        return

    final = ctx.allocator.allocate(signals_by_strategy)
    acct = ctx.api.get_account()
    cash = float(getattr(acct, "cash", 0))
    for sig in final:
        retries = 0
        price = 0.0
        data = None
        while price <= 0 and retries < 3:
            try:
                req = StockLatestQuoteRequest(symbol_or_symbols=[sig.symbol])
                quote: Quote = ctx.data_client.get_stock_latest_quote(req)
                price = float(getattr(quote, "ask_price", 0) or 0)
            except APIError as e:
                logger.warning("[run_all_trades] quote failed for %s: %s", sig.symbol, e)
                price = 0.0
            if price <= 0:
                time.sleep(2)
                try:
                    data = fetch_minute_df_safe(sig.symbol)
                except DataFetchError:
                    data = pd.DataFrame()
                if data is not None and not data.empty:
                    row = data.iloc[-1]
                    logger.debug(
                        "Fetched minute data for %s: %s",
                        sig.symbol,
                        row.to_dict(),
                    )
                    minute_close = float(row.get("close", 0))
                    logger.info(
                        "Using last_close=%.4f vs minute_close=%.4f",
                        utils.get_latest_close(data),
                        minute_close,
                    )
                    price = (
                        minute_close
                        if minute_close > 0
                        else utils.get_latest_close(data)
                    )
                if price <= 0:
                    logger.warning(
                        "Retry %s: price %.2f <= 0 for %s, refetching data",
                        retries + 1,
                        price,
                        sig.symbol,
                    )
                    retries += 1
            else:
                break
        if price <= 0:
            logger.critical(
                "Failed after retries: non-positive price for %s. Data context: %r",
                sig.symbol,
                (
                    data.tail(3).to_dict()
                    if hasattr(data, "tail") and hasattr(data, "to_dict")
                    else data
                ),
            )
            continue
        # Provide the account equity (cash) when sizing positions; this allows
        # CapitalScalingEngine.scale_position to use equity rather than raw size.
        if sig.side == "buy" and ctx.risk_engine.position_exists(ctx.api, sig.symbol):
            logger.info("SKIP_DUPLICATE_LONG", extra={"symbol": sig.symbol})
            continue

        # AI-AGENT-REF: Add validation and logging for signal processing
        logger.debug(
            "PROCESSING_SIGNAL",
            extra={
                "symbol": sig.symbol,
                "side": sig.side,
                "confidence": sig.confidence,
                "strategy": getattr(sig, "strategy", "unknown"),
                "weight": getattr(sig, "weight", 0.0),
            },
        )

        qty = ctx.risk_engine.position_size(sig, cash, price)
        if qty is None or not np.isfinite(qty) or qty <= 0:
            logger.warning(
                "SKIP_INVALID_QTY",
                extra={
                    "symbol": sig.symbol,
                    "side": sig.side,
                    "qty": qty,
                    "cash": cash,
                    "price": price,
                },
            )
            continue

        # AI-AGENT-REF: Validate signal side before execution to catch any corruption
        if sig.side not in ["buy", "sell"]:
            logger.error(
                "INVALID_SIGNAL_SIDE",
                extra={
                    "symbol": sig.symbol,
                    "side": sig.side,
                    "expected": "buy or sell",
                },
            )
            continue

        logger.info(
            "EXECUTING_ORDER",
            extra={"symbol": sig.symbol, "side": sig.side, "qty": qty, "price": price},
        )

        ctx.execution_engine.execute_order(
            sig.symbol, sig.side, qty, asset_class=sig.asset_class
        )
        ctx.risk_engine.register_fill(sig)

    # At the end of the strategy cycle, trigger trailing-stop checks if an ExecutionEngine is present.
    try:
        if hasattr(ctx, "execution_engine"):
            ctx.execution_engine.end_cycle()
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        logger.error("TRAILING_STOP_CHECK_FAILED", extra={"exc": str(exc)})


def _param(runtime, key, default):
    """Pull from runtime.params first, then cfg, else default."""
    if runtime and getattr(runtime, "params", None) and key in runtime.params:
        return runtime.params[key]
    if runtime and hasattr(runtime, "cfg") and runtime.cfg:
        return float(getattr(runtime.cfg, key.lower(), default))
    return default


def ensure_data_fetcher(runtime) -> DataFetcher:
    """Ensure a market data fetcher is attached to ``runtime``.

    Attempts to lazily rebuild the fetcher if it is missing. Raises
    ``DataFetchError`` if a fetcher cannot be constructed so callers fail
    fast during startup.
    """

    global data_fetcher
    fetcher = getattr(runtime, "data_fetcher", None)
    if fetcher is not None:
        return fetcher
    if data_fetcher is not None:
        logger_once.warning(
            "DATA_FETCHER_ATTACHED_LATE", key="data_fetcher_attached_late"
        )
        runtime.data_fetcher = data_fetcher
        return data_fetcher
    logger_once.warning("DATA_FETCHER_MISSING", key="data_fetcher_missing")
    try:
        fetcher = data_fetcher_module.build_fetcher(getattr(runtime, "params", {}))
    except COMMON_EXC as exc:  # pragma: no cover - best effort during startup
        logger_once.error(
            "DATA_FETCHER_INIT_FAILED", extra={"detail": str(exc)},
            key="data_fetcher_init_failed",
        )
        raise DataFetchError("data_fetcher not available") from exc
    runtime.data_fetcher = fetcher
    data_fetcher = fetcher
    return fetcher


def _prepare_run(
    runtime, state: BotState, tickers: list[str] | None
) -> tuple[float, bool, list[str]]:
    from ai_trading import portfolio
    from ai_trading.utils import portfolio_lock

    """Prepare trading run by syncing positions and generating symbols."""
    try:
        ensure_data_fetcher(runtime)
        cancel_all_open_orders(runtime)
        audit_positions(runtime)
    except APIError as e:
        logger.warning(
            "PREPARE_RUN_API_ERROR",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        raise DataFetchError("api error during pre-run") from e
    try:
        acct = safe_alpaca_get_account(runtime)
        equity = float(acct.equity) if acct else 0.0
    except (
        APIError,
        TimeoutError,
        ConnectionError,
    ) as e:  # AI-AGENT-REF: narrow account fetch errors
        logger.warning(
            "ACCOUNT_INFO_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        equity = 0.0
    update_if_present(runtime, equity)
    params["get_capital_cap()"] = _param(runtime, "get_capital_cap()", 0.04)
    compute_spy_vol_stats(runtime)

    full_watchlist = load_candidate_universe(runtime, tickers)
    try:
        pretrade_data_health(runtime, full_watchlist)
    except DataFetchError:
        time.sleep(1.0)
        return 0.0, False, []
    symbols = screen_candidates(runtime, full_watchlist)
    logger.info(
        "Number of screened candidates: %s", len(symbols)
    )  # AI-AGENT-REF: log candidate count
    if not symbols:
        logger.warning(
            "No candidates found after filtering, using top 5 tickers fallback."
        )
        if not full_watchlist:
            full_watchlist = load_universe() or FALLBACK_SYMBOLS
        symbols = full_watchlist[:5]
    logger.info("CANDIDATES_SCREENED", extra={"tickers": symbols})
    runtime.tickers = symbols  # AI-AGENT-REF: store screened tickers on runtime
    try:
        summary = pre_trade_health_check(runtime, symbols)
        logger.info("PRE_TRADE_HEALTH", extra=summary)
    except (
        APIError,
        TimeoutError,
        ConnectionError,
        KeyError,
        ValueError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: explicit error logging for data health
        logger.warning(
            "HEALTH_CHECK_FAILED",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        return 0.0, False, []
    with portfolio_lock:
        runtime.portfolio_weights = portfolio.compute_portfolio_weights(
            runtime, symbols
        )
    acct = safe_alpaca_get_account(runtime)
    if acct:
        current_cash = float(getattr(acct, "buying_power", acct.cash))
    else:
        logger.error("Failed to get account information from Alpaca")
        return 0.0, False, []
    regime_ok = check_market_regime(
        runtime, state
    )  # AI-AGENT-REF: runtime flows into regime check
    return current_cash, regime_ok, symbols


def _process_symbols(
    symbols: list[str],
    current_cash: float,
    model,
    regime_ok: bool,
    close_shorts: bool = False,
    skip_duplicates: bool = False,
) -> tuple[list[str], dict[str, int]]:
    processed: list[str] = []
    row_counts: dict[str, int] = {}

    # AI-AGENT-REF: bind lazy context for trade helpers
    ctx = get_ctx()

    if not hasattr(state, "trade_cooldowns"):
        state.trade_cooldowns = {}
    if not hasattr(state, "last_trade_direction"):
        state.last_trade_direction = {}

    now = datetime.now(UTC)

    filtered: list[str] = []
    cd_skipped: list[str] = []

    # AI-AGENT-REF: Add circuit breaker for symbol processing to prevent resource exhaustion
    max_symbols_per_cycle = min(50, len(symbols))  # Limit to 50 symbols per cycle
    processed_symbols = 0
    processing_start_time = time.monotonic()

    for symbol in symbols:
        # AI-AGENT-REF: Final-bar/session gating before strategy evaluation
        if not ensure_final_bar(symbol, "1min"):  # Default to 1min timeframe
            logger.info("SKIP_PARTIAL_BAR", extra={"symbol": symbol, "timeframe": "1min"})
            continue

        # Circuit breaker: limit processing time and symbol count
        if processed_symbols >= max_symbols_per_cycle:
            logger.warning(
                "SYMBOL_PROCESSING_CIRCUIT_BREAKER",
                extra={
                    "processed_count": processed_symbols,
                    "remaining_count": len(symbols) - processed_symbols,
                    "reason": "max_symbols_reached",
                },
            )
            break

        # Check processing time limit (max 5 minutes per cycle)
        if time.monotonic() - processing_start_time > 300:
            logger.warning(
                "SYMBOL_PROCESSING_CIRCUIT_BREAKER",
                extra={
                    "processed_count": processed_symbols,
                    "elapsed_seconds": time.monotonic() - processing_start_time,
                    "reason": "time_limit_reached",
                },
            )
            break

        processed_symbols += 1

        pos = state.position_cache.get(symbol, 0)
        if pos < 0 and close_shorts:
            logger.info(
                "SKIP_SHORT_CLOSE_QUEUED | symbol=%s qty=%s",
                symbol,
                -pos,
            )
            # AI-AGENT-REF: avoid submitting orders when short-close is skipped
            continue
        if skip_duplicates and pos != 0:
            log_skip_cooldown(symbol, reason="duplicate")
            skipped_duplicates.inc()
            continue
        if pos > 0:
            logger.info("SKIP_HELD_POSITION | already long, skipping close")
            skipped_duplicates.inc()
            continue
        if pos < 0:
            logger.info(
                "SHORT_CLOSE_QUEUED | symbol=%s  qty=%d",
                symbol,
                abs(pos),
            )
            try:
                submit_order(ctx, symbol, abs(pos), "buy")
            except (
                APIError,
                TimeoutError,
                ConnectionError,
            ) as e:  # AI-AGENT-REF: tighten order close errors
                logger.warning(
                    "SHORT_CLOSE_FAIL",
                    extra={
                        "symbol": symbol,
                        "cause": e.__class__.__name__,
                        "detail": str(e),
                    },
                )
            continue
        # AI-AGENT-REF: Add thread-safe locking for trade cooldown access
        with trade_cooldowns_lock:
            ts = state.trade_cooldowns.get(symbol)
        if ts and (now - ts).total_seconds() < 60:
            cd_skipped.append(symbol)
            skipped_cooldown.inc()
            continue
        filtered.append(symbol)

    symbols = filtered  # replace with filtered list

    _ensure_executors()  # AI-AGENT-REF: lazy executor creation

    if cd_skipped:
        log_skip_cooldown(cd_skipped)

    def process_symbol(symbol: str) -> None:
        try:
            logger.info(f"PROCESSING_SYMBOL | symbol={symbol}")
            if not is_market_open():
                logger.info("MARKET_CLOSED_SKIP_SYMBOL", extra={"symbol": symbol})
                return
            def _halt(reason: str) -> None:
                logger.error("DATA_FETCH_EMPTY", extra={"symbol": symbol, "reason": reason})
                halt_mgr = getattr(ctx, "halt_manager", None)
                if halt_mgr is not None:
                    try:
                        halt_mgr.manual_halt_trading(f"{symbol}:{reason}")
                    except Exception as hm_exc:  # noqa: BLE001
                        logger.error("HALT_MANAGER_ERROR", extra={"cause": str(hm_exc)})
            try:
                price_df = fetch_minute_df_safe(symbol)
            except DataFetchError:
                _halt("minute_data_unavailable")
                return
            # AI-AGENT-REF: record raw row count before validation
            row_counts[symbol] = len(price_df)
            logger.info(f"FETCHED_ROWS | {symbol} rows={len(price_df)}")
            if price_df.empty or "close" not in price_df.columns:
                _halt("empty_frame")
                return
            if symbol in state.position_cache:
                return  # AI-AGENT-REF: skip symbol with open position
            processed.append(symbol)
            _safe_trade(ctx, state, symbol, current_cash, model, regime_ok)
        except (
            KeyError,
            ValueError,
            TypeError,
        ) as e:  # AI-AGENT-REF: tighten symbol processing errors
            logger.error(
                "PROCESS_SYMBOL_FAILED",
                extra={
                    "symbol": symbol,
                    "cause": e.__class__.__name__,
                    "detail": str(e),
                },
                exc_info=True,
            )

    futures = [prediction_executor.submit(process_symbol, s) for s in symbols]
    for f in futures:
        f.result()
    return processed, row_counts


def _log_loop_heartbeat(loop_id: str, start: float) -> None:
    duration = time.monotonic() - start
    logger.info(
        "HEARTBEAT",
        extra={
            "loop_id": loop_id,
            "timestamp": utc_now_iso(),  # AI-AGENT-REF: Use UTC timestamp utility
            "duration": duration,
        },
    )


def _send_heartbeat() -> None:
    """Lightweight heartbeat when halted."""
    logger.info(
        "HEARTBEAT_HALTED",
        extra={"timestamp": utc_now_iso()},  # AI-AGENT-REF: Use UTC timestamp utility
    )


def manage_position_risk(ctx, position) -> None:
    """Adjust trailing stops and position size while halted."""
    symbol = position.symbol
    try:
        atr = utils.get_rolling_atr(symbol)
        vwap = utils.get_current_vwap(symbol)
        try:
            price_df = fetch_minute_df_safe(symbol)
        except DataFetchError:
            logger.critical(f"No minute data for {symbol}, skipping.")
            return
        logger.debug(f"Latest rows for {symbol}:\n{price_df.tail(3)}")
        if "close" in price_df.columns:
            price_series = price_df["close"].dropna()
            if not price_series.empty:
                price = price_series.iloc[-1]
                logger.debug(f"Final extracted price for {symbol}: {price}")
            else:
                logger.critical(f"No valid close prices found for {symbol}, skipping.")
                price = 0.0
        else:
            logger.critical(f"Close column missing for {symbol}, skipping.")
            price = 0.0
        if price <= 0 or pd.isna(price):
            logger.critical(f"Invalid price computed for {symbol}: {price}")
            return
        side = "long" if int(position.qty) > 0 else "short"
        if side == "long":
            new_stop = float(position.avg_entry_price) * (
                1 - min(0.01 + atr / 100, 0.03)
            )
        else:
            new_stop = float(position.avg_entry_price) * (
                1 + min(0.01 + atr / 100, 0.03)
            )
        update_trailing_stop(ctx, symbol, price, int(position.qty), atr)
        pnl = float(getattr(position, "unrealized_plpc", 0))
        kelly_scale = compute_kelly_scale(atr, 0.0)
        adjust_position_size(position, kelly_scale)
        volume_factor = utils.get_volume_spike_factor(symbol)
        ml_conf = utils.get_ml_confidence(symbol)
        if (
            (
                volume_factor > CFG.volume_spike_threshold
                and ml_conf > CFG.ml_confidence_threshold
            )
            and side == "long"
            and price > vwap
            and pnl > 0.02
        ):
            pyramid_add_position(ctx, symbol, CFG.pyramid_levels["low"], side)
        logger.info(
            f"HALT_MANAGE {symbol} stop={new_stop:.2f} vwap={vwap:.2f} vol={volume_factor:.2f} ml={ml_conf:.2f}"
        )
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # pragma: no cover - handle edge cases  # AI-AGENT-REF: narrow exception
        logger.warning(f"manage_position_risk failed for {symbol}: {exc}")


def pyramid_add_position(
    ctx: BotContext, symbol: str, fraction: float, side: str
) -> None:
    current_qty = _current_qty(ctx, symbol)
    add_qty = max(1, int(abs(current_qty) * fraction))
    submit_order(ctx, symbol, add_qty, "buy" if side == "long" else "sell")
    logger.info("PYRAMID_ADD", extra={"symbol": symbol, "qty": add_qty, "side": side})


def reduce_position_size(ctx: BotContext, symbol: str, fraction: float) -> None:
    current_qty = _current_qty(ctx, symbol)
    reduce_qty = max(1, int(abs(current_qty) * fraction))
    side = "sell" if current_qty > 0 else "buy"
    submit_order(ctx, symbol, reduce_qty, side)
    logger.info("REDUCE_POSITION", extra={"symbol": symbol, "qty": reduce_qty})


def _ensure_execution_engine(runtime) -> None:
    """Ensure an execution engine with check_stops is attached to runtime."""
    from ai_trading.execution.engine import ExecutionEngine  # deferred import

    global _exec_engine
    exec_engine = getattr(runtime, "execution_engine", None) or getattr(
        runtime, "exec_engine", None
    )
    if exec_engine is None:
        try:
            exec_engine = ExecutionEngine(runtime)
            runtime.execution_engine = exec_engine
            runtime.exec_engine = exec_engine
            _exec_engine = exec_engine
            logger.debug(
                "Execution engine initialized and attached to runtime"
            )
        except Exception as e:  # pragma: no cover - initialization rarely fails
            logger.warning(
                "Execution engine initialization failed: %s", e
            )
            return
    else:
        runtime.execution_engine = exec_engine
        runtime.exec_engine = exec_engine
        _exec_engine = exec_engine
    if not hasattr(exec_engine, "check_stops"):
        logger.warning(
            "Execution engine lacks check_stops(); risk-stop checks disabled"
        )


def _check_runtime_stops(runtime) -> None:
    """Run post-cycle stop checks if supported."""
    exec_engine = getattr(runtime, "exec_engine", None) or getattr(
        runtime, "execution_engine", None
    )
    if exec_engine is not None and hasattr(exec_engine, "check_stops"):
        try:
            exec_engine.check_stops()
        except (ValueError, TypeError) as e:  # AI-AGENT-REF: guard check_stops
            logger.info("check_stops raised but was suppressed: %s", e)
    else:
        logger.warning(
            "Execution engine missing check_stops; risk-stop checks skipped"
        )


@memory_profile  # AI-AGENT-REF: Monitor memory usage of main trading function
def run_all_trades_worker(state: BotState, runtime) -> None:
    """
    Execute the complete trading cycle for all candidate symbols.

    This is the core trading function that orchestrates the entire algorithmic
    trading process. It fetches market data, calculates technical indicators,
    generates trading signals, applies risk management, and executes trades
    based on the configured strategy parameters.

    The function implements comprehensive safety checks including market hours
    validation, risk limit enforcement, and overlap prevention to ensure safe
    and reliable trading operations.

    Parameters
    ----------
    state : BotState
        Current bot state containing position information, risk metrics,
        trading history, and operational flags. This object is modified
        during execution to track state changes.

    Returns
    -------
    None
        This function modifies the state object in-place and executes trades
        as side effects. Trading results are logged and stored in the state
        for subsequent analysis.

    Raises
    ------
    RuntimeError
        If critical trading conditions are not met or system is unhealthy
    ConnectionError
        If API connections fail during critical operations
    ValueError
        If invalid trading parameters or data quality issues are detected

    Trading Process Flow
    -------------------
    1. **Pre-flight Checks**
       - Verify market is open for trading
       - Check for overlapping execution (prevents race conditions)
       - Validate system health and API connectivity
       - Ensure risk limits are within acceptable ranges

    2. **Data Acquisition**
       - Fetch real-time and historical market data
       - Validate data quality and completeness
       - Handle data provider failover if needed
       - Cache data for performance optimization

    3. **Technical Analysis**
       - Calculate technical indicators across multiple timeframes
       - Generate trading signals using configured strategies
       - Apply machine learning predictions and meta-learning
       - Aggregate signals with confidence scoring

    4. **Risk Management**
       - Calculate optimal position sizes using Kelly criterion
       - Check portfolio heat and individual position limits
       - Apply drawdown protection and loss streak controls
       - Validate trades against PDT and margin rules

    5. **Trade Execution**
       - Generate order instructions for qualified signals
       - Execute trades through broker API with retry logic
       - Monitor fill status and handle partial fills
       - Update position tracking and performance metrics

    6. **State Management**
       - Update bot state with new positions and metrics
       - Record trade history and performance statistics
       - Set cooldown periods to prevent overtrading
       - Log results for monitoring and analysis

    Safety Features
    ---------------
    - **Overlap Prevention**: Uses locking to prevent concurrent execution
    - **Market Hours**: Only trades during official market hours
    - **Risk Limits**: Enforces position size and portfolio heat limits
    - **Health Checks**: Validates system health before trading
    - **Error Handling**: Graceful handling of API and data failures
    - **Cooldown Periods**: Prevents rapid-fire trading on same symbols
    - **Emergency Stops**: Automatic halt on critical errors or high losses

    Examples
    --------
    >>> import asyncio
    >>> from bot_engine import BotState, run_all_trades_worker
    >>>
    >>> state = BotState()
    >>> run_all_trades_worker(state, runtime)
    >>>
    >>> # Check results
    >>> logging.info(f"Trades executed: {len(state.position_cache)}")
    >>> logging.info(f"Last loop duration: {state.last_loop_duration:.2f}s")

    Performance Considerations
    -------------------------
    - Uses parallel processing for indicator calculations
    - Implements data caching to reduce API calls
    - Optimizes database queries for position tracking
    - Monitors memory usage and performs cleanup
    - Tracks execution time for performance optimization

    Notes
    -----
    - This function should only be called when markets are open
    - Execution is thread-safe and prevents overlapping runs
    - All trades are logged for audit and compliance purposes
    - Performance metrics are automatically collected and stored
    - The function will gracefully handle API rate limits and failures

    See Also
    --------
    BotState : Central state management
    pre_trade_health_check : System health validation
    BotContext : Global context and configuration
    trade_execution : Order execution and monitoring
    """
    _ensure_alpaca_classes()
    if _ALPACA_IMPORT_ERROR is not None:
        raise RuntimeError("Alpaca SDK is required") from _ALPACA_IMPORT_ERROR
    risk_engine = getattr(runtime, "risk_engine", None)
    if risk_engine is None:
        logger.error("RISK_ENGINE_MISSING")
        return
    _init_metrics()
    import uuid

    loop_id = str(uuid.uuid4())
    acquired = run_lock.acquire(blocking=False)
    if not acquired:
        logger.info("RUN_ALL_TRADES_SKIPPED_OVERLAP")
        return
    try:  # AI-AGENT-REF: ensure lock released on every exit
        try:
            risk_engine.wait_for_exposure_update(0.5)
        except (
            TimeoutError,
            ConnectionError,
            RuntimeError,
        ) as e:  # AI-AGENT-REF: tighten risk update errors
            logger.warning(
                "RISK_EXPOSURE_UPDATE_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
        _ensure_execution_engine(runtime)
        if not hasattr(state, "trade_cooldowns"):
            state.trade_cooldowns = {}
        if not hasattr(state, "last_trade_direction"):
            state.last_trade_direction = {}
        if state.running:
            logger.warning(
                "RUN_ALL_TRADES_SKIPPED_OVERLAP",
                extra={"last_duration": getattr(state, "last_loop_duration", 0.0)},
            )
            return
        now = datetime.now(UTC)
        for sym, ts in list(state.trade_cooldowns.items()):
            if (now - ts).total_seconds() > get_trade_cooldown_min() * 60:
                state.trade_cooldowns.pop(sym, None)
        if (
            state.last_run_at
            and (now - state.last_run_at).total_seconds() < RUN_INTERVAL_SECONDS
        ):
            logger.warning("RUN_ALL_TRADES_SKIPPED_RECENT")
            return
        if not is_market_open():
            logger.info("MARKET_CLOSED_NO_FETCH")
            return  # FIXED: skip work when market closed
        ensure_alpaca_attached(runtime)
        api = getattr(runtime, "api", None)
        if not _validate_trading_api(api):
            _send_heartbeat()
            return
        state.pdt_blocked = check_pdt_rule(runtime)
        if state.pdt_blocked:
            return
        state.running = True
        state.last_run_at = now
        loop_start = time.monotonic()
        if not getattr(state, "_strategies_loaded", False):
            runtime.strategies = get_strategies()
            state._strategies_loaded = True
        try:
            # AI-AGENT-REF: avoid overlapping cycles if any orders are pending
            try:
                open_orders = list_open_orders(api)
            except (
                APIError,
                TimeoutError,
                ConnectionError,
                AttributeError,
            ) as e:  # AI-AGENT-REF: tighten order check errors
                logger_once.warning(
                    "api.list_orders failed during order check",
                    key="alpaca_list_orders_failed",
                    extra={"cause": e.__class__.__name__, "detail": str(e)},
                )
                open_orders = []
            if any(o.status in ("new", "pending_new") for o in open_orders):
                logger.warning("Detected pending orders; skipping this trade cycle")
                return
            if get_verbose_logging():
                logger.info(
                    "RUN_ALL_TRADES_START",
                    extra={"timestamp": utc_now_iso()},
                )

            # Log standardized market fetch heartbeat (configurable)
            if CFG.log_market_fetch:
                logger.info("MARKET_FETCH")
            else:
                logger.debug("MARKET_FETCH")
            ctx = _get_runtime_context_or_none()
            if ctx and getattr(runtime, "data_fetcher", None) is None:
                runtime.data_fetcher = getattr(ctx, "data_fetcher", None)
            ensure_data_fetcher(runtime)

            for attempt in range(3):
                try:
                    current_cash, regime_ok, symbols = _prepare_run(
                        runtime, state, getattr(runtime, "tickers", None)
                    )
                    break
                except DataFetchError as e:
                    logger.warning(
                        "DATA_FETCHER_UNAVAILABLE",
                        extra={"detail": str(e), "attempt": attempt + 1},
                    )
                    if attempt == 2:
                        return
                    time.sleep(1.0)
                except APIError as e:
                    logger.warning(
                        "PREPARE_RUN_API_ERROR",
                        extra={"detail": str(e), "attempt": attempt + 1},
                    )
                    if attempt == 2:
                        return
                    time.sleep(1.0)

            # AI-AGENT-REF: Add memory monitoring and cleanup to prevent resource issues
            if MEMORY_OPTIMIZATION_AVAILABLE:
                try:
                    memory_stats = optimize_memory()
                    if (
                        memory_stats.get("memory_usage_mb", 0) > 512
                    ):  # If using more than 512MB
                        logger.warning(
                            "HIGH_MEMORY_USAGE_DETECTED",
                            extra={
                                "memory_usage_mb": memory_stats.get(
                                    "memory_usage_mb", 0
                                ),
                                "symbols_count": len(symbols),
                            },
                        )
                        # Emergency cleanup if memory is too high
                        if (
                            memory_stats.get("memory_usage_mb", 0) > 1024
                        ):  # 1GB threshold
                            logger.critical("EMERGENCY_MEMORY_CLEANUP_TRIGGERED")
                            emergency_memory_cleanup()
                except (
                    RuntimeError,
                    ValueError,
                    TypeError,
                ) as e:  # AI-AGENT-REF: tighten memory optimization errors
                    logger.debug(
                        "MEMORY_OPTIMIZATION_FAILED",
                        extra={"cause": e.__class__.__name__, "detail": str(e)},
                    )

            # AI-AGENT-REF: Update drawdown circuit breaker with current equity
            if runtime.drawdown_circuit_breaker:
                try:
                    acct = runtime.api.get_account()
                    current_equity = float(acct.equity) if acct else 0.0
                    trading_allowed = runtime.drawdown_circuit_breaker.update_equity(
                        current_equity
                    )

                    # AI-AGENT-REF: Get status once to avoid UnboundLocalError in else block
                    status = runtime.drawdown_circuit_breaker.get_status()

                    if not trading_allowed:
                        logger.critical(
                            "TRADING_HALTED_DRAWDOWN_PROTECTION",
                            extra={
                                "current_drawdown": status["current_drawdown"],
                                "max_drawdown": status["max_drawdown"],
                                "peak_equity": status["peak_equity"],
                                "current_equity": current_equity,
                            },
                        )
                        # Manage existing positions but skip new trades
                        try:
                            portfolio = runtime.api.list_positions()
                            for pos in portfolio:
                                manage_position_risk(runtime, pos)
                        except (
                            APIError,
                            TimeoutError,
                            ConnectionError,
                        ) as e:  # AI-AGENT-REF: tighten halt manage errors
                            logger.warning(
                                "HALT_MANAGE_FAIL",
                                extra={"cause": e.__class__.__name__, "detail": str(e)},
                            )
                        return
                    else:
                        # Log drawdown status for monitoring
                        logger.debug(
                            "DRAWDOWN_STATUS_OK",
                            extra={
                                "current_drawdown": status["current_drawdown"],
                                "max_drawdown": status["max_drawdown"],
                                "trading_allowed": status["trading_allowed"],
                            },
                        )
                except (
                    APIError,
                    TimeoutError,
                    ConnectionError,
                ) as e:  # AI-AGENT-REF: tighten circuit breaker update errors
                    logger.error(
                        "DRAWDOWN_CHECK_FAILED",
                        extra={"cause": e.__class__.__name__, "detail": str(e)},
                    )
                    # Continue trading but log the error for investigation

            # AI-AGENT-REF: honor global halt flag before processing symbols
            if check_halt_flag(runtime):
                _log_health_diagnostics(runtime, "halt_flag_loop")
                logger.info("TRADING_HALTED_VIA_FLAG: Managing existing positions only.")
                try:
                    portfolio = runtime.api.list_positions()
                    for pos in portfolio:
                        manage_position_risk(runtime, pos)
                except (
                    APIError,
                    TimeoutError,
                    ConnectionError,
                ) as e:  # AI-AGENT-REF: tighten halt manage errors
                    logger.warning(
                        "HALT_MANAGE_FAIL",
                        extra={"cause": e.__class__.__name__, "detail": str(e)},
                    )
                logger.info("HALT_SKIP_NEW_TRADES")
                _send_heartbeat()
                # log summary even when halted
                try:
                    acct = runtime.api.get_account()
                    cash = float(acct.cash)
                    equity = float(acct.equity)
                    positions = runtime.api.list_positions()
                    logger.debug("Raw Alpaca positions: %s", positions)
                    exposure = (
                        sum(abs(float(p.market_value)) for p in positions)
                        / equity
                        * 100
                        if equity > 0
                        else 0.0
                    )
                    logger.info(
                        f"Portfolio summary: cash=${cash:.2f}, equity=${equity:.2f}, exposure={exposure:.2f}%, positions={len(positions)}"
                    )
                    logger.info(
                        "POSITIONS_DETAIL",
                        extra={
                            "positions": [
                                {
                                    "symbol": p.symbol,
                                    "qty": int(p.qty),
                                    "avg_price": float(p.avg_entry_price),
                                    "market_value": float(p.market_value),
                                }
                                for p in positions
                            ],
                        },
                    )
                    logger.info(
                        "WEIGHTS_VS_POSITIONS",
                        extra={
                            "weights": runtime.portfolio_weights,
                            "positions": {p.symbol: int(p.qty) for p in positions},
                            "cash": cash,
                        },
                    )
                except (
                    APIError,
                    TimeoutError,
                    ConnectionError,
                ) as e:  # AI-AGENT-REF: tighten summary fetch errors
                    logger.warning(
                        "SUMMARY_FAIL",
                        extra={"cause": e.__class__.__name__, "detail": str(e)},
                    )
                return

            alpha_model = getattr(runtime, "model", None)
            if not alpha_model:
                logger.warning(
                    "ALPHA_MODEL_UNAVAILABLE - skipping compute stage for this cycle"
                )
                return

            if not symbols:
                logger.info("SKIP_MINUTE_FETCH", extra={"reason": "no_symbols"})
                return

            retries = 3
            processed, row_counts = [], {}
            for attempt in range(retries):
                with StageTimer(logger, "INDICATORS_COMPUTE", symbols=len(symbols)):
                    processed, row_counts = _process_symbols(
                        symbols, current_cash, alpha_model, regime_ok
                    )
                if processed:
                    if attempt:
                        logger.info(
                            "DATA_SOURCE_RETRY_SUCCESS",
                            extra={"attempt": attempt + 1, "symbols": symbols},
                        )
                    break
                time.sleep(2)

            # AI-AGENT-REF: abort only if all symbols returned zero rows
            if sum(row_counts.values()) == 0:
                last_ts = None
                for sym in symbols:
                    ts = runtime.data_fetcher._minute_timestamps.get(sym)
                    if last_ts is None or (ts and ts > last_ts):
                        last_ts = ts
                logger.critical(
                    "DATA_SOURCE_EMPTY",
                    extra={
                        "symbols": symbols,
                        "endpoint": "minute",
                        "last_success": last_ts.isoformat() if last_ts else "unknown",
                        "row_counts": row_counts,
                    },
                )
                logger.info(
                    "DATA_SOURCE_RETRY_FAILED",
                    extra={"attempts": retries, "symbols": symbols},
                )
                # AI-AGENT-REF: exit immediately on repeated data failure
                return
            else:
                logger.info(
                    "DATA_SOURCE_RETRY_FINAL",
                    extra={"success": True, "attempts": attempt + 1},
                )

            skipped = [s for s in symbols if s not in processed]
            if skipped:
                logger.info(
                    "CYCLE_SKIPPED_SUMMARY",
                    extra={"count": len(skipped), "symbols": skipped},
                )
                if len(skipped) == len(symbols):
                    state.skipped_cycles += 1
                else:
                    state.skipped_cycles = 0
            else:
                state.skipped_cycles = 0
            if state.skipped_cycles >= 2:
                logger.critical(
                    "ALL_SYMBOLS_SKIPPED_TWO_CYCLES",
                    extra={
                        "hint": "Check data provider API keys and entitlements; test data fetch manually from the server; review data fetcher logs",
                    },
                )

            run_multi_strategy(runtime)
            try:
                risk_engine.refresh_positions(runtime.api)
                pos_list = runtime.api.list_positions()
                state.position_cache = {p.symbol: int(p.qty) for p in pos_list}
                state.long_positions = {
                    s for s, q in state.position_cache.items() if q > 0
                }
                state.short_positions = {
                    s for s, q in state.position_cache.items() if q < 0
                }
                if runtime.execution_engine:
                    runtime.execution_engine.check_trailing_stops()
            except (
                APIError,
                TimeoutError,
                ConnectionError,
            ) as e:  # AI-AGENT-REF: tighten refresh errors
                logger.warning(
                    "REFRESH_POSITIONS_FAILED",
                    extra={"cause": e.__class__.__name__, "detail": str(e)},
                )
            logger.info(
                f"RUN_ALL_TRADES_COMPLETE | processed={len(row_counts)} symbols, total_rows={sum(row_counts.values())}"
            )
            try:
                acct = runtime.api.get_account()
                cash = float(acct.cash)
                equity = float(acct.equity)
                positions = runtime.api.list_positions()
                logger.debug("Raw Alpaca positions: %s", positions)
                # ai_trading.csv:9422 - Replace import guard with hard import (required dependencies)
                from ai_trading import portfolio
                from ai_trading.utils import portfolio_lock

                try:
                    with portfolio_lock:
                        runtime.portfolio_weights = portfolio.compute_portfolio_weights(
                            runtime, [p.symbol for p in positions]
                        )
                except (
                    ZeroDivisionError,
                    ValueError,
                    KeyError,
                ) as e:  # AI-AGENT-REF: tighten portfolio sizing errors
                    logger.warning(
                        "WEIGHT_RECOMPUTE_FAILED",
                        extra={"cause": e.__class__.__name__, "detail": str(e)},
                        exc_info=True,
                    )
                exposure = (
                    sum(abs(float(p.market_value)) for p in positions) / equity * 100
                    if equity > 0
                    else 0.0
                )
                logger.info(
                    f"Portfolio summary: cash=${cash:.2f}, equity=${equity:.2f}, exposure={exposure:.2f}%, positions={len(positions)}"
                )
                logger.info(
                    "POSITIONS_DETAIL",
                    extra={
                        "positions": [
                            {
                                "symbol": p.symbol,
                                "qty": int(p.qty),
                                "avg_price": float(p.avg_entry_price),
                                "market_value": float(p.market_value),
                            }
                            for p in positions
                        ],
                    },
                )
                logger.info(
                    "WEIGHTS_VS_POSITIONS",
                    extra={
                        "weights": runtime.portfolio_weights,
                        "positions": {p.symbol: int(p.qty) for p in positions},
                        "cash": cash,
                    },
                )
                try:
                    adaptive_cap = risk_engine._adaptive_global_cap()
                except (
                    ZeroDivisionError,
                    ValueError,
                    KeyError,
                ) as e:  # AI-AGENT-REF: tighten adaptive cap errors
                    logger.warning(
                        "ADAPTIVE_CAP_FAILED",
                        extra={"cause": e.__class__.__name__, "detail": str(e)},
                    )
                    adaptive_cap = 0.0
                logger.info(
                    "CYCLE SUMMARY: cash=$%.0f equity=$%.0f exposure=%.0f%% positions=%d adaptive_cap=%.1f",
                    cash,
                    equity,
                    exposure,
                    len(positions),
                    adaptive_cap,
                )
            except (
                APIError,
                TimeoutError,
                ConnectionError,
            ) as e:  # AI-AGENT-REF: tighten summary fetch errors
                logger.warning(
                    "SUMMARY_FAIL",
                    extra={"cause": e.__class__.__name__, "detail": str(e)},
                )
            try:
                acct = runtime.api.get_account()
                # Handle case where account object might not have last_equity attribute
                last_equity = getattr(acct, "last_equity", acct.equity)
                pnl = float(acct.equity) - float(last_equity)
                logger.info(
                    "LOOP_PNL",
                    extra={
                        "loop_id": loop_id,
                        "pnl": pnl,
                        "mode": "SHADOW" if CFG.shadow_mode else "LIVE",
                    },
                )
            except (
                APIError,
                TimeoutError,
                ConnectionError,
                ValueError,
            ) as e:  # AI-AGENT-REF: tighten PnL retrieval errors
                logger.warning(
                    "PNL_RETRIEVAL_FAILED",
                    extra={"cause": e.__class__.__name__, "detail": str(e)},
                )
        except APIError as e:  # AI-AGENT-REF: skip cycle on Alpaca API errors
            logger.warning(
                "TRADING_CYCLE_API_ERROR",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
            return
        except (
            TimeoutError,
            ConnectionError,
            ValueError,
            KeyError,
            TypeError,
        ) as e:  # AI-AGENT-REF: tighten trading cycle boundary
            logger.error(
                "TRADING_CYCLE_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
                exc_info=True,
            )
            raise
        finally:
            # Always reset running flag
            state.running = False
            state._strategies_loaded = False
            state.last_loop_duration = time.monotonic() - loop_start
            _log_loop_heartbeat(loop_id, loop_start)

            _check_runtime_stops(runtime)

            # AI-AGENT-REF: Perform memory cleanup after trading cycle
            if MEMORY_OPTIMIZATION_AVAILABLE:
                try:
                    gc_result = optimize_memory()
                    if gc_result.get("objects_collected", 0) > 50:
                        logger.info(
                            f"Post-cycle GC: {gc_result['objects_collected']} objects collected"
                        )
                except (
                    RuntimeError,
                    ValueError,
                    TypeError,
                ) as e:  # AI-AGENT-REF: tighten memory optimization errors
                    logger.warning(
                        "MEMORY_OPTIMIZATION_FAILED",
                        extra={"cause": e.__class__.__name__, "detail": str(e)},
                    )
    finally:
        if acquired:
            run_lock.release()


def schedule_run_all_trades(runtime):
    """Spawn run_all_trades_worker if market is open."""  # FIXED
    ensure_alpaca_attached(runtime)
    if not _validate_trading_api(getattr(runtime, "api", None)):
        return
    if is_market_open():
        t = threading.Thread(
            target=run_all_trades_worker,
            args=(
                state,
                runtime,
            ),
            daemon=True,
        )
        t.start()
    else:
        logger.info("Market closed—skipping run_all_trades.")


def schedule_run_all_trades_with_delay(runtime):
    time.sleep(30)
    schedule_run_all_trades(runtime)


def initial_rebalance(ctx: BotContext, symbols: list[str]) -> None:
    """Initial portfolio rebalancing."""

    if ctx.api is None:
        logger.warning("ctx.api is None - cannot perform initial rebalance")
        return

    try:
        datetime.now(UTC).astimezone(PACIFIC)
        acct = ctx.api.get_account()
        float(acct.equity)

        cash = float(acct.cash)
        buying_power = float(getattr(acct, "buying_power", cash))
        n = len(symbols)
        if n == 0 or cash <= 0 or buying_power <= 0:
            logger.info("INITIAL_REBALANCE_NO_SYMBOLS_OR_NO_CASH")
            return
    except (
        APIError,
        TimeoutError,
        ConnectionError,
    ) as e:  # AI-AGENT-REF: tighten rebalance account fetch errors
        logger.warning(
            "INITIAL_REBALANCE_ACCOUNT_FAIL",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        return

    # Determine current UTC time
    now_utc = datetime.now(UTC)
    # If it’s between 00:00 and 00:15 UTC, daily bars may not be published yet.
    if now_utc.hour == 0 and now_utc.minute < 15:
        logger.info("INITIAL_REBALANCE: Too early—daily bars not live yet.")
    else:
        # Gather all symbols that have a valid, nonzero close
        valid_symbols = []
        valid_prices = {}
        for symbol in symbols:
            df_daily = ctx.data_fetcher.get_daily_df(ctx, symbol)
            price = get_latest_close(df_daily)
            if price <= 0:
                # skip symbols with no real close data
                continue
            valid_symbols.append(symbol)
            valid_prices[symbol] = price

        if not valid_symbols:
            log_level = logging.ERROR if in_trading_hours(now_utc) else logging.WARNING
            logger.log(
                log_level,
                (
                    "INITIAL_REBALANCE: No valid prices for any symbol—skipping "
                    "rebalance. Possible data outage or market holiday. "
                    "Check data provider/API status."
                ),
            )
        else:
            # Compute equal weights on valid symbols only
            total_capital = cash
            weight_per = 1.0 / len(valid_symbols)

            if hasattr(ctx.api, "list_positions"):
                raw_positions = ctx.api.list_positions()
            elif hasattr(ctx.api, "get_all_positions"):
                raw_positions = ctx.api.get_all_positions()
            else:
                raw_positions = []
            positions = {p.symbol: int(p.qty) for p in raw_positions}

            for sym in valid_symbols:
                price = valid_prices[sym]
                target_qty = int((total_capital * weight_per) // price)
                current_qty = int(positions.get(sym, 0))

                if current_qty < target_qty:
                    qty_to_buy = target_qty  # AI-AGENT-REF: retry full amount
                    if qty_to_buy < 1:
                        continue
                    try:
                        # AI-AGENT-REF: preserve consistent client_order_id across retries
                        cid = ctx.rebalance_ids.get(sym)
                        if not cid:
                            cid = f"{sym}-{uuid.uuid4().hex[:8]}"
                            ctx.rebalance_ids[sym] = cid
                            ctx.rebalance_attempts[sym] = 0
                        order = submit_order(ctx, sym, qty_to_buy, "buy")
                        # AI-AGENT-REF: confirm order result before logging success
                        if order:
                            logger.info(f"INITIAL_REBALANCE: Bought {qty_to_buy} {sym}")
                            ctx.rebalance_buys[sym] = datetime.now(UTC)
                        else:
                            logger.error(
                                f"INITIAL_REBALANCE: Buy failed for {sym}: order not placed"
                            )
                    except (
                        APIError,
                        TimeoutError,
                        ConnectionError,
                    ) as e:  # AI-AGENT-REF: tighten rebalance buy errors
                        logger.error(
                            "INITIAL_REBALANCE_BUY_FAILED",
                            extra={
                                "symbol": sym,
                                "cause": e.__class__.__name__,
                                "detail": str(e),
                            },
                        )
                elif current_qty > target_qty:
                    qty_to_sell = current_qty - target_qty
                    if qty_to_sell < 1:
                        continue
                    try:
                        submit_order(ctx, sym, qty_to_sell, "sell")
                        logger.info(f"INITIAL_REBALANCE: Sold {qty_to_sell} {sym}")
                    except (
                        APIError,
                        TimeoutError,
                        ConnectionError,
                    ) as e:  # AI-AGENT-REF: tighten rebalance sell errors
                        logger.error(
                            "INITIAL_REBALANCE_SELL_FAILED",
                            extra={
                                "symbol": sym,
                                "cause": e.__class__.__name__,
                                "detail": str(e),
                            },
                        )

    ctx.initial_rebalance_done = True
    try:
        if hasattr(ctx.api, "list_positions"):
            pos_list = ctx.api.list_positions()
        elif hasattr(ctx.api, "get_all_positions"):
            pos_list = ctx.api.get_all_positions()
        else:
            pos_list = []
        state.position_cache = {p.symbol: int(p.qty) for p in pos_list}
        state.long_positions = {s for s, q in state.position_cache.items() if q > 0}
        state.short_positions = {s for s, q in state.position_cache.items() if q < 0}
    except (
        APIError,
        TimeoutError,
        ConnectionError,
    ) as e:  # AI-AGENT-REF: tighten rebalance position refresh errors
        logger.error(
            "Failed to refresh position cache after rebalance",
            extra={"cause": e.__class__.__name__, "detail": str(e)},
        )
        # Initialize empty cache to prevent AttributeError
        state.position_cache = {}
        state.long_positions = set()
        state.short_positions = set()


def main() -> None:
    logger.info("Main trading bot starting")

    # AI-AGENT-REF: Initialize runtime config and validate credentials
    try:
        init_runtime_config()
    except RuntimeError as e:
        logger.critical("Runtime configuration failed: %s", e)
        sys.exit(2)

    # AI-AGENT-REF: Validate Alpaca credentials using settings singleton
    cfg = get_settings()
    api_key, api_secret = cfg.get_alpaca_keys()
    if not api_key or not api_secret:
        logger.critical("Alpaca credentials missing – aborting startup")
        logger.critical(
            "Please set ALPACA_API_KEY and ALPACA_SECRET_KEY"
        )
        sys.exit(2)

    # Log masked config for verification (only once per process)
    logger_once.info(
        "Config: ALPACA_API_KEY=***MASKED***", extra={"present": bool(api_key)}
    )
    logger_once.info(
        "Config: ALPACA_SECRET_KEY=***MASKED***", extra={"present": bool(api_secret)}
    )
    logger_once.info(f"Config: ALPACA_BASE_URL={cfg.alpaca_base_url}")
    logger_once.info(f"Config: TRADING_MODE={cfg.trading_mode}")

    _reload_env()

    # Initialize Alpaca client before starting the main loop
    if not _initialize_alpaca_clients() or trading_client is None:
        logger.critical(
            "Alpaca client initialization failed; terminating startup"
        )
        sys.exit(1)

    # AI-AGENT-REF: Ensure only one bot instance is running
    try:
        from ai_trading.process_manager import (
            ProcessManager,
        )  # AI-AGENT-REF: normalized import

        pm = ProcessManager()
        pm.ensure_single_instance()
        logger.info("Single instance lock acquired successfully")
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
        RuntimeError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.error("Failed to acquire single instance lock", exc_info=e)
        raise RuntimeError("Single-instance lock acquisition failed") from e

    # AI-AGENT-REF: Add comprehensive health check on startup
    try:
        from health_check import log_health_summary

        log_health_summary()
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.warning("Health check failed on startup: %s", e)

    def _handle_term(signum, frame):
        logger.info("PROCESS_TERMINATION", extra={"signal": signum})
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["aggressive", "balanced", "conservative"],
        default=TRADING_MODE_ENV or "balanced",
    )
    args = parser.parse_args()
    if args.mode != state.mode_obj.mode:
        state.mode_obj = BotMode(args.mode)
        params.update(state.mode_obj.get_config())

    # Initialize context and data fetcher early in the startup sequence
    lbc = get_ctx()
    try:
        lbc._ensure_initialized()
        ctx = lbc._context
        ensure_data_fetcher(ctx)
    except DataFetchError as exc:
        logger.critical(
            "DATA_FETCHER_INIT_FAILED", extra={"detail": str(exc)}
        )
        sys.exit(1)

    try:
        logger.info(">>> BOT __main__ ENTERED – starting up")

        # --- Market hours check ---

        # pd.Timestamp.utcnow() already returns a timezone-aware UTC timestamp,
        # so calling tz_localize("UTC") would raise an error. Simply use the
        # timestamp directly to avoid "Cannot localize tz-aware Timestamp".
        now_utc = pd.Timestamp.now(tz="UTC")
        if is_holiday(now_utc):
            logger.warning(
                f"No NYSE market schedule for {now_utc.date()}; skipping market open/close check."
            )
            market_open = False
        else:
            cal = get_market_calendar()
            try:
                market_open = cal.open_at_time(get_market_schedule(), now_utc)
            except (AttributeError, ValueError) as e:
                logger.warning(
                    f"Invalid schedule time {now_utc}: {e}; assuming market closed"
                )
                market_open = False

        sleep_minutes = 60
        if not market_open:
            logger.info("Market is closed. Sleeping for %d minutes.", sleep_minutes)
            time.sleep(sleep_minutes * 60)
            # Return control to outer loop instead of exiting
            return

        logger.info("Market is open. Starting trade cycle.")

        # Start Prometheus metrics server on an available port
        start_metrics_server(9200)

        if RUN_HEALTH:
            Thread(target=start_healthcheck, daemon=True).start()

        # Daily jobs
        schedule.every().day.at("00:30").do(
            lambda: Thread(target=daily_summary, daemon=True).start()
        )
        schedule.every().day.at("00:05").do(
            lambda: Thread(target=daily_reset, args=(state,), daemon=True).start()
        )
        schedule.every().day.at("10:00").do(
            lambda: Thread(
                target=run_meta_learning_weight_optimizer, daemon=True
            ).start()
        )
        schedule.every().day.at("02:00").do(
            lambda: Thread(
                target=run_bayesian_meta_learning_optimizer, daemon=True
            ).start()
        )

        # Retraining after market close (~16:05 US/Eastern)
        close_time = (
            dt_.now(UTC)
            .astimezone(ZoneInfo("America/New_York"))
            .replace(hour=16, minute=5, second=0, microsecond=0)
            .astimezone(UTC)
            .strftime("%H:%M")
        )
        schedule.every().day.at(close_time).do(
            lambda: Thread(target=on_market_close, daemon=True).start()
        )

        # ai_trading/core/bot_engine.py:9768 - Convert import guard to settings-gated import

        settings = get_settings()
        if not settings.disable_daily_retrain:
            if settings.enable_sklearn:  # Meta-learning requires sklearn
                pass
            else:
                globals()["retrain_meta_learner"] = None
                logger.info("Daily retraining disabled: sklearn not enabled")
        else:
            logger.info("Daily retraining disabled via DISABLE_DAILY_RETRAIN")

        logger.info("BOT_LAUNCHED")
        cancel_all_open_orders(ctx)
        audit_positions(ctx)
        try:
            initial_list = load_tickers(TICKERS_FILE)
            summary = pre_trade_health_check(ctx, initial_list)
            logger.info("STARTUP_HEALTH", extra=summary)
            failures = (
                summary["failures"]
                or summary["insufficient_rows"]
                or summary["missing_columns"]
                or summary.get("invalid_values")
                or summary["timezone_issues"]
            )

            # AI-AGENT-REF: Add bypass for stale data during initial deployment
            stale_data = summary.get("stale_data", [])
            allow_stale_on_startup = (
                os.getenv("ALLOW_STALE_DATA_STARTUP", "true").lower() == "true"
            )

            if stale_data and allow_stale_on_startup:
                logger.warning(
                    "BYPASS_STALE_DATA_STARTUP: Allowing trading with stale data for initial deployment",
                    extra={"stale_symbols": stale_data, "count": len(stale_data)},
                )
            elif stale_data and not allow_stale_on_startup:
                failures = failures or stale_data

            health_ok = not failures
            if not health_ok:
                logger.error("HEALTH_CHECK_FAILED", extra=summary)
                sys.exit(1)
            else:
                logger.info("HEALTH_OK")
            # Prefetch minute history so health check rows are available
            for sym in initial_list:
                try:
                    ctx.data_fetcher.get_minute_df(
                        ctx, sym, lookback_minutes=CFG.min_health_rows
                    )
                except (
                    FileNotFoundError,
                    PermissionError,
                    IsADirectoryError,
                    JSONDecodeError,
                    ValueError,
                    KeyError,
                    TypeError,
                    OSError,
                ) as exc:  # AI-AGENT-REF: narrow exception
                    logger.warning("Initial minute prefetch failed for %s: %s", sym, exc)
        except (
            FileNotFoundError,
            OSError,
            KeyError,
            ValueError,
            TypeError,
            TimeoutError,
            ConnectionError,
        ) as e:  # AI-AGENT-REF: explicit error logging for data health
            logger.warning(
                "HEALTH_DATA_PROBE_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
            sys.exit(1)

        # ─── WARM-CACHE SENTIMENT FOR ALL TICKERS ─────────────────────────────────────
        # This will prevent the initial burst of NewsAPI calls and 429s
        all_tickers = load_tickers(TICKERS_FILE)
        now_ts = pytime.time()
        with sentiment_lock:
            for t in all_tickers:
                _SENTIMENT_CACHE[t] = (now_ts, 0.0)

        # Initial rebalance (once) only if health check passed
        try:
            if health_ok and not getattr(ctx, "_rebalance_done", False):
                universe = load_tickers(TICKERS_FILE)
                initial_rebalance(ctx, universe)
                ctx._rebalance_done = True
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.warning(f"[REBALANCE] aborted due to error: {e}")

        # Recurring jobs
        def gather_minute_data_with_delay():
            try:
                # delay can be configured via env SCHEDULER_SLEEP_SECONDS
                time.sleep(CFG.scheduler_sleep_seconds)
                schedule_run_all_trades(ctx)  # AI-AGENT-REF: runtime-based scheduling
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as e:  # AI-AGENT-REF: narrow exception
                logger.exception(f"gather_minute_data_with_delay failed: {e}")

        schedule.every(1).minutes.do(
            lambda: Thread(target=gather_minute_data_with_delay, daemon=True).start()
        )

        # --- run one fetch right away, before entering the loop ---
        try:
            gather_minute_data_with_delay()
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.exception("Initial data fetch failed", exc_info=e)
        schedule.every(1).minutes.do(
            lambda: Thread(
                target=validate_open_orders, args=(ctx,), daemon=True
            ).start()
        )
        schedule.every(1).minutes.do(
            lambda: Thread(target=_update_risk_engine_exposure, daemon=True).start()
        )
        # AI-AGENT-REF: Periodic metrics emission (gated by flag)
        schedule.every(5).minutes.do(
            lambda: Thread(target=_emit_periodic_metrics, daemon=True).start()
        )
        schedule.every(6).hours.do(
            lambda: Thread(target=update_signal_weights, daemon=True).start()
        )
        schedule.every(30).minutes.do(
            lambda: Thread(target=update_bot_mode, args=(state,), daemon=True).start()
        )
        schedule.every(30).minutes.do(
            lambda: Thread(
                target=adaptive_risk_scaling, args=(ctx,), daemon=True
            ).start()
        )
        schedule.every(get_rebalance_interval_min()).minutes.do(
            lambda: Thread(target=maybe_rebalance, args=(ctx,), daemon=True).start()
        )  # AI-AGENT-REF: standardize rebalance interval access
        schedule.every().day.at("23:55").do(
            lambda: Thread(target=check_disaster_halt, daemon=True).start()
        )

        # Start listening for trade updates in a background thread
        ctx.stream_event = asyncio.Event()
        ctx.stream_event.set()
        _, start_trade_updates_stream = _alpaca_symbols()  # AI-AGENT-REF: lazy stream import
        threading.Thread(
            target=lambda: asyncio.run(
                start_trade_updates_stream(
                    ALPACA_API_KEY,
                    ALPACA_SECRET_KEY,
                    trading_client,
                    state,
                    paper=True,
                    running=ctx.stream_event,
                )
            ),
            daemon=True,
        ).start()

    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.exception(f"Fatal error in main: {e}")
        raise


@profile
def prepare_indicators_simple(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        logger.error("Input dataframe is None or empty in prepare_indicators.")
        raise ValueError("Input dataframe is None or empty")

    try:
        macd_line, signal_line, hist = simple_calculate_macd(df["close"])
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.error(f"MACD calculation failed: {e}", exc_info=True)
        raise ValueError("MACD calculation failed") from e

    if macd_line is None or signal_line is None or hist is None:
        logger.error("MACD returned None")
        raise ValueError("MACD returned None")

    df["macd_line"] = macd_line
    df["signal_line"] = signal_line
    df["histogram"] = hist

    return df


def simple_calculate_macd(
    close_prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None]:
    if close_prices is None or close_prices.empty:
        logger.warning("Empty or None close_prices passed to calculate_macd.")
        return None, None, None

    try:
        exp1 = close_prices.ewm(span=fast, adjust=False).mean()
        exp2 = close_prices.ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.error(f"Exception in MACD calculation: {e}", exc_info=True)
        return None, None, None


def compute_ichimoku(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return Ichimoku lines and signal DataFrames."""
    try:
        ich_func = getattr(ta, "ichimoku", None)
        if ich_func is None:
            try:
                from ai_trading.indicators import ichimoku_fallback  # type: ignore

                ich_func = ichimoku_fallback
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ):  # pragma: no cover  # AI-AGENT-REF: narrow exception
                logger.warning("ichimoku indicators not available")
                ich_func = None

        if ich_func:
            ich = ich_func(high=high, low=low, close=close)
        else:
            # Return empty dataframes if no ichimoku function available
            return pd.DataFrame(index=high.index), pd.DataFrame(index=high.index)
        if isinstance(ich, tuple):
            ich_df = ich[0]
            signal_df = ich[1] if len(ich) > 1 else pd.DataFrame(index=ich_df.index)
        else:
            ich_df = ich
            signal_df = pd.DataFrame(index=ich_df.index)
        # AI-AGENT-REF: Use attribute check instead of isinstance to avoid type errors
        if not hasattr(ich_df, "iloc") or not hasattr(ich_df, "columns"):
            ich_df = pd.DataFrame(ich_df)
        if not hasattr(signal_df, "iloc") or not hasattr(signal_df, "columns"):
            signal_df = pd.DataFrame(signal_df)
        return ich_df, signal_df
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # pragma: no cover - defensive  # AI-AGENT-REF: narrow exception
        log_warning("INDICATOR_ICHIMOKU_FAIL", exc=exc)
        return pd.DataFrame(), pd.DataFrame()


def ichimoku_indicator(
    df: pd.DataFrame,
    symbol: str,
    state: BotState | None = None,
) -> tuple[pd.DataFrame, Any | None]:
    """Return Ichimoku indicator DataFrame and optional params."""
    try:
        ich_func = getattr(ta, "ichimoku", None)
        if ich_func is None:
            try:
                from ai_trading.indicators import ichimoku_fallback  # type: ignore

                ich_func = ichimoku_fallback
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ):  # pragma: no cover  # AI-AGENT-REF: narrow exception
                logger.warning("ichimoku indicators not available")
                ich_func = None

        if ich_func:
            ich = ich_func(high=df["high"], low=df["low"], close=df["close"])
        if isinstance(ich, tuple):
            ich_df = ich[0]
            params = ich[1] if len(ich) > 1 else None
        else:
            ich_df = ich
            params = None
        return ich_df, params
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # pragma: no cover - defensive  # AI-AGENT-REF: narrow exception
        log_warning("INDICATOR_ICHIMOKU_FAIL", exc=exc, extra={"symbol": symbol})
        if state:
            state.indicator_failures += 1
        return pd.DataFrame(), None


def _check_trade_frequency_limits(
    state: BotState, symbol: str, current_time: datetime
) -> bool:
    """
    Check if trading would exceed frequency limits.

    AI-AGENT-REF: Enhanced overtrading prevention with configurable frequency limits.
    Returns True if trade should be skipped due to frequency limits.
    """
    if not hasattr(state, "trade_history"):
        state.trade_history = []

    # Clean up old entries (older than 1 day)
    day_ago = current_time - timedelta(days=1)
    state.trade_history = [(sym, ts) for sym, ts in state.trade_history if ts > day_ago]

    # Count trades in different time windows
    hour_ago = current_time - timedelta(hours=TRADE_FREQUENCY_WINDOW_HOURS)

    # Count symbol-specific trades in last hour
    symbol_trades_hour = len(
        [
            (sym, ts)
            for sym, ts in state.trade_history
            if sym == symbol and ts > hour_ago
        ]
    )

    # Count total trades in last hour
    total_trades_hour = len(
        [(sym, ts) for sym, ts in state.trade_history if ts > hour_ago]
    )

    # Count total trades in last day
    total_trades_day = len(state.trade_history)

    # Check hourly limits
    if total_trades_hour >= MAX_TRADES_PER_HOUR:
        logger.warning(
            "FREQUENCY_LIMIT_HOURLY_EXCEEDED",
            extra={
                "symbol": symbol,
                "trades_last_hour": total_trades_hour,
                "max_per_hour": MAX_TRADES_PER_HOUR,
                "recommendation": "Reduce trading frequency to prevent overtrading",
            },
        )
        return True

    # Check daily limits
    if total_trades_day >= MAX_TRADES_PER_DAY:
        logger.warning(
            "FREQUENCY_LIMIT_DAILY_EXCEEDED",
            extra={
                "symbol": symbol,
                "trades_today": total_trades_day,
                "max_per_day": MAX_TRADES_PER_DAY,
                "recommendation": "Daily trade limit reached - consider reviewing strategy",
            },
        )
        return True

    # Check symbol-specific hourly limit (prevent rapid ping-pong on same symbol)
    symbol_hourly_limit = max(
        1, MAX_TRADES_PER_HOUR // 10
    )  # 10% of hourly limit per symbol
    if symbol_trades_hour >= symbol_hourly_limit:
        logger.info(
            "FREQUENCY_LIMIT_SYMBOL_HOURLY",
            extra={
                "symbol": symbol,
                "symbol_trades_hour": symbol_trades_hour,
                "symbol_hourly_limit": symbol_hourly_limit,
                "note": "Preventing rapid trading on single symbol",
            },
        )
        return True

    return False


def _record_trade_in_frequency_tracker(
    state: BotState, symbol: str, timestamp: datetime
) -> None:
    """
    Record a trade in the frequency tracking system.

    AI-AGENT-REF: Part of overtrading prevention system.
    """
    if not hasattr(state, "trade_history"):
        state.trade_history = []

    state.trade_history.append((symbol, timestamp))

    # Log frequency stats for monitoring
    hour_ago = timestamp - timedelta(hours=1)
    recent_trades = len([ts for _, ts in state.trade_history if ts > hour_ago])

    logger.debug(
        "TRADE_FREQUENCY_UPDATED",
        extra={
            "symbol": symbol,
            "trades_last_hour": recent_trades,
            "total_tracked_trades": len(state.trade_history),
        },
    )


def get_latest_price(symbol: str):
    try:
        alpaca_get, _ = _alpaca_symbols()  # AI-AGENT-REF: lazy fetch import
        data = alpaca_get(f"/v2/stocks/{symbol}/quotes/latest")
        price = float(data.get("ap", 0)) if data else None
        if price is None:
            raise ValueError(f"Price returned None for symbol {symbol}")
        return price
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as e:  # AI-AGENT-REF: narrow exception
        logger.error("Failed to get latest price for %s: %s", symbol, e, exc_info=True)
        return None


def initialize_bot(api=None, data_loader=None):
    """Return a minimal context and state for unit tests."""
    ctx = types.SimpleNamespace(api=api, data_loader=data_loader)
    state = {"positions": {}}
    return ctx, state


def generate_signals(df):
    """+1 if price rise, -1 if price fall, else 0."""
    price = df["price"]  # KeyError if missing
    diff = price.diff().fillna(0)  # NaN → 0 for the first row
    signals = diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return signals  # pandas Series → .items()


def execute_trades(ctx, signals: pd.Series) -> list[tuple[str, str]]:
    """Return orders inferred from ``signals`` without hitting real APIs."""
    orders = []
    for symbol, sig in signals.items():
        if sig == 0:
            continue
        side = "buy" if sig > 0 else "sell"
        api = getattr(ctx, "api", None)
        if api is not None and hasattr(api, "submit_order"):
            try:
                api.submit_order(symbol, 1, side)
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as e:  # AI-AGENT-REF: narrow exception
                # Order submission failed - log error and add to failed orders
                logger.error("Failed to submit test order for %s %s: %s", symbol, side, e)
        orders.append((symbol, side))
    return orders


def run_trading_cycle(ctx, df: pd.DataFrame) -> list[tuple[str, str]]:
    """Generate signals from ``df`` and execute trades via ``ctx``."""
    signals = generate_signals(df)
    return execute_trades(ctx, signals)


def health_check(df: pd.DataFrame, resolution: str) -> bool:
    """Delegate to :func:`utils.health_check` for convenience."""  # AI-AGENT-REF
    return _health_check(df, resolution)


def compute_atr_stop(df, atr_window=14, multiplier=2):
    # AI-AGENT-REF: helper for ATR-based trailing stop
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(atr_window).mean()
    stop_level = df["close"] - (atr * multiplier)
    return stop_level


if __name__ == "__main__":
    try:
        main()
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:  # AI-AGENT-REF: narrow exception
        logger.exception("Fatal error in main: %s", exc)
        raise

    import time

    import schedule

    while True:
        try:
            schedule.run_pending()
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as exc:  # AI-AGENT-REF: narrow exception
            logger.exception("Scheduler loop error: %s", exc)
        time.sleep(CFG.scheduler_sleep_seconds)
