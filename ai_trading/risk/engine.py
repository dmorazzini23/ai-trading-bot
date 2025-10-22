from __future__ import annotations
from ai_trading.logging import get_logger
import math
import random
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
import numpy as np
import importlib
from ai_trading.utils.lazy_imports import load_pandas, load_pandas_ta
from ai_trading.data.bars import safe_get_stock_bars, StockBarsRequest, TimeFrame
from ai_trading.utils.time import monotonic_time
from ai_trading.data.fetch import normalize_ohlcv_columns

try:
    from alpaca.common.exceptions import APIError
except ImportError:  # pragma: no cover - allow import without alpaca for tests

    class APIError(Exception):
        pass


try:
    from alpaca.trading.client import TradingClient
except ImportError:  # pragma: no cover - allow import without alpaca for tests
    TradingClient = object  # type: ignore[assignment]
from ai_trading.config.management import (
    SEED,
    TradingConfig,
    get_env,
    get_trading_config,
    _resolve_alpaca_env,
    validate_required_env,
)
from ai_trading.config.settings import get_settings
from ai_trading.settings import (
    POSITION_SIZE_MIN_USD_DEFAULT,
    get_alpaca_secret_key_plain,
    get_position_size_min_usd,
)

if not hasattr(np, "NaN"):
    np.NaN = np.nan

# Lazy pandas proxy
pd = load_pandas()


DEFAULT_VOLATILITY_FALLBACK = 0.02

try:  # pragma: no cover - optional dependency
    from requests import exceptions as _requests_exceptions  # type: ignore[import]
except (ImportError, ModuleNotFoundError):  # pragma: no cover - requests optional
    _requests_exceptions = None

_YF_HISTORY_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    OSError,
    RuntimeError,
    ConnectionError,
    TimeoutError,
)
if _requests_exceptions is not None:  # pragma: no cover - requests optional
    request_exc = getattr(_requests_exceptions, "RequestException", None)
    if isinstance(request_exc, type):
        _YF_HISTORY_EXCEPTIONS = _YF_HISTORY_EXCEPTIONS + (request_exc,)


def _is_finite_number(value: Any) -> bool:
    try:
        if hasattr(np, "isfinite"):
            return bool(np.isfinite(value))
    except (TypeError, ValueError):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _derive_minimum_quantity(engine: "RiskEngine", price: float) -> int:
    min_usd_raw = getattr(
        engine.config,
        "position_size_min_usd",
        POSITION_SIZE_MIN_USD_DEFAULT,
    )
    try:
        min_usd_value = float(min_usd_raw)
    except (TypeError, ValueError):
        min_usd_value = None

    fallback_usd = get_position_size_min_usd()
    if fallback_usd <= 0:
        fallback_usd = POSITION_SIZE_MIN_USD_DEFAULT
    fallback_qty = max(int(fallback_usd / price), 1)

    if min_usd_value is not None and _is_finite_number(min_usd_value) and min_usd_value > 0:
        return max(int(min_usd_value / price), 1)

    if not getattr(engine, "_invalid_min_size_logged", False):
        logger.warning(
            "Invalid position_size_min_usd=%s; using fallback of $%.2f",
            min_usd_raw,
            fallback_usd,
        )
        engine._invalid_min_size_logged = True
    return fallback_qty


def _calculate_position_size(
    engine: "RiskEngine",
    raw_qty: float,
    price: float,
    signal: Any,
) -> int:
    symbol = getattr(signal, "symbol", "UNKNOWN")
    min_qty = _derive_minimum_quantity(engine, price)

    if not _is_finite_number(raw_qty):
        logger.warning(
            "Non-finite raw_qty %s for %s; falling back to minimum position size",
            raw_qty,
            symbol,
        )
        return max(min_qty, 0)

    if raw_qty < 0:
        logger.warning("Negative raw_qty %s for %s, returning 0", raw_qty, symbol)
        return 0

    if raw_qty == 0:
        logger.warning(
            "Zero raw_qty for %s; falling back to minimum position size",
            symbol,
        )
        return max(min_qty, 0)

    qty = max(int(raw_qty), min_qty)
    return max(qty, 0)


try:
    from alpaca.data.historical.stock import StockHistoricalDataClient
except ImportError:  # pragma: no cover - allow import without alpaca for tests
    StockHistoricalDataClient = object  # type: ignore[assignment]


def _safe_call(fn, *a, **k):
    try:
        return fn(*a, **k)
    except AttributeError:
        return None


@dataclass
class TradeSignal:
    symbol: str
    side: str
    confidence: float
    strategy: str
    weight: float
    asset_class: str
    strength: float = 1.0


logger = get_logger(__name__)

random.seed(SEED)
np.random.seed(SEED)
if not hasattr(np, "NaN"):
    np.NaN = np.nan
MAX_DRAWDOWN = 0.05


class RiskEngine:
    """Cross-strategy risk manager."""

    _lock: object | None = None

    def __init__(self, cfg: TradingConfig | None = None) -> None:
        """Initialize the engine with an optional trading config."""
        self._validate_env()
        logger.info("Risk engine initialized")
        self.config = cfg if cfg is not None else get_trading_config()
        self._lock = threading.Lock()
        self.hard_stop = False
        self.max_trades = 10
        self.current_trades = 0
        settings = get_settings()
        self.enable_portfolio_features = settings.ENABLE_PORTFOLIO_FEATURES
        try:
            exposure_cap = getattr(self.config, "exposure_cap_aggressive", 0.8)
            if not isinstance(exposure_cap, int | float) or not 0 < exposure_cap <= 1.0:
                logger.warning("Invalid exposure_cap_aggressive %s, using default 0.8", exposure_cap)
                exposure_cap = 0.8
            self.global_limit = exposure_cap
        except (TypeError, ValueError) as e:  # config may contain non-numeric values
            logger.error("Error validating exposure_cap_aggressive: %s, using default", e)
            self.global_limit = 0.8
        self.asset_limits: dict[str, float] = {}
        self.strategy_limits: dict[str, float] = {}
        self.exposure: dict[str, float] = {}
        self.strategy_exposure: dict[str, float] = {}
        self._positions: dict[str, int] = {}
        self._atr_cache: dict[str, tuple] = {}
        self._volatility_cache: dict[str, tuple] = {}
        self._volatility_alerted = False
        self.data_client = None
        try:
            cfg_key, cfg_secret, _ = _resolve_alpaca_env()
            api_key = cfg_key or getattr(settings, "alpaca_api_key", None)
            secret = cfg_secret or get_alpaca_secret_key_plain()
            oauth = get_env("ALPACA_OAUTH")
            has_keypair = bool(api_key and secret)
            if has_keypair and oauth:
                raise RuntimeError("Provide either ALPACA_API_KEY/ALPACA_SECRET_KEY or ALPACA_OAUTH, not both")
            if has_keypair:
                self.data_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret)
            elif oauth:
                self.data_client = StockHistoricalDataClient(oauth_token=oauth)
        except (APIError, TypeError, AttributeError, OSError, ImportError) as e:
            logger.warning("Could not initialize data client: %s", e)
        self._returns: list[float] = []
        self._drawdowns: list[float] = []
        self._last_portfolio_cap: float | None = None
        self._last_equity_cap: float | None = None
        from threading import Event

        self._update_event = Event()
        self._last_update = 0.0
        self._invalid_min_size_logged = False
        try:
            max_drawdown = get_env("MAX_DRAWDOWN_THRESHOLD", "0.15", cast=float)
            if not 0 < max_drawdown <= 1.0:
                logger.warning("Invalid MAX_DRAWDOWN_THRESHOLD %s, using default 0.15", max_drawdown)
                max_drawdown = 0.15
            self.max_drawdown_threshold = float(max_drawdown)
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error("Error parsing MAX_DRAWDOWN_THRESHOLD: %s, using default 0.15", e)
            self.max_drawdown_threshold = 0.15
        try:
            cooldown = get_env("HARD_STOP_COOLDOWN_MIN", "10", cast=float)
            if cooldown < 0:
                logger.warning("Invalid HARD_STOP_COOLDOWN_MIN %s, using default 10", cooldown)
                cooldown = 10.0
            self.hard_stop_cooldown = float(cooldown)
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error("Error parsing HARD_STOP_COOLDOWN_MIN: %s, using default 10", e)
            self.hard_stop_cooldown = 10.0
        self._hard_stop_until: float | None = None

    def _init_data_client(self):
        """Return an initialized data client if available."""

        if self.data_client is not None:
            return self.data_client
        ctx = getattr(self, "ctx", None)
        if ctx is not None:
            candidate = getattr(ctx, "api", None)
            if candidate is not None:
                return candidate
        return None

    def _validate_env(self) -> None:
        """Validate required environment variables unless running tests."""
        if get_env("PYTEST_RUNNING", "0", cast=bool):
            return
        env_count = len(validate_required_env())
        logger.debug("Validated %d environment variables", env_count)

    def _dynamic_cap(self, asset_class: str, volatility: float | None = None, cash_ratio: float | None = None) -> float:
        """Return exposure cap for ``asset_class`` using adaptive rules."""
        base_cap = self.asset_limits.get(asset_class, self.global_limit)
        port_cap = self._adaptive_global_cap()
        vol = self._current_volatility()
        if (
            self._last_portfolio_cap is None
            or abs(self._last_portfolio_cap - port_cap) > 0.01
            or self._last_equity_cap is None
            or (abs(self._last_equity_cap - base_cap) > 0.01)
        ):
            logger.info(
                "Adaptive exposure caps: portfolio=%.1f, equity=%.1f (volatility=%.1f%%)", port_cap, base_cap, vol * 100
            )
            self._last_portfolio_cap = port_cap
            self._last_equity_cap = base_cap
        return min(base_cap, port_cap)

    def _current_volatility(self) -> float:
        recent = self._returns[-10:]
        fallback = DEFAULT_VOLATILITY_FALLBACK
        if not recent:
            self._log_volatility_anomaly("no_returns", fallback=fallback)
            return fallback
        try:
            vol = float(np.std(recent))
        except (TypeError, ValueError):
            self._log_volatility_anomaly("std_failed", fallback=fallback)
            return fallback
        if not math.isfinite(vol) or vol <= 0:
            self._log_volatility_anomaly(
                "non_positive",
                fallback=fallback,
                details={"volatility": vol, "samples": len(recent)},
            )
            return fallback
        self._volatility_alerted = False
        return vol

    def _log_volatility_anomaly(
        self,
        reason: str,
        *,
        fallback: float,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if getattr(self, "_volatility_alerted", False):
            return
        payload: dict[str, Any] = {"reason": reason, "fallback": fallback}
        if details:
            try:
                payload.update(details)
            except (TypeError, ValueError):
                payload["details_error"] = "unmergeable"
        logger.warning("PORTFOLIO_VOLATILITY_FALLBACK", extra=payload)
        self._volatility_alerted = True

    def _get_atr_data(self, symbol: str, lookback: int = 14) -> float | None:
        """Return ATR value for ``symbol``."""
        try:
            lookback = max(int(lookback), 1)
            if symbol in self._atr_cache:
                ts, val = self._atr_cache[symbol]
                if datetime.now(UTC) - ts < timedelta(minutes=30):
                    return val
            ctx = getattr(self, "ctx", None)
            client = getattr(ctx, "data_client", None) or self._init_data_client()

            def _safe_bar_value(bar: Any, names: Sequence[str]) -> Any:
                if isinstance(bar, dict):
                    for name in names:
                        if name in bar and bar[name] is not None:
                            return bar[name]
                for name in names:
                    if hasattr(bar, name):
                        value = getattr(bar, name)
                        if value is not None:
                            return value
                return None

            def _extract_arrays(df_in) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
                if df_in is None:
                    return (None, None, None)
                try:
                    if pd is not None and not isinstance(df_in, pd.DataFrame):
                        df_local = pd.DataFrame(df_in)
                    else:
                        df_local = df_in.copy() if isinstance(df_in, pd.DataFrame) else df_in
                except (ValueError, TypeError, AttributeError):
                    return (None, None, None)
                if getattr(df_local, "empty", True):
                    return (None, None, None)
                df_local = normalize_ohlcv_columns(df_local)
                if pd is not None and hasattr(pd, "RangeIndex") and isinstance(df_local.index, pd.RangeIndex):
                    df_local = df_local.reset_index(drop=True)

                def _series(df_obj: "pd.DataFrame", name: str) -> np.ndarray | None:
                    if name in df_obj:
                        try:
                            return df_obj[name].dropna().to_numpy()
                        except AttributeError:
                            return None
                    return None

                return (
                    _series(df_local, "high"),
                    _series(df_local, "low"),
                    _series(df_local, "close"),
                )

            def _sequence_to_records(seq: Sequence[Any]) -> list[dict[str, Any]]:
                records: list[dict[str, Any]] = []
                allowed_keys = (
                    "open",
                    "high",
                    "low",
                    "close",
                    "o",
                    "h",
                    "l",
                    "c",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                )
                for bar in seq:
                    if bar is None:
                        continue
                    if isinstance(bar, dict):
                        records.append(bar)
                        continue
                    if hasattr(bar, "__dict__"):
                        data = {key: value for key, value in vars(bar).items() if key in allowed_keys}
                        if data:
                            records.append(data)
                        continue
                    if isinstance(bar, Sequence) and not isinstance(bar, (str, bytes, bytearray)):
                        values = list(bar)
                        record: dict[str, Any] = {}
                        if len(values) >= 4:
                            record = {
                                "open": values[0],
                                "high": values[1],
                                "low": values[2],
                                "close": values[3],
                            }
                        elif len(values) >= 3:
                            record = {
                                "high": values[0],
                                "low": values[1],
                                "close": values[2],
                            }
                        if record:
                            records.append(record)
                        continue
                    record = {}
                    for attr in allowed_keys:
                        if hasattr(bar, attr):
                            record[attr] = getattr(bar, attr)
                    if record:
                        records.append(record)
                return records

            def _coerce_dataframe(candidate: Any) -> "pd.DataFrame | None":
                if pd is None or candidate is None:
                    return None
                try:
                    if isinstance(candidate, pd.DataFrame):
                        df_candidate = candidate.copy()
                    elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
                        try:
                            seq_list = list(candidate)
                        except TypeError:
                            return None
                        records = _sequence_to_records(seq_list)
                        if records:
                            df_candidate = pd.DataFrame(records)
                        else:
                            df_candidate = pd.DataFrame(seq_list)
                    elif hasattr(candidate, "to_dict"):
                        try:
                            records = candidate.to_dict("records")
                        except (TypeError, ValueError, AttributeError):
                            records = None
                        if records:
                            df_candidate = pd.DataFrame(records)
                        else:
                            df_candidate = pd.DataFrame(candidate)
                    else:
                        df_candidate = pd.DataFrame(candidate)
                except (ValueError, TypeError, AttributeError):
                    return None
                if getattr(df_candidate, "empty", True):
                    return None
                return df_candidate

            def _sequence_to_arrays(
                seq: Any,
            ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
                if seq is None:
                    return (None, None, None)
                df_from_seq = _coerce_dataframe(seq)
                if df_from_seq is not None:
                    try:
                        high_arr, low_arr, close_arr = _extract_arrays(df_from_seq)
                        if all(
                            arr is not None and len(arr) > 0 for arr in (high_arr, low_arr, close_arr)
                        ):
                            return high_arr, low_arr, close_arr
                    except (ValueError, TypeError, AttributeError) as exc:  # pragma: no cover - defensive
                        logger.debug("ATR sequence dataframe extraction failed for %s: %s", symbol, exc)
                records: list[dict[str, Any]] = []
                if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes, bytearray)):
                    records = _sequence_to_records(seq)
                elif pd is not None and hasattr(seq, "to_dict"):
                    try:
                        maybe_records = seq.to_dict("records")
                    except (TypeError, ValueError, AttributeError):
                        maybe_records = None
                    if maybe_records:
                        records = _sequence_to_records(maybe_records)
                if not records:
                    return (None, None, None)
                highs: list[float] = []
                lows: list[float] = []
                closes: list[float] = []
                for record in records:
                    high_val = None
                    for key in ("high", "High", "h", "H"):
                        if key in record and record[key] is not None:
                            high_val = record[key]
                            break
                    low_val = None
                    for key in ("low", "Low", "l", "L"):
                        if key in record and record[key] is not None:
                            low_val = record[key]
                            break
                    close_val = None
                    for key in ("close", "Close", "c", "C"):
                        if key in record and record[key] is not None:
                            close_val = record[key]
                            break
                    if None in (high_val, low_val, close_val):
                        continue
                    try:
                        highs.append(float(high_val))
                        lows.append(float(low_val))
                        closes.append(float(close_val))
                    except (TypeError, ValueError):
                        continue
                if highs and lows and closes:
                    return (
                        np.asarray(highs, dtype=float),
                        np.asarray(lows, dtype=float),
                        np.asarray(closes, dtype=float),
                    )
                return (None, None, None)

            def _candidate_to_arrays(candidate: Any) -> tuple[
                np.ndarray | None,
                np.ndarray | None,
                np.ndarray | None,
                Sequence[Any] | None,
            ]:
                if candidate is None:
                    return (None, None, None, None)
                cand_high = cand_low = cand_close = None
                df_candidate = _coerce_dataframe(candidate)
                if df_candidate is not None:
                    try:
                        cand_high, cand_low, cand_close = _extract_arrays(df_candidate)
                    except (ValueError, TypeError, AttributeError) as exc:  # pragma: no cover - defensive
                        logger.debug("ATR candidate dataframe extraction failed for %s: %s", symbol, exc)
                        cand_high = cand_low = cand_close = None
                seq_candidate: Sequence[Any] | None = None
                if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
                    seq_candidate = candidate
                elif not hasattr(candidate, "empty"):
                    try:
                        seq_list = list(candidate)
                    except TypeError:
                        seq_list = None
                    else:
                        if seq_list:
                            seq_candidate = seq_list
                seq_high = seq_low = seq_close = None
                if seq_candidate is not None:
                    seq_high, seq_low, seq_close = _sequence_to_arrays(seq_candidate)
                else:
                    seq_high, seq_low, seq_close = _sequence_to_arrays(candidate)
                if seq_high is not None and seq_low is not None and seq_close is not None:
                    cand_high, cand_low, cand_close = seq_high, seq_low, seq_close
                    if seq_candidate is None and isinstance(candidate, Sequence) and not isinstance(
                        candidate, (str, bytes, bytearray)
                    ):
                        seq_candidate = candidate
                return cand_high, cand_low, cand_close, seq_candidate

            high = low = close = None
            df = None
            bars_sequence: Sequence[Any] | None = None
            simple_get = None
            attempted_simple_get = False
            limit: int | None = None
            if client:
                has_stock_bars = callable(getattr(client, "get_stock_bars", None)) or callable(
                    getattr(client, "get_bars", None)
                )
                if has_stock_bars:
                    feed = getattr(get_settings(), "alpaca_data_feed", None)
                    limit = max(lookback + 10, lookback + 1, 2)
                    timeframe = getattr(TimeFrame, "Day", "1Day")
                    simple_get = getattr(client, "get_bars", None)
                    try:
                        request = StockBarsRequest(
                            symbol_or_symbols=symbol,
                            timeframe=timeframe,
                            limit=limit,
                            feed=feed,
                        )
                        bars_df = safe_get_stock_bars(client, request, symbol, context="risk_engine_atr")
                        if hasattr(bars_df, "empty"):
                            if not bars_df.empty:
                                df = bars_df.copy()
                        elif bars_df is not None and pd is not None:
                            df = pd.DataFrame(bars_df)
                    except (
                        APIError,
                        TimeoutError,
                        ConnectionError,
                        ValueError,
                        TypeError,
                        OSError,
                        AttributeError,
                    ) as exc:  # pragma: no cover - provider variance
                        logger.warning("ATR client fetch failed for %s: %s", symbol, exc)
                        if callable(simple_get):
                            attempted_simple_get = True
                            try:
                                candidate = simple_get(symbol, limit)
                            except TypeError:
                                candidate = None
                            except (
                                APIError,
                                TimeoutError,
                                ConnectionError,
                                ValueError,
                                OSError,
                            ) as fallback_exc:  # pragma: no cover - provider variance
                                logger.debug("ATR simple get_bars failed for %s: %s", symbol, fallback_exc)
                                candidate = None
                            if candidate is not None:
                                cand_high, cand_low, cand_close, seq_candidate = _candidate_to_arrays(candidate)
                                if cand_high is not None and cand_low is not None and cand_close is not None:
                                    high, low, close = cand_high, cand_low, cand_close
                                elif seq_candidate:
                                    bars_sequence = seq_candidate
                else:
                    logger.warning("missing stock bars fetch for %s", symbol)
            else:
                logger.warning("No data client available; attempting to use context data for %s", symbol)
            if df is not None:
                try:
                    high, low, close = _extract_arrays(df)
                except (ValueError, TypeError, AttributeError) as exc:  # pragma: no cover - defensive
                    logger.debug("ATR dataframe extraction failed for %s: %s", symbol, exc)
                    high = low = close = None
            if (
                any(x is None for x in (high, low, close))
                and simple_get is not None
                and callable(simple_get)
                and not attempted_simple_get
                and limit is not None
            ):
                try:
                    candidate = simple_get(symbol, limit)
                except TypeError:
                    candidate = None
                except (
                    APIError,
                    TimeoutError,
                    ConnectionError,
                    ValueError,
                    OSError,
                ) as fallback_exc:  # pragma: no cover - provider variance
                    logger.debug("ATR simple get_bars failed for %s: %s", symbol, fallback_exc)
                    candidate = None
                if candidate is not None:
                    cand_high, cand_low, cand_close, seq_candidate = _candidate_to_arrays(candidate)
                    if cand_high is not None and cand_low is not None and cand_close is not None:
                        high, low, close = cand_high, cand_low, cand_close
                    elif seq_candidate:
                        bars_sequence = seq_candidate
                attempted_simple_get = True
            if any(x is None for x in (high, low, close)) and bars_sequence:
                seq_high, seq_low, seq_close = _sequence_to_arrays(bars_sequence)
                if seq_high is not None and seq_low is not None and seq_close is not None:
                    high, low, close = seq_high, seq_low, seq_close
            if any(x is None for x in (high, low, close)):
                try:
                    from ai_trading.data.providers import yfinance_provider  # local import

                    provider_cls = getattr(yfinance_provider, "Provider", None)
                except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover - optional dependency missing
                    provider_cls = None
                else:
                    if provider_cls is not None and isinstance(client, provider_cls):
                        yf_module = yfinance_provider.get_yfinance()
                        if yf_module is not None and hasattr(yf_module, "Ticker"):
                            try:
                                yf_ticker = yf_module.Ticker(symbol)
                                base_days = max(int(math.ceil(lookback)), 1)
                                period_days = max(base_days + 10, base_days + 1, 2)
                                yf_df = yf_ticker.history(period=f"{period_days}d", interval="1d")
                            except _YF_HISTORY_EXCEPTIONS as exc:  # pragma: no cover - defensive network guard
                                logger.debug("ATR yfinance fallback failed for %s: %s", symbol, exc)
                            else:
                                if yf_df is not None and getattr(yf_df, "empty", True) is False:
                                    high, low, close = _extract_arrays(yf_df)
                data = None
                if ctx is not None:
                    data = getattr(ctx, "minute_data", {}).get(symbol)
                    if data is None:
                        data = getattr(ctx, "daily_data", {}).get(symbol)
                df_fallback = None
                if data is not None:
                    df_fallback = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
                if df_fallback is None:
                    fetcher = getattr(ctx, "data_fetcher", None)
                    if fetcher is not None and hasattr(fetcher, "get_daily_df"):
                        try:
                            df_fallback = fetcher.get_daily_df(ctx, symbol)
                        except (
                            APIError,
                            TimeoutError,
                            ConnectionError,
                            ValueError,
                            TypeError,
                            OSError,
                            AttributeError,
                        ) as exc:  # pragma: no cover - defensive
                            logger.debug("ATR fetcher fallback failed for %s: %s", symbol, exc)
                            df_fallback = None
                if df_fallback is not None and getattr(df_fallback, "empty", False) is False:
                    high, low, close = _extract_arrays(df_fallback)
            if any(x is None for x in (high, low, close)):
                logger.warning("Insufficient OHLC data for ATR calculation for %s", symbol)
                return None
            min_len = min(len(high), len(low), len(close))
            if min_len == 0:
                return None
            high = high[-min_len:]
            low = low[-min_len:]
            close = close[-min_len:]
            if len(high) < lookback or len(low) < lookback or len(close) < lookback:
                return None
            window_start = len(high) - lookback
            window_high = high[window_start:]
            window_low = low[window_start:]
            window_close = close[window_start:]
            if window_start > 0:
                prev_close = close[window_start - 1 : window_start - 1 + lookback]
            else:
                prev_close = np.concatenate(([window_close[0]], window_close[:-1])) if lookback > 1 else np.array([window_close[0]])
            tr_high_low = np.abs(window_high - window_low)
            tr_high_prev = np.abs(window_high - prev_close)
            tr_low_prev = np.abs(window_low - prev_close)
            tr = np.maximum.reduce([tr_high_low, tr_high_prev, tr_low_prev])
            atr = float(np.mean(tr))
            self._atr_cache[symbol] = (datetime.now(UTC), atr)
            return atr
        except (APIError, ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.warning("ATR calculation error for %s: %s", symbol, exc, extra={"cause": exc.__class__.__name__})
            return None

    def _adaptive_global_cap(self) -> float:
        base_cap = self.global_limit
        volatility_lookback_days = getattr(self.config, "volatility_lookback_days", 10)
        getattr(self.config, "exposure_cap_conservative", 1.0)
        if len(self._returns) < 3:
            return base_cap
        recent_returns = np.array(self._returns[-volatility_lookback_days:])
        mean_return = np.mean(recent_returns)
        vol = np.std(recent_returns) if np.std(recent_returns) > 0 else 0.01
        sharpe_proxy = mean_return / vol
        cumulative = np.cumprod(1 + recent_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        if sharpe_proxy > 0.5 and max_dd < 0.05:
            multiplier = min(1.2, 1 + sharpe_proxy * 0.3)
        elif sharpe_proxy < -0.3 or max_dd > 0.1:
            multiplier = max(0.3, 1 - max_dd * 2)
        else:
            multiplier = 1.0
        adaptive_cap = base_cap * multiplier
        return np.clip(adaptive_cap, base_cap * 0.3, base_cap * 1.5)

    def available_exposure(self, *, cap: float | None = None) -> float:
        """Return remaining exposure capacity against the effective cap."""

        try:
            total_exposure = sum(float(value) for value in self.exposure.values() if isinstance(value, (int, float)))
        except (TypeError, ValueError):
            total_exposure = 0.0

        if cap is None:
            try:
                adaptive = float(self._adaptive_global_cap())
            except (TypeError, ValueError):
                adaptive = float(self.global_limit)
        else:
            try:
                adaptive = float(cap)
            except (TypeError, ValueError):
                adaptive = float(self.global_limit)

        effective_cap = min(float(self.global_limit), adaptive)
        available = max(0.0, effective_cap - total_exposure)
        return float(available)

    def update_portfolio_metrics(self, returns: list[float] | None = None, drawdown: float | None = None) -> None:
        if not self.enable_portfolio_features:
            return
        if returns:
            self._returns.extend(list(returns))
        if drawdown is not None:
            self._drawdowns.append(float(drawdown))
            self._check_drawdown_and_update_stop(float(drawdown))

    def refresh_positions(self, api) -> None:
        """Synchronize exposure with live positions."""
        try:
            positions = api.list_positions()
            logger.debug("Raw Alpaca positions: %s", positions)
            acct = api.get_account()
            equity = float(getattr(acct, "equity", 0) or 0)
            exposure: dict[str, float] = {}
            for p in positions:
                asset = getattr(p, "asset_class", "equity")
                qty = float(getattr(p, "qty", 0) or 0)
                price = float(getattr(p, "avg_entry_price", 0) or 0)
                weight = qty * price / equity if equity > 0 else 0.0
                exposure[asset] = exposure.get(asset, 0.0) + weight
            self.exposure = exposure
        except (AttributeError, APIError) as exc:
            logger.warning("refresh_positions failed: %s", exc, extra={"cause": exc.__class__.__name__})

    def position_exists(self, api, symbol: str) -> bool:
        """Return True if ``symbol`` exists in current Alpaca positions."""
        try:
            for p in api.list_positions():
                if getattr(p, "symbol", "") == symbol:
                    return True
        except (AttributeError, APIError) as exc:
            logger.warning("position_exists failed for %s: %s", symbol, exc, extra={"cause": exc.__class__.__name__})
        return False

    def can_trade(
        self,
        signal: TradeSignal,
        *,
        pending: float = 0.0,
        volatility: float | None = None,
        cash_ratio: float | None = None,
        returns: list[float] | None = None,
        drawdowns: list[float] | None = None,
    ) -> bool:
        if returns:
            self._returns.extend(list(returns))
        if drawdowns:
            self._drawdowns.extend(list(drawdowns))
            try:
                current_dd = float(drawdowns[-1])
            except (ValueError, TypeError, IndexError):
                current_dd = 0.0
            self._check_drawdown_and_update_stop(current_dd)
        self._maybe_lift_hard_stop()
        if self.hard_stop:
            logger.error("TRADING_HALTED_RISK_LIMIT")
            return False
        if not isinstance(signal, TradeSignal):
            logger.error("can_trade called with invalid signal type")
            return False
        asset_exp = self.exposure.get(signal.asset_class, 0.0) + max(pending, 0.0)
        asset_cap = 1.1 * self._dynamic_cap(signal.asset_class, volatility, cash_ratio)
        signal = self.apply_risk_scaling(signal, volatility=volatility, returns=returns)
        try:
            signal_weight = float(signal.weight)
        except (ValueError, TypeError) as e:
            logger.warning(
                "Invalid signal.weight value '%s' for %s, defaulting to 0.0: %s", signal.weight, signal.symbol, e
            )
            signal_weight = 0.0
        if asset_exp + signal_weight > asset_cap:
            logger.warning(
                "Exposure cap breach: symbol=%s qty=%s alloc=%.3f exposure=%.2f vs cap=%.2f",
                signal.symbol,
                getattr(signal, "qty", "n/a"),
                signal_weight,
                asset_exp + signal_weight,
                asset_cap,
            )
            if not get_env("FORCE_CONTINUE_ON_EXPOSURE", "false", cast=bool):
                return False
            logger.warning("FORCE_CONTINUE_ON_EXPOSURE enabled; overriding cap")
        strat_cap = self.strategy_limits.get(signal.strategy, self.global_limit)
        if signal_weight > strat_cap:
            logger.warning("Strategy %s weight %.2f exceeds cap %.2f", signal.strategy, signal_weight, strat_cap)
            if not get_env("FORCE_CONTINUE_ON_EXPOSURE", "false", cast=bool):
                return False
            logger.warning("FORCE_CONTINUE_ON_EXPOSURE enabled; overriding cap")
        return True

    def register_fill(self, signal: TradeSignal) -> None:
        if not isinstance(signal, TradeSignal):
            logger.error("register_fill called with invalid signal type")
            return
        prev = self.exposure.get(signal.asset_class, 0.0)
        try:
            signal_weight = float(signal.weight)
        except (ValueError, TypeError) as e:
            logger.warning(
                "Invalid signal.weight value '%s' for %s in register_fill, defaulting to 0.0: %s",
                signal.weight,
                signal.symbol,
                e,
            )
            signal_weight = 0.0
        delta = signal_weight if signal.side.lower() == "buy" else -signal_weight
        new_exposure = prev + delta
        if new_exposure < 0 and signal.side.lower() == "sell":
            logger.warning(
                "EXPOSURE_NEGATIVE_PREVENTED",
                extra={
                    "asset": signal.asset_class,
                    "symbol": getattr(signal, "symbol", "UNKNOWN"),
                    "prev": prev,
                    "delta": delta,
                    "would_be": new_exposure,
                },
            )
            new_exposure = 0.0
            delta = -prev
        self.exposure[signal.asset_class] = new_exposure
        s_prev = self.strategy_exposure.get(signal.strategy, 0.0)
        self.strategy_exposure[signal.strategy] = s_prev + delta
        logger.info(
            "EXPOSURE_UPDATED",
            extra={
                "asset": signal.asset_class,
                "prev": prev,
                "new": self.exposure[signal.asset_class],
                "side": signal.side,
                "symbol": getattr(signal, "symbol", "UNKNOWN"),
            },
        )
        import time

        self._last_update = monotonic_time()
        self._update_event.set()

    def update_position(self, symbol: str, quantity: int, side: str) -> None:
        """Update exposure for a symbol."""
        if side == "buy":
            self._positions[symbol] = self._positions.get(symbol, 0) + quantity
        else:
            self._positions[symbol] = self._positions.get(symbol, 0) - quantity

    def update_returns(self, daily_return: float) -> None:
        """Append ``daily_return`` to history for adaptive calculations."""
        self._returns.append(daily_return)
        self._returns = self._returns[-90:]

    def _check_drawdown_and_update_stop(self, current_drawdown: float) -> None:
        """
        Evaluate the latest drawdown and update the hard stop flag if the
        drawdown exceeds the threshold.  When triggered, trading is
        disabled until a cooldown period has elapsed.

        Parameters
        ----------
        current_drawdown : float
            The most recent drawdown measurement (0–1).
        """
        if current_drawdown >= self.max_drawdown_threshold and (not self.hard_stop):
            self.hard_stop = True
            import time

            self._hard_stop_until = time.time() + self.hard_stop_cooldown * 60
            logger.error(
                "HARD_STOP_TRIGGERED", extra={"drawdown": current_drawdown, "threshold": self.max_drawdown_threshold}
            )

    def _maybe_lift_hard_stop(self) -> None:
        """
        Lift the hard stop if the cooldown period has expired.  This method
        should be called before evaluating new trades.
        """
        import time

        if self.hard_stop and self._hard_stop_until is not None:
            if time.time() >= self._hard_stop_until:
                self.hard_stop = False
                self._hard_stop_until = None
                logger.info("HARD_STOP_CLEARED")

    def acquire_trade_slot(self) -> bool:
        """Thread-safe check & increment of current_trades against max_trades."""
        if self._lock is None:
            self._lock = threading.Lock()
        with self._lock:
            if self.current_trades >= self.max_trades:
                return False
            self.current_trades += 1
            return True

    def release_trade_slot(self) -> None:
        """Decrement current_trades (no-op if already zero)."""
        if self._lock is None:
            self._lock = threading.Lock()
        with self._lock:
            if self.current_trades > 0:
                self.current_trades -= 1

    def trigger_hard_stop(self) -> None:
        self.hard_stop = True

    def wait_for_exposure_update(self, timeout: float = 0.5) -> None:
        """Block until an exposure update occurs or ``timeout`` elapses."""
        self._update_event.wait(timeout)
        self._update_event.clear()

    def update_exposure(self, context=None, *args, **kwargs):
        """
        Recalculate/update exposure. Prefer the provided context.
        Backward compatible: if context is None, fall back to self.ctx (if set).
        """
        ctx = context if context is not None else getattr(self, "ctx", None)
        if ctx is None:
            raise RuntimeError("RiskEngine.update_exposure: context is required")
        try:
            self.refresh_positions(ctx.api)
            logger.debug("Exposure updated successfully")
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as exc:
            logger.warning("Failed to update exposure: %s", exc)

    def apply_risk_scaling(
        self, signal: TradeSignal, *, volatility: float | None = None, returns: Sequence[float] | None = None
    ) -> TradeSignal:
        """
        Adjust a signal's weight based on volatility and CVaR.  This function
        uses a simple inverse‑volatility rule and CVaR scaling to shrink
        exposures when markets become turbulent.  The original signal is
        mutated and returned for convenience.
        """
        try:
            scale = 1.0
            if volatility and volatility > 0:
                scale *= max(0.5, min(1.0, 0.02 / volatility))
            if returns:
                import numpy as np
                from ai_trading.capital_scaling import cvar_scaling

                arr = np.asarray(list(returns), dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    cvar_metric = cvar_scaling(arr, alpha=0.05)
                    if cvar_metric > 1.0:
                        scale *= 1.0 / (1.0 + cvar_metric)
            signal.weight = max(0.0, float(signal.weight) * scale)
            return signal
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as exc:
            logger.error("Risk scaling failed: %s", exc)
            return signal

    def check_max_drawdown(self, api) -> bool:
        try:
            account = api.get_account()
            pnl = float(account.equity) - float(account.last_equity)
            if pnl < -MAX_DRAWDOWN * float(account.last_equity):
                logger.error("HARD_STOP_MAX_DRAWDOWN", extra={"pnl": pnl})
                self.hard_stop = True
                return False
            return True
        except (RuntimeError, AttributeError, ValueError) as exc:
            logger.error("check_max_drawdown failed: %s", exc)
            return False

    def position_size(self, signal: Any, cash: float, price: float, api=None) -> int:
        """
        Calculate optimal position size using Kelly criterion and risk management.

        This is the core position sizing algorithm that combines multiple risk factors:
        - Kelly criterion for optimal bet sizing
        - ATR-based volatility scaling
        - Maximum position limits
        - Account equity validation
        """
        if self.hard_stop:
            return 0
        if not self.can_trade(signal):
            return 0
        if api and (not self.check_max_drawdown(api)):
            return 0
        if price <= 0:
            logger.warning("Invalid price %s for %s", price, getattr(signal, "symbol", "UNKNOWN"))
            return 0
        if cash <= 0:
            logger.warning("Invalid cash amount %s for %s", cash, getattr(signal, "symbol", "UNKNOWN"))
            return 0
        try:
            if api:
                account = api.get_account()
                total_equity = float(getattr(account, "equity", cash))
            else:
                total_equity = cash
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as e:
            logger.warning("Error getting account equity: %s", e)
            total_equity = cash
        if total_equity <= 0:
            logger.warning("Invalid total equity %s for %s", total_equity, getattr(signal, "symbol", "UNKNOWN"))
            return 0
        try:
            if not hasattr(signal, "symbol"):
                logger.warning("Invalid signal object missing symbol attribute")
                return 0
            atr_data = self._get_atr_data(signal.symbol)
            if atr_data and atr_data > 0:
                risk_per_trade = total_equity * 0.01
                stop_distance = atr_data * self.config.atr_multiplier
                raw_qty = risk_per_trade / stop_distance
            else:
                weight = self._apply_weight_limits(signal)
                raw_qty = total_equity * weight / price
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as exc:
            logger.warning("ATR calculation failed for %s: %s", getattr(signal, "symbol", "UNKNOWN"), exc)
            try:
                weight = self._apply_weight_limits(signal)
                raw_qty = total_equity * weight / price
            except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError):
                logger.warning("Failed to calculate position size, returning 0")
                return 0
        try:
            qty = _calculate_position_size(self, raw_qty, price, signal)
            if getattr(signal, "strategy", "") == "default":
                qty = max(qty, 10)
            return max(qty, 0)
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as exc:
            logger.warning("Error calculating final quantity: %s", exc)
            return 0

    def _apply_weight_limits(self, sig: TradeSignal) -> float:
        """Apply confidence-based weight limits considering current exposure."""
        try:
            if (
                not hasattr(sig, "asset_class")
                or not hasattr(sig, "strategy")
                or (not hasattr(sig, "weight"))
                or (not hasattr(sig, "confidence"))
            ):
                logger.warning("Invalid signal object missing required attributes")
                return 0.0
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError):
            logger.warning("Error validating signal object")
            return 0.0
        current_asset_exposure = self.exposure.get(sig.asset_class, 0.0)
        current_strategy_exposure = self.strategy_exposure.get(sig.strategy, 0.0)
        asset_limit = self.asset_limits.get(sig.asset_class, self.global_limit)
        strategy_limit = self.strategy_limits.get(sig.strategy, self.global_limit)
        available_asset_capacity = max(0.0, float(asset_limit) - float(current_asset_exposure))
        available_strategy_capacity = max(0.0, float(strategy_limit) - float(current_strategy_exposure))
        available_global_capacity = self.available_exposure()
        max_allowed = min(
            available_asset_capacity,
            available_strategy_capacity,
            available_global_capacity,
        )
        try:
            requested_weight = float(sig.weight)
        except (ValueError, TypeError) as e:
            logger.warning(
                "Invalid signal.weight value '%s' for %s in _apply_weight_limits, defaulting to 0.0: %s",
                sig.weight,
                sig.symbol,
                e,
            )
            requested_weight = 0.0
        base_weight = min(requested_weight, max_allowed)
        return round(base_weight, 1)

    def compute_volatility(self, returns: np.ndarray) -> dict:
        """Return multiple volatility estimates."""
        if len(returns) == 0:
            return {"volatility": 0.0, "mad": 0.0, "garch_vol": 0.0}
        try:
            returns_array = np.asarray(returns)
            has_invalid = False
            try:
                if hasattr(np, "any") and hasattr(np, "isnan") and hasattr(np, "isinf"):
                    has_invalid = np.any(np.isnan(returns_array)) or np.any(np.isinf(returns_array))
                else:
                    for val in returns_array:
                        if str(val).lower() in ["nan", "inf", "-inf"]:
                            has_invalid = True
                            break
            except (AttributeError, TypeError):
                has_invalid = any((str(val).lower() in ["nan", "inf", "-inf"] for val in returns_array))
            if has_invalid:
                logger.error("compute_volatility: invalid values in returns array")
                return {"volatility": 0.0, "mad": 0.0, "garch_vol": 0.0}
            try:
                std_vol = float(np.std(returns_array))
            except (AttributeError, TypeError):
                mean_val = sum(returns_array) / len(returns_array)
                variance = sum(((x - mean_val) ** 2 for x in returns_array)) / len(returns_array)
                std_vol = variance**0.5
            try:
                if hasattr(np, "median") and hasattr(np, "abs"):
                    mad = float(np.median(np.abs(returns_array - np.median(returns_array))))
                else:
                    sorted_returns = sorted(returns_array)
                    n = len(sorted_returns)
                    median_val = (
                        sorted_returns[n // 2]
                        if n % 2 == 1
                        else (sorted_returns[n // 2 - 1] + sorted_returns[n // 2]) / 2
                    )
                    abs_deviations = [abs(x - median_val) for x in returns_array]
                    sorted_abs_dev = sorted(abs_deviations)
                    mad = (
                        sorted_abs_dev[len(sorted_abs_dev) // 2]
                        if len(sorted_abs_dev) % 2 == 1
                        else (sorted_abs_dev[len(sorted_abs_dev) // 2 - 1] + sorted_abs_dev[len(sorted_abs_dev) // 2])
                        / 2
                    )
            except (AttributeError, TypeError):
                mad = std_vol
        except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
            logger.error("compute_volatility: error in numpy operations: %s", exc)
            return {"volatility": 0.0, "mad": 0.0, "garch_vol": 0.0}
        try:
            alpha, beta = (0.1, 0.85)
            garch_vol = 0.0
            for i in range(1, len(returns_array)):
                garch_vol = alpha * returns_array[i - 1] ** 2 + beta * garch_vol
            try:
                garch_vol = float(np.sqrt(garch_vol))
            except (AttributeError, TypeError):
                garch_vol = float(garch_vol**0.5)
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError):
            garch_vol = std_vol
        primary_vol = std_vol
        return {
            "volatility": primary_vol,
            "std_vol": std_vol,
            "mad": mad,
            "mad_scaled": mad * 1.4826,
            "garch_vol": garch_vol,
        }

    def get_current_exposure(self) -> dict[str, float]:
        """
        Get current portfolio exposure by asset class.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping asset classes to exposure percentages.
            Values represent the portion of total equity allocated to each asset class.
        """
        return self.exposure.copy()

    def max_concurrent_orders(self) -> int:
        """
        Get maximum number of concurrent orders allowed.

        Returns
        -------
        int
            Maximum number of orders that can be active simultaneously.
            Prevents overwhelming the broker with too many pending orders.
        """
        return getattr(self.config, "max_concurrent_orders", 50)

    def max_exposure(self) -> float:
        """
        Get maximum total portfolio exposure limit.

        Returns
        -------
        float
            Maximum portfolio exposure as a fraction (0.0 to 1.0).
            Represents the maximum percentage of equity that can be at risk.
        """
        return self.global_limit

    def order_spacing(self) -> float:
        """
        Get minimum time spacing between orders in seconds.

        Returns
        -------
        float
            Minimum seconds to wait between submitting orders.
            Prevents rapid-fire order submission that could trigger rate limits.
        """
        return getattr(self.config, "order_spacing_seconds", 1.0)

    def check_position_limits(self, symbol: str, quantity: float) -> bool:
        """
        Check if a proposed position would exceed risk limits.

        Parameters
        ----------
        symbol : str
            Trading symbol to check limits for.
        quantity : float
            Proposed position size (positive for long, negative for short).

        Returns
        -------
        bool
            True if position is within limits, False if it would exceed limits.
        """
        try:
            current_exposure = self.exposure.get(symbol, 0.0)
            new_exposure = current_exposure + abs(quantity) * 0.001
            max_symbol_exposure = getattr(self.config, "max_symbol_exposure", 0.1)
            if new_exposure > max_symbol_exposure:
                logger.warning(
                    "Position for %s would exceed symbol exposure limit: %.3f > %.3f",
                    symbol,
                    new_exposure,
                    max_symbol_exposure,
                )
                return False
            total_exposure = sum(self.exposure.values()) + abs(quantity) * 0.001
            if total_exposure > self.global_limit:
                logger.warning(
                    "Position for %s would exceed total exposure limit: %.3f > %.3f",
                    symbol,
                    total_exposure,
                    self.global_limit,
                )
                return False
            return True
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as e:
            logger.error("Error checking position limits for %s: %s", symbol, e)
            return False

    def validate_order_size(self, symbol: str, quantity: float, price: float) -> bool:
        """
        Validate that an order size is appropriate for risk management.

        Parameters
        ----------
        symbol : str
            Trading symbol for the order.
        quantity : float
            Order quantity (shares).
        price : float
            Order price per share.

        Returns
        -------
        bool
            True if order size is valid, False if it should be rejected.
        """
        try:
            order_value = abs(quantity) * price
            min_order_value = getattr(self.config, "min_order_value", 100.0)
            if order_value < min_order_value:
                logger.warning("Order for %s below minimum value: $%.2f < $%.2f", symbol, order_value, min_order_value)
                return False
            max_order_value = getattr(self.config, "max_order_value", 50000.0)
            if order_value > max_order_value:
                logger.warning(
                    "Order for %s exceeds maximum value: $%.2f > $%.2f", symbol, order_value, max_order_value
                )
                return False
            if abs(quantity) < 1:
                logger.warning("Order quantity too small: %s shares", quantity)
                return False
            if abs(quantity) > 10000:
                logger.warning("Order quantity unusually large: %s shares", quantity)
            return True
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as e:
            logger.error("Error validating order size for %s: %s", symbol, e)
            return False


def dynamic_position_size(capital: float, volatility: float, drawdown: float) -> float:
    """Return position size using volatility and drawdown aware Kelly fraction.

    The base Kelly fraction of ``0.5 / volatility`` is throttled by current
    drawdown. When drawdown exceeds 10% the fraction is scaled down by 50%.
    """
    if capital <= 0:
        return 0.0
    vol = max(volatility, 1e-06)
    kelly_fraction = 0.5 / vol
    kelly_fraction = min(max(kelly_fraction, 0.0), 1.0)
    if drawdown > 0.1:
        kelly_fraction *= 0.5
    return capital * kelly_fraction


def calculate_position_size(*args, **kwargs) -> int:
    """
    Calculate optimal position size using Kelly criterion and risk management.

    This convenience wrapper function supports multiple calling patterns for
    calculating position sizes based on available capital, signal confidence,
    and risk parameters. It integrates Kelly criterion optimization with
    volatility-based risk scaling.

    Parameters
    ----------
    *args : tuple
        Variable arguments supporting multiple calling patterns:

        Pattern 1 (Simple): calculate_position_size(cash, price)
        - cash (float): Available trading capital
        - price (float): Current asset price per share

        Pattern 2 (Advanced): calculate_position_size(signal, cash, price, api=None)
        - signal (TradeSignal): Signal object with confidence and strategy info
        - cash (float): Available trading capital
        - price (float): Current asset price per share
        - api (optional): Broker API client for additional validation

    **kwargs : dict
        Optional keyword arguments:
        - api: Broker API client for account validation
        - max_position_pct (float): Maximum position as % of capital (default: 5%)
        - volatility_scaling (bool): Enable volatility-based sizing (default: True)

    Returns
    -------
    int
        Optimal number of shares to trade, considering:
        - Kelly criterion optimization for signal confidence
        - Risk-adjusted position sizing based on volatility
        - Maximum position limits and portfolio constraints
        - Available capital and margin requirements

    Raises
    ------
    TypeError
        If invalid argument patterns are provided
    ValueError
        If negative or invalid cash/price values are passed

    Examples
    --------
    >>> # Simple position sizing
    >>> shares = calculate_position_size(10000, 150.0)  # $10k capital, $150/share
    >>> logging.info(f"Buy {shares} shares")

    >>> # Advanced position sizing with signal
    >>> from ai_trading.strategies.base import StrategySignal as TradeSignal
    >>> signal = TradeSignal(symbol='AAPL', side='buy', confidence=0.8, strategy='momentum')
    >>> shares = calculate_position_size(signal, 10000, 150.0)
    >>> logging.info(f"Buy {shares} shares based on {signal.confidence:.1%} confidence")

    Notes
    -----
    - Returns 0 if insufficient capital or invalid parameters
    - Automatically applies risk management limits
    - Considers portfolio heat and correlation limits
    - Scales position size based on signal confidence
    """
    engine = RiskEngine(get_trading_config())
    if len(args) == 2 and (not kwargs):
        cash, price = args
        if not isinstance(cash, int | float) or cash <= 0:
            logger.warning(f"Invalid cash amount: {cash}")
            return 0
        if not isinstance(price, int | float) or price <= 0:
            logger.warning(f"Invalid price: {price}")
            return 0
        dummy = TradeSignal(symbol="DUMMY", side="buy", confidence=1.0, strategy="default")
        return engine.position_size(dummy, cash, price)
    if len(args) >= 3:
        signal, cash, price = args[:3]
        if not isinstance(cash, int | float) or cash <= 0:
            logger.warning(f"Invalid cash amount: {cash}")
            return 0
        if not isinstance(price, int | float) or price <= 0:
            logger.warning(f"Invalid price: {price}")
            return 0
        if not hasattr(signal, "confidence") or not hasattr(signal, "symbol"):
            logger.error(f"Invalid signal object: {type(signal)}")
            return 0
        api = args[3] if len(args) > 3 else kwargs.get("api")
        return engine.position_size(signal, cash, price, api)
    raise TypeError("Invalid arguments for calculate_position_size. Expected (cash, price) or (signal, cash, price)")


def check_max_drawdown(state: dict[str, float]) -> bool:
    """
    Validate if current portfolio drawdown exceeds maximum allowed threshold.

    This function checks portfolio performance against configured drawdown limits
    to implement risk management controls. When drawdown exceeds the threshold,
    trading may be halted or position sizes reduced.

    Parameters
    ----------
    state : Dict[str, float]
        Portfolio state dictionary containing:
        - 'current_drawdown' (float): Current drawdown as decimal (e.g., 0.05 = 5%)
        - 'max_drawdown' (float): Maximum allowed drawdown threshold
        - 'portfolio_value' (float): Current portfolio value (optional)
        - 'peak_value' (float): Historical peak portfolio value (optional)

    Returns
    -------
    bool
        True if current drawdown exceeds maximum allowed threshold,
        False if within acceptable limits or if data is insufficient.

    Notes
    -----
    - Returns False for missing or invalid state data
    - Drawdown values should be positive decimals (0.05 = 5%)
    - Used for automated risk management decisions
    """
    if not isinstance(state, dict):
        logger.warning(f"Invalid state type: {type(state)}")
        return False
    current_dd = state.get("current_drawdown", 0)
    max_dd = state.get("max_drawdown", 0)
    if not isinstance(current_dd, int | float) or current_dd < 0:
        logger.warning(f"Invalid current_drawdown: {current_dd}")
        return False
    if not isinstance(max_dd, int | float) or max_dd <= 0:
        logger.warning(f"Invalid max_drawdown: {max_dd}")
        return False
    return current_dd > max_dd


def can_trade(engine: RiskEngine) -> bool:
    """Return True if trading should proceed based on engine state."""
    return not engine.hard_stop and engine.current_trades < engine.max_trades


def register_trade(engine: RiskEngine, size: int) -> dict | None:
    """Register a trade and increment the count if allowed."""
    if size <= 0 or not engine.acquire_trade_slot():
        return None
    return {"size": size}


def check_exposure_caps(portfolio, exposure, cap):
    for sym, pos in portfolio.positions.items():
        if pos.quantity > 0 and exposure[sym] > cap:
            logger.warning("Exposure cap triggered, blocking new orders for %s", sym)
            return False


def apply_trailing_atr_stop(
    df: pd.DataFrame, entry_price: float, *, context: Any | None = None, symbol: str = "SYMBOL", qty: int | None = None
) -> None:
    """Exit ``qty`` at market if the trailing stop is triggered."""
    try:
        if entry_price <= 0:
            logger.warning("apply_trailing_atr_stop invalid entry price: %.2f", entry_price)
            return
        ta_mod = load_pandas_ta()
        if ta_mod is not None and hasattr(df, "ta"):
            atr = df.ta.atr()
        else:
            from ai_trading.indicators import atr as _atr

            atr = _atr(df["High"], df["Low"], df["Close"])
        trailing_stop = entry_price - 2 * atr
        last_valid_close = df["Close"].dropna()
        if not last_valid_close.empty:
            price = last_valid_close.iloc[-1]
        else:
            logger.critical("All NaNs in close column for ATR stop")
            price = 0.0
        logger.debug("Latest 5 rows for ATR stop:\n%s", df.tail(5))
        logger.debug("Computed price for ATR stop: %s", price)
        if price <= 0 or pd.isna(price):
            logger.critical("Invalid price computed for ATR stop: %s", price)
            return
        if price < trailing_stop.iloc[-1]:
            logger.info("ATR stop hit: price=%s vs stop=%s", price, trailing_stop.iloc[-1])
            if context is not None and qty:
                try:
                    if hasattr(context, "risk_engine") and (
                        not context.risk_engine.position_exists(context.api, symbol)
                    ):
                        logger.info("No position to sell for %s, skipping.", symbol)
                        return
                    bot_mod = importlib.import_module("ai_trading.core.bot_engine")
                    bot_mod.send_exit_order(context, symbol, abs(int(qty)), price, "atr_stop")
                except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as exc:
                    logger.error("ATR stop exit failed: %s", exc)
            else:
                logger.warning("ATR stop triggered but no context/qty provided")
            schedule_reentry_check(symbol, lookahead_days=2)
    except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as e:
        logger.error("ATR stop error: %s", e)


def schedule_reentry_check(symbol: str, lookahead_days: int) -> None:
    """Log a re-entry check after a stop out."""
    logger.info("Scheduling reentry for %s in %s days", symbol, lookahead_days)


def calculate_atr_stop(entry_price: float, atr: float, multiplier: float = 1.5, direction: str = "long") -> float:
    """Return ATR-based stop price."""
    stop = entry_price - multiplier * atr if direction == "long" else entry_price + multiplier * atr
    from ai_trading.telemetry import metrics_logger

    _safe_call(metrics_logger.log_atr_stop, symbol="generic", stop=stop)
    return stop


def calculate_bollinger_stop(price: float, upper_band: float, lower_band: float, direction: str = "long") -> float:
    """Return stop price using Bollinger band width."""
    mid = (upper_band + lower_band) / 2
    if direction == "long":
        stop = min(price, mid)
    else:
        stop = max(price, mid)
    from ai_trading.telemetry import metrics_logger

    _safe_call(metrics_logger.log_atr_stop, symbol="bb", stop=stop)
    return stop


def dynamic_stop_price(
    entry_price: float,
    atr: float | None = None,
    upper_band: float | None = None,
    lower_band: float | None = None,
    percent: float | None = None,
    direction: str = "long",
) -> float:
    """Return the tightest stop price based on ATR, Bollinger width or percent."""
    stops: list[float] = []
    if atr is not None:
        stops.append(calculate_atr_stop(entry_price, atr, direction=direction))
    if upper_band is not None and lower_band is not None:
        stops.append(calculate_bollinger_stop(entry_price, upper_band, lower_band, direction=direction))
    if percent is not None:
        pct_stop = entry_price * (1 - percent) if direction == "long" else entry_price * (1 + percent)
        stops.append(pct_stop)
    if not stops:
        return entry_price
    return max(stops) if direction == "long" else min(stops)


def compute_stop_levels(entry_price: float, atr: float, take_mult: float = 2.0) -> tuple[float, float]:
    """Return stop-loss and take-profit levels using ATR."""
    stop = entry_price - atr
    take = entry_price + take_mult * atr
    return (stop, take)


def correlation_position_weights(corr: pd.DataFrame, base: dict[str, float]) -> dict[str, float]:
    """Scale weights inversely proportional to asset correlations."""
    weights = {}
    for sym, w in base.items():
        if sym in corr.columns:
            c = corr[sym].abs().mean()
            scale = 1.0 / (1.0 + c)
            weights[sym] = w * scale
        else:
            weights[sym] = w
    return weights


def drawdown_circuit(drawdowns: Sequence[float], limit: float = 0.2) -> bool:
    """Return True if cumulative drawdown exceeds ``limit``."""
    dd = abs(min(0.0, *drawdowns)) if drawdowns else 0.0
    return dd > limit


def volatility_filter(atr: float, sma: float, threshold: float = 0.05) -> bool:
    """Return True when volatility below ``threshold`` relative to SMA."""
    if sma == 0:
        return True
    return atr / sma < threshold
