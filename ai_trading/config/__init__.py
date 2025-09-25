"""Configuration package exposing typed runtime settings."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
import threading
from dataclasses import dataclass
from typing import Sequence

from ai_trading.util.env_check import assert_dotenv_not_shadowed

assert_dotenv_not_shadowed()

from .runtime import (
    TradingConfig,
    CONFIG_SPECS,
    MODE_PARAMETERS,
    get_trading_config,
    reload_trading_config,
    generate_config_schema,
)
from .management import (
    get_env,
    reload_env,
    is_shadow_mode,
    validate_required_env,
    validate_alpaca_credentials,
    _resolve_alpaca_env,
    SEED,
    MAX_EMPTY_RETRIES,
)
from .settings import (
    Settings,
    get_settings,
    broker_keys,
    provider_priority,
    max_data_fallbacks,
    minute_data_freshness_tolerance,
    alpaca_feed_failover,
    alpaca_empty_to_backup,
)
from .alpaca import AlpacaConfig, get_alpaca_config
from ai_trading.validation.require_env import (
    _require_env_vars,
    require_env_vars,
)

_CFG = get_trading_config()
_CONFIG_LOGGED = False

_LOCK_TIMEOUT = int(os.getenv("CONFIG_LOCK_TIMEOUT", "30"))
_VALIDATION_LOCK = threading.Lock()
_LOCK_STATE = threading.local()

logger = logging.getLogger(__name__)


def _is_lock_held_by_current_thread() -> bool:
    """Return True when the calling thread currently holds the validation lock."""

    return bool(getattr(_LOCK_STATE, "held", False))


def _set_lock_held_by_current_thread(held: bool) -> None:
    """Record whether the calling thread owns the validation lock."""

    _LOCK_STATE.held = bool(held)


@contextmanager
def _validation_lock() -> Iterator[None]:
    """Acquire the validation lock with timeout handling.

    Re-entrant usage within the same thread is permitted to avoid the
    historical deadlock where ``validate_env_vars`` invoked
    ``validate_environment`` while already holding the lock.
    """

    if _is_lock_held_by_current_thread():
        yield
        return

    acquired = _VALIDATION_LOCK.acquire(timeout=_LOCK_TIMEOUT)
    if not acquired:
        raise TimeoutError("CONFIG_VALIDATION_LOCK_TIMEOUT")

    try:
        _set_lock_held_by_current_thread(True)
        yield
    finally:
        _set_lock_held_by_current_thread(False)
        _VALIDATION_LOCK.release()

SENTIMENT_API_KEY = _CFG.sentiment_api_key
SENTIMENT_API_URL = _CFG.sentiment_api_url
SENTIMENT_ENHANCED_CACHING = bool(_CFG.sentiment_enhanced_caching)
SENTIMENT_FALLBACK_SOURCES = tuple(_CFG.sentiment_fallback_sources)
SENTIMENT_SUCCESS_RATE_TARGET = float(_CFG.sentiment_success_rate_target)
SENTIMENT_RECOVERY_TIMEOUT_SECS = int(_CFG.sentiment_recovery_timeout_secs)

META_LEARNING_BOOTSTRAP_ENABLED = bool(_CFG.meta_learning_bootstrap_enabled)
META_LEARNING_BOOTSTRAP_WIN_RATE = float(_CFG.meta_learning_bootstrap_win_rate)
META_LEARNING_MIN_TRADES_REDUCED = int(_CFG.meta_learning_min_trades_reduced)

ORDER_TIMEOUT_SECONDS = int(_CFG.order_timeout_seconds)
ORDER_FILL_RATE_TARGET = float(_CFG.order_fill_rate_target)
LIQUIDITY_SPREAD_THRESHOLD = float(_CFG.liquidity_spread_threshold)
LIQUIDITY_VOL_THRESHOLD = float(_CFG.liquidity_vol_threshold)
LIQUIDITY_REDUCTION_AGGRESSIVE = float(_CFG.liquidity_reduction_aggressive)
LIQUIDITY_REDUCTION_MODERATE = float(_CFG.liquidity_reduction_moderate)
ORDER_STALE_CLEANUP_INTERVAL = int(_CFG.order_stale_cleanup_interval)
EXECUTION_MODE = str(getattr(_CFG, "execution_mode", "sim") or "sim").lower()
SHADOW_MODE = bool(getattr(_CFG, "shadow_mode", False))
DATA_FEED_INTRADAY = str(getattr(_CFG, "data_feed_intraday", getattr(_CFG, "alpaca_data_feed", "iex")) or "iex").lower()
SLIPPAGE_LIMIT_BPS = int(getattr(_CFG, "slippage_limit_bps", getattr(_CFG, "max_slippage_bps", 75)))
PRICE_PROVIDER_ORDER = tuple(getattr(_CFG, "price_provider_order", (
    "alpaca_quote",
    "alpaca_trade",
    "alpaca_minute_close",
    "yahoo",
    "bars",
)))


def _env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return None


TRADING_MODE = (
    os.getenv("TRADING_MODE")
    or os.getenv("AI_TRADING_TRADING_MODE")
    or "balanced"
).lower()
if TRADING_MODE not in MODE_PARAMETERS:
    TRADING_MODE = "balanced"


def _cfg_float(field: str, fallback: float) -> float:
    value = getattr(_CFG, field, None)
    if value in (None, ""):
        return float(fallback)
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        return float(fallback)


def _mode_default(field: str, fallback: float) -> float:
    defaults = MODE_PARAMETERS.get(TRADING_MODE, MODE_PARAMETERS["balanced"])
    return float(defaults.get(field, fallback))


CONF_THRESHOLD = _cfg_float("conf_threshold", _mode_default("conf_threshold", 0.75))
MAX_POSITION_SIZE = _cfg_float("max_position_size", _mode_default("max_position_size", 8000.0))
CAPITAL_CAP = _cfg_float("capital_cap", _mode_default("capital_cap", 0.25))
DOLLAR_RISK_LIMIT = _cfg_float("dollar_risk_limit", 0.05)

ALPACA_API_KEY = _env_value("ALPACA_API_KEY") or ""
ALPACA_SECRET_KEY = _env_value("ALPACA_SECRET_KEY") or ""
ALPACA_BASE_URL = _env_value("ALPACA_API_URL", "ALPACA_BASE_URL") or ""


def derive_cap_from_settings(
    settings: Settings | None = None,
    *,
    equity: float | None = None,
    fallback: float = 8000.0,
    capital_cap: float | None = None,
) -> float:
    """Calculate maximum capital allocation based on settings and equity."""

    s = settings or get_settings()
    cap = capital_cap if capital_cap is not None else float(getattr(s, "capital_cap", 0.25))
    if equity and equity > 0:
        return float(equity) * cap
    return float(fallback)


@dataclass(frozen=True)
class ExecutionSettingsSnapshot:
    """Lightweight snapshot of execution-related configuration."""

    mode: str
    shadow_mode: bool
    order_timeout_seconds: int
    slippage_limit_bps: int
    price_provider_order: tuple[str, ...]
    data_feed_intraday: str


def get_execution_settings() -> ExecutionSettingsSnapshot:
    """Return a cached snapshot of live execution configuration."""

    cfg = get_trading_config()
    provider_order = tuple(getattr(cfg, "price_provider_order", ()) or PRICE_PROVIDER_ORDER)
    return ExecutionSettingsSnapshot(
        mode=str(getattr(cfg, "execution_mode", EXECUTION_MODE) or EXECUTION_MODE).lower(),
        shadow_mode=bool(getattr(cfg, "shadow_mode", SHADOW_MODE)),
        order_timeout_seconds=int(getattr(cfg, "order_timeout_seconds", ORDER_TIMEOUT_SECONDS)),
        slippage_limit_bps=int(getattr(cfg, "slippage_limit_bps", SLIPPAGE_LIMIT_BPS)),
        price_provider_order=provider_order,
        data_feed_intraday=str(
            getattr(cfg, "data_feed_intraday", DATA_FEED_INTRADAY) or DATA_FEED_INTRADAY
        ).lower(),
    )


def validate_environment() -> None:
    """Validate required environment variables are present."""

    with _validation_lock():
        logger.debug("CONFIG_ENV_VALIDATION_START")
        reload_trading_config(allow_missing_drawdown=False)
        if _env_value("MAX_DRAWDOWN_THRESHOLD", "AI_TRADING_MAX_DRAWDOWN_THRESHOLD") is None:
            logger.error("CONFIG_ENV_MISSING_DRAWDOWN")
            raise RuntimeError("MAX_DRAWDOWN_THRESHOLD must be set")
        snapshot = {k: v for k, v in os.environ.items() if isinstance(v, str)}
        try:
            validate_required_env(env=snapshot)
        except Exception as exc:
            logger.exception("CONFIG_ENV_VALIDATION_FAILED")
            raise RuntimeError(f"Configuration validation failed: {exc}") from exc
        logger.debug("CONFIG_ENV_VALIDATION_SUCCESS")


def validate_env_vars(*names: str) -> None:
    """Ensure specific environment variables are defined."""

    with _validation_lock():
        reload_trading_config()
        if names:
            _require_env_vars(*names)
        validate_required_env()


def get_max_drawdown_threshold() -> float:
    raw = _env_value("MAX_DRAWDOWN_THRESHOLD", "AI_TRADING_MAX_DRAWDOWN_THRESHOLD")
    if raw is None:
        raise RuntimeError("MAX_DRAWDOWN_THRESHOLD must be set")
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - configuration error
        raise RuntimeError(f"Invalid MAX_DRAWDOWN_THRESHOLD value: {raw}") from exc


def validate_alpaca_credentials() -> None:
    missing = [
        name
        for name, value in {
            "ALPACA_API_KEY": ALPACA_API_KEY,
            "ALPACA_SECRET_KEY": ALPACA_SECRET_KEY,
            "ALPACA_API_URL": ALPACA_BASE_URL,
        }.items()
        if value in (None, "")
    ]
    if missing:
        raise RuntimeError(f"Missing required Alpaca credentials: {', '.join(missing)}")
    validate_required_env(("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_API_URL"))


def log_config(mask_fields: Sequence[str] | None = None) -> None:
    """Mark configuration as logged without emitting secrets."""

    global _CONFIG_LOGGED
    if _CONFIG_LOGGED:
        return
    _CONFIG_LOGGED = True


__all__ = sorted([
    "AlpacaConfig",
    "ALPACA_API_KEY",
    "ALPACA_BASE_URL",
    "ALPACA_SECRET_KEY",
    "CAPITAL_CAP",
    "CONF_THRESHOLD",
    "CONFIG_SPECS",
    "DATA_FEED_INTRADAY",
    "DOLLAR_RISK_LIMIT",
    "EXECUTION_MODE",
    "ExecutionSettingsSnapshot",
    "META_LEARNING_BOOTSTRAP_ENABLED",
    "META_LEARNING_BOOTSTRAP_WIN_RATE",
    "META_LEARNING_MIN_TRADES_REDUCED",
    "MODE_PARAMETERS",
    "MAX_POSITION_SIZE",
    "ORDER_FILL_RATE_TARGET",
    "ORDER_TIMEOUT_SECONDS",
    "ORDER_STALE_CLEANUP_INTERVAL",
    "PRICE_PROVIDER_ORDER",
    "SENTIMENT_API_KEY",
    "SENTIMENT_API_URL",
    "SENTIMENT_ENHANCED_CACHING",
    "SENTIMENT_FALLBACK_SOURCES",
    "SENTIMENT_RECOVERY_TIMEOUT_SECS",
    "SENTIMENT_SUCCESS_RATE_TARGET",
    "Settings",
    "SHADOW_MODE",
    "SLIPPAGE_LIMIT_BPS",
    "TradingConfig",
    "TRADING_MODE",
    "_require_env_vars",
    "broker_keys",
    "derive_cap_from_settings",
    "generate_config_schema",
    "get_alpaca_config",
    "get_env",
    "get_execution_settings",
    "get_max_drawdown_threshold",
    "get_settings",
    "get_trading_config",
    "log_config",
    "reload_env",
    "reload_trading_config",
    "require_env_vars",
    "validate_alpaca_credentials",
    "validate_env_vars",
    "validate_environment",
])
