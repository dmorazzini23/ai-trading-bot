"""Configuration package exposing typed runtime settings."""

from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import logging
from collections.abc import Iterator
from contextlib import contextmanager
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence, cast

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
    merged_env_snapshot,
    reload_env as _management_reload_env,
    is_shadow_mode,
    validate_required_env,
    validate_alpaca_credentials as _validate_alpaca_credentials_from_management,
    _resolve_alpaca_env,
    SEED,
    MAX_EMPTY_RETRIES,
)
from . import management as management
from .settings import (
    Settings,
    get_settings as _settings_get_settings,
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
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}

_LOCK_TIMEOUT = int(get_env("CONFIG_LOCK_TIMEOUT", 30, cast=int, resolve_aliases=False))
_VALIDATION_LOCK = threading.Lock()
_LOCK_STATE = threading.local()

logger = logging.getLogger(__name__)


_CACHED_SETTINGS: Settings | None = None


def _reset_cached_settings() -> None:
    """Clear this module's cached settings reference."""

    global _CACHED_SETTINGS
    _CACHED_SETTINGS = None
    clear_cache = getattr(_settings_get_settings, "cache_clear", None)
    if callable(clear_cache):
        clear_cache()


def _env_is_truthy(name: str) -> bool:
    value = str(get_env(name, "", cast=str, resolve_aliases=False) or "").strip().lower()
    return value in _TRUTHY_ENV_VALUES


def get_settings() -> Settings:
    """Return the cached Settings object."""

    global _CACHED_SETTINGS
    if _CACHED_SETTINGS is not None:
        return _CACHED_SETTINGS
    settings_obj = _settings_get_settings()
    maybe_settings: Any = settings_obj
    if maybe_settings is None:
        raise RuntimeError("settings unavailable")
    _CACHED_SETTINGS = settings_obj
    return settings_obj


def safe_settings() -> Settings:
    """Return settings with conservative defaults when unavailable."""

    try:
        settings_obj = get_settings()
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        settings_obj = None
    if settings_obj is None:
        pytest_running = _env_is_truthy("PYTEST_RUNNING")
        return cast(Settings, SimpleNamespace(
            ENABLE_PORTFOLIO_FEATURES=False,
            enable_memory_optimization=False,
            env="test" if pytest_running else "dev",
            api_port=9001,
            alpaca_data_feed=str(
                get_env("ALPACA_DATA_FEED", "iex", cast=str, resolve_aliases=False) or "iex"
            ),
            alpaca_execution_feed=str(
                get_env("ALPACA_EXECUTION_FEED", "iex", cast=str, resolve_aliases=False) or "iex"
            ),
            alpaca_reference_feed=str(
                get_env("ALPACA_REFERENCE_FEED", "delayed_sip", cast=str, resolve_aliases=False)
                or "delayed_sip"
            ),
            alpaca_adjustment=str(
                get_env("ALPACA_ADJUSTMENT", "raw", cast=str, resolve_aliases=False) or "raw"
            ),
        ))
    return settings_obj


def _normalize_intraday_feed(feed: str | None) -> str:
    normalized = str(feed or "iex").strip().lower()
    return normalized or "iex"


def _derive_intraday_feed(cfg: TradingConfig | None = None) -> str:
    source = cfg or _CFG
    return _normalize_intraday_feed(
        getattr(
            source,
            "data_feed_intraday",
            getattr(source, "alpaca_execution_feed", getattr(source, "alpaca_data_feed", "iex")),
        )
    )


def set_data_feed_intraday(feed: str | None) -> str:
    """Update the exported DATA_FEED_INTRADAY constant."""

    global DATA_FEED_INTRADAY, ALPACA_EXECUTION_FEED
    DATA_FEED_INTRADAY = _normalize_intraday_feed(feed)
    ALPACA_EXECUTION_FEED = DATA_FEED_INTRADAY
    try:
        setattr(_CFG, "data_feed_intraday", DATA_FEED_INTRADAY)
        setattr(_CFG, "alpaca_execution_feed", ALPACA_EXECUTION_FEED)
        setattr(_CFG, "alpaca_data_feed", ALPACA_EXECUTION_FEED)
    except AI_TRADING_FALLBACK_EXCEPTIONS:  # pragma: no cover - defensive fallback
        pass
    return DATA_FEED_INTRADAY


SENTIMENT_API_KEY = _CFG.sentiment_api_key
SENTIMENT_API_URL = _CFG.sentiment_api_url

ORDER_TIMEOUT_SECONDS = int(_CFG.order_timeout_seconds)
ORDER_STALE_CLEANUP_INTERVAL = int(_CFG.order_stale_cleanup_interval)
EXECUTION_MODE = str(getattr(_CFG, "execution_mode", "sim") or "sim").lower()
SHADOW_MODE = bool(getattr(_CFG, "shadow_mode", False))
DATA_FEED_INTRADAY: str = _derive_intraday_feed()
ALPACA_EXECUTION_FEED: str = str(
    getattr(_CFG, "alpaca_execution_feed", getattr(_CFG, "alpaca_data_feed", DATA_FEED_INTRADAY))
    or DATA_FEED_INTRADAY
).lower()
ALPACA_REFERENCE_FEED: str = str(
    getattr(_CFG, "alpaca_reference_feed", "delayed_sip") or "delayed_sip"
).lower()
SLIPPAGE_LIMIT_BPS = int(getattr(_CFG, "slippage_limit_bps", getattr(_CFG, "max_slippage_bps", 75)))
SAFE_MODE_ALLOW_PAPER = bool(getattr(_CFG, "safe_mode_allow_paper", False))
PRICE_PROVIDER_ORDER = tuple(getattr(_CFG, "price_provider_order", (
    "alpaca_quote",
    "alpaca_trade",
    "alpaca_minute_close",
    "yahoo",
    "bars",
)))


def _refresh_exported_runtime_constants(cfg: TradingConfig | None = None) -> TradingConfig:
    """Refresh module-level runtime exports after environment/config reloads."""

    global _CFG
    global EXECUTION_MODE, SHADOW_MODE, DATA_FEED_INTRADAY, ALPACA_EXECUTION_FEED
    global ALPACA_REFERENCE_FEED, SAFE_MODE_ALLOW_PAPER, PRICE_PROVIDER_ORDER

    _CFG = cfg or get_trading_config()
    EXECUTION_MODE = str(getattr(_CFG, "execution_mode", "sim") or "sim").lower()
    SHADOW_MODE = bool(getattr(_CFG, "shadow_mode", False))
    DATA_FEED_INTRADAY = _derive_intraday_feed(_CFG)
    ALPACA_EXECUTION_FEED = str(
        getattr(_CFG, "alpaca_execution_feed", getattr(_CFG, "alpaca_data_feed", DATA_FEED_INTRADAY))
        or DATA_FEED_INTRADAY
    ).lower()
    ALPACA_REFERENCE_FEED = str(
        getattr(_CFG, "alpaca_reference_feed", "delayed_sip") or "delayed_sip"
    ).lower()
    SAFE_MODE_ALLOW_PAPER = bool(getattr(_CFG, "safe_mode_allow_paper", False))
    PRICE_PROVIDER_ORDER = tuple(getattr(_CFG, "price_provider_order", PRICE_PROVIDER_ORDER))
    return _CFG


def reload_env(path: str | None = None, override: bool = True) -> str | None:
    """Reload environment variables and refresh exported config constants."""

    result = _management_reload_env(path=path, override=override)
    # Invalidate this package instance directly. A reimport can leave the
    # parent package attribute pointing at an older instance, so management's
    # best-effort lookup may have cleared a different settings cache.
    _reset_cached_settings()
    _refresh_exported_runtime_constants()
    return result


def _env_value(*names: str) -> str | None:
    for name in names:
        value = get_env(name, None, cast=str, resolve_aliases=False)
        if value not in (None, ""):
            return str(value)
    return None


TRADING_MODE = str(getattr(_CFG, "trading_mode", "balanced") or "balanced").lower()
if TRADING_MODE not in MODE_PARAMETERS:
    TRADING_MODE = "balanced"


def _cfg_float(field: str, fallback: float) -> float:
    value: Any = getattr(_CFG, field, None)
    if value in (None, ""):
        return float(fallback)
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
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
ALPACA_BASE_URL = _env_value("ALPACA_TRADING_BASE_URL") or ""


@dataclass(frozen=True)
class ExecutionSettingsSnapshot:
    """Lightweight snapshot of execution-related configuration."""

    mode: str
    shadow_mode: bool
    order_timeout_seconds: int
    slippage_limit_bps: int
    price_provider_order: tuple[str, ...]
    data_feed_intraday: str
    execution_feed: str = "iex"
    reference_feed: str = "delayed_sip"
    time_in_force: str | None = None
    order_ttl_seconds: int = 8
    marketable_limit_slippage_bps: int = 4
    max_participation_rate: float | None = None


def get_execution_settings() -> ExecutionSettingsSnapshot:
    """Return a cached snapshot of live execution configuration."""

    cfg = get_trading_config()
    provider_order = tuple(getattr(cfg, "price_provider_order", ()) or PRICE_PROVIDER_ORDER)
    raw_participation_cap: Any = getattr(cfg, "exec_max_participation_rate", None)
    if raw_participation_cap in (None, ""):
        raw_participation_cap = getattr(cfg, "participation_rate", None)
    participation_cap: float | None
    if isinstance(raw_participation_cap, (int, float, str)):
        try:
            participation_cap = float(raw_participation_cap)
        except (TypeError, ValueError):
            participation_cap = None
    else:
        participation_cap = None
    if participation_cap is not None:
        if not (0.0 < participation_cap <= 1.0):
            participation_cap = None
    return ExecutionSettingsSnapshot(
        mode=str(getattr(cfg, "execution_mode", EXECUTION_MODE) or EXECUTION_MODE).lower(),
        shadow_mode=bool(getattr(cfg, "shadow_mode", SHADOW_MODE)),
        order_timeout_seconds=int(getattr(cfg, "order_timeout_seconds", ORDER_TIMEOUT_SECONDS)),
        slippage_limit_bps=int(getattr(cfg, "slippage_limit_bps", SLIPPAGE_LIMIT_BPS)),
        price_provider_order=provider_order,
        data_feed_intraday=str(
            getattr(cfg, "data_feed_intraday", DATA_FEED_INTRADAY) or DATA_FEED_INTRADAY
        ).lower(),
        execution_feed=str(
            getattr(
                cfg,
                "alpaca_execution_feed",
                getattr(cfg, "alpaca_data_feed", ALPACA_EXECUTION_FEED),
            )
            or ALPACA_EXECUTION_FEED
        ).lower(),
        reference_feed=str(
            getattr(cfg, "alpaca_reference_feed", ALPACA_REFERENCE_FEED) or ALPACA_REFERENCE_FEED
        ).lower(),
        time_in_force=(
            (str(getattr(cfg, "execution_time_in_force", "")).strip().lower() or None)
        ),
        order_ttl_seconds=int(getattr(cfg, "order_ttl_seconds", 8)),
        marketable_limit_slippage_bps=int(
            getattr(cfg, "marketable_limit_slippage_bps", 4)
        ),
        max_participation_rate=participation_cap,
    )


def get_max_drawdown_threshold() -> float:
    raw = _env_value("MAX_DRAWDOWN_THRESHOLD")
    if raw is None:
        raise RuntimeError("MAX_DRAWDOWN_THRESHOLD must be set")
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - configuration error
        raise RuntimeError(f"Invalid MAX_DRAWDOWN_THRESHOLD value: {raw}") from exc


def validate_alpaca_credentials() -> None:
    _validate_alpaca_credentials_from_management()


__all__ = [
    "_require_env_vars",
    "AlpacaConfig",
    "get_max_drawdown_threshold",
    "MODE_PARAMETERS",
    "SENTIMENT_API_KEY",
    "SENTIMENT_API_URL",
    "Settings",
    "TradingConfig",
    "broker_keys",
    "get_alpaca_config",
    "get_env",
    "get_settings",
    "reload_env",
    "require_env_vars",
    "validate_alpaca_credentials",
]
