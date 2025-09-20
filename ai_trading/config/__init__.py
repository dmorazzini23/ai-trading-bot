"""Configuration package exposing typed runtime settings."""

from __future__ import annotations

from typing import Sequence

from .runtime import (
    TradingConfig,
    CONFIG_SPECS,
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
from ai_trading.settings import get_max_drawdown_threshold

_CFG = get_trading_config()
_CONFIG_LOGGED = False

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

MODE_PARAMETERS = {
    "conservative": {
        "kelly_fraction": 0.25,
        "conf_threshold": 0.85,
        "daily_loss_limit": 0.03,
        "max_position_size": 5000.0,
    },
    "balanced": {
        "kelly_fraction": 0.6,
        "conf_threshold": 0.75,
        "daily_loss_limit": 0.05,
        "max_position_size": 8000.0,
    },
    "aggressive": {
        "kelly_fraction": 0.75,
        "conf_threshold": 0.65,
        "daily_loss_limit": 0.08,
        "max_position_size": 12000.0,
    },
}


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


def validate_environment() -> None:
    """Validate required environment variables are present."""

    validate_required_env()


def validate_env_vars(*names: str) -> None:
    """Ensure specific environment variables are defined."""

    _require_env_vars(*names)


def log_config(mask_fields: Sequence[str] | None = None) -> None:
    """Mark configuration as logged without emitting secrets."""

    global _CONFIG_LOGGED
    if _CONFIG_LOGGED:
        return
    _CONFIG_LOGGED = True


__all__ = sorted([
    "AlpacaConfig",
    "META_LEARNING_BOOTSTRAP_ENABLED",
    "META_LEARNING_BOOTSTRAP_WIN_RATE",
    "META_LEARNING_MIN_TRADES_REDUCED",
    "MODE_PARAMETERS",
    "ORDER_FILL_RATE_TARGET",
    "SENTIMENT_API_KEY",
    "SENTIMENT_API_URL",
    "SENTIMENT_ENHANCED_CACHING",
    "SENTIMENT_FALLBACK_SOURCES",
    "SENTIMENT_RECOVERY_TIMEOUT_SECS",
    "SENTIMENT_SUCCESS_RATE_TARGET",
    "Settings",
    "TradingConfig",
    "_require_env_vars",
    "broker_keys",
    "derive_cap_from_settings",
    "get_alpaca_config",
    "get_env",
    "get_max_drawdown_threshold",
    "get_settings",
    "log_config",
    "reload_env",
    "require_env_vars",
    "validate_alpaca_credentials",
    "validate_env_vars",
    "validate_environment",
])
