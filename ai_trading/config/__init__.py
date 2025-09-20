"""Configuration package exposing typed runtime settings."""

from __future__ import annotations

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

__all__ = [
    "TradingConfig",
    "CONFIG_SPECS",
    "get_trading_config",
    "reload_trading_config",
    "generate_config_schema",
    "get_env",
    "reload_env",
    "is_shadow_mode",
    "validate_required_env",
    "validate_alpaca_credentials",
    "_resolve_alpaca_env",
    "SEED",
    "MAX_EMPTY_RETRIES",
    "Settings",
    "get_settings",
    "broker_keys",
    "provider_priority",
    "max_data_fallbacks",
    "minute_data_freshness_tolerance",
    "alpaca_feed_failover",
    "alpaca_empty_to_backup",
]
