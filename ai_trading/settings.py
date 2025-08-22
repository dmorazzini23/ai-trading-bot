"""Runtime settings with env aliases and safe defaults."""

# ruff: noqa: F821,F841,I001
from __future__ import annotations

from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from ai_trading.env import ensure_dotenv_loaded  # AI-AGENT-REF: re-export for CLI  # noqa: F401

from pydantic import Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:  # AI-AGENT-REF: tolerate pydantic internals missing
    from pydantic.fields import FieldInfo
except Exception:  # noqa: BLE001  pragma: no cover
    FieldInfo = object


def _secret_to_str(val: Any) -> str | None:
    """Return a plain string for SecretStr or str; None if unset."""
    # AI-AGENT-REF: safe secret unwrap
    if val is None or isinstance(val, FieldInfo):
        return None
    if isinstance(val, SecretStr):
        return val.get_secret_value()
    if isinstance(val, str):
        return val
    return str(val)


def _to_int(val: Any, default: int | None = None) -> int:
    """Robust int conversion handling FieldInfo and bool."""
    if isinstance(val, FieldInfo) or val is None:
        if default is None:
            raise ValueError("int value missing")
        return int(default)
    if isinstance(val, bool):
        return int(val)
    try:
        return int(val)
    except Exception:
        if default is None:
            raise
        return int(default)


def _to_float(val: Any, default: float | None = None) -> float:
    """Robust float conversion handling FieldInfo."""
    if isinstance(val, FieldInfo) or val is None:
        if default is None:
            raise ValueError("float value missing")
        return float(default)
    try:
        return float(val)
    except Exception:
        if default is None:
            raise
        return float(default)


def _to_bool(val: Any, default: bool | None = None) -> bool:
    """Best effort bool conversion."""  # AI-AGENT-REF: bool normalization
    if isinstance(val, FieldInfo) or val is None:
        return bool(default) if default is not None else False
    if isinstance(val, str):
        return val.strip().lower() not in ("0", "false", "no", "")
    return bool(val)


class Settings(BaseSettings):
    # General runtime configuration
    env: str = Field(default="test", alias="APP_ENV")  # AI-AGENT-REF: environment name
    market_calendar: str = Field(default="XNYS", alias="MARKET_CALENDAR")  # AI-AGENT-REF: trading calendar
    data_provider: str = Field(default="mock", alias="DATA_PROVIDER")  # AI-AGENT-REF: data provider source
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")  # AI-AGENT-REF: log level
    enable_memory_optimization: bool = Field(default=True)  # AI-AGENT-REF: memory tweaks
    log_compact_json: bool = Field(False, alias="LOG_COMPACT_JSON")  # AI-AGENT-REF: compact log format
    alpaca_api_key: str | None = Field(default=None, alias="ALPACA_API_KEY")  # AI-AGENT-REF: Alpaca key
    alpaca_secret_key: SecretStr | None = Field(
        default=None, alias="ALPACA_SECRET_KEY"
    )  # AI-AGENT-REF: Alpaca secret
    redis_url: str | None = Field(default=None, alias="REDIS_URL")  # AI-AGENT-REF: redis URL
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets", alias="ALPACA_BASE_URL"
    )  # AI-AGENT-REF: Alpaca base URL
    trading_mode: str = Field(default="balanced", alias="TRADING_MODE")  # AI-AGENT-REF: trading mode
    webhook_secret: str | None = Field(default=None, alias="WEBHOOK_SECRET")  # AI-AGENT-REF: webhook secret
    testing: bool = Field(False, alias="TESTING")  # AI-AGENT-REF: test mode toggle
    shadow_mode: bool = Field(False, alias="SHADOW_MODE")  # AI-AGENT-REF: shadow trades
    log_market_fetch: bool = Field(True, alias="LOG_MARKET_FETCH")  # AI-AGENT-REF: fetch logging
    healthcheck_port: int = Field(9001, alias="HEALTHCHECK_PORT")  # AI-AGENT-REF: health server port
    min_health_rows: int = Field(120, alias="MIN_HEALTH_ROWS")  # AI-AGENT-REF: health rows threshold

    # --- API server configuration ---
    api_host: str = Field("0.0.0.0", alias="API_HOST")  # AI-AGENT-REF: API bind host
    api_port: int = Field(9001, alias="API_PORT")  # AI-AGENT-REF: API bind port

    # Finnhub API calls per minute (rate limit budget)
    finnhub_rpm: int = Field(default=55, env="AI_TRADER_FINNHUB_RPM")
    # Upper bound on trade submissions per day (rate limit)
    max_trades_per_day: int = Field(default=200, env="AI_TRADER_MAX_TRADES_PER_DAY")
    # Upper bound on trade submissions per hour (rate limit)
    max_trades_per_hour: int = Field(default=30, env="AI_TRADER_MAX_TRADES_PER_HOUR")
    # Min confidence threshold
    # Confidence thresholds tuned to test expectations
    conf_threshold: float = Field(default=0.75, env="AI_TRADER_CONF_THRESHOLD")
    # Min model buy score
    buy_threshold: float = Field(default=0.4, env="AI_TRADER_BUY_THRESHOLD")
    # Max daily loss fraction before halt
    daily_loss_limit: float = Field(default=0.03, env="AI_TRADER_DAILY_LOSS_LIMIT")
    # Max running drawdown before action
    max_drawdown_threshold: float = Field(
        default=0.08, env="AI_TRADER_MAX_DRAWDOWN_THRESHOLD"
    )
    # Rebalance drift threshold
    portfolio_drift_threshold: float = Field(
        default=0.15, env="AI_TRADER_PORTFOLIO_DRIFT_THRESHOLD"
    )
    # Max fraction of equity in one sector
    sector_exposure_cap: float = Field(
        default=0.33, env="AI_TRADER_SECTOR_EXPOSURE_CAP"
    )
    # Upper bound on concurrent open positions
    max_portfolio_positions: int = Field(
        default=10, env="AI_TRADER_MAX_PORTFOLIO_POSITIONS"
    )
    disaster_dd_limit: float = Field(default=0.25, env="AI_TRADER_DISASTER_DD_LIMIT")
    # Toggle dataframe/bars cache in data_fetcher
    data_cache_enable: bool = Field(default=True, env="AI_TRADER_DATA_CACHE_ENABLE")
    data_cache_ttl_seconds: int = Field(
        default=300, env="AI_TRADER_DATA_CACHE_TTL_SECONDS"
    )
    data_cache_dir: str = Field(
        default=str(Path.home() / ".cache" / "ai_trader"),
        env="AI_TRADER_DATA_CACHE_DIR",
    )  # AI-AGENT-REF: cache directory
    data_cache_disk_enable: bool = Field(
        True, env="AI_TRADER_DATA_CACHE_DISK_ENABLE"
    )  # AI-AGENT-REF: disk cache toggle
    pretrade_lookback_days: int = Field(120, alias="PRETRADE_LOOKBACK_DAYS")  # AI-AGENT-REF: safety lookback
    verbose_logging: bool = Field(default=False, env="AI_TRADER_VERBOSE_LOGGING")
    # Plotting (matplotlib) allowed in environments that support it
    enable_plotting: bool = Field(default=False, env="AI_TRADER_ENABLE_PLOTTING")
    # Minimum absolute USD size for a position
    position_size_min_usd: float = Field(
        default=0.0, env="AI_TRADER_POSITION_SIZE_MIN_USD"
    )
    # Global volume threshold used by bot_engine init
    volume_threshold: float = Field(default=0.0, env="AI_TRADER_VOLUME_THRESHOLD")

    # Data fetching knobs used by main.py
    alpaca_data_feed: Literal["iex", "sip"] = Field(
        "iex", env="ALPACA_DATA_FEED"
    )  # AI-AGENT-REF: market data feed
    alpaca_adjustment: Literal["all", "raw"] = Field(
        "all", env="ALPACA_ADJUSTMENT"
    )  # AI-AGENT-REF: bar adjustment

    # Risk knobs used during runtime validation
    capital_cap: float = Field(0.04, env="CAPITAL_CAP")  # AI-AGENT-REF: equity cap
    dollar_risk_limit: float = Field(
        0.05, env="DOLLAR_RISK_LIMIT"
    )  # AI-AGENT-REF: per-position risk
    """Single source of truth for runtime configuration."""

    # loop control
    interval: int = Field(
        60, alias="AI_TRADING_INTERVAL"
    )  # AI-AGENT-REF: interval alias
    iterations: int = Field(
        0, alias="AI_TRADING_ITERATIONS"
    )  # AI-AGENT-REF: iterations alias
    scheduler_iterations: int = Field(0, validation_alias="SCHEDULER_ITERATIONS")
    scheduler_sleep_seconds: int = Field(60, validation_alias="SCHEDULER_SLEEP_SECONDS")
    ai_trading_seed: int = Field(
        42, alias="AI_TRADING_SEED"
    )  # AI-AGENT-REF: seed alias

    # paths
    model_path: str = Field(
        "trained_model.pkl", alias="AI_TRADING_MODEL_PATH"
    )  # AI-AGENT-REF: model path
    halt_flag_path: str = Field(
        "halt.flag", alias="HALT_FLAG_PATH"
    )  # AI-AGENT-REF: halt flag path
    rl_model_path: str = Field(
        "rl_agent.zip", alias="AI_TRADING_RL_MODEL_PATH"
    )  # AI-AGENT-REF: RL model path

    # RL switch
    use_rl_agent: bool = Field(False, alias="USE_RL_AGENT")  # AI-AGENT-REF: RL toggle

    # cooling
    trade_cooldown_min: int = Field(
        15, alias="TRADE_COOLDOWN_MIN"
    )  # AI-AGENT-REF: cooldown minutes
    health_tick_seconds: int = Field(
        default=300, env="AI_TRADER_HEALTH_TICK_SECONDS"
    )  # AI-AGENT-REF: scheduler heartbeat
    cpu_only: bool = Field(
        default=False, validation_alias="CPU_ONLY"
    )  # AI-AGENT-REF: ML device override

    # External APIs
    news_api_key: str | None = None  # AI-AGENT-REF: optional News API key
    rebalance_interval_min: int = Field(
        60,
        ge=1,
        description="Minutes between portfolio rebalances",
        alias="REBALANCE_INTERVAL_MIN",
    )  # AI-AGENT-REF: rebalance interval

    @field_validator("model_path", "halt_flag_path", "rl_model_path", mode="before")
    @classmethod
    def _empty_to_default(cls, v, info):  # AI-AGENT-REF: blank string to default
        if v in (None, "", "None"):
            return info.field_info.default
        return v

    @field_validator("alpaca_data_feed", mode="before")
    @classmethod
    def _norm_feed(cls, v):  # AI-AGENT-REF: normalize feed
        return str(v).lower().strip()

    @field_validator("alpaca_adjustment", mode="before")
    @classmethod
    def _norm_adj(cls, v):  # AI-AGENT-REF: normalize adjustment
        return str(v).lower().strip()

    @field_validator("capital_cap", "dollar_risk_limit")
    @classmethod
    def _risk_in_range(cls, v, info):  # AI-AGENT-REF: risk bounds check
        if not (0.0 < float(v) <= 1.0):
            raise ValueError(f"{info.field_name} must be in (0, 1], got {v}")
        return float(v)

    # AI-AGENT-REF: derive timedelta from minutes
    @computed_field
    @property
    def trade_cooldown(self) -> timedelta:
        return timedelta(minutes=_to_int(getattr(self, "trade_cooldown_min", 15), 15))

    model_config = SettingsConfigDict(  # noqa: F841
        env_prefix="AI_TRADER_",
        extra="ignore",
        case_sensitive=False,
    )  # AI-AGENT-REF: AI_TRADER_ env prefix


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return module-level Settings singleton."""  # AI-AGENT-REF: cached settings

    return Settings()


def get_news_api_key() -> str | None:
    """Lazy accessor for optional News API key."""  # AI-AGENT-REF: runtime News API key
    return getattr(get_settings(), "news_api_key", None)


def get_rebalance_interval_min() -> int:
    """Lazy accessor for rebalance interval."""
    s = get_settings()
    val = getattr(s, "rebalance_interval_min", 60)
    try:
        return int(val)
    except Exception:  # AI-AGENT-REF: tolerate FieldInfo during early imports  # noqa: BLE001
        return 60


# ---- Lazy getters (access only at runtime; never at module import) ----
def get_disaster_dd_limit() -> float:
    return _to_float(getattr(get_settings(), "disaster_dd_limit", 0.25), 0.25)


# ---- Lazy getters (access only at runtime; never at module import) ----
def get_max_portfolio_positions() -> int:
    return _to_int(getattr(get_settings(), "max_portfolio_positions", 10), 10)


def get_sector_exposure_cap() -> float:
    return _to_float(getattr(get_settings(), "sector_exposure_cap", 0.33), 0.33)


def get_capital_cap() -> float:
    return _to_float(getattr(get_settings(), "capital_cap", 0.04), 0.04)


def get_dollar_risk_limit() -> float:
    return _to_float(getattr(get_settings(), "dollar_risk_limit", 0.05), 0.05)


def get_portfolio_drift_threshold() -> float:
    return _to_float(getattr(get_settings(), "portfolio_drift_threshold", 0.15), 0.15)


def get_max_drawdown_threshold() -> float:
    return _to_float(getattr(get_settings(), "max_drawdown_threshold", 0.08), 0.08)


def get_daily_loss_limit() -> float:
    return _to_float(getattr(get_settings(), "daily_loss_limit", 0.03), 0.03)


def get_buy_threshold() -> float:
    return _to_float(getattr(get_settings(), "buy_threshold", 0.4), 0.4)


def get_conf_threshold() -> float:
    from ai_trading.config.management import TradingConfig

    mode = getattr(get_settings(), "trading_mode", "balanced")
    if not isinstance(mode, str):  # AI-AGENT-REF: guard FieldInfo from raw Settings
        mode = str(getattr(mode, "default", mode))
    cfg = TradingConfig.from_env(mode=mode)
    return _to_float(
        getattr(cfg, "conf_threshold", 0.75), 0.75
    )  # AI-AGENT-REF: mode-aware


def get_trade_cooldown_min() -> int:
    return _to_int(getattr(get_settings(), "trade_cooldown_min", 15), 15)


# ---- Lazy getters ----
def get_max_trades_per_hour() -> int:
    return _to_int(getattr(get_settings(), "max_trades_per_hour", 30), 30)


# ---- Lazy getters ----
def get_max_trades_per_day() -> int:
    return _to_int(getattr(get_settings(), "max_trades_per_day", 200), 200)


# ---- Lazy getters ----
def get_finnhub_rpm() -> int:
    return _to_int(getattr(get_settings(), "finnhub_rpm", 55), 55)


def get_data_cache_enable() -> bool:
    return _to_bool(getattr(get_settings(), "data_cache_enable", True), True)


def get_data_cache_ttl_seconds() -> int:
    return _to_int(getattr(get_settings(), "data_cache_ttl_seconds", 300), 300)


def get_verbose_logging() -> bool:
    return _to_bool(getattr(get_settings(), "verbose_logging", False), False)


def get_enable_plotting() -> bool:
    return _to_bool(getattr(get_settings(), "enable_plotting", False), False)


def get_position_size_min_usd() -> float:
    return _to_float(getattr(get_settings(), "position_size_min_usd", 0.0), 0.0)


def get_volume_threshold() -> float:
    return _to_float(getattr(get_settings(), "volume_threshold", 0.0), 0.0)


def get_alpaca_secret_key_plain() -> str | None:
    """Return Alpaca secret key as plain string if present."""
    s = get_settings()
    return _secret_to_str(getattr(s, "alpaca_secret_key", None))


def get_seed_int(default: int = 42) -> int:
    """Fetch deterministic seed as int."""  # AI-AGENT-REF: robust seed accessor
    s = get_settings()
    return _to_int(getattr(s, "ai_trading_seed", default), default)
