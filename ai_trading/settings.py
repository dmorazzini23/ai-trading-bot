"""Runtime settings with env aliases and safe defaults."""
from __future__ import annotations
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
import os
from pydantic import AliasChoices, Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from ai_trading.logging import logger
try:
    from pydantic.fields import FieldInfo
except Exception:  # pragma: no cover - pydantic may be missing in tests
    FieldInfo = object

def _secret_to_str(val: Any) -> str | None:
    """Return a plain string for SecretStr or str; None if unset."""
    if val is None or isinstance(val, FieldInfo):
        return None
    if isinstance(val, SecretStr):
        return val.get_secret_value()
    if isinstance(val, str):
        return val
    return str(val)

def _to_int(val: Any, default: int | None=None) -> int:
    """Robust int conversion handling FieldInfo and bool."""
    if isinstance(val, FieldInfo) or val is None:
        if default is None:
            raise ValueError('int value missing')
        return int(default)
    if isinstance(val, bool):
        return int(val)
    try:
        return int(val)
    except (ValueError, TypeError):
        if default is None:
            raise
        return int(default)

def _to_float(val: Any, default: float | None=None) -> float:
    """Robust float conversion handling FieldInfo."""
    if isinstance(val, FieldInfo) or val is None:
        if default is None:
            raise ValueError('float value missing')
        return float(default)
    try:
        return float(val)
    except (ValueError, TypeError):
        if default is None:
            raise
        return float(default)

def _to_bool(val: Any, default: bool | None=None) -> bool:
    """Best effort bool conversion."""
    if isinstance(val, FieldInfo) or val is None:
        return bool(default) if default is not None else False
    if isinstance(val, str):
        return val.strip().lower() not in ('0', 'false', 'no', '')
    return bool(val)

class Settings(BaseSettings):
    env: str = Field(default='test', alias='APP_ENV')
    market_calendar: str = Field(default='XNYS', alias='MARKET_CALENDAR')
    data_provider: str = Field(default='mock', alias='DATA_PROVIDER')
    log_level: str = Field(default='INFO', alias='LOG_LEVEL')
    enable_memory_optimization: bool = Field(default=True)
    log_compact_json: bool = Field(False, alias='LOG_COMPACT_JSON')
    log_level_http: str = Field('WARNING', alias='LOG_LEVEL_HTTP')
    alpaca_api_key: str | None = Field(default=None, alias='ALPACA_API_KEY')
    alpaca_secret_key: SecretStr | None = Field(default=None, alias='ALPACA_SECRET_KEY')
    redis_url: str | None = Field(default=None, alias='REDIS_URL')
    enable_finnhub: bool = Field(True, alias='ENABLE_FINNHUB')
    finnhub_api_key: str | None = Field(default=None, alias='FINNHUB_API_KEY')
    backup_data_provider: Literal['yahoo', 'none'] = Field(
        'yahoo', alias='BACKUP_DATA_PROVIDER'
    )
    alpaca_base_url: str = Field(
        default='https://paper-api.alpaca.markets',
        alias='ALPACA_API_URL',
        validation_alias=AliasChoices('ALPACA_API_URL', 'ALPACA_BASE_URL'),
    )
    trading_mode: str = Field(default='balanced', alias='TRADING_MODE')
    WEBHOOK_SECRET: str | None = Field(
        default=None,
        validation_alias=AliasChoices('WEBHOOK_SECRET', 'AI_TRADING_WEBHOOK_SECRET'),
    )
    ENABLE_PORTFOLIO_FEATURES: bool = Field(
        True,
        validation_alias=AliasChoices(
            'ENABLE_PORTFOLIO_FEATURES', 'AI_TRADING_ENABLE_PORTFOLIO_FEATURES'
        ),
    )
    testing: bool = Field(False, alias='TESTING')
    shadow_mode: bool = Field(False, alias='SHADOW_MODE')
    force_trades: bool = Field(False, alias='FORCE_TRADES')
    disable_daily_retrain: bool = Field(False, alias='DISABLE_DAILY_RETRAIN')
    log_market_fetch: bool = Field(True, alias='LOG_MARKET_FETCH')
    healthcheck_port: int = Field(
        9001,
        validation_alias=AliasChoices('HEALTHCHECK_PORT', 'AI_TRADING_HEALTHCHECK_PORT'),
    )
    min_health_rows: int = Field(120, alias='MIN_HEALTH_ROWS')
    api_host: str = Field('0.0.0.0', alias='API_HOST')
    api_port: int = Field(
        9001,
        validation_alias=AliasChoices('API_PORT', 'AI_TRADING_API_PORT'),
    )
    api_port_wait_seconds: float = Field(
        30.0,
        validation_alias=AliasChoices('API_PORT_WAIT_SECONDS', 'AI_TRADING_API_PORT_WAIT_SECONDS'),
    )
    # Support AUTO sizing mode from either MAX_POSITION_MODE or AI_TRADING_MAX_POSITION_MODE
    max_position_mode: str = Field(
        'STATIC',
        validation_alias=AliasChoices('MAX_POSITION_MODE', 'AI_TRADING_MAX_POSITION_MODE'),
    )
    finnhub_rpm: int = Field(default=55, env='AI_TRADING_FINNHUB_RPM')
    max_trades_per_day: int = Field(default=200, env='AI_TRADING_MAX_TRADES_PER_DAY')
    max_trades_per_hour: int = Field(default=30, env='AI_TRADING_MAX_TRADES_PER_HOUR')
    conf_threshold: float = Field(default=0.75, env='AI_TRADING_CONF_THRESHOLD')
    score_confidence_min: float | None = Field(default=None, alias='SCORE_CONFIDENCE_MIN')
    score_size_max_boost: float = Field(1.0, alias='SCORE_SIZE_MAX_BOOST', description='Upper bound of raw size multiplier at confidence=1.0')
    score_size_gamma: float = Field(1.0, alias='SCORE_SIZE_GAMMA', description='Shape parameter: 1.0 linear, <1 concave, >1 convex')
    buy_threshold: float = Field(default=0.4, env='AI_TRADING_BUY_THRESHOLD')
    sector_exposure_cap: float = Field(default=0.33, env='AI_TRADING_SECTOR_EXPOSURE_CAP')
    max_portfolio_positions: int = Field(default=10, env='AI_TRADING_MAX_PORTFOLIO_POSITIONS')
    disaster_dd_limit: float = Field(default=0.25, env='AI_TRADING_DISASTER_DD_LIMIT')
    data_cache_enable: bool = Field(default=True, env='AI_TRADING_DATA_CACHE_ENABLE')
    data_cache_ttl_seconds: int = Field(default=300, env='AI_TRADING_DATA_CACHE_TTL_SECONDS')
    data_cache_dir: str = Field(default=str(Path.home() / '.cache' / 'ai_trading'), env='AI_TRADING_DATA_CACHE_DIR')
    data_cache_disk_enable: bool = Field(True, env='AI_TRADING_DATA_CACHE_DISK_ENABLE')
    pretrade_lookback_days: int = Field(120, alias='PRETRADE_LOOKBACK_DAYS')
    verbose_logging: bool = Field(default=False, env='AI_TRADING_VERBOSE_LOGGING')
    enable_plotting: bool = Field(default=False, env='AI_TRADING_ENABLE_PLOTTING')
    position_size_min_usd: float = Field(default=0.0, env='AI_TRADING_POSITION_SIZE_MIN_USD')
    volume_threshold: float = Field(default=0.0, env='AI_TRADING_VOLUME_THRESHOLD')
    alpaca_data_feed: Literal['iex', 'sip'] = Field('iex', env='ALPACA_DATA_FEED')
    alpaca_feed_failover: tuple[str, ...] = Field(('sip',), env='ALPACA_FEED_FAILOVER')
    alpaca_empty_to_backup: bool = Field(True, env='ALPACA_EMPTY_TO_BACKUP')
    alpaca_adjustment: Literal['all', 'raw'] = Field('all', env='ALPACA_ADJUSTMENT')
    data_provider_priority: tuple[str, ...] = Field(
        ('alpaca_iex', 'alpaca_sip', 'yahoo'), env='DATA_PROVIDER_PRIORITY'
    )
    max_data_fallbacks: int = Field(2, env='MAX_DATA_FALLBACKS')
    daily_loss_limit: float = Field(default=0.05, env='AI_TRADING_DAILY_LOSS_LIMIT')
    max_drawdown_threshold: float = Field(default=0.08, env='AI_TRADING_MAX_DRAWDOWN_THRESHOLD')
    portfolio_drift_threshold: float = Field(default=0.15, env='AI_TRADING_PORTFOLIO_DRIFT_THRESHOLD')
    capital_cap: float = Field(
        0.25,
        validation_alias=AliasChoices('capital_cap', 'CAPITAL_CAP', 'AI_TRADING_CAPITAL_CAP'),
    )
    dollar_risk_limit: float = Field(
        0.05,
        validation_alias=AliasChoices('dollar_risk_limit', 'DOLLAR_RISK_LIMIT'),
    )
    max_position_size: float | None = Field(
        8000.0,
        description=(
            'Absolute max dollars per position. If None, derive from equity * '
            'capital_cap; if equity unknown, use static fallback.'
        ),
        alias='MAX_POSITION_SIZE',
    )
    max_position_equity_fallback: float = Field(
        200000.0,
        alias='MAX_POSITION_EQUITY_FALLBACK',
        description='Equity used when deriving max_position_size when real equity is unavailable.',
    )
    interval: int = Field(60, alias='AI_TRADING_INTERVAL')
    iterations: int = Field(0, alias='AI_TRADING_ITERATIONS')
    scheduler_iterations: int = Field(0, validation_alias='SCHEDULER_ITERATIONS')
    scheduler_sleep_seconds: int = Field(60, validation_alias='SCHEDULER_SLEEP_SECONDS')
    ai_trading_seed: int = Field(42, alias='AI_TRADING_SEED')
    model_path: str = Field('trained_model.pkl', alias='AI_TRADING_MODEL_PATH')
    halt_flag_path: str = Field('halt.flag', alias='HALT_FLAG_PATH')
    rl_model_path: str = Field('rl_agent.zip', alias='AI_TRADING_RL_MODEL_PATH')
    use_rl_agent: bool = Field(False, alias='USE_RL_AGENT')
    trade_cooldown_min: int = Field(15, alias='TRADE_COOLDOWN_MIN')
    health_tick_seconds: int = Field(default=300, env='AI_TRADING_HEALTH_TICK_SECONDS')
    cpu_only: bool = Field(default=False, validation_alias='CPU_ONLY')
    news_api_key: str | None = None
    sentiment_api_key: str | None = Field(
        default=None,
        alias='SENTIMENT_API_KEY',
        validation_alias=AliasChoices('SENTIMENT_API_KEY', 'NEWS_API_KEY'),
    )
    sentiment_api_url: str | None = Field(default=None, alias='SENTIMENT_API_URL')
    sentiment_max_retries: int = Field(5, alias='SENTIMENT_MAX_RETRIES')
    sentiment_backoff_base: float = Field(5.0, alias='SENTIMENT_BACKOFF_BASE')
    sentiment_backoff_strategy: str = Field('exponential', alias='SENTIMENT_BACKOFF_STRATEGY')
    rebalance_interval_min: int = Field(60, ge=1, description='Minutes between portfolio rebalances', alias='REBALANCE_INTERVAL_MIN')
    # HTTP client/session tuning (used by main._init_http_session)
    http_pool_maxsize: int = Field(32, alias='HTTP_POOL_MAXSIZE')
    http_total_retries: int = Field(3, alias='HTTP_TOTAL_RETRIES')
    http_backoff_factor: float = Field(0.3, alias='HTTP_BACKOFF_FACTOR')
    http_connect_timeout: float = Field(5.0, alias='HTTP_CONNECT_TIMEOUT')
    http_read_timeout: float = Field(10.0, alias='HTTP_READ_TIMEOUT')
    # Closed-hours HTTP overrides (optional)
    http_connect_timeout_closed: float | None = Field(None, alias='HTTP_CONNECT_TIMEOUT_CLOSED')
    http_read_timeout_closed: float | None = Field(None, alias='HTTP_READ_TIMEOUT_CLOSED')
    # Dynamic sizing refresh TTL (seconds) to limit equity polling in AUTO mode
    dynamic_size_refresh_secs: float = Field(3600.0, alias='DYNAMIC_SIZE_REFRESH_SECS')
    # Execution policy (safe rollout toggles)
    exec_prefer_limit: bool = Field(False, alias='EXECUTION_PREFER_LIMIT')
    exec_max_participation_rate: float = Field(0.05, alias='EXECUTION_MAX_PARTICIPATION_RATE', description='Cap per-order market participation rate [0,1] when available')
    exec_log_slippage: bool = Field(False, alias='EXECUTION_LOG_SLIPPAGE')
    # Slow the main loop when market is closed
    interval_when_closed: int = Field(300, alias='INTERVAL_WHEN_CLOSED')

    @field_validator('model_path', 'halt_flag_path', 'rl_model_path', mode='before')
    @classmethod
    def _empty_to_default(cls, v, info):
        """Map empty values to the field's declared default (Pydantic v2).

        In Pydantic v2, ``info`` is a ``FieldValidationInfo`` which no longer
        exposes ``field_info``. Retrieve the default from ``cls.model_fields``
        using the current ``field_name``.
        """
        if v in (None, '', 'None'):
            try:
                # Pydantic v2: defaults are stored on model_fields
                return cls.model_fields[info.field_name].default  # type: ignore[index]
            except Exception:
                return v
        return v

    @field_validator('alpaca_data_feed', mode='before')
    @classmethod
    def _norm_feed(cls, v):
        return str(v).lower().strip()

    @field_validator('alpaca_data_feed', mode='after')
    @classmethod
    def _enforce_allowed_feed(cls, v: str) -> str:
        """Force IEX feed unless SIP explicitly allowed."""
        allow_sip = os.getenv('ALPACA_ALLOW_SIP', '').strip().lower() in {
            '1',
            'true',
            'yes',
        }
        if v == 'sip' and not allow_sip:
            logger.warning('SIP_FEED_DISABLED', extra={'requested': 'sip', 'using': 'iex'})
            return 'iex'
        return v

    @field_validator('alpaca_feed_failover', mode='before')
    @classmethod
    def _split_feed_failover(cls, v):
        if isinstance(v, FieldInfo) or v is None:
            return tuple()
        if isinstance(v, str):
            if not v.strip():
                return tuple()
            return tuple(i.strip() for i in v.split(',') if i.strip())
        return tuple(v)

    @field_validator('alpaca_feed_failover', mode='after')
    @classmethod
    def _normalize_feed_failover(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        allow_sip = os.getenv('ALPACA_ALLOW_SIP', '').strip().lower() in {
            '1',
            'true',
            'yes',
        }
        normalized: list[str] = []
        for feed in v:
            feed_norm = str(feed).lower().strip()
            if feed_norm not in {'iex', 'sip'}:
                continue
            if feed_norm == 'sip' and not allow_sip:
                continue
            if feed_norm not in normalized:
                normalized.append(feed_norm)
        return tuple(normalized)

    @field_validator('alpaca_adjustment', mode='before')
    @classmethod
    def _norm_adj(cls, v):
        return str(v).lower().strip()

    @field_validator('data_provider_priority', mode='before')
    @classmethod
    def _split_priority(cls, v):
        if isinstance(v, str):
            return tuple(i.strip() for i in v.split(',') if i.strip())
        return tuple(v)

    @field_validator('capital_cap', mode='before')
    @classmethod
    def _risk_in_range_cap(cls, v):
        # Ensure explicit kwargs are validated even with validation_alias present
        if v is None:
            return v
        if not 0.0 < float(v) <= 1.0:
            raise ValueError(f'capital_cap must be in (0, 1], got {v}')
        return float(v)

    @field_validator('dollar_risk_limit')
    @classmethod
    def _risk_in_range(cls, v):
        if not 0.0 < float(v) <= 1.0:
            raise ValueError(f'dollar_risk_limit must be in (0, 1], got {v}')
        return float(v)

    @field_validator('exec_max_participation_rate')
    @classmethod
    def _part_rate_range(cls, v):
        v = float(v)
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    @field_validator('max_position_size')
    @classmethod
    def _max_pos_positive(cls, v, info):
        if v is not None and float(v) <= 0.0:
            raise ValueError(f"{info.field_name} must be positive")
        return v

    @field_validator('force_trades', mode='before')
    @classmethod
    def _force_trades_cast(cls, v):
        return _to_bool(v, False)

    @field_validator('alpaca_base_url', mode='after')
    @classmethod
    def _validate_alpaca_base_url(cls, value: str) -> str:
        """Ensure Alpaca base URL is normalized and explicitly https://."""

        from ai_trading.config import management as config_management

        env_key = 'ALPACA_API_URL' if os.getenv('ALPACA_API_URL') else 'ALPACA_BASE_URL'
        normalized, message = config_management._normalize_alpaca_base_url(
            value, source_key=env_key
        )
        if normalized:
            return normalized
        guidance = message or config_management.ALPACA_URL_GUIDANCE
        raise ValueError(guidance)

    @computed_field
    @property
    def alpaca_secret_key_plain(self) -> str | None:
        """Return the Alpaca secret key as a plain string."""
        return _secret_to_str(self.alpaca_secret_key)

    @computed_field
    @property
    def trade_cooldown(self) -> timedelta:
        return timedelta(minutes=_to_int(getattr(self, 'trade_cooldown_min', 15), 15))
    model_config = SettingsConfigDict(env_prefix='AI_TRADING_', extra='ignore', case_sensitive=False)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return module-level Settings singleton."""
    return Settings()

def get_news_api_key() -> str | None:
    """Lazy accessor for optional News API key."""
    val = getattr(get_settings(), 'news_api_key', None)
    if val:
        return val
    import os
    return (
        os.getenv('NEWS_API_KEY')
        or os.getenv('AI_TRADING_NEWS_API_KEY')
    )

def get_rebalance_interval_min() -> int:
    """Lazy accessor for rebalance interval.

    Considers Settings.rebalance_interval_min (minutes) and env overrides
    AI_TRADING_REBALANCE_INTERVAL_MIN (minutes) or
    AI_TRADING_REBALANCE_INTERVAL_HOURS (hours). Hours are normalized to minutes
    and all candidates must be positive integers. Returns the smallest valid
    value or ``60`` if none are provided.
    """

    def _parse_pos_int(val: Any) -> int | None:
        try:
            iv = int(val)
        except (TypeError, ValueError):
            return None
        return iv if iv > 0 else None

    env_vals: list[int] = []
    if (m := _parse_pos_int(os.getenv("AI_TRADING_REBALANCE_INTERVAL_MIN"))) is not None:
        env_vals.append(m)
    if (h := _parse_pos_int(os.getenv("AI_TRADING_REBALANCE_INTERVAL_HOURS"))) is not None:
        env_vals.append(h * 60)
    if env_vals:
        return min(env_vals)

    s = get_settings()
    if (m := _parse_pos_int(getattr(s, "rebalance_interval_min", None))) is not None:
        return m
    return 60

def get_disaster_dd_limit() -> float:
    return _to_float(getattr(get_settings(), 'disaster_dd_limit', 0.25), 0.25)

def get_max_portfolio_positions() -> int:
    return _to_int(getattr(get_settings(), 'max_portfolio_positions', 10), 10)

def get_sector_exposure_cap() -> float:
    return _to_float(getattr(get_settings(), 'sector_exposure_cap', 0.33), 0.33)

def get_capital_cap() -> float:
    return _to_float(getattr(get_settings(), 'capital_cap', 0.25), 0.25)

def get_dollar_risk_limit() -> float:
    return _to_float(getattr(get_settings(), 'dollar_risk_limit', 0.05), 0.05)

def get_portfolio_drift_threshold() -> float:
    return _to_float(getattr(get_settings(), 'portfolio_drift_threshold', 0.15), 0.15)

def get_max_drawdown_threshold() -> float:
    return _to_float(getattr(get_settings(), 'max_drawdown_threshold', 0.08), 0.08)

def get_daily_loss_limit() -> float:
    return _to_float(getattr(get_settings(), 'daily_loss_limit', 0.05), 0.05)

def get_buy_threshold() -> float:
    return _to_float(getattr(get_settings(), 'buy_threshold', 0.4), 0.4)

def get_conf_threshold() -> float:
    """Return confidence threshold based on trading mode or explicit setting."""
    s = get_settings()
    val = getattr(s, 'conf_threshold', None)
    if val is not None:
        return _to_float(val, 0.75)
    mode = str(getattr(s, 'trading_mode', 'balanced')).lower()
    defaults = {'conservative': 0.85, 'balanced': 0.75, 'aggressive': 0.65}
    return float(defaults.get(mode, 0.75))

def get_trade_cooldown_min() -> int:
    return _to_int(getattr(get_settings(), 'trade_cooldown_min', 15), 15)

def get_max_trades_per_hour() -> int:
    return _to_int(getattr(get_settings(), 'max_trades_per_hour', 30), 30)

def get_max_trades_per_day() -> int:
    return _to_int(getattr(get_settings(), 'max_trades_per_day', 200), 200)

def get_finnhub_rpm() -> int:
    return _to_int(getattr(get_settings(), 'finnhub_rpm', 55), 55)

def get_backup_data_provider() -> str:
    return getattr(get_settings(), 'backup_data_provider', 'yahoo')

def get_data_cache_enable() -> bool:
    return _to_bool(getattr(get_settings(), 'data_cache_enable', True), True)

def get_data_cache_ttl_seconds() -> int:
    return _to_int(getattr(get_settings(), 'data_cache_ttl_seconds', 300), 300)

def get_verbose_logging() -> bool:
    return _to_bool(getattr(get_settings(), 'verbose_logging', False), False)

def get_enable_plotting() -> bool:
    return _to_bool(getattr(get_settings(), 'enable_plotting', False), False)

def get_position_size_min_usd() -> float:
    return _to_float(getattr(get_settings(), 'position_size_min_usd', 0.0), 0.0)

def get_volume_threshold() -> float:
    return _to_float(getattr(get_settings(), 'volume_threshold', 0.0), 0.0)

def get_alpaca_secret_key_plain() -> str | None:
    """Return Alpaca secret key as plain string if present."""
    s = get_settings()
    return _secret_to_str(getattr(s, 'alpaca_secret_key', None))

def get_seed_int(default: int=42) -> int:
    """Fetch deterministic seed as int."""  # AI-AGENT-REF: stable default accessor
    s = get_settings()
    return _to_int(getattr(s, 'ai_trading_seed', default), default)
