"""Runtime settings with env aliases and safe defaults."""
from __future__ import annotations
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from pydantic import Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
try:
    from pydantic.fields import FieldInfo
except (ValueError, TypeError):
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
    alpaca_api_key: str | None = Field(default=None, alias='ALPACA_API_KEY')
    alpaca_secret_key: SecretStr | None = Field(default=None, alias='ALPACA_SECRET_KEY')
    redis_url: str | None = Field(default=None, alias='REDIS_URL')
    alpaca_base_url: str = Field(default='https://paper-api.alpaca.markets', alias='ALPACA_BASE_URL')
    trading_mode: str = Field(default='balanced', alias='TRADING_MODE')
    webhook_secret: str | None = Field(default=None, alias='WEBHOOK_SECRET')
    testing: bool = Field(False, alias='TESTING')
    shadow_mode: bool = Field(False, alias='SHADOW_MODE')
    disable_daily_retrain: bool = Field(False, alias='DISABLE_DAILY_RETRAIN')
    log_market_fetch: bool = Field(True, alias='LOG_MARKET_FETCH')
    healthcheck_port: int = Field(9001, alias='HEALTHCHECK_PORT')
    min_health_rows: int = Field(120, alias='MIN_HEALTH_ROWS')
    api_host: str = Field('0.0.0.0', alias='API_HOST')
    api_port: int = Field(9001, alias='API_PORT')
    finnhub_rpm: int = Field(default=55, env='AI_TRADING_FINNHUB_RPM')
    max_trades_per_day: int = Field(default=200, env='AI_TRADING_MAX_TRADES_PER_DAY')
    max_trades_per_hour: int = Field(default=30, env='AI_TRADING_MAX_TRADES_PER_HOUR')
    conf_threshold: float = Field(default=0.75, env='AI_TRADING_CONF_THRESHOLD')
    score_confidence_min: float | None = Field(default=None, alias='SCORE_CONFIDENCE_MIN')
    score_size_max_boost: float = Field(1.0, alias='SCORE_SIZE_MAX_BOOST', description='Upper bound of raw size multiplier at confidence=1.0')
    score_size_gamma: float = Field(1.0, alias='SCORE_SIZE_GAMMA', description='Shape parameter: 1.0 linear, <1 concave, >1 convex')
    buy_threshold: float = Field(default=0.4, env='AI_TRADING_BUY_THRESHOLD')
    daily_loss_limit: float = Field(default=0.03, env='AI_TRADING_DAILY_LOSS_LIMIT')
    max_drawdown_threshold: float = Field(default=0.08, env='AI_TRADING_MAX_DRAWDOWN_THRESHOLD')
    portfolio_drift_threshold: float = Field(default=0.15, env='AI_TRADING_PORTFOLIO_DRIFT_THRESHOLD')
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
    alpaca_adjustment: Literal['all', 'raw'] = Field('all', env='ALPACA_ADJUSTMENT')
    capital_cap: float = Field(0.04, env='CAPITAL_CAP')
    dollar_risk_limit: float = Field(0.05, env='DOLLAR_RISK_LIMIT')
    max_position_size: float | None = Field(default=None, description='Absolute max dollars per position. If None, derive from equity * capital_cap; if equity unknown, use static fallback.', alias='MAX_POSITION_SIZE')
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
    rebalance_interval_min: int = Field(60, ge=1, description='Minutes between portfolio rebalances', alias='REBALANCE_INTERVAL_MIN')

    @field_validator('model_path', 'halt_flag_path', 'rl_model_path', mode='before')
    @classmethod
    def _empty_to_default(cls, v, info):
        if v in (None, '', 'None'):
            return info.field_info.default
        return v

    @field_validator('alpaca_data_feed', mode='before')
    @classmethod
    def _norm_feed(cls, v):
        return str(v).lower().strip()

    @field_validator('alpaca_adjustment', mode='before')
    @classmethod
    def _norm_adj(cls, v):
        return str(v).lower().strip()

    @field_validator('capital_cap', 'dollar_risk_limit')
    @classmethod
    def _risk_in_range(cls, v, info):
        if not 0.0 < float(v) <= 1.0:
            raise ValueError(f'{info.field_name} must be in (0, 1], got {v}')
        return float(v)

    @field_validator('max_position_size')
    @classmethod
    def _max_pos_positive(cls, v):
        if v is not None and float(v) <= 0.0:
            raise ValueError('max_position_size must be positive')
        return v

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
        or os.getenv('AI_TRADER_NEWS_API_KEY')
    )

def get_rebalance_interval_min() -> int:
    """Lazy accessor for rebalance interval.
    Prefers Settings.rebalance_interval_min, else env AI_TRADING_REBALANCE_INTERVAL_MIN, else 60.
    """
    s = get_settings()
    val = getattr(s, 'rebalance_interval_min', 60)
    try:
        v = int(val)
    except (ValueError, TypeError):
        v = 60
    if v == 60:
        import os
        alt = os.getenv('AI_TRADING_REBALANCE_INTERVAL_MIN') or os.getenv('AI_TRADER_REBALANCE_INTERVAL_MIN')
        if alt is not None:
            try:
                return int(alt)
            except (ValueError, TypeError):
                pass
    return v

def get_disaster_dd_limit() -> float:
    return _to_float(getattr(get_settings(), 'disaster_dd_limit', 0.25), 0.25)

def get_max_portfolio_positions() -> int:
    return _to_int(getattr(get_settings(), 'max_portfolio_positions', 10), 10)

def get_sector_exposure_cap() -> float:
    return _to_float(getattr(get_settings(), 'sector_exposure_cap', 0.33), 0.33)

def get_capital_cap() -> float:
    return _to_float(getattr(get_settings(), 'capital_cap', 0.04), 0.04)

def get_dollar_risk_limit() -> float:
    return _to_float(getattr(get_settings(), 'dollar_risk_limit', 0.05), 0.05)

def get_portfolio_drift_threshold() -> float:
    return _to_float(getattr(get_settings(), 'portfolio_drift_threshold', 0.15), 0.15)

def get_max_drawdown_threshold() -> float:
    return _to_float(getattr(get_settings(), 'max_drawdown_threshold', 0.08), 0.08)

def get_daily_loss_limit() -> float:
    return _to_float(getattr(get_settings(), 'daily_loss_limit', 0.03), 0.03)

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


