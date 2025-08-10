# Ensure .env is loaded before constructing settings (test expectation)
from ai_trading.env import ensure_dotenv_loaded

ensure_dotenv_loaded()

import functools

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Runtime toggles
    trading_mode: str = Field("balanced", env="TRADING_MODE")
    shadow_mode: bool = Field(False, env="SHADOW_MODE")
    disable_daily_retrain: bool = Field(False, env="DISABLE_DAILY_RETRAIN")

    # Portfolio features (default disabled to reduce startup noise)
    ENABLE_PORTFOLIO_FEATURES: bool = Field(False, env="ENABLE_PORTFOLIO_FEATURES")
    REGIME_MIN_ROWS: int = Field(50, env="REGIME_MIN_ROWS")

    # Worker sizing (None/0 => auto)
    executor_workers: int | None = Field(None, env="EXECUTOR_WORKERS")
    prediction_workers: int | None = Field(None, env="PREDICTION_WORKERS")

    # Alpaca credentials (dual-schema)
    alpaca_api_key: str | None = Field(
        None, validation_alias=AliasChoices("ALPACA_API_KEY", "APCA_API_KEY_ID")
    )
    alpaca_secret_key: str | None = Field(
        None, validation_alias=AliasChoices("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY")
    )
    alpaca_base_url: str | None = Field(
        None, validation_alias=AliasChoices("ALPACA_BASE_URL", "APCA_API_BASE_URL")
    )
    # Additional settings used by runtime
    alpaca_data_feed: str | None = Field(None, env="ALPACA_DATA_FEED")
    finnhub_api_key: str | None = Field(None, env="FINNHUB_API_KEY")
    halt_flag_path: str | None = Field(None, env="HALT_FLAG_PATH")

    # Bot engine environment variables
    disaster_dd_limit: float = Field(0.2, env="DISASTER_DD_LIMIT")
    model_path: str = Field("meta_model.pkl", env="MODEL_PATH")
    model_rf_path: str = Field("model_rf.pkl", env="MODEL_RF_PATH")
    model_xgb_path: str = Field("model_xgb.pkl", env="MODEL_XGB_PATH")
    model_lgb_path: str = Field("model_lgb.pkl", env="MODEL_LGB_PATH")
    max_portfolio_positions: int = Field(20, env="MAX_PORTFOLIO_POSITIONS")
    sector_exposure_cap: float = Field(0.4, env="SECTOR_EXPOSURE_CAP")
    max_open_positions: int = Field(10, env="MAX_OPEN_POSITIONS")
    weekly_drawdown_limit: float = Field(0.15, env="WEEKLY_DRAWDOWN_LIMIT")
    volume_threshold: int = Field(50000, env="VOLUME_THRESHOLD")
    finnhub_rpm: int = Field(60, env="FINNHUB_RPM")
    trade_cooldown_min: int = Field(5, env="TRADE_COOLDOWN_MIN")
    max_trades_per_hour: int = Field(10, env="MAX_TRADES_PER_HOUR")
    max_trades_per_day: int = Field(50, env="MAX_TRADES_PER_DAY")
    minute_cache_ttl: int = Field(60, env="MINUTE_CACHE_TTL")
    healthcheck_port: int = Field(8080, env="HEALTHCHECK_PORT")

    # Rebalancer environment variables
    rebalance_interval_min: int = Field(10, env="REBALANCE_INTERVAL_MIN")
    rebalance_sleep_seconds: int = Field(600, env="REBALANCE_SLEEP_SECONDS")

    # Webhook secret for API security
    webhook_secret: str | None = Field(None, env="WEBHOOK_SECRET")

    # Risk knobs (align defaults with current behavior)
    capital_cap: float = Field(0.04, env="CAPITAL_CAP")
    dollar_risk_limit: float = Field(0.05, env="DOLLAR_RISK_LIMIT")
    max_position_size: int = Field(8000, env="MAX_POSITION_SIZE")

    # Additional environment variables commonly used
    news_api_key: str | None = Field(None, env="NEWS_API_KEY")
    sentiment_api_key: str | None = Field(None, env="SENTIMENT_API_KEY")
    sentiment_api_url: str = Field(
        "https://newsapi.org/v2/everything", env="SENTIMENT_API_URL"
    )
    finnhub_api_key: str | None = Field(None, env="FINNHUB_API_KEY")
    bot_mode: str = Field("balanced", env="BOT_MODE")
    bot_log_dir: str = Field("logs", env="BOT_LOG_DIR")
    api_port: int = Field(9001, env="API_PORT")
    scheduler_sleep_seconds: int = Field(30, env="SCHEDULER_SLEEP_SECONDS")
    scheduler_iterations: int = Field(0, env="SCHEDULER_ITERATIONS")

    # --- Data cache controls ---
    data_cache_enable: bool = Field(False, env="DATA_CACHE_ENABLE")
    data_cache_ttl_seconds: int = Field(90, env="DATA_CACHE_TTL_SECONDS")
    data_cache_dir: str = Field("data_cache", env="DATA_CACHE_DIR")
    data_cache_disk_enable: bool = Field(False, env="DATA_CACHE_DISK_ENABLE")

    # --- Data warm-up controls ---
    data_warmup_enable: bool = Field(False, env="DATA_WARMUP_ENABLE")
    data_warmup_symbols: int = Field(
        25, env="DATA_WARMUP_SYMBOLS"
    )  # top N from tickers.csv
    data_warmup_timeframe: str = Field(
        "1Min", env="DATA_WARMUP_TIMEFRAME"
    )  # "1Min" or "1D"
    data_warmup_lookback_days: int = Field(
        5, env="DATA_WARMUP_LOOKBACK_DAYS"
    )  # historical window

    # --- Regime configuration ---
    regime_symbols_csv: str = Field("SPY", env="REGIME_SYMBOLS_CSV")

    # --- Batch sizing ---
    pretrade_batch_size: int = Field(50, env="PRETRADE_BATCH_SIZE")
    # Fallback used if ctx.lookback_start/end are unset
    pretrade_lookback_days: int = Field(120, env="PRETRADE_LOOKBACK_DAYS")
    # If ctx.lookback_start/end are not present, fall back to this many days

    # --- Intraday batching controls ---
    intraday_batch_enable: bool = Field(True, env="INTRADAY_BATCH_ENABLE")
    intraday_batch_size: int = Field(40, env="INTRADAY_BATCH_SIZE")
    intraday_lookback_minutes: int = Field(120, env="INTRADAY_LOOKBACK_MINUTES")

    # --- Bounded concurrency for per-symbol fallbacks ---
    batch_fallback_workers: int = Field(4, env="BATCH_FALLBACK_WORKERS")

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}

    @model_validator(mode="after")
    def _normalize_alpaca(self):
        def clean(v: str | None) -> str | None:
            if not v:
                return None
            return v.strip().strip('"').strip("'") or None

        self.alpaca_api_key = clean(self.alpaca_api_key)
        self.alpaca_secret_key = clean(self.alpaca_secret_key)
        self.alpaca_base_url = (
            clean(self.alpaca_base_url) or "https://paper-api.alpaca.markets"
        )
        return self

    # Enforcement
    def require_alpaca_or_raise(self) -> None:
        if self.shadow_mode:
            return
        if not (self.alpaca_api_key and self.alpaca_secret_key):
            raise RuntimeError("Missing Alpaca API credentials")

    # Policy: worker sizing
    def effective_executor_workers(self, cpu_count: int) -> int:
        return (self.executor_workers or 0) or max(2, min(4, cpu_count or 2))

    def effective_prediction_workers(self, cpu_count: int) -> int:
        return (self.prediction_workers or 0) or max(2, min(4, cpu_count or 2))

    # NOTE: Intentionally no uppercase alias properties; callers should use lower-case fields via get_settings().


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
