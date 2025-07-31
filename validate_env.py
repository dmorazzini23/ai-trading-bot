"""Environment validation using pydantic-settings."""

import logging

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from the environment."""

    FLASK_PORT: int = 9000
    ALPACA_API_KEY: str
    ALPACA_SECRET_KEY: str
    
    # AI-AGENT-REF: Support for live vs paper trading configuration
    TRADING_MODE: str = "paper"  # paper, live
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"  # Default to paper for safety
    ALPACA_DATA_FEED: str = "iex"
    FINNHUB_API_KEY: str | None = None
    FUNDAMENTAL_API_KEY: str | None = None
    NEWS_API_KEY: str | None = None
    IEX_API_TOKEN: str | None = None
    BOT_MODE: str = "balanced"
    MODEL_PATH: str = "trained_model.pkl"
    HALT_FLAG_PATH: str = "halt.flag"
    MAX_PORTFOLIO_POSITIONS: int = 20
    LIMIT_ORDER_SLIPPAGE: float = 0.005
    HEALTHCHECK_PORT: int = 8081
    RUN_HEALTHCHECK: str = "0"
    BUY_THRESHOLD: float = 0.5
    WEBHOOK_SECRET: str = ""
    WEBHOOK_PORT: int = 9000
    SLIPPAGE_THRESHOLD: float = 0.003
    REBALANCE_INTERVAL_MIN: int = 1440
    SHADOW_MODE: bool = False
    DRY_RUN: bool = False
    DISABLE_DAILY_RETRAIN: bool = False
    TRADE_LOG_FILE: str = "data/trades.csv"
    FORCE_TRADES: bool = False
    DISASTER_DD_LIMIT: float = 0.2
    MODEL_RF_PATH: str = "model_rf.pkl"
    MODEL_XGB_PATH: str = "model_xgb.pkl"
    MODEL_LGB_PATH: str = "model_lgb.pkl"
    RL_MODEL_PATH: str = "rl_agent.zip"
    USE_RL_AGENT: bool = False
    SECTOR_EXPOSURE_CAP: float = 0.4
    MAX_OPEN_POSITIONS: int = 10
    WEEKLY_DRAWDOWN_LIMIT: float = 0.15
    VOLUME_THRESHOLD: int = 50000
    DOLLAR_RISK_LIMIT: float = 0.02
    FINNHUB_RPM: int = 60
    MINUTE_CACHE_TTL: int = 60
    EQUITY_EXPOSURE_CAP: float = 2.5
    PORTFOLIO_EXPOSURE_CAP: float = 2.5
    SEED: int = 42
    RATE_LIMIT_BUDGET: int = 190

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # allow unknown env vars (e.g. SLACK_WEBHOOK)
    )


settings = Settings()


def validate_trading_mode() -> None:
    """Validate trading mode configuration and set appropriate URLs."""
    # AI-AGENT-REF: Trading mode validation and URL configuration
    if settings.TRADING_MODE not in ["paper", "live"]:
        raise ValueError(f"Invalid TRADING_MODE: {settings.TRADING_MODE}. Must be 'paper' or 'live'")
    
    # Auto-configure base URL based on trading mode if not explicitly set
    if settings.TRADING_MODE == "live":
        if settings.ALPACA_BASE_URL == "https://paper-api.alpaca.markets":
            logger.warning("TRADING_MODE is 'live' but ALPACA_BASE_URL is paper trading URL")
            logger.warning("Auto-configuring for live trading. Set ALPACA_BASE_URL explicitly to override.")
            settings.ALPACA_BASE_URL = "https://api.alpaca.markets"
        logger.critical("LIVE TRADING MODE ENABLED - Real money at risk!")
    else:
        if settings.ALPACA_BASE_URL != "https://paper-api.alpaca.markets":
            logger.info("TRADING_MODE is 'paper', ensuring paper trading URL")
            settings.ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
        logger.info("Paper trading mode enabled - Safe for testing")


def get_alpaca_config() -> dict:
    """Get Alpaca client configuration based on current settings."""
    validate_trading_mode()
    return {
        "api_key": settings.ALPACA_API_KEY,
        "secret_key": settings.ALPACA_SECRET_KEY,
        "base_url": settings.ALPACA_BASE_URL,
        "paper": settings.TRADING_MODE == "paper"
    }


def generate_schema() -> dict:
    """Return JSONSchema for the environment settings."""
    # AI-AGENT-REF: expose env schema for validation in CI
    return Settings.model_json_schema()


def _main() -> None:  # pragma: no cover - simple CLI helper
    logger.info("Environment variables successfully validated")


if __name__ == "__main__":
    _main()
