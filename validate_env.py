"""Environment validation using pydantic-settings."""

import logging
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from the environment."""

    FLASK_PORT: int = 9000
    ALPACA_API_KEY: str
    ALPACA_SECRET_KEY: str
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"
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
    SLACK_WEBHOOK: str | None = None
    SLIPPAGE_THRESHOLD: float = 0.003
    REBALANCE_INTERVAL_MIN: int = 1440
    SHADOW_MODE: bool = False
    DISABLE_DAILY_RETRAIN: bool = False
    TRADE_LOG_FILE: str = "trades.csv"
    DISASTER_DD_LIMIT: float = 0.2
    MODEL_RF_PATH: str = "model_rf.pkl"
    MODEL_XGB_PATH: str = "model_xgb.pkl"
    MODEL_LGB_PATH: str = "model_lgb.pkl"
    SECTOR_EXPOSURE_CAP: float = 0.4
    MAX_OPEN_POSITIONS: int = 10
    WEEKLY_DRAWDOWN_LIMIT: float = 0.15
    VOLUME_THRESHOLD: int = 50000
    DOLLAR_RISK_LIMIT: float = 0.02
    FINNHUB_RPM: int = 60
    MINUTE_CACHE_TTL: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()


def _main() -> None:  # pragma: no cover - simple CLI helper
    logger.info("Environment variables successfully validated")


if __name__ == "__main__":
    _main()
