from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Any, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from ai_trading.settings import _secret_to_str  # AI-AGENT-REF: centralized normalization


TICKERS_FILE = os.getenv("AI_TRADER_TICKERS_FILE", "tickers.csv")
TICKERS_CSV = os.getenv("AI_TRADER_TICKERS_CSV")  # optional literal CSV list
UNIVERSE_LIMIT = int(os.getenv("AI_TRADER_UNIVERSE_LIMIT", "0") or "0")  # 0 => no cap

MODEL_PATH = os.getenv("AI_TRADER_MODEL_PATH")
MODEL_MODULE = os.getenv("AI_TRADER_MODEL_MODULE")


def model_config_source() -> str:
    """Return 'path', 'module', or ''."""
    if MODEL_PATH:
        return "path"
    if MODEL_MODULE:
        return "module"
    return ""


class Settings(BaseSettings):
    """Minimal project settings for tests and CI."""

    env: str = Field(default="test", alias="APP_ENV")
    # AI-AGENT-REF: allow market calendar override via droplet .env
    market_calendar: str = Field(default="XNYS", alias="MARKET_CALENDAR")
    data_provider: str = Field(default="mock", alias="DATA_PROVIDER")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    enable_memory_optimization: bool = Field(default=True)
    alpaca_api_key: str = Field(default="test_key", alias="ALPACA_API_KEY")
    alpaca_secret_key: SecretStr = Field(
        default=SecretStr("test_secret"), alias="ALPACA_SECRET_KEY"
    )
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")
    # AI-AGENT-REF: include Alpaca base URL
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets", alias="ALPACA_BASE_URL"
    )
    # Expose canonical env key; alias resolution handled by resolver
    trading_mode: str = Field(default="balanced", alias="TRADING_MODE")

    REGIME_MIN_ROWS: int = Field(200, alias="REGIME_MIN_ROWS")
    data_warmup_lookback_days: int = Field(60, alias="DATA_WARMUP_LOOKBACK_DAYS")
    disable_daily_retrain: bool = Field(False, alias="DISABLE_DAILY_RETRAIN")
    enable_sklearn: bool = Field(False, alias="ENABLE_SKLEARN")
    intraday_lookback_minutes: int = Field(120, alias="INTRADAY_LOOKBACK_MINUTES")
    enable_numba_optimization: bool = Field(False, alias="ENABLE_NUMBA_OPTIMIZATION")
    alpaca_data_feed: str | None = Field(default=None, alias="ALPACA_DATA_FEED")
    scheduler_sleep_seconds: int = Field(60, alias="SCHEDULER_SLEEP_SECONDS")
    # Feature flags and thresholds
    data_cache_enable: bool = Field(True, env="AI_TRADER_DATA_CACHE_ENABLE")
    enable_plotting: bool = Field(False, env="AI_TRADER_ENABLE_PLOTTING")
    position_size_min_usd: float = Field(
        0.0, env="AI_TRADER_POSITION_SIZE_MIN_USD"
    )
    volume_threshold: float = Field(0.0, env="AI_TRADER_VOLUME_THRESHOLD")

    # AI-AGENT-REF: rebalancer defaults
    rebalance_interval_min: int = Field(
        15,
        env="AI_TRADER_REBALANCE_INTERVAL_MIN",
        description="How often (minutes) to consider portfolio rebalancing.",
    )
    rebalance_on_fill: bool = Field(
        True,
        env="AI_TRADER_REBALANCE_ON_FILL",
        description="Trigger a rebalance pass after order fills.",
    )
    rebalance_max_trades_per_cycle: int = Field(
        10,
        env="AI_TRADER_REBALANCE_MAX_TRADES_PER_CYCLE",
        description="Safety cap to limit rebalance churn.",
    )

    # AI-AGENT-REF: runtime defaults for deterministic behavior and loops
    seed: int | None = 42
    loop_interval_seconds: int = 60
    iterations: int = 0  # 0 => run forever
    api_port: int | None = 9001

    # AI-AGENT-REF: optional Finnhub API config
    # --- Finnhub (optional; tests may reference these) ---
    finnhub_api_key: str | None = Field(
        default=None,
        alias="FINNHUB_API_KEY",
        description="Finnhub API key; optional for local/dev & tests",
    )
    finnhub_base_url: str = Field(
        default="https://finnhub.io/api/v1",
        alias="FINNHUB_BASE_URL",
        description="Finnhub REST base URL",
    )

    model_config = SettingsConfigDict(
        env_prefix="AI_TRADING_",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def alpaca_secret_key_plain(self) -> str:
        return _secret_to_str(self.alpaca_secret_key) or ""

    @property
    def alpaca_headers(self) -> dict[str, str]:
        """Canonical Alpaca auth headers as plain strings."""  # AI-AGENT-REF: standard header builder
        return {
            "APCA-API-KEY-ID": self.alpaca_api_key or "",
            "APCA-API-SECRET-KEY": self.alpaca_secret_key_plain or "",
        }

    @property
    def scheduler_iterations(self) -> int:
        """Back-compat alias used by main.py and tests."""
        return self.iterations


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# AI-AGENT-REF: return mapping of broker keys with optional Finnhub
def broker_keys(s: Settings | None = None) -> dict[str, str]:
    s = s or get_settings()
    keys = {
        "ALPACA_API_KEY": s.alpaca_api_key,
        "ALPACA_SECRET_KEY": s.alpaca_secret_key_plain,
    }
    if s.finnhub_api_key:
        keys["finnhub"] = s.finnhub_api_key
    return keys

