from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Minimal project settings for tests and CI."""

    env: str = Field(default="test", alias="APP_ENV")
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
    bot_mode: str = Field(default="test", alias="BOT_MODE")

    # AI-AGENT-REF: runtime defaults for deterministic behavior and loops
    seed: int | None = 42
    loop_interval_seconds: int = 60
    iterations: int | None = 0  # 0 => run forever
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
        return self.alpaca_secret_key.get_secret_value()

    # AI-AGENT-REF: allow missing attributes to default to None
    def __getattr__(self, name: str) -> Any:  # pragma: no cover - simple fallback
        return None


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
    if getattr(s, "finnhub_api_key", None):
        keys["finnhub"] = s.finnhub_api_key
    return keys

